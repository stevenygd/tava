import os
import sys
import torch
import trimesh
import os.path as osp
from tqdm import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra import initialize, compose
from torchmcubes import marching_cubes
from tava.utils.structures import namedtuple_map
from tava.utils.training import resume_from_ckpt
from tava.models.basic.mipnerf import cylinder_to_gaussian
# from tava.utils.bone import closest_distance_to_points

device = "cuda"

PROJECT_DIR = "/home/guandao/tava"

def build_model(subject_id = 2):
    # ARGS for method
    ARGS_TAVA_HUMANRF = [
        "dataset=humanrf", 
        f"dataset.subject_id={subject_id:02d}",
        # "dataset.post_fix='_qual'",
        "eval_splits=['test']",
        "pos_enc=snarf",
        "loss_bone_w_mult=1.0",
        "loss_bone_offset_mult=0.1",
        "engine=evaluator",
        "pos_enc.offset_net_enabled=true", 
        "model.shading_mode=implicit_AO",
        f"hydra.run.dir=/home/guandao/tava/pretrained/actor{subject_id:02d}"
    ]
    overrides = ["resume=True"] + ARGS_TAVA_HUMANRF 

    # create the cfg
    with initialize(config_path="../configs"):
        cfg = compose(config_name="mipnerf_dyn_humanrf", overrides=overrides, return_hydra_config=True)
        OmegaConf.resolve(cfg.hydra)
        save_dir = cfg.hydra.run.dir
        ckpt_dir = os.path.join(save_dir, "checkpoints")
        
    # initialize model and load ckpt
    model = instantiate(cfg.model).to(device)
    _ = resume_from_ckpt(
        path=ckpt_dir, model=model, step=cfg.resume_step, strict=True,
    )
    assert os.path.exists(ckpt_dir), "ckpt does not exist! Please check."

    # initialize dataset
    dataset = instantiate(
        cfg.dataset, split="train", num_rays=None, cache_n_repeat=None,
    )
    meta_data = dataset.build_pose_meta_info()
    return cfg, model, dataset, meta_data

# Create a grid coordinates to be queried in canonical.
def create_grid3D(min, max, steps, device="cpu"):
    if type(min) is int:
        min = (min, min, min) # (x, y, z)
    if type(max) is int:
        max = (max, max, max) # (x, y)
    if type(steps) is int:
        steps = (steps, steps, steps) # (x, y, z)
    arrangeX = torch.linspace(min[0], max[0], steps[0]).to(device)
    arrangeY = torch.linspace(min[1], max[1], steps[1]).to(device)
    arrangeZ = torch.linspace(min[2], max[2], steps[2]).to(device)
    gridX, girdY, gridZ = torch.meshgrid([arrangeX, arrangeY, arrangeZ], indexing="ij")
    coords = torch.stack([gridX, girdY, gridZ]) # [3, steps[0], steps[1], steps[2]]
    coords = coords.view(3, -1).t() # [N, 3]
    return coords


# cfg, model, dataset, meta_data = build_model(subject_id)
def extract_mesh(
    idx, model, dataset, meta_data, 
    res:int = 256, 
    chunk_size:int = 20000 * 25,
    thr=50
):
    # As the Mip-NeRF requires a covariance for density & color querying,
    # we here *estimate* a `cov` based on the size of the subject and the 
    # number of sampled points. It can be estimated in other ways such as
    # average of cov during training.
    radii = dataset[0]["rays"].radii.mean().to(device)
    rest_verts = torch.from_numpy(
        dataset.parser.load_meta_data()["verts"][idx]
    ).to(device)
    bboxs_min = rest_verts.min(dim=0).values
    bboxs_max = rest_verts.max(dim=0).values
    subject_size = torch.prod(bboxs_max - bboxs_min) ** (1./3.)
    t0, t1 = 0, subject_size / model.num_samples
    d = torch.tensor([0., 0., 1.]).to(device)
    _, cov = cylinder_to_gaussian(d, t0, t1, radii, model.diag)
    
    _center = (bboxs_max + bboxs_min) / 2.0
    _scale = (bboxs_max - bboxs_min) / 2.0
    bboxs_min_large = _center - _scale * 1.5
    bboxs_max_large = _center + _scale * 1.5
    coords = create_grid3D(bboxs_min_large, bboxs_max_large, res)
    
    # Query the density grid in the canonical
    bones_rest = namedtuple_map(lambda x: x.to(device), meta_data["bones_rest"])
    bones_posed = namedtuple_map(
        lambda x: x.to(device),
        meta_data["bones_posed"][idx])
    pose_latent = meta_data["pose_latent"][idx].to(device)
    rigid_clusters = torch.tensor(dataset.parser.RIGID_BONE_IDS).long().to(device)
    
    densities = []
    with torch.no_grad():
        for i in tqdm(range(0, coords.shape[0], chunk_size)):
            coords_chunk = coords[i: i + chunk_size].to(device)
            
            x_enc, _, mask, valid = model.pos_enc(
                coords_chunk[None, ...], 
                cov.expand_as(coords_chunk)[None, ...], 
                bones_posed, 
                bones_rest,
                rigid_clusters=rigid_clusters,
                pose_latent=pose_latent
            )
            
            _, raw_density, _ = model.mlp(
                x_enc[0], cond_extra=pose_latent, masks=mask[0])
            density = model.density_activation(raw_density + model.density_bias)
            density = torch.max(density, -2).values
            densities.append(density.cpu())
        densities = torch.cat(densities, dim=0).reshape(res, res, res)
    
        verts, faces = marching_cubes(densities, thr)
        verts = verts[..., [2, 1, 0]] / res * (bboxs_max_large - bboxs_min_large).cpu() + bboxs_min_large.cpu()

    return verts.cpu().numpy(), faces.cpu().numpy(), densities


if __name__ == "__main__":
    subject_id = int(sys.argv[1])
    output_dir = f"/home/guandao/tava/data/actorhq_dataset/evaluations/tava_meshes/actor{subject_id:02d}"
    cfg, model, dataset, meta_data = build_model(subject_id)
    n_shapes = dataset.parser.load_meta_data()["verts"].shape[0]
    
    all_fids = set()
    for idx in tqdm(range(n_shapes)):
        fid = meta_data["meta_ids"][idx]
        all_fids.add(fid)
        
    for fid in tqdm(all_fids):
        verts, faces, densities = extract_mesh(
            idx, model, dataset, meta_data, res=256, thr=5)
        if fid in all_fids:
            continue
        out_fname = osp.join(output_dir, f"mesh_{fid:06d}.obj")
        trimesh.export(out_fname, trimesh.Trimesh(verts, faces))