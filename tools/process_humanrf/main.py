import os
import sys
import json
import tqdm
import glob
import torch
import pickle
import numpy as np
import os.path as osp
from camera import read_calibration_csv 


def process_camera(root_dir, downscale_factor : int =4):
    """Process camera representation from HumanRF dataset to ZJU dataset.
       ZJU dataset camera is stored in the following format:
       {
           "K": 3x3 intrinsic
           "D": distortion (5,)=[k1, k2, p1, p2, k3]
           "R": 3x3 rotation,
           "T": 3x1 translation
           "w3c": 4x4
       }
    """
    cameras = read_calibration_csv(osp.join(root_dir, "calibration.csv"))
   
    # NOTE: use full resolution
    meta_info = {"cam_names": []}
    cameras_info =  {"D": [], "K": [], "R": [], "T": [], "w2c": []}
    for camera in cameras:
        # camera.camera
        camera = camera.get_downscaled_camera(downscale_factor=downscale_factor)
        c2w = camera.extrinsic_matrix_cam2world()   # (4, 4)
        w2c = np.array(np.linalg.inv(c2w))          # (4, 4)
        R = w2c[:3, :3].reshape(1, 3, 3)
        T = w2c[:3, -1].reshape(1, 3, 1)
        K = np.array(camera.intrinsic_matrix()).reshape(1, 3, 3)  # (3, 3) 
        w2c = w2c.reshape(1, 4, 4)
        D = np.array(
            [float(camera.k1), float(camera.k2), 0., 0., float(camera.k3)]
        ).reshape(1, 5)
        cameras_info["D"].append(D)
        cameras_info["K"].append(K)
        cameras_info["w2c"].append(w2c)
        cameras_info["R"].append(R)
        cameras_info["T"].append(T)
        print("=" * 80)
        print(camera.name)
        print("-" * 80)
        print(camera)
        print("=" * 80)
       
        # Add name to the list 
        meta_info["cam_names"].append(camera.name)
    for k in cameras_info.keys():
        cameras_info[k] = np.concatenate(cameras_info[k], axis=0)
    return cameras_info, meta_info


def process_imgs(root_dir, camera_names): 
    """ Put images file names (absolute path) into array of 
         #frames x #cameras of strings
    """
    
    # Figure out how many frames do we have
    # nframes = len(os.listdir(osp.join(root_dir, "rgbs", camera_names[0])))
    fids = set()
    pbar = tqdm.tqdm(glob.glob(
        "%s/*.jpg" % osp.join(root_dir, "rgbs", camera_names[0])))
    pbar.set_description("Valid:0")
    for fname in pbar:
        fid = int(osp.split(fname)[-1][len(f"{camera_names[0]}_rgb"):-len(".jpg")])
        has_all_data = True
        for cname in camera_names:
            rgb_fname = osp.join(
                root_dir, "rgbs", cname, f"{cname}_rgb{fid:06d}.jpg")
            mask_fname = osp.join(
                root_dir, "masks", cname, f"{cname}_mask{fid:06d}.png")
            if not (osp.isfile(rgb_fname) and osp.isfile(mask_fname)):
                has_all_data = False
                break
            
            pose_fname = osp.join(
                root_dir, "tava", "smpl", f"{fid:06d}.pkl"
            )
            try:
                np.load(pose_fname, allow_pickle=True)
            except:
                has_all_data = False
                break
        if has_all_data:
            fids.add(fid) 
            pbar.set_description("Valid:%d" % len(fids))
            
    fids = list(fids)
    fids = sorted(fids) # ascending order
    print("Found %d frame ids." % (len(fids)))
            
    img_out = []
    for fid in fids:
        curr = []
        for cname in camera_names:
            fname = osp.join(
                "rgbs", cname, f"{cname}_rgb{fid:06d}.jpg")
            assert osp.isfile(osp.join(root_dir, fname))
            curr.append(fname)
        img_out.append(curr)
    img_out = np.array(img_out, dtype=str)
    print(img_out.shape)
    
    mask_out = []
    for fid in fids:
        curr = []
        for cname in camera_names:
            fname = osp.join(
                "masks", cname, f"{cname}_mask{fid:06d}.png")
            assert osp.isfile(osp.join(root_dir, fname)), osp.join(
                root_dir, fname)
            curr.append(fname)
        mask_out.append(curr)
    mask_out = np.array(mask_out, dtype=str)
    print(mask_out.shape)
    return img_out, mask_out, {"fids": fids}


def process_pose(pose_dir, fids, rest_frame=0):
    """Processing pose into a format the codebase can use. Output format:
    
    """
    pose_data = {
        # Rest pose
        "lbs_weights": None,    # 6890 x 24
        "rest_verts": None,     # 6890 x 3
        "rest_joints": None,    # 24x3
        # "rest_tfs": None,       # 24x4x4,
        "rest_tfs_bone": None,  # 24x4x4
        # Per frame pose,
        "verts": [],            # N x 6890 x 3,
        "joints": [],           # N x 24 x 3,
        # "tfs": [],              # N x 24 x 4 x 4,
        "tf_bones": [],         # N x 24 x 4 x 4,
        "params": [],           # N x 78 (or dim input) - use beta here
    }
    # Load shape
    bodymodel = pickle.load(open(
        "/home/guandao/lagrangian_gaussian_splatting/smplx/body_models/smpl/SMPL_NEUTRAL.pkl", 
        "rb"), 
        encoding="latin1")
    pose_data["rest_verts"] = np.array(bodymodel["v_template"]) # 6890 x 3
    pose_data["rest_joints"] = np.array(bodymodel["J"]) # 24 x 3
    pose_data["rest_tfs_bone"] = np.repeat(np.eye(4)[None, ...], 24, axis=0)
    print("Processing pose...")
    for fid in tqdm.tqdm(fids):
        pose = np.load(osp.join(pose_dir, f"{fid:06d}.pkl"), allow_pickle=True)
        # All other frames
        # 1 x 6890 x 3
        pose_data["verts"].append(pose["vertices"].cpu().detach().numpy()) 
        # 1 x 45->24 x 3
        pose_data["joints"].append(pose["joints"][:, :24].cpu().detach().numpy()) 
        # 1 x 24 x 4 x 4
        pose_data["tf_bones"].append(pose["transform_mat"].cpu().detach().numpy())
        # 1 x 10
        pose_data["params"].append(pose["betas"].cpu().detach().numpy())
    # for k in ["verts", "joints", "tfs", "tf_bones", "params"]:
    for k in ["verts", "joints", "tf_bones", "params"]:
        # pose_data[k] = torch.cat(pose_data[k], axis=0)
        pose_data[k] = np.concatenate(pose_data[k], axis=0)
    return pose_data


def process_split(out_dir, subject_id, fids, cnames):
    os.makedirs(osp.join(out_dir, "splits"), exist_ok=True)
    if subject_id == 1:
        _train_fid_filter_ = lambda fid: fid < 460
        _train_cam_filter_ = lambda cname: int(cname[len("Cam"):]) not in [7]
        
        train_fids = [i for i, fid in enumerate(fids) 
                      if _train_fid_filter_(fid)]
        train_cams = [cid for cid, cname in enumerate(cnames)
                      if _train_cam_filter_(cname)]
        train_split = {"fids": train_fids, "cams": train_cams}
        np.save(osp.join(out_dir, "splits", "train.npy"), train_split)
        
        _test_fid_filter_ = lambda fid: fid >= 460 and fid <= 660
        _test_cam_filter_ = lambda cname: int(cname[len("Cam"):]) in [7]
        test_fids = [i for i, fid in enumerate(fids) 
                     if _test_fid_filter_(fid)]
        test_cams = [cid for cid, cname in enumerate(cnames)
                     if _test_cam_filter_(cname)]
        test_split = {"fids": test_fids, "cams": test_cams}
        np.save(osp.join(out_dir, "splits", "test.npy"), test_split)
        return {
            "train": train_split, 
            "test": test_split
        }
    elif subject_id == 2:
        _train_fid_filter_ = lambda fid: not (fid <= 930 and fid <= 1130)
        _train_cam_filter_ = lambda cname: int(cname[len("Cam"):]) not in [7]
        
        train_fids = [i for i, fid in enumerate(fids) if _train_fid_filter_(fid)]
        train_cams = [cid for cid, cname in enumerate(cnames)
                      if _train_cam_filter_(cname)]
        train_split = {"fids": train_fids, "cams": train_cams}
        np.save(osp.join(out_dir, "splits", "train.npy"), train_split)
        
        _test_fid_filter_ = lambda fid: fid >= 930 and fid <= 1130 
        _test_cam_filter_ = lambda cname: int(cname[len("Cam"):]) in [7]
        test_fids = [i for i, fid in enumerate(fids) if _test_fid_filter_(fid)]
        test_cams = [cid for cid, cname in enumerate(cnames)
                     if _test_cam_filter_(cname)]
        test_split = {"fids": test_fids, "cams": test_cams}
        np.save(osp.join(out_dir, "splits", "test.npy"), test_split)
        return {
            "train": train_split, 
            "test": test_split
        }
    else:
        raise NotImplemented


if __name__ == "__main__":
    # Process HumanRF data into a format similar to ZJU
    subject_id = int(sys.argv[1])
    root_dir = "data/actorhq_dataset/Actor%02d/Sequence1/4x/" % subject_id
    pose_dir = "data/actorhq_dataset/Actor%02d/Sequence1/4x/tava/smpl" % subject_id
    out_dir = "data/actorhq_dataset/Actor%02d/Sequence1/4x/tava/" % subject_id
    os.makedirs(out_dir, exist_ok=True)
    camera_info, camera_meta_info = process_camera(root_dir)
    img_fnames, mask_fnames, frames_meta_info = process_imgs(
        root_dir, camera_meta_info["cam_names"])
    annots = {
        "cams": camera_info,
        "ims": img_fnames,
        "masks": mask_fnames
    }
    np.save(osp.join(out_dir, "annots.npy"), annots)
    
    # Handle pose
    pose_data = process_pose(pose_dir, frames_meta_info["fids"])
    np.save(osp.join(out_dir, "pose_data.npy"), pose_data)
    
    # Split
    split_data = process_split(
        out_dir, subject_id, 
        frames_meta_info["fids"], 
        camera_meta_info["cam_names"])
    
    # Save all meta
    np.save(osp.join(out_dir, "meta_data.npy"), 
            {**camera_meta_info, **frames_meta_info, **split_data})
    print("Out dir:", out_dir)
    