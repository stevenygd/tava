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
    cameras_info =  {"D": [], "K": [], "R": [], "T": [], "w2c": [], 
                     "heights": [], "widths": []}
    for camera in cameras:
        # camera.camera
        camera = camera.get_downscaled_camera(downscale_factor=downscale_factor)
        c2w = camera.extrinsic_matrix_cam2world()   # (4, 4)
        w2c = np.array(np.linalg.inv(c2w))          # (4, 4)
        R = w2c[:3, :3].reshape(1, 3, 3)
        T = w2c[:3, -1].reshape(1, 3, 1)
        K = np.array(camera.intrinsic_matrix()).reshape(1, 3, 3)  # (3, 3) 
        w2c = np.array(
            [[1, 0, 0, 2.9691],
             [0, 1, 0, 0.4892],
             [0, 0, 1, -0.539],
             [0, 0, 0, 1],
            ]
        )
        w2c.reshape(1, 4, 4)
        D = np.array(
            [float(camera.k1), float(camera.k2), 0., 0., float(camera.k3)]
        ).reshape(1, 5)
        cameras_info["D"].append(D)
        cameras_info["K"].append(K)
        cameras_info["w2c"].append(w2c)
        cameras_info["R"].append(R)
        cameras_info["T"].append(T)
        cameras_info["heights"].append(camera.height * np.ones((1,)))
        cameras_info["widths"].append(camera.width * np.ones((1,)))
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


def process_imgs(root_dir, camera_names, actor_id: int = 2): 
    """ Put images file names (absolute path) into array of 
         #frames x #cameras of strings
    """
    if actor_id == 2:
        fids = [60, 100, 205]
    else:
        raise NotImplemented
    
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
            
    return img_out, mask_out, {"fids": fids}


def process_pose(pose_dir, fids, rest_frame=0):
    """Processing pose into a format the codebase can use. """
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
    # # Load shape
    bodymodel = pickle.load(open(
        "/home/guandao/lagrangian_gaussian_splatting/smplx/body_models/smpl/SMPL_NEUTRAL.pkl", 
        "rb"), 
        encoding="latin1")
    pose_data["lbs_weights"] = bodymodel["weights"].astype(np.float32)
    print("Processing pose...")
    for fid in tqdm.tqdm(fids):
        if rest_frame == fid:
            pose = np.load(osp.join(pose_dir, f"{fid:04d}.pkl"), allow_pickle=True)
            pose_data["rest_verts"] = pose["vertices"].cpu().detach().numpy().reshape(6890, 3)
            pose_data["rest_joints"] = pose["joints"][:, :24].cpu().detach().numpy().reshape(24, 3)
            tf_bones = pose["bone_transform_mat"][0] # 24x4x4
            global_transform = torch.eye(4)
            global_transform[:3, -1] = pose["transl"][0]
            global_transform = global_transform[None, ...].repeat(
                tf_bones.shape[0], 1, 1)
            tf_bones = torch.bmm(global_transform.to(tf_bones.device), tf_bones)
            tf_bones = tf_bones.cpu().detach().numpy().reshape(24, 4, 4)
            pose_data["rest_tfs_bone"] = tf_bones
        pose = np.load(osp.join(pose_dir, f"{fid:04d}.pkl"), allow_pickle=True)
        # All other frames
        # 1 x 6890 x 3
        pose_data["verts"].append(pose["vertices"].cpu().detach().numpy()) 
        # 1 x 45->24 x 3
        pose_data["joints"].append(pose["joints"][:, :24].cpu().detach().numpy()) 
        # 1 x 24 x 4 x 4
        tf_bones = pose["bone_transform_mat"][0] # 24x4x4
        global_transform = torch.eye(4)
        global_transform[:3, -1] = pose["transl"][0]
        global_transform = global_transform[None, ...].repeat(
            tf_bones.shape[0], 1, 1)
        tf_bones = torch.bmm(global_transform.to(tf_bones.device), tf_bones)
        pose_data["tf_bones"].append(tf_bones[None, ...].cpu().detach().numpy())
        # 1 x 10
        pose_data["params"].append(pose["betas"].cpu().detach().numpy())
    for k in ["verts", "joints", "tf_bones", "params"]:
        pose_data[k] = np.concatenate(pose_data[k], axis=0)
    return pose_data


def process_split(out_dir, subject_id, fids, cnames):
    os.makedirs(osp.join(out_dir, "splits"), exist_ok=True)
    
    def make_split(train_fid_filter, train_cam_filter):
        train_fids = [i for i, fid in enumerate(fids) 
                      if train_fid_filter(fid)]
        train_cams = [cid for cid, cname in enumerate(cnames)
                      if train_cam_filter(cname)]
        train_split = {"fids": train_fids, "cams": train_cams}
        
        test_fids = [i for i, fid in enumerate(fids) 
                     if not train_fid_filter(fid)]
        test_cams = [cid for cid, cname in enumerate(cnames)
                     if not train_cam_filter(cname)]
        test_split = {"fids": test_fids, "cams": test_cams}
        return {
            "train": train_split, 
            "test": test_split
        }
    if subject_id == 2:
        _train_fid_filter_ = lambda fid: True
        _train_cam_filter_ = lambda cname: True
        
    else:
        raise NotImplemented
    
    split_info = make_split(_train_fid_filter_, _train_cam_filter_)
    np.save(osp.join(out_dir, "splits", "train.npy"), split_info["train"])
    np.save(osp.join(out_dir, "splits", "test.npy"), split_info["test"])
    return split_info


if __name__ == "__main__":
    # Process HumanRF data into a format similar to ZJU
    path = "/home/guandao/tava/data/actorhq_dataset"
    subject_id = int(sys.argv[1])
    root_dir = f"{path}/Actor{subject_id:02d}_qual/Sequence1/4x/"
    pose_dir = f"{path}/Actor{subject_id:02d}_qual/Sequence1/4x/tava/smpl"
    out_dir = f"{path}/Actor{subject_id:02d}_qual/Sequence1/4x/tava/"
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
    pose_data = process_pose(pose_dir, frames_meta_info["fids"], 
                             rest_frame=frames_meta_info["fids"][0])
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
    