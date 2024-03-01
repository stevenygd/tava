# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import re
import cv2
import numpy as np
import torch
from tava.datasets.abstract import CachedIterDataset
from tava.datasets.humanrf_parser import SubjectParser
from tava.utils.camera import generate_rays, transform_cameras
from tava.utils.structures import Bones, Cameras, namedtuple_map
from tava.utils.transforms import axis_angle_to_matrix, matrix_to_rotation_6d


def _dataset_view_split(parser: SubjectParser, split: str):
    if split == "all":
        camera_ids = parser._camera_ids
    elif re.match("cam[0-9]*_fid[0-9]*", split):
        meta_data = np.load(
            os.path.join(parser.root_dir, "tava/meta_data.npy"),
            allow_pickle=True
        ).item()
        cname = "Cam%03d" % int(split.split("_")[0][len("cam"):])
        cid = meta_data["cam_names"].index(cname)
        camera_ids = [cid]
    else:
        splits_fp = os.path.join(parser.root_dir, f"tava/splits/{split}.npy")
        splits_info = np.load(splits_fp, allow_pickle=True).item()
        camera_ids = splits_info["cams"]
    return camera_ids


def _dataset_frame_split(parser: SubjectParser, split: str):
    if split == "all":
        frame_ids = parser._frame_ids
    elif re.match("cam[0-9]*_fid[0-9]*", split):
        meta_data = np.load(
            os.path.join(parser.root_dir, "tava/meta_data.npy"),
            allow_pickle=True
        ).item()
        fname = int(split.split("_")[0][len("fid"):])
        fid = meta_data["fids"].index(fname)
        frame_ids = [fid]
    else:
        splits_fp = os.path.join(parser.root_dir, f"tava/splits/{split}.npy")
        splits_info = np.load(splits_fp, allow_pickle=True).item()
        frame_ids = splits_info["fids"]
    return frame_ids

def _dataset_index_list(parser, split):
    camera_ids = _dataset_view_split(parser, split)
    frame_list = _dataset_frame_split(parser, split)
    index_list = []
    for frame_id in frame_list:
        index_list.extend([(frame_id, camera_id) for camera_id in camera_ids])
    return index_list


class SubjectLoader(CachedIterDataset):
    """Single subject data loader for training and evaluation."""

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        resize_factor: float = 1.0,
        color_bkgd_aug: str = None,
        num_rays: int = None,
        cache_n_repeat: int = 0,
        near: float = None,
        far: float = None,
        legacy: bool = False,
        **kwargs,
    ):
        assert color_bkgd_aug in ["white", "black", "random"]
        # self.resize_factor = resize_factor
        self.resize_factor = 1
        assert self.resize_factor == 1
        self.split = split
        self.num_rays = num_rays
        self.near = near
        self.far = far
        self.legacy = legacy
        self.training = (num_rays is not None) and (split in ["train", "all"])
        self.color_bkgd_aug = color_bkgd_aug
        self.parser = SubjectParser(subject_id=subject_id, root_fp=root_fp)
        self.index_list = _dataset_index_list(self.parser, split)
        self.dtype = torch.get_default_dtype()
        super().__init__(self.training, cache_n_repeat)

    def __len__(self):
        return len(self.index_list)

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        image, alpha = torch.split(rgba, [3, 1], dim=-1)
        # print("rgba", rgba.shape)
        # "origins", "directions", "viewdirs"
        # print("rays", rays.origins.shape, 
        #       rays.directions.shape, rays.viewdirs.shape)
        # print("image", image.shape)
        # print("alpha", alpha.shape)

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, dtype=rgba.dtype)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, dtype=rgba.dtype)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, dtype=rgba.dtype)
        else:
            # just use black during inference
            color_bkgd = torch.zeros(3, dtype=rgba.dtype)

        # only replace regions with `alpha == 0` to `color_bkgd`
        image = image * (alpha != 0) + color_bkgd * (alpha == 0)

        if self.num_rays is not None:  # usually this is in the training phase
            resolution = image.shape[0] * image.shape[1]
            # only sample rays in regions with `alpha == 0 or 1`
            indices = torch.where(
                ((alpha == 0) | (alpha == 1)).reshape(resolution)
            )[0]
            ray_indices = indices[torch.randperm(len(indices))][: self.num_rays]
            pixels = image.reshape(resolution, 3)[ray_indices]
            rays = namedtuple_map(
                lambda r: r.reshape([resolution] + list(r.shape[2:])), rays
            )
            rays = namedtuple_map(lambda x: x[ray_indices], rays)
        else:
            pixels = image

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        # load data
        frame_id, camera_id = self.index_list[index]
        K = self.parser.cameras["K"][camera_id].copy()
        w2c = self.parser.cameras["w2c"][camera_id].copy()
        D = self.parser.cameras["D"][camera_id].copy()
        height = int(self.parser.cameras["heights"][camera_id].copy())
        width = int(self.parser.cameras["widths"][camera_id].copy())

        # create pixels
        rgba = np.concatenate(
            [
                self.parser.load_image(frame_id, camera_id),
                self.parser.load_mask(frame_id, camera_id, trimap=True)[
                    ..., None
                ],
            ],
            axis=-1,
        )
        rgba = (
            torch.from_numpy(
                cv2.resize(
                    cv2.undistort(rgba, K, D),
                    (0, 0),
                    fx=self.resize_factor,
                    fy=self.resize_factor,
                    interpolation=cv2.INTER_AREA,
                )
            ).to(self.dtype)
            / 255.0
        )

        # create rays from camera
        cameras = Cameras(
            intrins=torch.from_numpy(K).to(self.dtype),
            extrins=torch.from_numpy(w2c).to(self.dtype),
            distorts=None,
            # width=self.parser.WIDTH,
            # height=self.parser.HEIGHT,
            width=width,
            height=height,
        )
        cameras = transform_cameras(cameras, self.resize_factor)
        rays = generate_rays(
            cameras, opencv_format=True, near=self.near, far=self.far
        )

        return {
            "subject_id": self.parser.subject_id,
            "camera_id": camera_id,
            # `meta_id` is used to query pose info from `pose_meta_info`
            "meta_id": frame_id,
            "rgba": rgba,  # [h, w, 4]
            "rays": rays,  # [h, w]
            "rigid_clusters": torch.tensor(
                self.parser.RIGID_BONE_IDS
            ).long(),  # bone cluster ids
        }

    def build_pose_meta_info(self):
        # create indexing for this split
        meta_ids = [int(frame_id) for frame_id, _ in self.index_list]
        meta_data = self.parser.load_meta_data(frame_ids=meta_ids)

        # load canonical meta info.
        rest_matrixs = meta_data["rest_tfs_bone"][1:]  # [23, 4, 4]
        rest_tails = meta_data["rest_joints"][
            [
                self.parser.JOINT_NAMES.index(tail_name)
                for _, tail_name in self.parser.BONE_NAMES
            ]
        ]  # [23, 3]
        rest_heads = meta_data["rest_joints"][
            [
                self.parser.JOINT_NAMES.index(head_name)
                for head_name, _ in self.parser.BONE_NAMES
            ]
        ]  # [23, 3]
        bones_rest = Bones(
            heads=torch.from_numpy(rest_heads).to(self.dtype),
            tails=torch.from_numpy(rest_tails).to(self.dtype),
            transforms=torch.from_numpy(rest_matrixs).to(self.dtype),
        )  # real bones [23,]

        # load view space meta info.
        pose_matrixs = meta_data["tf_bones"][:, 1:]  # [N, 23, 4, 4]
        pose_heads = meta_data["joints"][
            :,
            [
                self.parser.JOINT_NAMES.index(head_name)
                for head_name, _ in self.parser.BONE_NAMES
            ],
        ]  # [N, 23, 3]
        pose_tails = meta_data["joints"][
            :,
            [
                self.parser.JOINT_NAMES.index(tail_name)
                for _, tail_name in self.parser.BONE_NAMES
            ],
        ]  # [N, 23, 3]
        bones_posed = [
            Bones(
                heads=torch.from_numpy(pose_heads[i]).to(self.dtype),
                tails=torch.from_numpy(pose_tails[i]).to(self.dtype),
                transforms=torch.from_numpy(pose_matrixs[i]).to(self.dtype),
            )
            for i in range(len(meta_ids))
        ]  # [23,] * N
        pose_latent = torch.from_numpy(meta_data["params"]).to(self.dtype)
        # if self.legacy:
        #     # The paper uses axis-angles as pose latent. However it can
        #     # be discontinuous.
        #     breakpoint()
        #     pose_latent = torch.from_numpy(meta_data["params"]).to(self.dtype)
        # else:
        #     raise NotImplemented
        #     # An improved version is to use the 6D rotation? (not verified)
        #     pose_latent = torch.from_numpy(meta_data["params"]).to(self.dtype)
        #     _aa, _g_aa, _g_transl = torch.split(pose_latent, [72, 3, 3], dim=-1)
        #     assert (
        #         _aa[:, 0:3] == 0
        #     ).all()  # ZJU diable root rotation by default.
        #     _aa[
        #         :, 0:3
        #     ] = _g_aa  # write the global rotation into the root rotation.
        #     pose_latent = torch.cat(
        #         [
        #             matrix_to_rotation_6d(
        #                 axis_angle_to_matrix(_aa.reshape(-1, 24, 3))
        #             ).reshape(-1, 24 * 6),
        #             _g_transl,
        #         ],
        #         dim=-1,
        #     )
        return {
            "meta_ids": meta_ids,
            "bones_rest": bones_rest,
            "bones_posed": bones_posed,
            "pose_latent": pose_latent,
        }
