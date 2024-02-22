# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

import cv2
import imageio
import numpy as np
import torch


class SubjectParser:
    """Single subject data parser."""

    WIDTH = 747 
    HEIGHT = 1022

    # Joint names from SMLP
    JOINT_NAMES = [
        "root",
        "lhip",
        "rhip",
        "belly",
        "lknee",
        "rknee",
        "spine",
        "lankle",
        "rankle",
        "chest",
        "ltoes",
        "rtoes",
        "neck",
        "linshoulder",
        "rinshoulder",
        "head",
        "lshoulder",
        "rshoulder",
        "lelbow",
        "relbow",
        "lwrist",
        "rwrist",
        "lhand",
        "rhand",
    ]

    BONE_NAMES = [
        ("root", "lhip"),
        ("root", "rhip"),
        ("root", "belly"),
        ("lhip", "lknee"),
        ("rhip", "rknee"),
        ("belly", "spine"),
        ("lknee", "lankle"),
        ("rknee", "rankle"),
        ("spine", "chest"),
        ("lankle", "ltoes"),
        ("rankle", "rtoes"),
        ("chest", "neck"),
        ("chest", "linshoulder"),
        ("chest", "rinshoulder"),
        ("neck", "head"),
        ("linshoulder", "lshoulder"),
        ("rinshoulder", "rshoulder"),
        ("lshoulder", "lelbow"),
        ("rshoulder", "relbow"),
        ("lelbow", "lwrist"),
        ("relbow", "rwrist"),
        ("lwrist", "lhand"),
        ("rwrist", "rhand"),
    ]

    RIGID_BONE_IDS = [
        0,
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        9,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    ]

    def __init__(self, subject_id: int, root_fp: str):

        if not root_fp.startswith("/"):
            # allow relative path. e.g., "./data/zju/"
            root_fp = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                "..", "..",
                root_fp,
            )

        self.subject_id = subject_id
        self.root_fp = root_fp
        self.root_dir = os.path.join(
            root_fp, "Actor0%d" % int(subject_id), "Sequence1", "4x")

        self.mask_dir = os.path.join(self.root_dir)
        self.splits_dir = os.path.join(self.root_dir, "splits")

        annots_fp = os.path.join(self.root_dir, "tava", "annots.npy")
        annots_data = np.load(annots_fp, allow_pickle=True).item()
        # K/D/R/T/w2c : #came x (3x3) / (5,) / (3,3) / (3,1) / (4x4)
        self.cameras = annots_data["cams"]
        # [1000 x 160], #frames x #cameras
        self.image_files = annots_data["ims"]
        self._frame_ids = list(range(self.image_files.shape[0]))
        self._camera_ids = list(range(self.image_files.shape[1]))
        
        # [1000 x 160], #frames x #cameras
        self.mask_files = annots_data["masks"]

    @property
    def camera_ids(self):
        return self._camera_ids

    @property
    def frame_ids(self):
        return self._frame_ids

    def load_image(self, frame_id, camera_id):
        path = os.path.join(
            self.root_dir, self.image_files[frame_id, camera_id]
        )
        image = imageio.imread(path)
        return image  # shape [HEIGHT, WIDTH, 3], value 0 ~ 255

    def load_mask(self, frame_id, camera_id, trimap=True):
        path = os.path.join(
            self.mask_dir, self.mask_files[frame_id, camera_id]
        )
        mask = (imageio.imread(path) != 0).astype(np.uint8) * 255
        if trimap:
            mask = self._process_mask(mask)
        return mask  # shape [HEIGHT, WIDTH], value [0, (128,) 255]

    def load_meta_data(self, frame_ids=None):
        data = np.load(
            os.path.join(self.root_dir, "tava", "pose_data.npy"),
            allow_pickle=True
        ).item()
        keys = [
            # "lbs_weights",
            "rest_verts",
            "rest_joints",
            # "rest_tfs",
            "rest_tfs_bone",
            "verts",
            "joints",
            # "tfs",
            "tf_bones",
            "params",
        ]
        return {
            key: (
                data[key][frame_ids]
                if (
                    frame_ids is not None
                    and key in ["verts", "joints", "tf_bones", "params"]
                )
                else data[key]
            )
            for key in keys
        }

    def _process_mask(self, mask, border: int = 5, ignore_value: int = 128):
        kernel = np.ones((border, border), np.uint8)
        mask_erode = cv2.erode(mask.copy(), kernel)
        mask_dilate = cv2.dilate(mask.copy(), kernel)
        mask[mask_dilate != mask_erode] = ignore_value
        return mask

