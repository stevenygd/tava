from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

try:
    # If this file is imported via blender, we can't use `scipy`
    import bpy
except ModuleNotFoundError:
    from scipy.spatial.transform import Rotation


@dataclass
class CameraData:
    """
    Our camera coordinate system uses right-down-forward (RDF) convention similarly to COLMAP.

    Right-handed coordinate system is adopted throughout this project, and vectors are represented as columns.
    So, the transformations need to be applied from the left-hand side, e.g., `t_vector = Matrix @ vector`

    Extrinsics represent the transformation from camera-space to world-space,
    i.e., `world_space = Rotation @ camera_space + Translation`.
    * The magnitude of `rotation_axisangle` defines the rotation angle in radians.
    * `translation` is typically stored in meters [m].
    """

    name: str
    width: int
    height: int

    # Extrinsics
    rotation_axisangle: np.array
    translation: np.array

    # Intrinsics
    focal_length: np.array
    principal_point: np.array

    # Optional distortion coefficients
    k1: float = 0
    k2: float = 0
    k3: float = 0

    @property
    def fx_pixel(self):
        return self.width * self.focal_length[0]

    @property
    def fy_pixel(self):
        return self.height * self.focal_length[1]

    @property
    def cx_pixel(self):
        return self.width * self.principal_point[0]

    @property
    def cy_pixel(self):
        return self.height * self.principal_point[1]

    def intrinsic_matrix(self):
        return np.array(
            [
                [self.fx_pixel, 0, self.cx_pixel],
                [0, self.fy_pixel, self.cy_pixel],
                [0, 0, 1],
            ]
        )

    def rotation_matrix_cam2world(self) -> np.array:
        """Set up the camera to world rotation matrix from the axis-angle representation.

        Returns:
            np.array (3 x 3): Rotation matrix going from camera to world space.
        """
        return Rotation.from_rotvec(self.rotation_axisangle).as_matrix()

    def extrinsic_matrix_cam2world(self) -> np.array:
        """Set up the camera to world transformation matrix to be applied on homogeneous coordinates.

        Returns:
            np.array (4 x 4): Transformation matrix going from camera to world space.
        """
        tfm_cam2world = np.eye(4)
        tfm_cam2world[:3, :3] = self.rotation_matrix_cam2world()
        tfm_cam2world[:3, 3] = self.translation

        return tfm_cam2world

    def projection_matrix_world2pixel(self):
        """Set up the world to pixel transformation matrix to project homogeneous coordinates onto image plane.

        Returns:
            np.array (4 x 4): Transformation matrix going from world to pixel space (division by Z-coordinate must be applied as the last step)
        """
        tfm_world2pixel = np.eye(4)
        tfm_world2pixel[:3] = self.intrinsic_matrix() @ np.linalg.inv(self.extrinsic_matrix_cam2world())[:3]

        return tfm_world2pixel

    def get_downscaled_camera(self, downscale_factor: int) -> CameraData:
        """Get a new `CameraData` object with the same parameters but downscaled by `downscale_factor` along each axis.
        This corresponds to our pre-processing of downscaled versions of the dataset.

        Args:
            scale (int): Downscale factor.

        Returns:
            CameraData: New `CameraData` object with downscaled parameters.
        """
        return CameraData(
            name=self.name,
            width=self.width // downscale_factor,
            height=self.height // downscale_factor,
            rotation_axisangle=self.rotation_axisangle,
            translation=self.translation,
            focal_length=self.focal_length,
            principal_point=self.principal_point,
            k1=self.k1,
            k2=self.k2,
            k3=self.k3,
        )


def write_calibration_csv(cameras: List[CameraData], output_csv_path: Path) -> None:
    """Write camera intrinsics and extrinsics to a calibration CSV file.

    Args:
        cameras (List[CameraData]): List of `CameraData` objects describing camera parameters.
        output_csv_path (Path): Path to the output CSV file.
    """
    csv_field_names = ["name", "w", "h", "rx", "ry", "rz", "tx", "ty", "tz", "fx", "fy", "px", "py"]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_field_names)
        writer.writeheader()

        for camera in cameras:
            csv_row = {}
            csv_row["name"] = camera.name
            csv_row["w"] = camera.width
            csv_row["h"] = camera.height
            csv_row["rx"] = camera.rotation_axisangle[0]
            csv_row["ry"] = camera.rotation_axisangle[1]
            csv_row["rz"] = camera.rotation_axisangle[2]
            csv_row["tx"] = camera.translation[0]
            csv_row["ty"] = camera.translation[1]
            csv_row["tz"] = camera.translation[2]
            csv_row["fx"] = camera.focal_length[0]
            csv_row["fy"] = camera.focal_length[1]
            csv_row["px"] = camera.principal_point[0]
            csv_row["py"] = camera.principal_point[1]

            assert len(csv_row) == len(csv_field_names)
            writer.writerow(csv_row)


def read_calibration_csv(input_csv_path: Path) -> List[CameraData]:
    """Read camera intrinsics and extrinsics from a calibration CSV file.

    Args:
        input_csv_path (Path): Path to a CSV file that contains camera calibration data.

    Returns:
        List[CameraData]: A list of `CameraData` objects that describe multiple camera intrinsics and extrinsics.
    """
    cameras = []
    with open(input_csv_path, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            camera = CameraData(
                name=row["name"],
                width=int(row["w"]),
                height=int(row["h"]),
                rotation_axisangle=np.array([float(row["rx"]), float(row["ry"]), float(row["rz"])]),
                translation=np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])]),
                focal_length=np.array([float(row["fx"]), float(row["fy"])]),
                principal_point=np.array([float(row["px"]), float(row["py"])]),
            )
            cameras.append(camera)
    return cameras