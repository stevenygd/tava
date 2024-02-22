import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math
import torch
import numpy as np
import cv2
import smplx
import pickle
import trimesh
import os
from tqdm import tqdm

import trimesh

from sklearn.preprocessing import normalize
from human_body_prior.train.vposer_smpl import VPoser
import time
import torch.nn.functional as F


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))



class SmplxDeformer():
    def __init__(self, gender='neutral'):
        # self.renderer = pyrender.OffscreenRenderer(viewport_width=img_size[0], viewport_height=img_size[1])
        self.gender = gender
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.smplx_model = smplx.SMPLX(model_path='./data/body_models/smplx', ext='npz', gender=gender,
                                      num_betas=300, num_expression_coeffs=100, use_face_contour=False,
                                      use_pca=False).eval().to(self.device)
        self.extra_jregressor = torch.tensor(np.load('./data/body_models/J_regressor_extra_smplx.npy'),
                                             dtype=torch.float32).to(self.device)
        self.vposer = VPoser(512, 32, [3, 21]).eval().to(self.device)
        self.vposer.load_state_dict(torch.load('./data/body_models/TR00_E096.pt', map_location='cpu'))

    def read_obj(self, filename):
        # Parse the .obj file and return vertices and indices
        # This is a placeholder function, you need to implement the parsing
        # according to the .obj file format.
        vertices = []
        indices = []
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith('v '):  # This line describes a vertex
                    parts = line.strip().split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):  # This line describes a face
                    parts = line.strip().split()
                    face_indices = [int(p.split('/')[0]) - 1 for p in parts[1:]]  # OBJ indices start at 1
                    indices.append(face_indices)
            vertices = np.array(vertices, dtype=np.float32)
            indices = np.array(indices, dtype=np.int32)

        return vertices, indices

    def save_obj(self, filename, v, f):
        with open(filename, 'w') as fp:
            for vi in v:
                fp.write('v %f %f %f\n' % (vi[0], vi[1], vi[2]))
            for fi in f:
                fp.write('f %d %d %d\n' % (fi[0] + 1, fi[1] + 1, fi[2] + 1))
        fp.close()

    def smplx_forward(self, smplx_param):

        body_pose = matrix_to_axis_angle(
            self.vposer.decode(smplx_param['latent']).view(-1, 3, 3)
        ).view(1, -1)
        smplx_out = self.smplx_model.forward(
            transl=smplx_param['trans'],
            global_orient=smplx_param['orient'],
            body_pose=body_pose,
            betas=smplx_param['beta'],
            left_hand_pose=smplx_param['left_hand_pose'],
            right_hand_pose=smplx_param['right_hand_pose'],
            expression=smplx_param['expr'],
            jaw_pose=smplx_param['jaw_pose'],
            leye_pose=smplx_param['left_eye_pose'],
            reye_pose=smplx_param['right_eye_pose']
        )
        breakpoint()
        extra_joints = torch.einsum('bik,ji->bjk', smplx_out.vertices, self.extra_jregressor)

        smplx_out.joints = torch.cat([smplx_out.joints, extra_joints], dim=1)

        scale = smplx_param['scale']
        if len(scale.shape):
            scale = scale.unsqueeze(dim=1)
        smplx_out.joints *= scale
        smplx_out.vertices *= scale
        return smplx_out

    def export(self, smplx_param, save_path):
        smplx_out = self.smplx_forward(smplx_param)
        with open(save_path, 'w') as smplx_save:
            for v in smplx_out.vertices[0].detach().cpu().numpy():
                smplx_save.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
            for f in self.smplx_model.faces:
                smplx_save.write('f {} {} {}\n'.format(f[0] + 1, f[1] + 1, f[2] + 1))

    def transform_to_t_pose(self, vertices, smplx_output, global_transl=None, 
                            scale=None, lbs_w=None):
        """
        Transform the given mesh to the T-pose using the inverse transformations of the SMPLX model.

        Args:
            vertices (torch.Tensor): The vertices of the mesh to be transformed.
            smplx_output (smplx.SMPLXOutput): The output of the SMPLX forward pass.
            weights (torch.Tensor): The blend skinning weights for each vertex.
            point_idx (torch.Tensor): Indices of the closest points in the SMPLX model.
        Returns:
            torch.Tensor: The transformed vertices in the T-pose.
        """
        if lbs_w is None:
            transformation_matrix = smplx_output.transform_mat.to(self.device)
            lbs_weights_packed = self.smplx_model.lbs_weights  # (V, J+1)
            point_v_weights, point_v_idxs = self.weights_from_k_closest_verts(vertices, smplx_output.vertices, k=5, p=2)

            lbs_weights_pnts = torch.stack(
                [lbs_weights_packed[idxs] for idxs in point_v_idxs]
            )  # (bs, P, K, J+1)

            lbs_weights_pnts = torch.einsum(
                "npkj,npk->npj", lbs_weights_pnts, point_v_weights
            )
            B, num_points = point_v_weights.shape[:2]
            assert lbs_weights_pnts.shape == (
                B,
                num_points,
                self.smplx_model.lbs_weights.shape[-1],
            )
            # perform lbs
            # (N x V x (J + 1)) x (N x (J + 1) x 16)
            num_joints = self.smplx_model.J_regressor.shape[0]
            W = lbs_weights_pnts
        else:
            W = lbs_w
            transformation_matrix = smplx_output.transform_mat.to(self.device)
            num_joints = self.smplx_model.J_regressor.shape[0]
            B, num_points, _ = lbs_w.shape


        T = torch.matmul(W, transformation_matrix.reshape((-1, B, num_joints, 16))).reshape((-1, B, num_points, 4, 4))
        T = torch.inverse(T)

        if len(scale.shape):
            scale = scale.unsqueeze(dim=1)
        vertices = vertices / scale
        if len(vertices.shape) == 3:
            global_transl = global_transl.unsqueeze(dim=1)
        vertices = vertices - global_transl
        src_verts = vertices.reshape((-1, num_points, 3)) #self.smplx_model.v_template.unsqueeze(dim=0).repeat(B, 1, 1)

        num_verts = src_verts.shape[-2]

        homogen_coord = torch.ones_like(
            src_verts.reshape((-1, B, num_verts, 3))[..., :1],
            dtype=src_verts.dtype,
            device=src_verts.device,
        )

        v_posed_homo = torch.cat([src_verts.reshape((-1, B, num_verts, 3)), homogen_coord], dim=-1)

        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))


        deformed_pnts = v_homo[..., :3, 0]

        return deformed_pnts, T, W

    def transform_to_pose(self, vertices, lbs_weights_pnts, smplx_output,  
                          global_transl=None, scale=None):
        """
        Transform the given mesh to the T-pose using the inverse transformations of the SMPLX model.

        Args:
            vertices (torch.Tensor): The vertices of the mesh to be transformed.
            smplx_output (smplx.SMPLXOutput): The output of the SMPLX forward pass.
            weights (torch.Tensor): The blend skinning weights for each vertex.
            point_idx (torch.Tensor): Indices of the closest points in the SMPLX model.
        Returns:
            torch.Tensor: The transformed vertices in the T-pose.
        """

        transformation_matrix = smplx_output.transform_mat.to(self.device)
        B, num_points, _ = lbs_weights_pnts.shape
        # perform lbs
        # (N x V x (J + 1)) x (N x (J + 1) x 16)
        num_joints = self.smplx_model.J_regressor.shape[0]
        W = lbs_weights_pnts

        T = torch.matmul(W, transformation_matrix.reshape((-1, B, num_joints, 16))).reshape((-1, B, num_points, 4, 4))

        src_verts = vertices.reshape((-1, num_points, 3)) #self.smplx_model.v_template.unsqueeze(dim=0).repeat(B, 1, 1)

        num_verts = src_verts.shape[-2]

        homogen_coord = torch.ones_like(
            src_verts.reshape((-1, B, num_verts, 3))[..., :1],
            dtype=src_verts.dtype,
            device=src_verts.device,
        )

        v_posed_homo = torch.cat([src_verts.reshape((-1, B, num_verts, 3)), homogen_coord], dim=-1)

        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))


        deformed_pnts = v_homo[..., :3, 0]

        if global_transl is not None:
            deformed_pnts += global_transl.to(deformed_pnts.device).unsqueeze(dim=1)

        if scale is not None:
            if len(scale.shape):
                scale = scale.unsqueeze(dim=1)
            deformed_pnts *= scale

        return deformed_pnts, T

if __name__ == '__main__':

    import sys
    folder = sys.argv[1]
    smplx_deformer = SmplxDeformer(gender='neutral')


    smplx_param0 = torch.load(f'{folder}/smplx_icp_param.pth')
    smplx_param0 = {k: torch.from_numpy(v).to(smplx_deformer.device) for k, v in smplx_param0.items()}
    smplx_deformer.export(smplx_param0, './output/exp1_cloth/a1_s1_lbs/smplx_icp_ori.obj')

    smplx_param = torch.load(f'./output/exp1_cloth/a1_s1_lbs/smplx_param.pth')
    smplx_param0 = {k: torch.from_numpy(v[141:142]).to(smplx_deformer.device) for k, v in smplx_param.items()}
    smplx_deformer.export(smplx_param0, './output/exp1_cloth/a1_s1_lbs/smplx_icp_opt.obj')


    lbs_w = torch.load(f'./output/exp1_cloth/a1_s1_lbs/lbs_w.pth')['lbs_w']
    human_v, human_f = smplx_deformer.read_obj(f'./output/exp1_cloth/a1_s1_lbs/extracted_mesh/141.obj')
    human_v = torch.from_numpy(human_v).to(smplx_deformer.device).unsqueeze(0)

    smplx = smplx_deformer.smplx_forward(smplx_param0)

    t_human_v, transform_matrix, _ = smplx_deformer.transform_to_t_pose(human_v, smplx, smplx_param0['trans'],
                                                                            smplx_param0['scale'], lbs_w.unsqueeze(0))
    print(lbs_w.shape, human_v.shape, t_human_v.shape)
    smplx_deformer.save_obj('./output/exp1_cloth/a1_s1_lbs/human_tpose_.obj', t_human_v.squeeze().detach().cpu().numpy(), human_f)

    # smplx_param1 = torch.load(f'./data/a1_s1/smplx_fitted_icp/001209/smplx_icp_param.pth')
    smplx_param1 = {k: torch.from_numpy(v[:1]).to(smplx_deformer.device) for k, v in smplx_param.items()} #{k: torch.from_numpy(v).to(smplx_deformer.device) for k, v in smplx_param1.items()}
    smplx1 = smplx_deformer.smplx_forward(smplx_param1)
    t_human_v = t_human_v.squeeze().unsqueeze(0)
    t_human_v1, transform_matrix1 = smplx_deformer.transform_to_pose(t_human_v, lbs_w.unsqueeze(0), smplx1, smplx_param1['trans'], smplx_param1['scale'])
    smplx_deformer.save_obj('./output/exp1_cloth/a1_s1_lbs/human_409.obj', t_human_v1.squeeze().detach().cpu().numpy(), human_f)
    # smplx_model.export(smplx_param0, '../data/a1_s1/smplx_icp.obj')
    # for i in tqdm(range(10)):
    #     smplx_param0 = torch.load(f'./data/a1_s1/smplx_fitted_icp/000{409}/smplx_icp_param.pth')
    #     smplx_param0 = {k: torch.from_numpy(v).to(smplx_deformer.device) for k, v in smplx_param0.items()}
    #
    #     smplx_param1 = torch.load(f'./data/a1_s1/smplx_fitted_icp/000{409+i+1}/smplx_icp_param.pth')
    #     smplx_param1 = {k: torch.from_numpy(v).to(smplx_deformer.device) for k, v in smplx_param1.items()}
    #     with torch.no_grad():
    #         smplx = smplx_deformer.smplx_forward(smplx_param0)
    #         smplx1 = smplx_deformer.smplx_forward(smplx_param1)
    #
    #
    #         human_mesh = smplx_deformer.read_obj(f'./data/a1_s1/Frame000{409}_50k.obj')
    #         # human_mesh = smplx_deformer.read_obj('../data/a1_s1/human_tpose_.obj')
    #
    #         human_v, human_f = human_mesh
    #
    #         human_v = torch.from_numpy(human_v).to(smplx_deformer.device).unsqueeze(0)
    #
    #
    #         # transform the human mesh to the T pose
    #         t_human_v, transform_matrix, lbs_w = smplx_deformer.transform_to_t_pose(human_v, smplx, smplx_param0['trans'], smplx_param0['scale'])
    #
    #         t_human_v = t_human_v.squeeze().unsqueeze(0)
    #         t_human_v1, transform_matrix1 = smplx_deformer.transform_to_pose(t_human_v, lbs_w, smplx1, smplx_param1['trans'], smplx_param1['scale'])
    #         smplx_deformer.save_obj(f'../data/a1_s1/Frame000{409+i+1}_50k.obj', t_human_v1.squeeze().numpy(), human_f)