import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

import util, constants, draw, cam

# THIS IS ADITYA'S CODE
def so3_log_map_impl(R, eps=1e-5):
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, W = torch.eig(R33.t(), eigenvectors=True)
    i = torch.where(abs(w[:, 0]-1.0) < eps)[0]
    if not len(i):
        raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
    direction = W[:, i[-1]].squeeze()

    # rotation angle depending on direction
    cosa = (torch.trace(R33) - 1.0) / 2.0
    if torch.abs(direction[2]) > eps:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif torch.abs(direction[1]) > eps:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = torch.atan2(sina, cosa)
    return direction* angle
def so3_log_map(R, eps=1e-5):
    is_numpy = type(R) is np.ndarray
    if is_numpy:
        R = torch.from_numpy(R)
    bs = R.shape[:-2]
    bs_tot = np.prod(bs, dtype=int)
    R = R.view(bs_tot, 3, 3)
    axangs = torch.zeros(bs_tot, 3)
    for i in range(bs_tot):
        axangs[i] = so3_log_map_impl(R[i], eps)
    axangs = axangs.reshape(*bs, 3)
    if is_numpy:
        axangs = axangs.numpy()
    return axangs


def so3_exponential_map(axang, homogeneous=False):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    is_numpy = type(axang) is np.ndarray
    if is_numpy:
        axang = torch.from_numpy(axang)
    
    #axis = np.asarray(axis)
    theta = axang.norm(dim=-1)
    axis = axang/theta[..., None]

    # TODO: hacky broadcasting
    theta = theta[..., None]
    axis, theta = torch.broadcast_tensors(axis, theta)
    theta = theta[..., 0] # the last dimension gets broadcasted to "3", so truncate

    axis = axis / torch.norm(axis, 2, dim=-1)[..., None]
    a = torch.cos(theta / 2.0)
    # Transpose so we can unpack into variables
    b, c, d = torch.unbind(-axis * torch.sin(theta / 2.0)[..., None], dim=-1)

    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    mat =  torch.stack([torch.stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)], dim=-1),
                        torch.stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)], dim=-1),
                        torch.stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc], dim=-1)],
                       dim=-2)
    if homogeneous:
        extra_dims = axis.shape[:-1]
        zeros_column = torch.zeros_like(mat)[..., :, 0:1] # shape: (..., 3, 1)
        hstacked = torch.cat([mat, zeros_column], dim=-1) # shape: (..., 3, 4)

        hom_row = torch.cat((torch.zeros_like(hstacked)[..., :1, :3],
                             torch.ones_like(hstacked)[..., :1, 3:]), dim=-1) # shape: (..., 1, 4)
        mat = torch.cat((hstacked, hom_row), dim=-2)
        
    if is_numpy:
        mat = mat.numpy()
    return mat
        

def to_homo(X, dim=-1):
    if type(X) is torch.Tensor:
        X = X.transpose(dim, -1)
        X = torch.cat((X, torch.ones(X.shape[:-1]+(1,))), dim=-1)
        X = X.transpose(dim, -1)
    else:
        X = np.swapaxes(X, dim, -1)
        X = np.concatenate([X, np.ones(X.shape[:-1]+(1,))], axis=-1).astype(X.dtype)
        X = np.swapaxes(X, dim, -1)
    return X

def from_homo(X, dim=-1):
    if type(X) is torch.Tensor:
        X = X.transpose(dim, -1)
        X = X[..., :-1]/X[..., -1, None]
        X = X.transpose(dim, -1)
    else:
        X = np.swapaxes(X, dim, -1)
        X = (X[..., :-1]/X[..., -1, None]).astype(X.dtype)
        X = np.swapaxes(X, dim, -1)
    return X

# def translation_matrix(tvec: torch.Tensor) -> torch.Tensor:
#     extra_dims = tvec.shape[:-1]
#     eye_tiled = torch.eye(3, dtype=tvec.dtype).to(tvec.device).repeat(extra_dims + (1, 1))
#     hstacked = torch.cat([eye_tiled, tvec[..., :, None]], dim=-1)

#     hom_row = torch.cat((torch.zeros_like(hstacked)[..., :1, :3],
#                          torch.ones_like(hstacked)[..., :1, 3:]), dim=-1) # shape: (..., 1, 4)
#     return torch.cat((hstacked, hom_row), dim=-2)
# def dofs2mat(dofs):
#     T = dofs[..., :3]
#     angle = dofs[..., 3:].norm(dim=-1)
#     axis = dofs[..., 3:]/angle[..., None]
#     rot_mat = rotation_matrix(axis, angle, homogeneous=True)
#     trans_mat = translation_matrix(T)
#     return trans_mat @ rot_mat

