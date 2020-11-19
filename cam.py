import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

import util, constants, draw, cam

"""
This method takes in 2 DoFs that mean "log fx" and "log fy".
This is useful for having direct representation.

dofs_f has shape (..., 2)
returns fxy of shape (..., 2)
"""
def calc_fxy_direct(dofs_f):
    return np.e**dofs_f[..., :]

"""
This method takes in 2 DoFs that mean "log fx" and "log (fy/fx)".
This is useful for keeping the ratio of fy/fx consistent.

dofs_f has shape (..., 2)
returns fxy of shape (..., 2)
"""
def calc_fxy_ratio(dofs_f):
    fx = np.e**dofs_f[..., [0]]
    fy = fx*np.e**dofs_f[..., [1]]
    return torch.cat([fx, fy], dim=-1)
    

def calc_dofs_cam(cam_position, cam_points_to, dofs_f):
    bs = cam_position.shape[:-1]

    Zp = cam_points_to - cam_position
    Zp = Zp / Zp.norm(dim=-1, keepdim=True)
    
    Xp = -torch.cross(Zp, constants.Z_global.repeat(bs+(1,)), dim=-1)
    Xp = Xp/Xp.norm(dim=-1, keepdim=True)
    
    Yp = torch.cross(Zp, Xp, dim=-1)
    Yp = Yp/Yp.norm(dim=-1, keepdim=True)
    # Zp is our forward. Xp is our left, Yp is our up
    R = torch.stack([Xp, Yp, Zp], dim=-1)
    
    axang = util.so3_log_map(R)
    return torch.cat((cam_position, axang, dofs_f), dim=-1)

# returns shape (cam, point, 2)
def project_to_cam(X_w, dofs_cam, calc_fxy=calc_fxy_ratio):
    bs_X_w, bs_dofs_cam = X_w.shape[:-1], dofs_cam.shape[:-1]
    for _ in range(len(bs_X_w)):
        dofs_cam = dofs_cam[..., None, :]
    for _ in range(len(bs_dofs_cam)):
        X_w = X_w[None, ...]
    
    R = util.so3_exponential_map(dofs_cam[..., 3:6]) # this goes from camera->world
    Rinv = R.transpose(-1, -2) # world->camera
    T = dofs_cam[..., :3]
    
    X_c = (Rinv @ ((X_w-T)[..., None]))[..., 0]
    
    dofs_f = dofs_cam[..., 6:]
    fxy = calc_fxy(dofs_f)
    
    X_i = fxy * ((X_c/X_c[..., [-1]])[..., :2])
    
    vis_mask = torch.logical_and(X_i>=-1., X_i<=1).all(dim=-1)
    vis_mask = torch.logical_and(vis_mask, X_c[..., 2]>0.)
    return X_i, vis_mask


# returns a crop such that:
# x is cropped (xmin, xmax) left to right
# y is cropped (ymin, ymax) down to up
def get_intrinsic_mat(xmin, xmax, ymin, ymax):
    u_0 = (xmin+xmax)/2.
    v_0 = (ymin+ymax)/2.

    xlim = 1.
    # m_x should map [-xlim, xlim] to [xmin, xmax]
    m_x = (xmax-xmin)/(xlim--xlim)

    # m_y should map [-ylim, ylim] to [ymin, ymax] 
    # such that the aspect ratio of (ymax-ymin)/(xmax-xmin) = (ylim--ylim)/(xlim--xlim)
    # ylim = xlim * (ymax-ymin)/(xmax-xmin)
    ylim = np.abs(xlim * (ymax-ymin)/(xmax-xmin))
    m_y = (ymax-ymin)/(ylim--ylim)
    
    # -m_x is needed to make +X axis point right and not left in camera/projection coordinates
    P = np.array([[ -m_x,     0, u_0, 0],
                  [    0,   m_y, v_0, 0],
                  [    0,     0,   1, 0]])
    return P

def apply_focus(f, P):
    f = np.array(f)
    P = np.tile(P, (*f.shape, 1, 1))
    P[..., [0, 1], [0, 1]] = f[..., None]*P[..., [0, 1], [0, 1]]
    return P

def get_intrinsic_mat_default():
    return get_intrinsic_mat(-1., 1., -1., 1.)

def get_intrinsic_mat_for_img_shape(img_shape):
    return get_intrinsic_mat(0, img_shape[1], img_shape[0], 0)

