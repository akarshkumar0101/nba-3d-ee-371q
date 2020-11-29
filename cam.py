import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

import util, constants, draw, cam

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

def get_mat_view(dofs_cam):
    R = util.so3_exponential_map(dofs_cam[..., 3:6]) # this goes from camera->world
    T = dofs_cam[..., :3]
    if type(dofs_cam) is torch.Tensor:
        Rinv = R.transpose(-1, -2) # world->camera
        mat_view = torch.eye(4).repeat(*dofs_cam.shape[:-1], 1, 1)
    else:
        Rinv = R.swapaxes(-1, -2) # world->camera
        mat_view = np.tile(np.eye(4, dtype=np.float32), (*dofs_cam.shape[:-1], 1, 1))
    mat_view[..., :3, :3] = Rinv[..., :3, :3]
    mat_view[..., :3, 3] = (-Rinv @ (T[..., :, None]))[..., 0]
    return mat_view

def calc_perspective_image(img, dofs_cam, mat_model, img_shape_xy):
    if type(dofs_cam) is torch.Tensor:
        dofs_cam = dofs_cam.numpy()
    img_shape_yx = img_shape_xy[::-1]
    mat_view = get_mat_view(dofs_cam)
    f = np.e**dofs_cam[..., 6]
    mat_int = cam.apply_focus(f, cam.get_intrinsic_mat_for_img_shape(img_shape_xy))
    
    H = mat_int@mat_view@mat_model[:, [0, 1, 3]]
    
    bs = dofs_cam.shape[:-1]
    results = np.zeros(bs + img_shape_yx+img.shape[2:], dtype=img.dtype)
    results = results.reshape((-1,)+img_shape_yx+img.shape[2:])
    for i, Hi in enumerate(H.reshape(-1, 3, 3)):
        results[i, ...] = cv2.warpPerspective(img, Hi, img_shape_xy)
    results = results.reshape(bs+img_shape_yx+img.shape[2:])
    return results

# returns shape (cam, point, 2)
def project_to_cam(X_w, dofs_cam, img_shape_xy):
    mat_view = get_mat_view(dofs_cam)
    f = np.e**dofs_cam[..., 6]
    mat_int = cam.apply_focus(f, cam.get_intrinsic_mat_for_img_shape(img_shape_xy))
    
    bs_dofs = dofs_cam.shape[:-1]
    bs_p = X_w.shape[:-1]
    for _ in range(len(bs_p)):
        mat_int = mat_int[..., None, :, :]
        mat_view = mat_view[..., None, :, :]
    
    X_i = (mat_int @ mat_view @ (util.to_homo(X_w)[..., :, None]))[..., 0]
    z_buf = X_i[..., 2]
    X_i = util.from_homo(X_i)
    
    xmin, xmax, ymin, ymax = 0, img_shape_xy[0], 0, img_shape_xy[1]
    
    pack = torch if (type(dofs_cam) is torch.Tensor) else np
    
    viz_mask_x = pack.logical_and(X_i[..., 0]>xmin, X_i[..., 0]<xmax)
    viz_mask_y = pack.logical_and(X_i[..., 1]>ymin, X_i[..., 1]<ymax)
    viz_mask = pack.logical_and(viz_mask_x, viz_mask_y)
    viz_mask = pack.logical_and(viz_mask, z_buf>0)
    return X_i, viz_mask


# returns a crop such that:
# x is cropped (xmin, xmax) left to right
# y is cropped (ymin, ymax) down to up
def get_mat_intrinsic(xmin=-1., xmax=1., ymin=-1., ymax=1.):
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
    uf_mat_int = np.array([[ -m_x,     0, u_0, 0],
                           [    0,   m_y, v_0, 0],
                           [    0,     0,   1, 0]], dtype=np.float32)
    return uf_mat_int

def apply_focus(f, uf_mat_int):
    if type(f) is torch.Tensor:
        mat_int = torch.from_numpy(uf_mat_int).repeat(*f.shape, 1, 1)
        mat_int[..., [0, 1], [0, 1]] = f*mat_int[..., [0, 1], [0, 1]]
    else:
        f = np.array(f)
        mat_int = np.tile(uf_mat_int, (*f.shape, 1, 1))
        mat_int[..., [0, 1], [0, 1]] = f[..., None]*mat_int[..., [0, 1], [0, 1]]
    return mat_int

def get_intrinsic_mat_for_img_shape(img_shape_xy):
    return get_mat_intrinsic(0, img_shape_xy[0], img_shape_xy[1], 0)

def get_intrinsic_mat_for_dofs_cam(dofs_cam, img_shape_xy):
    return apply_focus(np.e**dofs_cam[..., 6], get_intrinsic_mat_for_img_shape(img_shape_xy))

