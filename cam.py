import torch

import util, constants

def calc_dofs_cam(cam_position, cam_points_to, fxy):
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
    return torch.cat((cam_position, axang, fxy), dim=-1)