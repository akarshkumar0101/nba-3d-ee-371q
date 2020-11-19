import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

import util, constants, draw, cam

def multi_view(views, region=None, rows=1):
    plt.figure(figsize=(7*len(views), 7*rows))
    axs = []
    i=0
    for row in range(rows):
        axs_row = []
        for view in views:
            ax = plt.subplot(rows, len(views), i+1, projection='3d')
            if region is not None:
                ax.set_xlim3d(*region[0]);ax.set_ylim3d(*region[1]);ax.set_zlim3d(*region[2])
            ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
            ax.view_init(*view)
            axs_row.append(ax)
            i+=1
        axs.append(axs_row)
    plt.tight_layout()
    return axs

def show_cam_view(X_w, dofs_cam, ax=None, calc_fxy=cam.calc_fxy_ratio, **kwargs):
    if ax is None:
        ax = plt.gca()
    X_i, vis_mask = cam.project_to_cam(X_w, dofs_cam, calc_fxy)
    X_i = X_i[vis_mask]
    ax.scatter(-X_i[:, 0], X_i[:, 1], marker='.', **kwargs)
    ax.set_xlim(-1, 1);ax.set_ylim(-1,1)
    plt.tick_params(axis='both', labelsize=0, length = 0)

    
keypoint_parents = [0,0,0,1,2, 0,0,5,6,7,8,5,6,11,12,13,14]

"""
kp is of shape (num_people, 17, 2)
"""
def draw_people2D(img, kp, color=255, thickness=5):
    newimg = img.copy()
    for kpi in kp:
        for ji in range(17):
            pji = keypoint_parents[ji]
            x, px = kpi[ji].astype(int), kpi[pji].astype(int)
            newimg = cv2.line(newimg, (x[0], x[1]),(px[0], px[1]), color, thickness)
    return newimg

    
def set_img_bounds(img_shape, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.set_xlim(0, img_shape[1]);ax.set_ylim(img_shape[0], 0)
    