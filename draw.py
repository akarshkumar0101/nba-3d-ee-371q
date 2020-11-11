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
    
    
    