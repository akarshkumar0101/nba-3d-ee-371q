import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import cv2, json, torch, os, re

import util, constants, draw, cam, model, loader, img_proc

np.random.seed(0)
torch.manual_seed(0)

VID_ROOT = '/home/akarshkumar0101/Insync/akarshkumar0101@gmail.com/Google Drive/nba-3d-data/harden/'
DATA_ROOT = '/home/akarshkumar0101/Insync/akarshkumar0101@gmail.com/Google Drive/nba-3d-data/'

img_shape_yx = plt.imread(VID_ROOT+'/all_views/frame_00001.png').shape[:2];img_shape_xy = img_shape_yx[::-1]

uf_mat_int_default = cam.get_mat_intrinsic()
uf_mat_int = cam.get_intrinsic_mat_for_img_shape(img_shape_xy)

with open(VID_ROOT+'/md.json') as f:
    data = json.load(f)
    num_views, num_frames = data['num_views'], data['num_frames']
    
print(f'{num_views} views, {num_frames} frames')
print(f'img_shape_xy: {img_shape_xy}')