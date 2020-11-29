import numpy as np
import matplotlib.pyplot as plt
import json
import cv2

import util

import img_proc

"""
Loads the model corners from models directory.
"""
def load_model_corners(data_root):
    with open(data_root+'/models/court_corners.json') as f:
        # coordinates on the outside of corners,
        # order: top left, bot left, bot right, top right
        # (x, y) not (y, x)
        return np.array(json.load(f)).astype(np.float32)
    
"""
Loads the model image from models directory.
"""
def load_model_img(data_root):
    court_img = plt.imread(data_root+'/models/court.png')
    court_bin = court_img.mean(axis=-1)>.9
    return court_img, court_bin
"""
Loads the model from models directory as a vector of shape (N, 3) into model space.
"""
def get_model(data_root):
    _, court_bin = load_model_img(data_root)
    y, x = np.where(court_bin)
    X_m = np.stack([x, y, np.zeros_like(x)], axis=-1)
    return X_m.astype(np.float32)

"""
Loads the model from models directory as a vector of shape (N, 3) into model space.
"""
def get_model_canny(data_root):
    court_img, _ = load_model_img(data_root)
    court_img = cv2.cvtColor(court_img, cv2.COLOR_RGB2GRAY)
    court_bin = img_proc.process_img_canny(court_img)>128
    y, x = np.where(court_bin)
    X_m = np.stack([x, y, np.zeros_like(x)], axis=-1)
    return X_m.astype(np.float32)

"""
Gets the model matrix that goes from model to world space based on court_corners.
"""
def get_mat_model(model_corners):
    M = np.eye(4, dtype=np.float32)
    scale = 1/(model_corners[1, 1] -model_corners[0, 1])
    M[1, 1] *= -1
    M[:2, :2] = scale*M[:2, :2]
    M[:2, 3] = -M[[0,1], [0,1]]*model_corners.mean(axis=0)
    return M
"""
Loads the court model from models directory as a vector of shape (N, 3) into world space.
"""
def calc_model_world_coordinates(data_root):
    X_m = get_model_canny(data_root)
    model_corners = load_model_corners(data_root)
    M = get_mat_model(model_corners)
    
    X_m = util.to_homo(X_m)
    X_w = (M @ X_m[..., None])[..., 0]
    X_w = util.from_homo(X_w)
    return X_w
