import numpy as np
import cv2

import draw

def to_uint8(img):
    return (img.astype(np.float32)/img.max()*255.).astype(np.uint8)

def process_img(img):
    img_ep = cv2.edgePreservingFilter(to_uint8(img))
    gray_ep = cv2.cvtColor(img_ep, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray_ep, cv2.CV_32F, 1, 1, ksize=3)
    sobel = cv2.equalizeHist(to_uint8(np.abs(sobel)))
    sobel = (sobel>190).astype(np.uint8)*255
    return sobel

def process_img_canny(img, ep=True):
    img = to_uint8(img)
    if ep:
        img = cv2.edgePreservingFilter(img)
#     canny = cv2.Canny(img, 50, 150, apertureSize=3)
    canny = cv2.Canny(img, 25, 50, apertureSize=3)
    return canny

def erase_players(img, kp, width=25):
    img = img.copy()
    blank = np.zeros(img.shape[:2], dtype=np.uint8)
    blank = draw.draw_people_img(blank, kp, color=255, thickness=1)
    blank = cv2.dilate(blank, np.ones((width, width), dtype=np.uint8))
    img[blank>100] = 0
    return img
def erase_players_points(img_shape_xy, points, kp, width=25):
    blank = np.zeros(img_shape_xy[::-1], dtype=np.uint8)
    blank = draw.draw_people_img(blank, kp, color=255, thickness=1)
    blank = cv2.dilate(blank, np.ones((width, width), dtype=np.uint8))
    mask = blank>100
    points = points.astype(int)
    pointmask = ~mask[points[:, 1], points[:, 0]]
    return pointmask


def erase_scorebox(img, score_box):
    img = img.copy()
    ((x0,y0), (x1,y1)) = score_box
    img[y0:y1, x0: x1] = 0
    return img

def erase_scorebox_points(points, score_box):
    x, y = points[:, 0], points[:, 1]
    ((x0,y0), (x1,y1)) = score_box
    return ~np.logical_and(np.logical_and(x>x0, x<x1), np.logical_and(y>y0, y<y1))

