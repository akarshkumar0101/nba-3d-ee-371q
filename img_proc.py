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
    img = cv2.edgePreservingFilter(img)
    canny = cv2.Canny(img, 50, 150, apertureSize=3)
    return canny

def erase_players(img, kp, width=25):
    img = img.copy()
    blank = np.zeros(img.shape[:2], dtype=np.uint8)
    blank = draw.draw_people_img(blank, kp, color=255, thickness=1)
    blank = cv2.dilate(blank, np.ones((width, width), dtype=np.uint8))
    img[blank>100] = 0
    return img
