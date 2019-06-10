import cv2
import numpy as np

def im_processor(im):
    """预处理图片"""
    assert im.shape[2] == 3
    im = cv2.cvtColor(cv2.resize(im, (80, 80)), cv2.COLOR_BGR2GRAY)
    im = im[...,np.newaxis]
    return im
