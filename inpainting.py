import cv2 as cv
import numpy as np

def classic_inpainting(img, mask):
    return cv.inpaint(img, np.uint8(mask), 3, cv.INPAINT_NS)