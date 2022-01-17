import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def thresolding(img):
    #blur = cv.GaussianBlur(img,(5,5),0)
    blur = cv.bilateralFilter(img, 9, 75, 75)
    ret3,th3 = cv.threshold(blur, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    return th3

def small_structure_killer(img):
    th = thresolding(img)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv.morphologyEx(th, cv.MORPH_CLOSE, kernel)
    blur = cv.bilateralFilter(closing, 9, 75, 75)
    cv.imshow('closed img', blur)
    cv.waitKey()

    return closing