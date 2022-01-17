import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def thresolding(img):
    #blur = cv.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    return th3