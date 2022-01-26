import cv2 as cv
import numpy as np


def draw_reactangle(event, x, y, flags, params):
    params.rectangle[2] = x
    params.rectangle[3] = y
    if event == cv.EVENT_LBUTTONDOWN:
        params.drawing = True
        params.rectangle[0] = x
        params.rectangle[1] = y

    elif event == cv.EVENT_MOUSEMOVE:
        if params.drawing == True:
            cv.rectangle(params.img, pt1=(params.rectangle[0], params.rectangle[1]), pt2=(x, y), color=(0, 255, 255),
                         thickness=10)

    elif event == cv.EVENT_LBUTTONUP:
        params.drawing = False
        cv.rectangle(params.mask, pt1=(params.rectangle[0], params.rectangle[1]), pt2=(x, y), color=(255, 255, 255),
                     thickness=-1)

class DrawRect:
    def __init__(self, img = None):
        self.rectangle = [-1, -1, -1, -1]
        self.drawing = False
        self.img = img
        if img.shape.__len__() == 3:
            size_x, size_y, _ = img.shape
        else:
            size_x, size_y = img.shape
        self.mask = np.zeros((size_x, size_y))

    def setup_Mouse_callback(self, windowname):
        cv.namedWindow(winname=windowname)
        cv.setMouseCallback(windowname, draw_reactangle, self)
