import cv2 as cv
import numpy as np

import pattern_matching
import preprocessing

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img1 = cv.imread("images/Feuerloescher.png", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("images/cutted_feuerplan.jpg", cv.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print('Could not open or find the images!')
        exit(0)

    #img1 = preprocessing.thresolding(img1)
    #img2 = preprocessing.thresolding(img2)
    pattern_matching.pattern_matching(object=img1, scene=img2)

