import cv2 as cv
import numpy as np

import pattern_matching
import preprocessing

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    original = cv.imread("images/cutted_feuerplan.jpg")
    img1 = cv.imread("images/Brandmelder.png", cv.IMREAD_GRAYSCALE)
    img2 = cv.imread("images/cutted_feuerplan.jpg", cv.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print('Could not open or find the images!')
        exit(0)

    img1 = preprocessing.thresolding(img1)
    img2 = preprocessing.thresolding(img2)
    corners = pattern_matching.pattern_matching(object=img1, scene=img2)

    cv.rectangle(original, corners[0], corners[1], (0, 0, 0), -1)

    cv.imshow("original", original)
    cv.waitKey()

    #img2 = cv.COLOR_RGB2GRAY(img2)

    x, y = original.shape
    mask = np.zeros((x, y))
    cv.rectangle(mask, corners[0], corners[1], (255, 255, 255), -1)
    dst = cv.inpaint(img2, np.uint8(mask), 3, cv.INPAINT_NS)

    cv.imshow("inpainted", dst)
    cv.waitKey()
    _ = preprocessing.small_structure_killer(dst)




