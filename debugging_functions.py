import math
import cv2 as cv

def show_image(img, name = 'default'):
    """
    :type name: string
    """
    copy = img.copy()
    x = 0
    y = 0
    if copy.shape.__len__() == 3:
        y, x, _ = copy.shape
    else:
        y, x = copy.shape

    scale_x = 1
    scale_y = 1
    if x > 750:
        scale_x = math.ceil(x/759)
    if y > 1500:
        scale_y = math.ceil(y/1500)

    if scale_x > scale_y:
        x /= scale_x
        y /= scale_x
    else:
        x /= scale_y
        y /= scale_y

    x = math.ceil(x)
    y = math.ceil(y)

    copy = cv.resize(copy, (x, y))
    cv.imshow(name, copy)
    cv.waitKey()
    cv.destroyWindow(name)
