import cv2 as cv
import numpy as np

import debugging_functions
import pattern_matching
import preprocessing
import draw_rect


BaseSymbols = ["images/BaseSymbols/Feuerloescher.png",
               "images/BaseSymbols/Brandmelder.png",
               "images/BaseSymbols/Feuerleiter.png",
               "images/BaseSymbols/Loeschschlauch.png",
               "images/BaseSymbols/mittel_und_geraete.png",
               "images/BaseSymbols/Pfeil.png"]

if __name__ == '__main__':
    original = cv.imread("images/Krankenhaus_5OG.jpg")
    img2 = cv.imread("images/Krankenhaus_5OG.jpg", cv.IMREAD_GRAYSCALE)
    img1 = cv.imread(BaseSymbols[0], cv.IMREAD_GRAYSCALE)

    masked_img = img2.copy()
    mask_maker = draw_rect.DrawRect(masked_img)

    mask_maker.setup_Mouse_callback("test")
    while True:
        cv.imshow("test", masked_img)
        if cv.waitKey(10) == 27:
            break
    cv.destroyWindow("test")

    scale = debugging_functions.show_image(mask_maker.mask, "mask")

    cv.imwrite("lama/LaMa_test_images/cuttedfeuerplan_mask.png", mask_maker.mask)
    cv.imwrite("lama/LaMa_test_images/cuttedfeuerplan.png", img2)

    exit(0)
    if img1 is None or img2 is None:
        print('Could not open or find the images!')
        exit(0)

    #img1 = preprocessing.denoise(img1)
    #img2 = preprocessing.denoise(img2)
    img1 = preprocessing.thresolding(img1)
    img2 = preprocessing.thresolding(img2)
    for i in range(0, 1):
        corners = pattern_matching.pattern_matching(object=img1, scene=img2)

        cv.rectangle(original, corners[0], corners[1], (0, 0, 0), -1)

        debugging_functions.show_image(original, 'new show')
        #img1 = preprocessing.thresolding(img1)
        #img2 = preprocessing.thresolding(img2)

        #img2 = cv.COLOR_RGB2GRAY(img2)

        if original.shape.__len__() == 3:
            x, y, _ = original.shape
        else:
            x, y = original.shape
        mask = np.zeros((x, y))
        cv.rectangle(mask, corners[0], corners[1], (255, 255, 255), -1)
        dst = cv.inpaint(img2, np.uint8(mask), 3, cv.INPAINT_NS)
        img2 = cv.cvtColor(original.copy(), cv.COLOR_BGR2GRAY)
        img2 = preprocessing.thresolding(img2)

    debugging_functions.show_image(dst, 'result after inpaint')
    _ = preprocessing.small_structure_killer(dst)




