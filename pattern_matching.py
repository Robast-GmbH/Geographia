from __future__ import print_function
import cv2 as cv
import numpy as np


def pattern_matching(object, scene):
    # -- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    detector = cv.SIFT_create()
    keypoints1, descriptors1 = detector.detectAndCompute(object, None)
    keypoints2, descriptors2 = detector.detectAndCompute(scene, None)
    # -- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
    # -- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.85
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    # -- Draw matches
    img_matches = np.empty((max(object.shape[0], scene.shape[0]), object.shape[1] + scene.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(object, keypoints1, scene, keypoints2, good_matches, img_matches,
                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # -- Show detected matches
    #resized = cv.resize(img_matches, (1500, 1000))
    cv.imshow('Good Matches', img_matches)
    cv.waitKey()
    # -- Localize the object
    obj = np.empty((len(good_matches), 2), dtype=np.float32)
    scene = np.empty((len(good_matches), 2), dtype=np.float32)
    for i in range(len(good_matches)):
        # -- Get the keypoints from the good matches
        obj[i, 0] = keypoints1[good_matches[i].queryIdx].pt[0]
        obj[i, 1] = keypoints1[good_matches[i].queryIdx].pt[1]
        scene[i, 0] = keypoints2[good_matches[i].trainIdx].pt[0]
        scene[i, 1] = keypoints2[good_matches[i].trainIdx].pt[1]
    H, _ = cv.findHomography(obj, scene, cv.RANSAC)
    # -- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4, 1, 2), dtype=np.float32)
    obj_corners[0, 0, 0] = 0
    obj_corners[0, 0, 1] = 0
    obj_corners[1, 0, 0] = object.shape[1]
    obj_corners[1, 0, 1] = 0
    obj_corners[2, 0, 0] = object.shape[1]
    obj_corners[2, 0, 1] = object.shape[0]
    obj_corners[3, 0, 0] = 0
    obj_corners[3, 0, 1] = object.shape[0]
    scene_corners = cv.perspectiveTransform(obj_corners, H)
    # -- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv.line(img_matches, (int(scene_corners[0, 0, 0] + object.shape[1]), int(scene_corners[0, 0, 1])), \
            (int(scene_corners[1, 0, 0] + object.shape[1]), int(scene_corners[1, 0, 1])), (0, 255, 0), 4)
    cv.line(img_matches, (int(scene_corners[1, 0, 0] + object.shape[1]), int(scene_corners[1, 0, 1])), \
            (int(scene_corners[2, 0, 0] + object.shape[1]), int(scene_corners[2, 0, 1])), (0, 255, 0), 4)
    cv.line(img_matches, (int(scene_corners[2, 0, 0] + object.shape[1]), int(scene_corners[2, 0, 1])), \
            (int(scene_corners[3, 0, 0] + object.shape[1]), int(scene_corners[3, 0, 1])), (0, 255, 0), 4)
    cv.line(img_matches, (int(scene_corners[3, 0, 0] + object.shape[1]), int(scene_corners[3, 0, 1])), \
            (int(scene_corners[0, 0, 0] + object.shape[1]), int(scene_corners[0, 0, 1])), (0, 255, 0), 4)
    # -- Show detected matches
    #resized = cv.resize(img_matches, (1500, 1000))
    cv.imshow('Good Matches', img_matches)
    cv.waitKey()

    return [(int(scene_corners[0, 0, 0]), int(scene_corners[0, 0, 1])), (int(scene_corners[2, 0, 0]), int(scene_corners[2, 0, 1]))]
    #a = cv.rectangle(img_matches, (int(scene_corners[0, 0, 0] + object.shape[1]), int(scene_corners[0, 0, 1])),(int(scene_corners[2, 0, 0] + object.shape[1]), int(scene_corners[2, 0, 1])),(0, 255, 255), -1)

