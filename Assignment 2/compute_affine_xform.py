# Author: TK
import cv2
import numpy as np
import random
import math

def generate_affine_xform(sample_matches, features1, features2):
    A = []
    b = []
    for match in sample_matches:
        A.append([features2[match[1]][1], features2[match[1]][0], 1, 0, 0, 0])
        A.append([0, 0, 0, features2[match[1]][1], features2[match[1]][0], 1])

        b.extend([features1[match[0]][1], features1[match[0]][0]])

    t = np.linalg.lstsq(np.array(A), np.array(b).transpose())[0]

    H = np.array([[t[0], t[1], t[2]], [t[3], t[4], t[5]], [0, 0, 1]])
    #H = np.array([[t[0], t[1], t[2]], [t[3], t[4], t[5]], [0, 0, 1]])

    return H

def compute_affine_xform(matches,features1,features2,img1,img2, threshold=5.0, times=20, filename=None):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
                                  in feature_coords2 are determined to be matches, the list should contain (4,0).
        features1 (list of tuples) : list of feature coordinates corresponding to image1
        features2 (list of tuples) : list of feature coordinates corresponding to image2
        image1 (numpy.ndarray): The input image corresponding to features_coords1
        image2 (numpy.ndarray): The input image corresponding to features_coords2
    Returns:
        affine_xform (numpy.ndarray): a 3x3 Affine transformation matrix between the two images, computed using the matches.
    """
    
    affine_xform = np.zeros((3,3))

    # RANSAC
    maximum = 0
    final_inliers = []
    for i in xrange(0, times):
        sample_matches = random.sample(matches, 3)
        H = generate_affine_xform(sample_matches, features1, features2)

        current_inliers = []
        for match in matches:
            x = features2[match[1]][1] * H[0][0] + features2[match[1]][0] * H[0][1] + H[0][2]
            y = features2[match[1]][1] * H[1][0] + features2[match[1]][0] * H[1][1] + H[1][2]

            temp = np.array([x, y]) - np.array([features1[match[0]][1], features1[match[0]][0]])
            dist = math.sqrt(temp.dot(temp))
            if dist < threshold:
                current_inliers.append(match)

        if len(current_inliers) > maximum:
            maximum = len(current_inliers)
            final_inliers = current_inliers

    image1 = img1.copy()
    image2 = img2.copy()

    black = np.zeros((image1.shape[0], image1.shape[1], image1.shape[2]))
    for i in xrange(0, image2.shape[0]):
        for j in xrange(0, image2.shape[1]):
            black[i, j] = image2[i, j]

    sbs = np.concatenate((image1, black), axis=1)
    for match in final_inliers:
        coord1 = features1[match[0]]
        coord2 = features2[match[1]]
        coord1 = (coord1[1], coord1[0])
        coord2 = (coord2[1] + image1.shape[1], coord2[0])
        cv2.circle(sbs, coord1, 4, (255, 0, 0), -1)
        cv2.circle(sbs, coord2, 4, (255, 0, 0), -1)
        cv2.line(sbs, coord1, coord2, (0, 255, 0), 1)

    for match in matches:
        if match not in final_inliers:
            coord1 = features1[match[0]]
            coord2 = features2[match[1]]
            coord1 = (coord1[1], coord1[0])
            coord2 = (coord2[1] + image1.shape[1], coord2[0])
            cv2.circle(sbs, coord1, 4, (0, 0, 255), -1)
            cv2.circle(sbs, coord2, 4, (0, 0, 255), -1)
            cv2.line(sbs, coord1, coord2, (0, 0, 255), 1)

    if filename != None:
        cv2.imwrite(filename, sbs)

    return generate_affine_xform(final_inliers, features1, features2)
