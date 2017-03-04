import cv2
import numpy
import math

def match_features(feature_coords1, feature_coords2, image1, image2, radius=20):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        feature_coords1 (list of tuples): list of (row,col) tuple feature coordinates from image1
        feature_coords2 (list of tuples): list of (row,col) tuple feature coordinates from image2
        image1 (numpy.ndarray): The input image corresponding to features_coords1
        image2 (numpy.ndarray): The input image corresponding to features_coords2
    Returns:
        matches (list of tuples): list of index pairs of possible matches. For example, if the 4-th feature in feature_coords1 and the 0-th feature
                                  in feature_coords2 are determined to be matches, the list should contain (4,0).
    """
	
    matches = list()

    feature_list1 = []
    idx = 0
    for feature_coord in feature_coords1:
        if feature_coord[0] - radius < 0 or feature_coord[1] - radius < 0:
            idx += 1
            continue
        if feature_coord[0] + radius >= image1.shape[0] or feature_coord[1] + radius >= image1.shape[1]:
            idx += 1
            continue

        feature = image1[feature_coord[0] - radius : feature_coord[0] + radius, feature_coord[1] - radius : feature_coord[1] + radius]
        feature = feature - feature.mean()
        temp = numpy.sqrt(numpy.sum(feature ** 2))
        feature_list1.append((idx, feature, temp))
        idx += 1

    feature_list2 = []
    idx = 0
    for feature_coord in feature_coords2:
        if feature_coord[0] - radius < 0 or feature_coord[1] - radius < 0:
            idx += 1
            continue
        if feature_coord[0] + radius >= image2.shape[0] or feature_coord[1] + radius >= image2.shape[1]:
            idx += 1
            continue

        feature = image2[feature_coord[0] - radius : feature_coord[0] + radius, feature_coord[1] - radius : feature_coord[1] + radius]

        feature = feature - feature.mean()
        temp = numpy.sqrt(numpy.sum(feature ** 2))
        feature_list2.append((idx, feature, temp))
        idx += 1

    ncr = numpy.zeros((len(feature_list1), len(feature_list2)))
    matches1 = []
    for i in xrange(0, len(feature_list1)):
        maximum = -2.0
        idx = -1
        for j in xrange(0, len(feature_list2)):
            numerator = feature_list1[i][1] * feature_list2[j][1]
            cr = numpy.sum(numerator) / (feature_list1[i][2] * feature_list2[j][2]) 

            ncr[i, j] = cr

            if cr > maximum:
                maximum = cr
                idx = feature_list2[j][0]

        matches1.append((feature_list1[i][0], idx))

    matches2 = []
    for i in xrange(0, len(feature_list2)):
        maximum = -2.0
        idx = -1
        for j in xrange(0, len(feature_list1)):
            cr = ncr[j, i]
            if cr > maximum:
                maximum = cr
                idx = feature_list1[j][0]

        matches2.append((idx, feature_list2[i][0]))

    for match in matches1:
        if match in matches2:
            matches.append(match)

    return matches
