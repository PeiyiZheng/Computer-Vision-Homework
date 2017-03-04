import cv2
import numpy as np
import math
import heapq
from operator import itemgetter

def match_features_ssift(feature_descriptors1,feature_descriptors2, threshold=0.6):
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

    for key1, val1 in feature_descriptors1.iteritems():
        dist_list = []
        for key2, val2 in feature_descriptors2.iteritems():
            temp = val1 - val2
            dist = np.sqrt(temp.dot(temp))
            dist_list.append((dist, key1, key2))

        dist_list = heapq.nsmallest(2, dist_list, key=itemgetter(0))

        if dist_list[0][0] == 0.0 or dist_list[1][0] == 0.0:
            matches.append((dist_list[0][1], dist_list[0][2]))
        else:
            ratio = dist_list[0][0] / dist_list[1][0]

            if ratio <= threshold:
                matches.append((dist_list[0][1], dist_list[0][2]))

    return matches
