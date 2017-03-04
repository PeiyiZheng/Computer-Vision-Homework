import cv2
import numpy as np
from scipy import signal

def gauss_derivative_kernels(size, sigma):
    radius = size / 2
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]

    gx = - x * np.exp(-(x * x + y * y) / (2.0 * sigma ** 2))
    gy = - y * np.exp(-(x * x + y * y) / (2.0 * sigma ** 2))

    return gx,gy

def ssift_descriptor(feature_coords,image,radius=20, kernel_size=5, sigma=1.5, filename=None, color_image=None):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        feature_coords (list of tuples): list of (row,col) tuple feature coordinates from image
        image (numpy.ndarray): The input image to compute ssift descriptors on. Note: this is NOT the image name or image path.
    Returns:
        descriptors (dictionary{(row,col): 128 dimensional list}): the keys are the feature coordinates (row,col) tuple and
                                                                   the values are the 128 dimensional ssift feature descriptors.
    """
	
    descriptors = dict()

    gx, gy = gauss_derivative_kernels(kernel_size, sigma)
    I_x = signal.convolve2d(image, gx, mode='same')
    I_y = signal.convolve2d(image, gy, mode='same')
    I_mag = np.sqrt(I_x * I_x + I_y * I_y)

    bins = np.array([-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.5])
    for feature_coord in feature_coords:
        if feature_coord[0] - radius < 0 or feature_coord[1] - radius < 0:
            continue
        if feature_coord[0] + radius >= image.shape[0] or feature_coord[1] + radius >= image.shape[1]:
            continue

        feature_vec = []
        grid_size = radius / 2
        shift_row = -radius
        for i in xrange(0, 4):
            shift_col = -radius
            for j in xrange(0, 4):
                top_left_x = feature_coord[0] + shift_row
                top_left_y = feature_coord[1] + shift_col

                grid_Ix = I_x[top_left_x : top_left_x + grid_size, top_left_y : top_left_y + grid_size]
                grid_Iy = I_y[top_left_x : top_left_x + grid_size, top_left_y : top_left_y + grid_size]
                grid_angle = np.arctan2(grid_Iy, grid_Ix) * 180.0 / np.pi

                grid_angle = np.digitize(grid_angle, bins)

                hist = np.zeros(8)
                for p in xrange(0, grid_angle.shape[0]):
                    for q in xrange(0, grid_angle.shape[1]):
                        hist[grid_angle[p][q]] += I_mag[top_left_x + p][top_left_y + q]

                feature_vec.extend(hist)

                shift_col += grid_size

            shift_row += grid_size

        feature_vec = np.array(feature_vec)
        temp = np.sqrt(feature_vec.dot(feature_vec))
        if temp == 0.0:
            continue
        feature_vec /= temp
        for i in xrange(0, feature_vec.shape[0]):
            if feature_vec.item(i) > 0.2:
                feature_vec.itemset(i, 0.2)
        temp = np.sqrt(feature_vec.dot(feature_vec))
        if temp == 0.0:
            continue
        feature_vec /= temp

        descriptors[(feature_coord[0], feature_coord[1])] = feature_vec

    if filename != None:
        img = image.copy()
        if color_image != None:
            img = color_image.copy()
        for key in descriptors:
            if color_image != None:
                cv2.circle(img, (key[1], key[0]), 3, (255, 0, 0), -1)
            else:
                cv2.circle(img, (key[1], key[0]), 3, (255, 255, 255))

        cv2.imwrite(filename, img)

    return descriptors
