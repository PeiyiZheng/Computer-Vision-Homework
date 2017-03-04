import cv2
import math
import numpy as np
import heapq
from scipy import signal
from nonmaxsuppts import nonmaxsuppts

def generateGaussianKernel(size, sigma):
	kernel = np.zeros((size, size))
	sum_kernel = 0.0
	radius = size / 2
	for i in xrange(-radius, radius + 1):
		p = i + radius
		q = 0
		for j in xrange(-radius, radius + 1):
			kernel.itemset((p, q), math.exp(-(i * i + j * j) / (2.0 * sigma ** 2)))
			sum_kernel += kernel.item(p, q)
			q += 1

	for i in xrange(0, kernel.shape[0]):
		for j in xrange(0, kernel.shape[1]):
			kernel.itemset((i, j), kernel.item(i, j) / sum_kernel)

	return kernel

def gauss_derivative_kernels(size, sigma):
    radius = size / 2
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    gx = - x * np.exp(-(x * x + y * y) / (2.0 * sigma ** 2))
    gy = - y * np.exp(-(x * x + y * y) / (2.0 * sigma ** 2))

    return gx,gy

def detect_features(image, kernel_size=5, sigma=1.5, nLimit=8, radius=3, filename=None, color_image=None):
    """
    Computer Vision 600.461/661 Assignment 2
    Args:
        image (numpy.ndarray): The input image to detect features on. Note: this is NOT the image name or image path.
    Returns:
        pixel_coords (list of tuples): A list of (row,col) tuples of detected feature locations in the image
    """
    pixel_coords = list()

    image.astype(float)

    gx, gy = gauss_derivative_kernels(kernel_size, sigma)
    I_x = signal.convolve2d(image, gx, mode='same')
    I_y = signal.convolve2d(image, gy, mode='same')
   
    I_x2 = I_x * I_x
    I_y2 = I_y * I_y
    I_xy = I_x * I_y

    window = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    a = signal.convolve2d(I_x2, window, mode='same')
    b = signal.convolve2d(2.0 * I_xy, window, mode='same')
    c = signal.convolve2d(I_y2, window, mode='same')
    temp = np.sqrt(b * b + (a - c) ** 2)
    lambda1 = 0.5 * (a + c + temp)
    lambda2 = 0.5 * (a + c - temp)

    R = lambda1 * lambda2 - 0.04 * (lambda1 + lambda2) ** 2

    all_r = []
    for row in R.tolist():
    	all_r.extend([x for x in row])


    result = heapq.nlargest(len(all_r) / nLimit, all_r)
    threshold = result[-1]

    pixel_coords = nonmaxsuppts(R, radius, threshold)

    if filename != None:
        img = image.copy()
        if color_image != None:
            img = color_image.copy()
        for pixel in pixel_coords:
            if color_image != None:
                cv2.circle(img, (pixel[1], pixel[0]), 3, (255, 0, 0), -1)
            else:
                cv2.circle(img, (pixel[1], pixel[0]), 3, (255, 255, 255))

        cv2.imwrite(filename, img)

    return pixel_coords
