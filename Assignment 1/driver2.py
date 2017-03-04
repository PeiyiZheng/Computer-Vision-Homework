#!/usr/bin/python
import numpy
import cv2
import math
from matplotlib import pyplot as plt
from p5 import p5
from p6 import p6
from p7 import p7
from p8 import p8

def img_scale(image_in):
	mx = 0
	for (x, y), val in numpy.ndenumerate(image_in):
		if val != 0:
			if val > mx:
				mx = val

	if mx == 0:
		return image_in

	mx = 255 / mx
	for (x, y), val in numpy.ndenumerate(image_in):
		if val != 0:
			image_in.itemset((x, y), image_in.item(x, y) * mx)

	return image_in

## case 1
img = cv2.imread('hough_simple_1.pgm', cv2.IMREAD_UNCHANGED)
copy_img1 = img.copy()
copy_img2 = img.copy()
img = p5(img)

cv2.imwrite('p5_hough_simpe_1.jpg', img)

result = p6(img, 25)
cv2.imwrite('p6_simple_1_edge_thresh.jpg', img_scale(result[0]))
cv2.imwrite('p6_simple_1_hough_image.jpg', result[1])

hough_copy = result[1].copy()
lines = p7(copy_img1, result[1], 120)
cv2.imwrite('p7_hough_simple_1.jpg', lines)

segments = p8(copy_img2, hough_copy, 20, 100)
cv2.imwrite('p8_hough_simple_1.jpg', segments)

## case 2
img = cv2.imread('hough_simple_2.pgm', cv2.IMREAD_UNCHANGED)
copy_img1 = img.copy()
copy_img2 = img.copy()
img = p5(img)

cv2.imwrite('p5_hough_simpe_2.jpg', img)

result = p6(img, 25)
cv2.imwrite('p6_simple_2_edge_thresh.jpg', img_scale(result[0]))
cv2.imwrite('p6_simple_2_hough_image.jpg', result[1])

hough_copy = result[1].copy()
lines = p7(copy_img1, result[1], 120)
cv2.imwrite('p7_hough_simple_2.jpg', lines)

segments = p8(copy_img2, hough_copy, 20, 100)
cv2.imwrite('p8_hough_simple_2.jpg', segments)

## case 3
img = cv2.imread('hough_complex_1.pgm', cv2.IMREAD_UNCHANGED)
copy_img1 = img.copy()
copy_img2 = img.copy()
img = p5(img)

cv2.imwrite('p5_hough_complex_1.jpg', img)

result = p6(img, 55)
cv2.imwrite('p6_complex_1_edge_thresh.jpg', img_scale(result[0]))
cv2.imwrite('p6_complex_1_hough_image.jpg', result[1])

hough_copy = result[1].copy()
lines = p7(copy_img1, result[1], 100)
cv2.imwrite('p7_hough_complex_1.jpg', lines)

segments = p8(copy_img2, hough_copy, 20, 100)
cv2.imwrite('p8_hough_complex_1.jpg', segments)