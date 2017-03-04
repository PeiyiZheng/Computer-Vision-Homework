#!/usr/bin/python
import numpy
import cv2
import math
from p1 import p1

def p6(edge_image_in, edge_thresh): #return [edge_image_thresh_out, hough_image_out]
	edge_image_in = p1(edge_image_in, edge_thresh)

	dig = int(math.sqrt((edge_image_in.shape[0] ** 2) + (edge_image_in.shape[1] ** 2)))

	theta_sin = []
	theta_cos = []
	half_pi = math.pi * 0.5
	factor = math.pi / 180.0
	for i in xrange(0, 180):
		theta_sin.append(math.sin(float(i) * factor))
		theta_cos.append(math.cos(float(i) * factor))

	y_center = edge_image_in.shape[0] * 0.5
	x_center = edge_image_in.shape[1] * 0.5

	accumulator = numpy.zeros((dig * 2, 180), dtype = numpy.float)
	for (y, x), val in numpy.ndenumerate(edge_image_in):
		if val == 0:
			continue

		for i in xrange(0, 180):
			p = (x - x_center) * theta_cos[i] + (y - y_center) * theta_sin[i]
			j = int(round(p + dig))
			accumulator.itemset((j, i), accumulator.item(j, i) + 1.0)

	min_val = 1000000
	max_val = -1
	for (x, y), val in numpy.ndenumerate(accumulator):
		if val == 0:
			continue

		if val < min_val:
			min_val = val

		if val > max_val:
			max_val = val

	max_val = (max_val - min_val) / 255.0
	for (x, y), val in numpy.ndenumerate(accumulator):
		if val == 0:
			continue

		accumulator.itemset((x, y), (val - min_val) / max_val)

	return (edge_image_in, accumulator)