#!/usr/bin/python
import numpy
import cv2
import math
from p7 import p7
from p1 import p1
from p5 import p5

def p8(image_in, hough_image_in, edge_thresholded_in, hough_thresh): #return cropped_lines_image_out
	row = hough_image_in.shape[0]
	col = hough_image_in.shape[1]

	lines = []
	for (p, t), val in numpy.ndenumerate(hough_image_in):
		if val < hough_thresh:
			hough_image_in.itemset((p, t), 0)
			continue

		local_maxima = val
		for i in xrange(-1, 2):
			for j in xrange(-1, 2):
				p_ = p + i
				t_ = t + j
				if p_ < 0 or t_ < 0 or p_ == row or t_ == col:
					continue

				if local_maxima < hough_image_in.item(p_, t_):
					local_maxima = hough_image_in.item(p_, t_)

		if local_maxima == val:
			lines.append((p, t))
	
	img = image_in.copy()
	edge_image_in = p5(image_in)
	edge_image = p1(edge_image_in, edge_thresholded_in)

	segments = []
	factor = math.pi / 180.0
	x_center = image_in.shape[1] / 2
	y_center = image_in.shape[0] / 2
	dig = int(math.sqrt((edge_image.shape[0] ** 2) + (edge_image.shape[1] ** 2)))
	for line in lines:
		segment = []
		for (y, x), val in numpy.ndenumerate(edge_image):
			if val == 0:
				continue

			p_ = (y - y_center) * math.sin(float(line[1]) * factor) + (x - x_center) * math.cos(float(line[1]) * factor)

			if abs(p_ + dig - line[0]) <= 1:
				segment.append((y, x))

		segments.append(segment)

	for segment in segments:
		for pt in segment:
			cv2.circle(img, (pt[1], pt[0]), 1, (255, 0, 0))

	return img