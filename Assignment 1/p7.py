#!/usr/bin/python
import numpy
import cv2
import math

def p7(image_in, hough_image_in, hough_thresh): #return line_image_ou
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

	factor = math.pi / 180.0

	for line in lines:
		start_x = start_y = end_x = end_y = 0
		if line[1] >= 45 and line[1] <= 135:
			start_x = 0
			end_x = image_in.shape[1]
			part1 = line[0] - hough_image_in.shape[0] / 2
			
			part2 = (start_x - image_in.shape[1] / 2) * math.cos(line[1] * factor)
			start_y = (part1 - part2) / math.sin(line[1] * factor) + image_in.shape[0] / 2
			
			part2 = (end_x - image_in.shape[1] / 2) * math.cos(line[1] * factor)
			end_y = (part1 - part2) / math.sin(line[1] * factor) + image_in.shape[0] / 2
		else:
			start_y = 0
			end_y = image_in.shape[0]
			part_1 = line[0] - hough_image_in.shape[0] / 2

			part_2 = (start_y - image_in.shape[0] / 2) * math.sin(line[1] * factor)

			start_x = (part_1 - part_2) / math.cos(line[1] * factor) + image_in.shape[1] / 2
			part_2 = (end_y - image_in.shape[0] / 2) * math.sin(line[1] * factor)
			end_x = (part_1 - part_2) / math.cos(line[1] * factor) + image_in.shape[1] / 2

		cv2.line(image_in, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (255, 0, 0), 2)


	return image_in