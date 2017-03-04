#!/usr/bin/python
import numpy
import cv2
import math
from p3 import p3

def p4(labels_in, database_in): # return overlays_out
	result = labels_in.copy()
	candidates = p3(labels_in)

	matches = []
	for data in database_in:
		min_val = 10000.0
		match = {}
		for candidate in candidates[0]:
			area_ratio = float(candidate['area'] - data['area']) / float(data['area'])
			round_ratio = float(candidate['roundness'] - data['roundness']) / float(data['roundness'])

			dist = math.fabs(area_ratio) * 0.7 + math.fabs(round_ratio) * 0.3
			if dist > 0.3:
				continue
			if dist < min_val:
				min_val = dist
				match = candidate

		if len(match) != 0:
			matches.append(match)

	length = 35.0
	for match in matches:
		x = match['x_position'] + length * math.cos(match['orientation'] * math.pi / 180.0)
		y = match['y_position'] + length * math.sin(match['orientation'] * math.pi / 180.0)
		cv2.circle(result, (match['y_position'], match['x_position']), 4, len(candidates) + 10)
		cv2.line(result, (match['y_position'], match['x_position']), (int(y), int(x)), len(candidates) + 10, 2)

	return result

