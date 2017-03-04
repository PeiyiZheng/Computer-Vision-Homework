#!/usr/bin/python
import numpy
import cv2
import math

def p3(labels_in):
	objects = {}

	for (i, j), val in numpy.ndenumerate(labels_in):
		if val == 0:
			continue

		if val in objects:
			objects[val].append((i, j))
		else:
			objects[val] = [(i, j)]

	database_out = []
	for key, val in objects.items():
		if len(val) < 20:
			continue
		object_label = key
		
		x_center = y_center = a_ = b_ = c_ = 0

		A = len(val)

		for xy in val:
			x_center += xy[1]
			y_center += xy[0]

			a_ += xy[0] * xy[0]
			b_ += xy[0] * xy[1]
			c_ += xy[1] * xy[1]

		x_center /= len(val)
		y_center /= len(val)
		b_ *= 2


		a = a_ - (y_center ** 2) * A
		b = b_ - 2 * x_center * y_center * A
		c = c_ - (x_center ** 2) * A

		lamda1 = 0.5 * math.atan2(b, a - c)
		lamda2 = lamda1 + math.pi * 0.5

		orientation = lamda1 / math.pi * 180.0

		min_moment = (a * (math.sin(lamda1) ** 2)) - (b * math.sin(lamda1) * math.cos(lamda1)) + (c * (math.cos(lamda1) ** 2))
		max_moment = (a * (math.sin(lamda2) ** 2)) - (b * math.sin(lamda2) * math.cos(lamda2)) + (c * (math.cos(lamda2) ** 2))

		roundness = min_moment / max_moment

		database_out.append({'object_label':object_label, 'x_position':y_center, 'y_position':x_center, 'min_moment':min_moment, 'orientation':orientation,'roundness':roundness,'area':A})

	length = 35.0
	for data in database_out:
		x = data['x_position'] + length * math.cos(data['orientation'] * math.pi / 180.0)
		y = data['y_position'] + length * math.sin(data['orientation'] * math.pi / 180.0)
		cv2.circle(labels_in, (data['y_position'], data['x_position']), 4, len(database_out) + 2)
		cv2.line(labels_in, (data['y_position'], data['x_position']), (int(y), int(x)), len(database_out) + 2, 2)

	return (database_out, labels_in)