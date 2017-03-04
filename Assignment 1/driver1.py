#!/usr/bin/python
import numpy
import cv2
import math
from p1 import p1
from p2 import p2
from p3 import p3
from p4 import p4

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

## p1
img = cv2.imread('two_objects.pgm', cv2.IMREAD_UNCHANGED)
copy_img = img.copy()

img = p1(img, 110)
p1_output = img.copy()

p1_output = img_scale(p1_output)

cv2.imwrite('p1_two_objects.jpg', p1_output)

## p2
img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
img = p2(img)
img = img[1 : img.shape[0] - 1, 1 : img.shape[1] - 1]

p2_output = img.copy()
p2_output = img_scale(p2_output)
cv2.imwrite('p2_two_objects.jpg', p2_output)

## p3
result = p3(img)

p3_output = result[1].copy()
p3_output = img_scale(p3_output)
cv2.imwrite('p3_two_objects.jpg', p3_output)

## p4
query = cv2.imread('many_objects_1.pgm', cv2.IMREAD_UNCHANGED)
query = p1(query, 81)
query = cv2.copyMakeBorder(query, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
query = p2(query)
query = query[1 : query.shape[0] - 1, 1 : query.shape[1] - 1]

p4_output = p4(query, result[0])

p4_output = img_scale(p4_output)
cv2.imwrite('p4_many_objects_1.jpg', p4_output)

query = cv2.imread('many_objects_2.pgm', cv2.IMREAD_UNCHANGED)
query = p1(query, 81)
query = cv2.copyMakeBorder(query, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
query = p2(query)
query = query[1 : query.shape[0] - 1, 1 : query.shape[1] - 1]

p4_output = p4(query, result[0])

p4_output = img_scale(p4_output)
cv2.imwrite('p4_many_objects_2.jpg', p4_output)









