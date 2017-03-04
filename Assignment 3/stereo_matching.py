#! /usr/bin/python
import cv2
import numpy as np
import math

def img_scale(image_in):
	mx = 0
	for (x, y), val in np.ndenumerate(image_in):
		if val != 0:
			if val > mx:
				mx = val

	if mx == 0:
		return image_in

	mx = 255 / mx
	for (x, y), val in np.ndenumerate(image_in):
		if val != 0:
			image_in.itemset((x, y), image_in.item(x, y) * mx)

	return image_in

np.set_printoptions(threshold=np.nan)
img_l = cv2.imread('scene_l.bmp', cv2.IMREAD_UNCHANGED)
img_r = cv2.imread('scene_r.bmp', cv2.IMREAD_UNCHANGED)

radius = 7
ncc_l = []
ncc_r = []

for i in xrange(img_l.shape[0]):
	ncc_l_row = []
	ncc_r_row = []
	for j in xrange(img_l.shape[1]):
		if i - radius < 0 or j - radius < 0 or i + radius >= img_l.shape[0] or j + radius >= img_l.shape[1]:
			ncc_l_row.append((np.array([[]]), -1))
			ncc_r_row.append((np.array([[]]), -1))
			continue

		template_l = img_l[i - radius : i + radius, j - radius : j + radius]
		template_r = img_r[i - radius : i + radius, j - radius : j + radius]

		template_l = template_l - template_l.mean()
		template_r = template_r - template_r.mean()

		ncc_l_row.append((template_l, np.sqrt(np.sum(template_l ** 2))))
		ncc_r_row.append((template_r, np.sqrt(np.sum(template_r ** 2))))

	ncc_l.append(ncc_l_row)
	ncc_r.append(ncc_r_row)

d_map = np.zeros((img_l.shape[0], img_l.shape[1]))
for i in xrange(img_l.shape[0]):
	for j in xrange(img_l.shape[1]):
		if ncc_l[i][j][1] < 0:
			continue

		maximum = -2.0
		best_offset = 1
		offset = -1
		while offset > -60 and j + offset >= 0:
			if ncc_r[i][j + offset][1] < 0:
				break

			numerator = ncc_l[i][j][0] * ncc_r[i][j + offset][0]
			cr = np.sum(numerator) / (ncc_l[i][j][1] * ncc_r[i][j + offset][1])

			if cr > maximum:
				maximum = cr
				best_offset = offset

			offset -= 1

		if maximum > -2.0:
			d_map[i][j] = 40000 / -best_offset

f = open('2b_3d_points.txt', 'w')
for i in xrange(img_l.shape[0]):
	for j in xrange(img_r.shape[1]):
		if d_map[i][j] == 0:
			continue

		z = d_map[i][j]
		x = z * i / 400.0
		y = z * j / 400.0

		f.write(str(x) + ' ' + str(y) + ' ' + str(z) + '\n')
f.close()
d_map = img_scale(d_map)
cv2.imwrite('2a_depth_map.jpg', d_map)
