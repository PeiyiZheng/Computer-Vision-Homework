#! /usr/bin/python
import cv2
import numpy as np
import math
import random
np.set_printoptions(threshold=np.nan)
def drawMatches(img1, kpts_1, img2, kpts_2, matches, name):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    output = np.zeros((max([rows1, rows2]), cols1 + cols2, 3))
    output[:rows1, :cols1] = np.dstack([img1])
    output[:rows2, cols1:] = np.dstack([img2])

    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = kpts_1[img1_idx].pt
        (x2, y2) = kpts_2[img2_idx].pt

        cv2.circle(output, (int(x1), int(y1)), 3, (0, 255, 0), -1)
        cv2.circle(output, (int(x2) + cols1, int(y2)), 3, (0, 255, 0), -1)
        cv2.line(output, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (0, 255, 0), 1)

    cv2.imwrite(name, output)

def getF(kpts_1, kpts_2, matches):
	uv = []
	for i in xrange(len(matches)):
		img1_idx = matches[i].queryIdx
		img2_idx = matches[i].trainIdx
		(x1, y1) = kpts_1[img1_idx].pt
		(x2, y2) = kpts_2[img2_idx].pt
		uv.append([x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1])
	
	w, v = np.linalg.eig(np.mat(uv).transpose() * np.mat(uv))
	sort_idx = w.argsort()
	v = v[:, sort_idx]

	F = np.array([[v.item(0, 0), v.item(1, 0), v.item(2, 0)], [v.item(3, 0), v.item(4, 0), v.item(5, 0)], [v.item(6, 0), v.item(7, 0), v.item(8, 0)]]).real
	return F

img1 = cv2.imread('hopkins1.JPG')
img2 = cv2.imread('hopkins2.JPG')

orb = cv2.ORB()
keypoints1 = orb.detect(img1, None)
keypoints1, descriptors1 = orb.compute(img1, keypoints1)

orb = cv2.ORB()
keypoints2 = orb.detect(img2, None)
keypoints2, descriptors2 = orb.compute(img2, keypoints2)

bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bfm.match(descriptors1, descriptors2)
dist = [m.distance for m in matches]
thres_dist = (sum(dist) / len(dist)) * 0.6
matches = [m for m in matches if m.distance < thres_dist]

drawMatches(img1, keypoints1, img2, keypoints2, matches, '4a_matches.jpg')

f = getF(keypoints1, keypoints2, matches)
print f

sample_keypoints = [keypoints1[matches[i].queryIdx] for i in sorted(random.sample(xrange(len(matches)), 8))]

for keypoint in sample_keypoints:
	(u, v) = keypoint.pt
	cv2.circle(img1, (int(u), int(v)), 4, (0, 255, 0), -1)
	a = f[0][0] * u + f[1][0] * v + f[2][0]
	b = f[0][1] * u + f[1][1] * v + f[2][1]
	c = f[0][2] * u + f[1][2] * v + f[2][2]
	
	p1_x = 0
	p1_y = (-c - a * p1_x) / b

	p2_x = img1.shape[1]
	p2_y = (-c - a * p2_x) / b
	cv2.line(img2, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), (0, 255, 0), 1)

sbs = np.concatenate((img1, img2), axis=1)
cv2.imwrite('4c_epipolar.jpg', sbs)
 

