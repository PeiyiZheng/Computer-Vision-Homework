#! /usr/bin/python
import sys
import os
import cv2
import numpy as np
import math
from sklearn.svm import LinearSVC

np.set_printoptions(threshold=np.nan)
mode = 0
if len(sys.argv) == 2:
	if sys.argv[1] == '1':
		mode = 1

features = []
feature_list = {}
feature_len = 0
num_of_doc = 0
for root, dirs, files in os.walk("train"):
	for dir_name in dirs:
		for sub_root, sub_dirs, sub_files in os.walk("train/" + dir_name):
			feature_list[dir_name] = []
			for sub_name in sub_files:
				img = cv2.imread(sub_root + '/' + sub_name)
				num_of_doc += 1

				min_len = min(img.shape[0], img.shape[1])
				if min_len > 800:
					factor = 800.0 / min_len
					img = cv2.resize(img, (0,0), fx=factor, fy=factor) 

				orb = cv2.ORB()
				keypoints = orb.detect(img, None)
				keypoints, descriptors = orb.compute(img, keypoints)
				if descriptors is not None:
					features.extend(descriptors)
					feature_idx = [i for i in xrange(feature_len,len(features))]
					feature_list[dir_name].append(feature_idx)
					feature_len = len(features)
				else:
					feature_list[dir_name].append([])

features = np.float32(features)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center = cv2.kmeans(features, 800, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

np.savetxt('cluster_center.txt', center, delimiter=', ', fmt='%.4f')

idf = [0.0 for x in xrange(800)]
tf = {}
for key in feature_list:
	tf[key] = []
	for feature in feature_list[key]:
		if len(feature) == 0:
			tf[key].append([])
			continue

		histogram = [0.0 for x in xrange(800)]
		for i in xrange(len(feature)):
			feature[i] = label[feature[i]]
			histogram[feature[i][0]] += 1.0


		for i in xrange(800):
			if histogram[i] != 0.0:
				idf[i] += 1.0

		for i in xrange(800):
			histogram[i] /= float(len(feature))

		tf[key].append(histogram)

for i in xrange(800):
	idf[i] = math.log(num_of_doc / idf[i]) / math.log(2)

database = []
categories = []
for key in tf:
	for hist in tf[key]:
		if len(hist) == 0:
			continue

		for i in xrange(800):
			hist[i] *= idf[i]

		hist = np.array(hist)
		hist /= np.sqrt(hist.dot(hist))

		database.append(hist)
		categories.append(key)

np.savetxt('bow_encoding.txt', database, delimiter=', ', fmt='%.4f')

knn = cv2.KNearest()
category = np.array([i for i in xrange(800)])
knn.train(center, category)

correct = wrong = 0
test_imgs = []
test_labels = []
for root, dirs, files in os.walk("test"):
	for dir_name in dirs:
		for sub_root, sub_dirs, sub_files in os.walk("test/" + dir_name):
			for sub_name in sub_files:
				img = cv2.imread(sub_root + '/' + sub_name)

				min_len = min(img.shape[0], img.shape[1])
				if min_len > 800:
					factor = 800.0 / min_len
					img = cv2.resize(img, (0,0), fx=factor, fy=factor) 

				orb = cv2.ORB()
				keypoints = orb.detect(img, None)
				keypoints, descriptors = orb.compute(img, keypoints)
				if descriptors is not None:
					temp = []
					temp.extend(descriptors)
					temp = np.matrix(temp).astype(np.float32)
					ret, results, neighbours, dist = knn.find_nearest(temp, 1)

					hist = [0.0 for x in xrange(800)]
					for idx in results:
						hist[int(idx)] += 1.0

					for i in xrange(800):
						hist[i] = hist[i] / float(len(results))
						hist[i] *= idf[i]

					hist = np.array(hist)
					hist /= np.sqrt(hist.dot(hist))

					test_imgs.append(hist)
					test_labels.append(dir_name)

lbs = np.array([i for i in xrange(len(database))])
database = np.matrix(database).astype(np.float32)
test_imgs = np.matrix(test_imgs).astype(np.float32)

if mode == 0:
	knn1 = cv2.KNearest()
	knn1.train(database, lbs)
	ret, results, neighbours, dist = knn1.find_nearest(test_imgs, 1)

if mode == 1:
	classifier = LinearSVC(random_state=0, C=1.0, loss='hinge', penalty='l2')
	classifier.fit(database, categories)
	results = classifier.predict(test_imgs)

for i in xrange(len(test_imgs)):
	if mode == 1:
		if test_labels[i] == results[i]:
			correct += 1.0
		else:
			wrong += 1.0
	else:
		if test_labels[i] == categories[int(results[i])]:
			correct += 1.0
		else:
			wrong += 1.0

print correct / (correct + wrong)