import cv2
import math
import numpy as np
from detect_features import detect_features
from match_features import match_features
from compute_affine_xform import compute_affine_xform
from compute_proj_xform import compute_proj_xform
from ssift_descriptor import ssift_descriptor
from match_features_ssift import match_features_ssift

def ssift_matches_to_harris(matches, result1, result2):
	temp = []
	for match in matches:
		idx1 = idx2 = 0
		feature_coordniate = match[0]
		for i in xrange(0, len(result1)):
			if feature_coordniate == result1[i]:
				idx1 = i
				break

		feature_coordniate = match[1]
		for i in xrange(0, len(result2)):
			if feature_coordniate == result2[i]:
				idx2 = i
				break

		temp.append((idx1, idx2))

	return temp

def draw_matches(matches, img1, img2, corners1, corners2, filename):
	image1 = img1.copy()
	image2 = img2.copy()
	black = np.zeros((image1.shape[0], image1.shape[1], image1.shape[2]))
	for i in xrange(0, image2.shape[0]):
		for j in xrange(0, image2.shape[1]):
			black[i, j] = image2[i, j]

	sbs = np.concatenate((image1, black), axis=1)
	for match in matches:
		coord1 = corners1[match[0]]
		coord2 = corners2[match[1]]
		coord1 = (coord1[1], coord1[0])
		coord2 = (coord2[1] + image1.shape[1], coord2[0])
		cv2.circle(sbs, coord1, 3, (255, 0, 0), -1)
		cv2.circle(sbs, coord2, 3, (255, 0, 0), -1)
		cv2.line(sbs, coord1, coord2, (0, 255, 0), 1)

	cv2.imwrite(filename, sbs)

images = {}
images['graf_1'] = cv2.imread('graf1.png')
images['graf_2'] = cv2.imread('graf2.png')
images['graf_3'] = cv2.imread('graf3.png')

images['bikes_1'] = cv2.imread('bikes1.png')
images['bikes_2'] = cv2.imread('bikes2.png')
images['bikes_3'] = cv2.imread('bikes3.png')

images['leuven_1'] = cv2.imread('leuven1.png')
images['leuven_2'] = cv2.imread('leuven2.png')
images['leuven_3'] = cv2.imread('leuven3.png')

images['wall_1'] = cv2.imread('wall1.png')
images['wall_2'] = cv2.imread('wall2.png')
images['wall_3'] = cv2.imread('wall3.png')

gray_images = {}
corners = {}
for key, value in images.items():
	gray_image = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
	gray_images[key] = gray_image
	corners[key] = detect_features(gray_image, filename=key+'_corners.png', color_image=value)

matches = {}
matches[('bikes_1', 'bikes_2')] = match_features(corners['bikes_1'], corners['bikes_2'], gray_images['bikes_1'], gray_images['bikes_2'])
draw_matches(matches[('bikes_1', 'bikes_2')], images['bikes_1'], images['bikes_2'], corners['bikes_1'], corners['bikes_2'], 'bikes1_bikes2_match.png')

matches[('bikes_1', 'bikes_3')] = match_features(corners['bikes_1'], corners['bikes_3'], gray_images['bikes_1'], gray_images['bikes_3'])
draw_matches(matches[('bikes_1', 'bikes_3')], images['bikes_1'], images['bikes_3'], corners['bikes_1'], corners['bikes_3'], 'bikes1_bikes3_match.png')

matches[('leuven_1', 'leuven_2')] = match_features(corners['leuven_1'], corners['leuven_2'], gray_images['leuven_1'], gray_images['leuven_2'])
draw_matches(matches[('leuven_1', 'leuven_2')], images['leuven_1'], images['leuven_2'], corners['leuven_1'], corners['leuven_2'], 'leuven1_leuven2_match.png')

matches[('leuven_1', 'leuven_3')] = match_features(corners['leuven_1'], corners['leuven_3'], gray_images['leuven_1'], gray_images['leuven_3'])
draw_matches(matches[('leuven_1', 'leuven_3')], images['leuven_1'], images['leuven_3'], corners['leuven_1'], corners['leuven_3'], 'leuven1_leuven3_match.png')

matches[('graf_1', 'graf_2')] = match_features(corners['graf_1'], corners['graf_2'], gray_images['graf_1'], gray_images['graf_2'])
draw_matches(matches[('graf_1', 'graf_2')], images['graf_1'], images['graf_2'], corners['graf_1'], corners['graf_2'], 'graf1_graf2_match.png')

matches[('graf_1', 'graf_3')] = match_features(corners['graf_1'], corners['graf_3'], gray_images['graf_1'], gray_images['graf_3'])
draw_matches(matches[('graf_1', 'graf_3')], images['graf_1'], images['graf_3'], corners['graf_1'], corners['graf_3'], 'graf1_graf3_match.png')

matches[('wall_1', 'wall_2')] = match_features(corners['wall_1'], corners['wall_2'], gray_images['wall_1'], gray_images['wall_2'])
draw_matches(matches[('wall_1', 'wall_2')], images['wall_1'], images['wall_2'], corners['wall_1'], corners['wall_2'], 'wall1_wall2_match.png')

matches[('wall_1', 'wall_3')] = match_features(corners['wall_1'], corners['wall_3'], gray_images['wall_1'], gray_images['wall_3'])
draw_matches(matches[('wall_1', 'wall_3')], images['wall_1'], images['wall_3'], corners['wall_1'], corners['wall_3'], 'wall1_wall3_match.png')

H = compute_affine_xform(matches[('bikes_1', 'bikes_2')], corners['bikes_1'], corners['bikes_2'], images['bikes_1'], images['bikes_2'], filename='bikes1_bikes2_af_ransac.png')
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['bikes_1'].shape[0], images['bikes_1'].shape[1], images['bikes_1'].shape[2]))
sbs = np.concatenate((images['bikes_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['bikes_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('bikes1_bikes2_affine.png', wrap_image)

H = compute_affine_xform(matches[('bikes_1', 'bikes_3')], corners['bikes_1'], corners['bikes_3'], images['bikes_1'], images['bikes_3'], filename='bikes1_bikes3_af_ransac.png')
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['bikes_1'].shape[0], images['bikes_1'].shape[1], images['bikes_1'].shape[2]))
sbs = np.concatenate((images['bikes_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['bikes_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('bikes1_bikes3_affine.png', wrap_image)

H = compute_affine_xform(matches[('leuven_1', 'leuven_2')], corners['leuven_1'], corners['leuven_2'], images['leuven_1'], images['leuven_2'], filename='leuven1_leuven2_af_ransac.png')
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['leuven_1'].shape[0], images['leuven_1'].shape[1], images['leuven_1'].shape[2]))
sbs = np.concatenate((images['leuven_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['leuven_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('leuven1_leuven2_affine.png', wrap_image)

H = compute_affine_xform(matches[('leuven_1', 'leuven_3')], corners['leuven_1'], corners['leuven_3'], images['leuven_1'], images['leuven_3'], filename='leuven1_leuven3_af_ransac.png')
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['leuven_1'].shape[0], images['leuven_1'].shape[1], images['leuven_1'].shape[2]))
sbs = np.concatenate((images['leuven_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['leuven_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('leuven1_leuven3_affine.png', wrap_image)

H = compute_affine_xform(matches[('graf_1', 'graf_2')], corners['graf_1'], corners['graf_2'], images['graf_1'], images['graf_2'], filename='graf1_graf2_af_ransac.png', threshold=5, times=30)
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['graf_1'].shape[0], images['graf_1'].shape[1], images['graf_1'].shape[2]))
sbs = np.concatenate((images['graf_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['graf_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('graf1_graf2_affine.png', wrap_image)

H = compute_affine_xform(matches[('graf_1', 'graf_3')], corners['graf_1'], corners['graf_3'], images['graf_1'], images['graf_3'], filename='graf1_graf3_af_ransac.png', threshold=12, times=100)
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['graf_1'].shape[0], images['graf_1'].shape[1], images['graf_1'].shape[2]))
sbs = np.concatenate((images['graf_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['graf_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('graf1_graf3_affine.png', wrap_image)

H = compute_affine_xform(matches[('wall_1', 'wall_2')], corners['wall_1'], corners['wall_2'], images['wall_1'], images['wall_2'], filename='wall1_wall2_af_ransac.png', threshold=5, times=30)
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['wall_1'].shape[0], images['wall_1'].shape[1], images['wall_1'].shape[2]))
sbs = np.concatenate((images['wall_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['wall_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('wall1_wall2_affine.png', wrap_image)

H = compute_affine_xform(matches[('wall_1', 'wall_3')], corners['wall_1'], corners['wall_3'], images['wall_1'], images['wall_3'], filename='wall1_wall3_af_ransac.png', threshold=12, times=100)
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['wall_1'].shape[0], images['wall_1'].shape[1], images['wall_1'].shape[2]))
sbs = np.concatenate((images['wall_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['wall_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('wall1_wall3_affine.png', wrap_image)

#####  

H = compute_proj_xform(matches[('bikes_1', 'bikes_2')], corners['bikes_1'], corners['bikes_2'], images['bikes_1'], images['bikes_2'], filename='bikes1_bikes2_pj_ransac.png')
black = np.zeros((images['bikes_1'].shape[0], images['bikes_1'].shape[1], images['bikes_1'].shape[2]))
sbs = np.concatenate((images['bikes_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['bikes_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('bikes1_bikes2_projective.png', wrap_image)

H = compute_proj_xform(matches[('bikes_1', 'bikes_3')], corners['bikes_1'], corners['bikes_3'], images['bikes_1'], images['bikes_3'], filename='bikes1_bikes3_pj_ransac.png')
black = np.zeros((images['bikes_1'].shape[0], images['bikes_1'].shape[1], images['bikes_1'].shape[2]))
sbs = np.concatenate((images['bikes_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['bikes_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('bikes1_bikes3_projective.png', wrap_image)

H = compute_proj_xform(matches[('leuven_1', 'leuven_2')], corners['leuven_1'], corners['leuven_2'], images['leuven_1'], images['leuven_2'], filename='leuven1_leuven2_pj_ransac.png')
black = np.zeros((images['leuven_1'].shape[0], images['leuven_1'].shape[1], images['leuven_1'].shape[2]))
sbs = np.concatenate((images['leuven_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['leuven_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('leuven1_leuven2_projective.png', wrap_image)

H = compute_proj_xform(matches[('leuven_1', 'leuven_3')], corners['leuven_1'], corners['leuven_3'], images['leuven_1'], images['leuven_3'], filename='leuven1_leuven3_pj_ransac.png')
black = np.zeros((images['leuven_1'].shape[0], images['leuven_1'].shape[1], images['leuven_1'].shape[2]))
sbs = np.concatenate((images['leuven_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['leuven_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('leuven1_leuven3_projective.png', wrap_image)

H = compute_proj_xform(matches[('graf_1', 'graf_2')], corners['graf_1'], corners['graf_2'], images['graf_1'], images['graf_2'], filename='graf1_graf2_pj_ransac.png', threshold=5, times=30)
black = np.zeros((images['graf_1'].shape[0], images['graf_1'].shape[1], images['graf_1'].shape[2]))
sbs = np.concatenate((images['graf_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['graf_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('graf1_graf2_projective.png', wrap_image)

H = compute_proj_xform(matches[('graf_1', 'graf_3')], corners['graf_1'], corners['graf_3'], images['graf_1'], images['graf_3'], filename='graf1_graf3_pj_ransac.png', threshold=12, times=100)
black = np.zeros((images['graf_1'].shape[0], images['graf_1'].shape[1], images['graf_1'].shape[2]))
sbs = np.concatenate((images['graf_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['graf_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('graf1_graf3_projective.png', wrap_image)

H = compute_proj_xform(matches[('wall_1', 'wall_2')], corners['wall_1'], corners['wall_2'], images['wall_1'], images['wall_2'], filename='wall1_wall2_pj_ransac.png', threshold=5, times=30)
black = np.zeros((images['wall_1'].shape[0], images['wall_1'].shape[1], images['wall_1'].shape[2]))
sbs = np.concatenate((images['wall_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['wall_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('wall1_wall2_projective.png', wrap_image)

H = compute_proj_xform(matches[('wall_1', 'wall_3')], corners['wall_1'], corners['wall_3'], images['wall_1'], images['wall_3'], filename='wall1_wall3_pj_ransac.png', threshold=12, times=100)
black = np.zeros((images['wall_1'].shape[0], images['wall_1'].shape[1], images['wall_1'].shape[2]))
sbs = np.concatenate((images['wall_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['wall_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('wall1_wall3_projective.png', wrap_image)

ssift = {}
for key, value in corners.items():
	ssift[key] = ssift_descriptor(value, gray_images[key], filename=key+'_ssift.png', color_image=images[key])

ssift_matches = {}

ssift_matches[('bikes_1', 'bikes_2')] = match_features_ssift(ssift['bikes_1'], ssift['bikes_2'], threshold=0.7)
ssift_matches[('bikes_1', 'bikes_2')] = ssift_matches_to_harris(ssift_matches[('bikes_1', 'bikes_2')], corners['bikes_1'], corners['bikes_2'])
draw_matches(ssift_matches[('bikes_1', 'bikes_2')], images['bikes_1'], images['bikes_2'], corners['bikes_1'], corners['bikes_2'], 'bikes1_bikes2_sift_match.png')

ssift_matches[('bikes_1', 'bikes_3')] = match_features_ssift(ssift['bikes_1'], ssift['bikes_3'], threshold=0.7)
ssift_matches[('bikes_1', 'bikes_3')] = ssift_matches_to_harris(ssift_matches[('bikes_1', 'bikes_3')], corners['bikes_1'], corners['bikes_3'])
draw_matches(ssift_matches[('bikes_1', 'bikes_3')], images['bikes_1'], images['bikes_3'], corners['bikes_1'], corners['bikes_3'], 'bikes1_bikes3_sift_match.png')

ssift_matches[('leuven_1', 'leuven_2')] = match_features_ssift(ssift['leuven_1'], ssift['leuven_2'], threshold=0.7)
ssift_matches[('leuven_1', 'leuven_2')] = ssift_matches_to_harris(ssift_matches[('leuven_1', 'leuven_2')], corners['leuven_1'], corners['leuven_2'])
draw_matches(ssift_matches[('leuven_1', 'leuven_2')], images['leuven_1'], images['leuven_2'], corners['leuven_1'], corners['leuven_2'], 'leuven1_leuven2_sift_match.png')

ssift_matches[('leuven_1', 'leuven_3')] = match_features_ssift(ssift['leuven_1'], ssift['leuven_3'], threshold=0.7)
ssift_matches[('leuven_1', 'leuven_3')] = ssift_matches_to_harris(ssift_matches[('leuven_1', 'leuven_3')], corners['leuven_1'], corners['leuven_3'])
draw_matches(ssift_matches[('leuven_1', 'leuven_3')], images['leuven_1'], images['leuven_3'], corners['leuven_1'], corners['leuven_3'], 'leuven1_leuven3_sift_match.png')

ssift_matches[('wall_1', 'wall_2')] = match_features_ssift(ssift['wall_1'], ssift['wall_2'], threshold=0.85)
ssift_matches[('wall_1', 'wall_2')] = ssift_matches_to_harris(ssift_matches[('wall_1', 'wall_2')], corners['wall_1'], corners['wall_2'])
draw_matches(ssift_matches[('wall_1', 'wall_2')], images['wall_1'], images['wall_2'], corners['wall_1'], corners['wall_2'], 'wall1_wall2_sift_match.png')

ssift_matches[('wall_1', 'wall_3')] = match_features_ssift(ssift['wall_1'], ssift['wall_3'], threshold=0.95)
ssift_matches[('wall_1', 'wall_3')] = ssift_matches_to_harris(ssift_matches[('wall_1', 'wall_3')], corners['wall_1'], corners['wall_3'])
draw_matches(ssift_matches[('wall_1', 'wall_3')], images['wall_1'], images['wall_3'], corners['wall_1'], corners['wall_3'], 'wall1_wall3_sift_match.png')

ssift_matches[('graf_1', 'graf_2')] = match_features_ssift(ssift['graf_1'], ssift['graf_2'], threshold=0.85)
ssift_matches[('graf_1', 'graf_2')] = ssift_matches_to_harris(ssift_matches[('graf_1', 'graf_2')], corners['graf_1'], corners['graf_2'])
draw_matches(ssift_matches[('graf_1', 'graf_2')], images['graf_1'], images['graf_2'], corners['graf_1'], corners['graf_2'], 'graf1_graf2_sift_match.png')

ssift_matches[('graf_1', 'graf_3')] = match_features_ssift(ssift['graf_1'], ssift['graf_3'], threshold=0.95)
ssift_matches[('graf_1', 'graf_3')] = ssift_matches_to_harris(ssift_matches[('graf_1', 'graf_3')], corners['graf_1'], corners['graf_3'])
draw_matches(ssift_matches[('graf_1', 'graf_3')], images['graf_1'], images['graf_3'], corners['graf_1'], corners['graf_3'], 'graf1_graf3_sift_match.png')

H = compute_affine_xform(ssift_matches[('bikes_1', 'bikes_2')], corners['bikes_1'], corners['bikes_2'], images['bikes_1'], images['bikes_2'], filename='bikes1_bikes2_sift_af_ransac.png')
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['bikes_1'].shape[0], images['bikes_1'].shape[1], images['bikes_1'].shape[2]))
sbs = np.concatenate((images['bikes_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['bikes_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('bikes1_bikes2_sift_affine.png', wrap_image)

H = compute_affine_xform(ssift_matches[('bikes_1', 'bikes_3')], corners['bikes_1'], corners['bikes_3'], images['bikes_1'], images['bikes_3'], filename='bikes1_bikes3_sift_af_ransac.png')
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['bikes_1'].shape[0], images['bikes_1'].shape[1], images['bikes_1'].shape[2]))
sbs = np.concatenate((images['bikes_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['bikes_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('bikes1_bikes3_sift_affine.png', wrap_image)

H = compute_affine_xform(ssift_matches[('leuven_1', 'leuven_2')], corners['leuven_1'], corners['leuven_2'], images['leuven_1'], images['leuven_2'], filename='leuven1_leuven2_sift_af_ransac.png')
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['leuven_1'].shape[0], images['leuven_1'].shape[1], images['leuven_1'].shape[2]))
sbs = np.concatenate((images['leuven_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['leuven_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('leuven1_leuven2_sift_affine.png', wrap_image)

H = compute_affine_xform(ssift_matches[('leuven_1', 'leuven_3')], corners['leuven_1'], corners['leuven_3'], images['leuven_1'], images['leuven_3'], filename='leuven1_leuven3_sift_af_ransac.png')
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['leuven_1'].shape[0], images['leuven_1'].shape[1], images['leuven_1'].shape[2]))
sbs = np.concatenate((images['leuven_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['leuven_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('leuven1_leuven3_sift_affine.png', wrap_image)

H = compute_affine_xform(ssift_matches[('graf_1', 'graf_2')], corners['graf_1'], corners['graf_2'], images['graf_1'], images['graf_2'], filename='graf1_graf2_sift_af_ransac.png', threshold=5, times=30)
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['graf_1'].shape[0], images['graf_1'].shape[1], images['graf_1'].shape[2]))
sbs = np.concatenate((images['graf_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['graf_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('graf1_graf2_sift_affine.png', wrap_image)

H = compute_affine_xform(ssift_matches[('graf_1', 'graf_3')], corners['graf_1'], corners['graf_3'], images['graf_1'], images['graf_3'], filename='graf1_graf3_sift_af_ransac.png', threshold=10, times=100)
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['graf_1'].shape[0], images['graf_1'].shape[1], images['graf_1'].shape[2]))
sbs = np.concatenate((images['graf_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['graf_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('graf1_graf3_sift_affine.png', wrap_image)

H = compute_affine_xform(ssift_matches[('wall_1', 'wall_2')], corners['wall_1'], corners['wall_2'], images['wall_1'], images['wall_2'], filename='wall1_wall2_sift_af_ransac.png', threshold=5, times=30)
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['wall_1'].shape[0], images['wall_1'].shape[1], images['wall_1'].shape[2]))
sbs = np.concatenate((images['wall_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['wall_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('wall1_wall2_sift_affine.png', wrap_image)

H = compute_affine_xform(ssift_matches[('wall_1', 'wall_3')], corners['wall_1'], corners['wall_3'], images['wall_1'], images['wall_3'], filename='wall1_wall3_sift_af_ransac.png', threshold=10, times=100)
H = np.array([[H[0][0], H[0][1], H[0][2]], [H[1][0], H[1][1], H[1][2]]])
black = np.zeros((images['wall_1'].shape[0], images['wall_1'].shape[1], images['wall_1'].shape[2]))
sbs = np.concatenate((images['wall_1'], black), axis=1)
wrap_image = cv2.warpAffine(images['wall_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('wall1_wall3_sift_affine.png', wrap_image)

#####  

H = compute_proj_xform(ssift_matches[('bikes_1', 'bikes_2')], corners['bikes_1'], corners['bikes_2'], images['bikes_1'], images['bikes_2'], filename='bikes1_bikes2_sift_pj_ransac.png')
black = np.zeros((images['bikes_1'].shape[0], images['bikes_1'].shape[1], images['bikes_1'].shape[2]))
sbs = np.concatenate((images['bikes_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['bikes_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('bikes1_bikes2_sift_projective.png', wrap_image)

H = compute_proj_xform(ssift_matches[('bikes_1', 'bikes_3')], corners['bikes_1'], corners['bikes_3'], images['bikes_1'], images['bikes_3'], filename='bikes1_bikes3_sift_pj_ransac.png')
black = np.zeros((images['bikes_1'].shape[0], images['bikes_1'].shape[1], images['bikes_1'].shape[2]))
sbs = np.concatenate((images['bikes_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['bikes_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('bikes1_bikes3_sift_projective.png', wrap_image)

H = compute_proj_xform(ssift_matches[('leuven_1', 'leuven_2')], corners['leuven_1'], corners['leuven_2'], images['leuven_1'], images['leuven_2'], filename='leuven1_leuven2_sift_pj_ransac.png')
black = np.zeros((images['leuven_1'].shape[0], images['leuven_1'].shape[1], images['leuven_1'].shape[2]))
sbs = np.concatenate((images['leuven_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['leuven_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('leuven1_leuven2_sift_projective.png', wrap_image)

H = compute_proj_xform(ssift_matches[('leuven_1', 'leuven_3')], corners['leuven_1'], corners['leuven_3'], images['leuven_1'], images['leuven_3'], filename='leuven1_leuven3_sift_pj_ransac.png')
black = np.zeros((images['leuven_1'].shape[0], images['leuven_1'].shape[1], images['leuven_1'].shape[2]))
sbs = np.concatenate((images['leuven_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['leuven_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('leuven1_leuven3_sift_projective.png', wrap_image)

H = compute_proj_xform(ssift_matches[('graf_1', 'graf_2')], corners['graf_1'], corners['graf_2'], images['graf_1'], images['graf_2'], filename='graf1_graf2_sift_pj_ransac.png', threshold=5, times=30)
black = np.zeros((images['graf_1'].shape[0], images['graf_1'].shape[1], images['graf_1'].shape[2]))
sbs = np.concatenate((images['graf_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['graf_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('graf1_graf2_sift_projective.png', wrap_image)

H = compute_proj_xform(ssift_matches[('graf_1', 'graf_3')], corners['graf_1'], corners['graf_3'], images['graf_1'], images['graf_3'], filename='graf1_graf3_sift_pj_ransac.png', threshold=10, times=100)
black = np.zeros((images['graf_1'].shape[0], images['graf_1'].shape[1], images['graf_1'].shape[2]))
sbs = np.concatenate((images['graf_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['graf_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('graf1_graf3_sift_projective.png', wrap_image)

H = compute_proj_xform(ssift_matches[('wall_1', 'wall_2')], corners['wall_1'], corners['wall_2'], images['wall_1'], images['wall_2'], filename='wall1_wall2_sift_pj_ransac.png', threshold=5, times=30)
black = np.zeros((images['wall_1'].shape[0], images['wall_1'].shape[1], images['wall_1'].shape[2]))
sbs = np.concatenate((images['wall_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['wall_2'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('wall1_wall2_sift_projective.png', wrap_image)

H = compute_proj_xform(ssift_matches[('wall_1', 'wall_3')], corners['wall_1'], corners['wall_3'], images['wall_1'], images['wall_3'], filename='wall1_wall3_sift_pj_ransac.png', threshold=10, times=100)
black = np.zeros((images['wall_1'].shape[0], images['wall_1'].shape[1], images['wall_1'].shape[2]))
sbs = np.concatenate((images['wall_1'], black), axis=1)
wrap_image = cv2.warpPerspective(images['wall_3'], H, (sbs.shape[1], sbs.shape[0]))
wrap_image = wrap_image * 0.5 + sbs * 0.5
cv2.imwrite('wall1_wall3_sift_projective.png', wrap_image)