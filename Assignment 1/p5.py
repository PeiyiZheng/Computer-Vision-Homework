#!/usr/bin/python
import numpy
import cv2
from p1 import p1

def p5(image_in): #return edge_image_out
	image_in = cv2.GaussianBlur(image_in,(3,3),0)
	#image_in = numpy.uint16(image_in)
	laplacian = cv2.filter2D(image_in, -1, numpy.array([[-1, -4, -1], [-4, 20, -4], [-1, -4, -1]]))

	return laplacian
