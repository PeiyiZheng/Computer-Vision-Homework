#!/usr/bin/python
import numpy
import cv2
 
def p1(gray_in, thresh_val): # return binary_out
	#gray_in = cv2.GaussianBlur(gray_in,(3,3),0)
	gray_in[gray_in <= thresh_val] = 0
	gray_in[gray_in > thresh_val] = 1
	
	return gray_in