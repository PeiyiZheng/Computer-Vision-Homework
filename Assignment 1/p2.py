#!/usr/bin/python
import numpy
import cv2

parent = {}

def findParent(node):
	if parent[node] != node:
		parent[node] = findParent(parent[node])

	return parent[node]

def mergeSet(node_a, node_b):
	node_a = findParent(node_a)
	node_b = findParent(node_b)

	if node_a < node_b:
		parent[node_b] = node_a
	else:
		parent[node_a] = node_b
 
def p2(binary_in): # return labels_out
	binary_in = numpy.uint16(binary_in)
	label = 2

	# pass 1
	for (x, y), val in numpy.ndenumerate(binary_in):
		if val == 0:
			continue

		D = binary_in.item(x - 1, y - 1)
		if D != 0:
			binary_in.itemset((x, y), D)
			continue

		B = binary_in.item(x - 1, y)
		C = binary_in.item(x, y - 1)
		if B == 0 and C == 0:
			binary_in.itemset((x, y), label)
			parent[label] = label
			label += 1
		else:
			if B == 0:
				binary_in.itemset((x, y), C)

			if C == 0:
				binary_in.itemset((x, y), B)

			if C != 0 and B != 0:
				if B < C:
					binary_in.itemset((x, y), B)
				else:
					binary_in.itemset((x, y), C)
				
				mergeSet(B, C)
				

	lb_set = set()

	# pass 2
	for (x, y), val in numpy.ndenumerate(binary_in):
		if val == 0:
			continue

		binary_in.itemset((x, y), findParent(val))

	for (x, y), val in numpy.ndenumerate(binary_in):
		if val == 0:
			continue

 		lb_set.add(val)
		
	label_map = {}
	idx = 1
	for lb in lb_set:
		label_map[lb] = idx
		idx += 1

	for (x, y), val in numpy.ndenumerate(binary_in):
		if val == 0:
			continue

		binary_in.itemset((x, y), label_map[binary_in.item(x, y)])

	return binary_in





