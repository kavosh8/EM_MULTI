import numpy


def action_index_to_one_hot(index):
	arr=numpy.zeros((1,4))
	arr[0,index]=1
	return arr
