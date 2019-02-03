import numpy
import grid
import random
import sys
import matplotlib.pyplot as plt
import utils

s_li=[]
a_li=[]
sp_li=[]

env=grid.grid_env(True)
s=env.reset()

for count in range(5000):
	
	a=numpy.random.randint(0,4)
	sp,r,done = env.step([a])
	a_arr=utils.action_index_to_one_hot(a)

	#add stuff to list
	s_li.append(s[-2:])#just ghost

	if done==True:
		s=env.reset()
	else:
		s=sp

	a_li.append(a)
	sp_li.append(sp[-2:])#just ghost
	#add stuff to list

s_li=numpy.array(s_li)
a_li=numpy.array(a_li)
sp_li=numpy.array(sp_li)

numpy.savetxt("s_li.data",s_li)
numpy.savetxt("a_li.data",a_li)
numpy.savetxt("sp_li.data",sp_li)


###test see if save and load works
s_li=numpy.loadtxt("s_li.data")
a_li=numpy.loadtxt("a_li.data")
sp_li=numpy.loadtxt("sp_li.data")
print(s_li.shape)
