import numpy
import grid
import random
import sys
import matplotlib.pyplot as plt

class QLearning(object):
	def __init__(self,policy="eps-greedy",lr=0.1,epsilon=0.1,gamma=0.99,grid_range=5,action_range=4):
		self.lr=lr
		self.epsilon=epsilon
		self.gamma=gamma
		self.policy=policy
		self.initialize_q_table(grid_range,action_range)

	def initialize_q_table(self,grid_range,action_range):
		self.table=numpy.zeros((grid_range,grid_range,grid_range,grid_range,action_range))

	def update(self,s,sp,r,a,done):
		if done==True:
			self.table[tuple(list(s)+[a])] = self.table[tuple(list(s)+[a])]+self.lr*(r-self.table[tuple(list(s)+[a])])
		elif done==False:
			self.table[tuple(list(s)+[a])] = self.table[tuple(list(s)+[a])]+\
											self.lr*(r+self.gamma*numpy.max(self.table[tuple(list(sp))])-self.table[tuple(list(s)+[a])])

	def act(self,s):
		if self.policy=="eps-greedy":
			return self.e_greedy(s)
		else:
			print("other policies not implemented yet")
			sys.exit(1)

	def e_greedy(self,s):
		if numpy.random.random()<self.epsilon:
			return random.sample(range(4),1)[0]
		else:
			return numpy.argmax(self.table[tuple(list(s))])