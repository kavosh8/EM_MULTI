import grid, q_learning
import matplotlib.pyplot as plt
import numpy

max_seeds=10
max_episode=1000
lr=0.025


env=grid.grid_env(True)
G_list_all_seeds=[]

for seed in range(max_seeds):
	print("starting seed number {}".format(seed))
	numpy.random.seed(seed)
	Q_learner=q_learning.QLearning(lr=lr)
	G_list_one_seed=[]

	for episode in range(max_episode):
		s=env.reset()
		G=0
		while True:
			a=Q_learner.act(s)
			sp,r,done = env.step([a])
			if episode==max_episode-1:
				env.plot()
				plt.pause(0.2)
			G=G+r
			Q_learner.update(s,sp,r,a,done)
			if done:
				G_list_one_seed.append(G)
				break
			s=sp
	G_list_all_seeds.append(G_list_one_seed)
plt.close()
plt.plot(numpy.mean(G_list_all_seeds,axis=0))
plt.show()
plt.pause(5)
plt.close()