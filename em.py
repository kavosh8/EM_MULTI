import numpy
import sys
import numpy.linalg

class em_learner:
	"""docstring for ClassName"""
	def __init__(self, params):
		self.num_iterations=params['num_iterations']
		self.gaussian_variance=params['gaussian_variance']
		self.N=params['num_models']
		self.learned_priors=(self.N)*[1.0/self.N]
		self.observation_size=params['observation_size']



	def log_likelihood(self,sample_labels,line_labels):
		"""  
			output log-likehood of each sample coming from each line
			sample_labels: actual labels from environement (y)
			line_labels: labels predicted by functions (y_hat) one per function
		"""
		X,Y=len(sample_labels),len(line_labels)
		out=numpy.zeros((X,Y))

		for x_index in range(X):
			for y_index in range(Y):

				nm=numpy.linalg.norm( sample_labels[x_index] - line_labels[y_index][x_index] )
				out[x_index,y_index]=-(nm*nm)/(2.*self.gaussian_variance)

		return out



	def posterior(self,log_p_sample_given_z):
		'''
		compute p(z|sample) and output, for each z, a numpy array containint p(sample) via p_li
		takes p(sample|z) as input: log_p_sample_given_z is actually log of this quantity
		'''
		num_modes=log_p_sample_given_z.shape[1]
		p_z_given_sample=numpy.zeros_like(log_p_sample_given_z)
		obj=0
		num_samples=log_p_sample_given_z.shape[0]
		for i in range(num_samples):
			#compute probabilities for each sample -- p(z|sample)=p(sample|z)p(z)/sum_z (p(sample|z)p(z))
			clipped=numpy.clip(log_p_sample_given_z[i,:],a_min=-100, a_max=0)
			exped=numpy.exp(clipped)
			top=numpy.multiply(exped,numpy.array(self.learned_priors))
			p_z_given_sample[i,:]=top/numpy.sum(top)
			#compute probabilities for each sample



			#compute em objective note that E[log xy] is E[log x] + E[log y]
			obj=obj+numpy.dot(p_z_given_sample[i,:],log_p_sample_given_z[i,:])+\
					numpy.dot(p_z_given_sample[i,:],numpy.log(self.learned_priors))
			#compute em objective



			
		p_z_given_sample=numpy.clip(p_z_given_sample,a_min=1e-5, a_max=1)
		li=[p_z_given_sample[:,i] for i in range(num_modes)]

		return li,obj


	def compute_learned_prior(self,w_li):# compute best priors, where best is defined as the prior that maximizes lower bound
		for n in range(self.N):#in this case \mean_x p(z|x)
			self.learned_priors[n]=numpy.mean(w_li[n])

	def m_step(self,tm,phi,y,w_li,iteration):
		tm.regression(phi,y,w_li,iteration)#M step fits however many functions (fn)
		self.compute_learned_prior(w_li)# and the prior to maximize the lower bound.

	def e_step(self,tm,phi,y,iteration_number):
		
		line_labels=tm.predict(phi)# get y_hat from each fn, given x values. a list of numpy arrays
		p_sample_given_z=self.log_likelihood(y,line_labels)# get log-likelihood of each sample=(x,y) coming from each fn
														   # so this outputs a 2D matrix of size (num_samples,num_modes)
		p_z_given_sample,obj=self.posterior(p_sample_given_z)#get posterior p(z|(x,y))
		return p_z_given_sample,obj

	def e_step_m_step(self,tm,phi,y,iteration):
		w_li,obj=self.e_step(tm,phi,y,iteration)#E step computes posteriors w_li and EM objective obj
		print(iteration,obj)
		self.m_step(tm,phi,y,w_li,iteration)
		return obj