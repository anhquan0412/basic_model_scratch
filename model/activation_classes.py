import numpy as np
class Sigmoid():
	def __call__(self,x):
		return 1/(1+np.exp(-x))
	def grad(self,x):
		x_acted = self.__call__(x)
		return x_acted*(1-x_acted)

class Softmax():
	def __call__(self,x):
		return np.exp(x) / np.sum(np.exp(x), axis=1)[:,None]

class ReLU():
	def __call__(self, x):
		# return np.maximum(x,0)
		return np.where(x >= 0, x, 0)

	def grad(self, x):
		return np.where(x >= 0, 1, 0)

class LeakyReLU():
	def __init__(self, alpha=0.2):
		self.alpha = alpha

	def __call__(self, x):
		return np.where(x >= 0, x, self.alpha * x)

	def grad(self, x):
		return np.where(x >= 0, 1, self.alpha)
