import numpy as np
from model.utils import get_train_val,batch_iterator,plot_learning_curve
from model.metrics import neg_multi_logloss
from model.gradients import MSE_grad
from model.activation_classes import Softmax,ReLU,LeakyReLU
from model.gradients import neg_logloss_softmax_grad

def initialize_weight(shape):
	'''
	Kaiming He normal initialization
	'''
	return np.random.rand(shape[0],shape[1]) * np.sqrt(2/shape[0])
	

class CustomNeuralNetwork():
	'''
	Simple neural network for binary classification
	'''
	def __init__(self,layers,act_class):
		self.act_class= act_class
		self.weights = [[initialize_weight((layers[i],layers[i+1])) , 
						initialize_weight((1,layers[i+1]))] for i in range(len(layers)-1)]
		self.train_losses=[]
		self.val_losses=[]
	def forward_pass(self,X,eval=False):
		self.X_inputs = [X]
		inp = X
		for i,w in enumerate(self.weights):
			inp = inp @ w[0] + w[1]
			if eval: self.X_inputs.append(inp)
			if i<len(self.weights)-1: inp = self.act_class()(inp)
		
		#output layer
		y_outp = Softmax()(inp)
		return y_outp

	def backward_pass(self,y,y_pred,l2):
		# assuming we have 2 weights with shape (400,200) and (200,10)
		# grad of rightmost layer
		bs = len(y)
		grad_wrt_input = neg_logloss_softmax_grad(y,y_pred) # (n,10)
		grad_wbias = (1/bs) * grad_wrt_input
		grad_w = (1/bs) * self.X_inputs[-1].T @ grad_wrt_input # (200,10)
		grad_w += (l2/bs) * 
		# TODO: add l2
		self.X_inputs=[]
	def fit_epoch(self,X,y,lr,epochs,bs,l2=0,val_ratio=0.2):
		'''
		Fit data using stochastic gradient descent and l2 regularization
		'''
		X_train,y_train,X_val,y_val = get_train_val(X,y,val_ratio)
		for epoch in range(epochs):
			train_cumloss,val_cumloss = 0,0
			# get batch from train set
			for xb,yb in batch_iterator(X_train,y_train,bs):
				y_pred = self.forward_pass(xb)
				train_cumloss+= -1 * neg_multi_logloss(yb,y_pred) * len(xb)

				self.backward_pass(yb,y_pred,l2)

			# get double of bs from validation set (since there's less calculation for prediction)
			for xb,yb in batch_iterator(X_val,y_val,bs*2):
				y_pred = self.forward_pass(xb,eval=True)
				val_cumloss += -1 * neg_multi_logloss(yb,y_pred) * len(xb)

			self.train_losses.append(train_cumloss/ len(X_train))
			self.val_losses.append(val_cumloss / len(X_val))
			print(f'Epoch {epoch+1}. Training loss: {self.train_losses[-1]}, Val loss:{self.val_losses[-1]}')
		plot_learning_curve(self.train_losses,self.val_losses)

	def predict(self,X,thres=0.5):
		if X.shape[1] == self.dim:
			X0 = np.array([[1]*X.shape[0]]).T # nx1
			X = np.concatenate((X0,X),axis=1)
		y_pred= self.act_fn(np.squeeze(X @ self.W))
		if not self.is_reg:
			y_pred = (y_pred >= thres).astype(np.uint8)
		return y_pred
	def predict_proba(self,X):
		if self.is_reg:
			raise Exception('Cannot predict probability for regression')
		if X.shape[1] == self.dim:
			X0 = np.array([[1]*X.shape[0]]).T # nx1
			X = np.concatenate((X0,X),axis=1)
		return self.act_fn(np.squeeze(X @ self.W))