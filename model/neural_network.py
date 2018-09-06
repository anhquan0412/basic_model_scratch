from model.imports import *
from model.utils import get_train_val,batch_iterator,plot_learning_curve
from model.metrics import MSE
from model.gradients import MSE_grad

def initialize_weight(shape):
	'''
	Kaiming He normal initialization
	'''
	return np.random.rand(shape[0]+1,shape[1]) * np.sqrt(2/shape[0])

class CustomNeuralNetwork():
	def __init__(self,layers,loss_fn,grad_fn,act_fn = lambda x: x):
		self.act_fn,self.loss_fn,self.grad_fn = act_fn,loss_fn,grad_fn
		self.weights = [initialize_weight((layers[i],layers[i+1])) for i in range(len(layers)-1)]
		self.train_losses=[]
		self.val_losses=[]
	
	def forward_pass(self,X,y):
		self.X_inputs = []
		X_ones= np.ones([X.shape[0],1])
		inp = X
		for w in self.weights:
			inp = np.concatenate((X_ones,inp),axis=1)
			self.X_inputs.append(inp)
			inp = self.act_fn(inp @ w)
	def fit_epoch(self,X,y,lr,epochs,bs,l2=0,val_ratio=0.2):
		'''
		Fit data using stochastic gradient descent and l2 regularization
		'''
		X_train,y_train,X_val,y_val = get_train_val(X,y,val_ratio)
		for epoch in range(epochs):
			train_cumloss,val_cumloss = 0,0
			# get batch from train set
			for xb,yb in batch_iterator(X_train,y_train,bs):
				y_pred = self.act_fn(np.squeeze(xb @ self.W))
				train_cumloss+= self.loss_fn(yb,y_pred) * len(xb)

				grad = self.grad_fn(yb,y_pred)
				grad_w = xb.T @ grad
				if len(grad_w.shape)==1: grad_w = grad_w[:,None]
				grad_w[1:,:]+= 2*(l2/len(xb))*self.W[1:,:]
				self.W-= lr*grad_w 

			# get double of bs from validation set (since there's less calculation for prediction)
			for xb,yb in batch_iterator(X_val,y_val,bs*2):
				y_pred = self.act_fn(np.squeeze(xb @ self.W))
				val_cumloss += self.loss_fn(yb,y_pred) * len(xb)

			self.train_losses.append(train_cumloss/ len(X_train))
			self.val_losses.append(val_cumloss / len(X_val))
			print(f'Epoch {epoch+1}. Training loss: {self.train_losses[-1]}, Val loss:{self.val_losses[-1]}')
		plot_learning_curve(self.train_losses,self.val_losses)

	def get_weight(self):
		return self.W
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