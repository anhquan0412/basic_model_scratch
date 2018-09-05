from model.imports import *
from model.utils import get_train_val,batch_iterator,plot_learning_curve
from model.metrics import MSE
from model.gradients import MSE_grad

def initialize_weight(dim):
	W0 = np.array([[0]]) # bias, 1x1
	W= np.random.rand(dim,1)
	return np.concatenate((W0,W))

class CustomLinearModel():
	def __init__(self,dim,is_reg,loss_fn,grad_fn,act_fn = lambda x: x):
		self.dim,self.act_fn,self.loss_fn,self.grad_fn,self.is_reg = dim,act_fn,loss_fn,grad_fn,is_reg

		self.W = initialize_weight(self.dim)
		self.train_losses=[]
		self.val_losses=[]
	def fit(self,X,y,lr,l2=0,n_iteration=50,val_ratio=.2):
		'''
		Fit data using gradient descent and l2 regularization
		'''
		X_train,y_train,X_val,y_val = get_train_val(X,y,val_ratio)
		for i in range(n_iteration):      
			y_pred = self.act_fn(np.squeeze(X_train @ self.W))
			# MSE loss for regression
			loss = self.loss_fn(y_train,y_pred)
			grad = self.grad_fn(y_train,y_pred) # shape (n,)
			grad_w = X_train.T @ grad # shape (dim,)
			
			if len(grad_w.shape)==1: grad_w = grad_w[:,None] # turn (dim,) to (dim,1)
			#ignore update of grad_w0 (bias term) since w0 does not contribute to regularization process
			grad_w[1:,:]+= 2*(l2/len(X_train))*self.W[1:,:] # (2 *lambda / m)* weight

			self.W-= lr*grad_w 

			#save training loss
			self.train_losses.append(loss)
			#predict validation set
			y_pred = self.act_fn(np.squeeze(X_val @ self.W))
			val_loss = self.loss_fn(y_val,y_pred)
			self.val_losses.append(val_loss)
			if (i+1) % 20 == 0:
				print(f'{i+1}. Training loss: {loss}, Val loss:{val_loss}')

		plot_learning_curve(self.train_losses,self.val_losses)

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
	def predict(self,X_test,thres=0.5):
		if X_test.shape[1] == self.dim:
			X0 = np.array([[1]*X_test.shape[0]]).T # nx1
			X_test = np.concatenate((X0,X_test),axis=1)
		y_pred= self.act_fn(np.squeeze(X_test @ self.W))
		if not self.is_reg:
			y_pred = (y_pred >= thres).astype(np.uint8)
		return y_pred
	def predict_proba(self,X_test):
		if self.is_reg:
			raise Exception('Cannot predict probability for regression')
		if X_test.shape[1] == self.dim:
			X0 = np.array([[1]*X_test.shape[0]]).T # nx1
			X_test = np.concatenate((X0,X_test),axis=1)
		return self.act_fn(np.squeeze(X_test @ self.W))