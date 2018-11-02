import numpy as np
from model.utils import get_train_val,batch_iterator,plot_learning_curve
from model.metrics import multi_logloss
from model.gradients import MSE_grad
from model.activation_classes import Softmax,ReLU,LeakyReLU
from model.gradients import logloss_softmax_grad
from IPython.core.debugger import set_trace
from model.optimizers import GradientDescent

def init_weight(shape):
	'''
	Kaiming He normal initialization
	'''
	np.random.seed(42)
	return [np.random.uniform(size=shape) * np.sqrt(2/shape[0]), np.zeros((1,shape[1]))]


def drop_out(X_act,keep_prob):
	'''
	Inverted dropout implementation
	'''
	mask = np.random.rand(*X_act.shape) <= keep_prob
	X_act= (X_act*mask)/keep_prob
	return X_act,mask
class CustomNeuralNetwork():
	'''
	Simple neural network for binary classification
	'''
	def __init__(self,layers,act_obj,opt= GradientDescent,keep_prob=1.0):
		'''
		Layers include output layer 
		E.g for 10 output classification, input layer size 400 and 1 hidden layer size 200: [400,200,10]

		act_obj: object (or list of objects) from activation_classes module to apply before reaching each hidden layer (exclude the last SoftMax activation before loss calculation)
		keep_prob: keep probability (or list) for drop out at each hidden layer. Note that we don't drop out the first layer
		'''
		self.act_objs= [act_obj]*(len(layers)-2) if type(act_obj)!=list else act_obj
		self.keep_probs = [keep_prob]*(len(layers)-2) if type(keep_prob)!=list else keep_prob
		# list of [weight,bias]
		self.weights = [init_weight((layers[i],layers[i+1])) for i in range(len(layers)-1)]
		self.opt = opt(layers)
		assert len(self.act_objs) == len(self.keep_probs),'# of activation objs and # of keep probs must be equal'
		assert len(self.act_objs) == len(layers)-2, 'We only need (# of hidden layers) activation objects'
		
		# class params
		self.train_losses,self.val_losses=[],[]
		self.X_inputs,self.X_acts,self.X_masks=[],[],[]

	def forward_pass(self,X,train):
		if train: 
			self.X_inputs,self.X_acts,self.X_masks = [X],[X],[]
		inp = X
		for i,w in enumerate(self.weights):
			inp = inp @ w[0] + w[1]
			if i<len(self.weights)-1:
				if train: self.X_inputs.append(inp) 
				#activation
				inp = self.act_objs[i](inp)				
				if train:
					#dropout
					inp,mask = drop_out(inp,self.keep_probs[i])
					self.X_masks.append(mask)
					# note that this activation inp already includes dropout
					self.X_acts.append(inp)
		
		#output layer
		y_outp = Softmax()(inp)
		return y_outp

	def backward_pass(self,y,y_pred,l2,lr,**kwargs):
		'''
		kwargs should only contain beta1 (adam,rmsprop) and/or beta2 (adam)
		'''
		# assuming we have 2 weights with shape (400,200) and (200,10)
		# grad of rightmost layer
		bs = len(y)
		grad_wrt_input = logloss_softmax_grad(y,y_pred) # (n,10)

		grad_wbias = np.sum(grad_wrt_input,axis=0) # (1,10)

		grad_w = self.X_acts[-1].T @ grad_wrt_input # (200,10)
		grad_w += (l2/bs) * self.weights[-1][0] # l2 reg
		
		#optimizer
		grad_w,grad_wbias = self.opt.step(grad_w,grad_wbias,-1,**kwargs)
		self.weights[-1][0]-= lr*grad_w #update weight
		self.weights[-1][1]-= lr*grad_wbias #update bias

		for i in range(len(self.weights)-2,-1,-1):
			grad_wrt_input = grad_wrt_input @ self.weights[i+1][0].T #(n,200) # this is grad_wrt to (activation->dropout)
			grad_wrt_input = grad_wrt_input * self.X_masks[i] / self.keep_probs[i] # grad_wrt_activation, after factoring in the inverted dropout
			grad_wrt_input = grad_wrt_input * self.act_objs[i].grad(self.X_inputs[i+1]) # actual grad_wrt_input

			grad_wbias = np.sum(grad_wrt_input,axis=0) # (1,200)

			grad_w = self.X_acts[i].T @ grad_wrt_input # (400,200)
			grad_w+= (l2/bs) * self.weights[i][0] # l2 reg

			grad_w,grad_wbias = self.opt.step(grad_w,grad_wbias,i,**kwargs)
			self.weights[i][0]-= lr*grad_w #update weight
			self.weights[i][1]-= lr*grad_wbias #update bias

	def fit_epoch(self,X_train,y_train,X_val,y_val,lr,epochs,bs=64,l2=0,beta1=0.9,beta2=0.99):
		'''
		Fit data using stochastic gradient descent and l2 regularization
		'''
		# set_trace()
		for epoch in range(epochs):
			train_cumloss,val_cumloss = 0,0
			# get batch from train set
			for xb,yb in batch_iterator(X_train,y_train,bs):
				y_pred = self.forward_pass(xb,True)
				train_cumloss+= multi_logloss(yb,y_pred) * len(xb)

				self.backward_pass(yb,y_pred,l2,lr,beta1=beta1,beta2=beta2)

			# get double of bs from validation set (since there's less calculation for prediction)
			for xb,yb in batch_iterator(X_val,y_val,bs*2):
				y_pred = self.forward_pass(xb,False)
				val_cumloss += multi_logloss(yb,y_pred) * len(xb)

			self.train_losses.append(train_cumloss/ len(X_train))
			self.val_losses.append(val_cumloss / len(X_val))
			print(f'Epoch {epoch+1}. Training loss: {self.train_losses[-1]}, Val loss:{self.val_losses[-1]}')
		plot_learning_curve(self.train_losses,self.val_losses)

	def predict(self,X):
		y_proba = self.forward_pass(X,False)
		return np.argmax(y_proba,axis=1)
	def predict_proba(self,X):
		return self.forward_pass(X,False)