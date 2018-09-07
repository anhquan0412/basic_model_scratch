import numpy as np
from model.utils import onehot_array
def MSE_grad(y,y_pred):
	'''
	Derivative of MSE loss w.r.t y_pred (not w)
	'''
	return (2/len(y))*(y_pred-y)

def logloss_sigmoid_grad(y,y_pred):
	'''
	Derivative of sigmoid + log loss combination is equivalent to derivative of MSE 
	'''
	return MSE_grad(y,y_pred)/2

def neg_logloss_softmax_grad(y,y_pred):
	y_onehot = onehot_array(y)
	# gradient of neg log loss is positive
	return (-1)* y_onehot * (y_pred-1)