import numpy as np
from model.utils import onehot_array
def MSE(y,y_pred):
	return np.mean((y_pred -y)**2)
def logloss(y,y_pred):
	y_pred= np.clip(y_pred,1e-5,1-1e-5)
	return np.mean(-np.log(y_pred)*y - np.log(1-y_pred)*(1-y))
def neg_multi_logloss(y,y_pred):
	y_pred= np.clip(y_pred,1e-5,1-1e-5)
	y_onehot = onehot_array(y)
	return np.mean(np.log(np.sum(y_onehot * y_pred,axis=1)))
def accuracy(y,y_pred):
	return np.mean(y==y_pred)