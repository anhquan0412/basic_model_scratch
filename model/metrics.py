from model.imports import *

def MSE(y,y_pred):
	return np.mean((y_pred -y)**2)

def logloss(y,y_pred):
	y_pred= np.clip(y_pred,1e-5,1-1e-5)
	return np.mean(-np.log(y_pred)*y - np.log(1-y_pred)*(1-y))

def accuracy(y,y_pred):
	return np.mean(y==y_pred)