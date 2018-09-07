from model.imports import *

def MSE(y,y_pred):
	return np.mean((y_pred -y)**2)

def logloss(y,y_pred):
	y_pred= np.clip(y_pred,1e-5,1-1e-5)
	return np.mean(-np.log(y_pred)*y - np.log(1-y_pred)*(1-y))
def multi_logloss(y,y_pred):
	n_class = np.max(y)+1
	y_onehot = np.eye(n_values)[y]
	return -1*np.mean(np.log(np.sum(y_onehot * y_pred,axis=1)))
def accuracy(y,y_pred):
	return np.mean(y==y_pred)