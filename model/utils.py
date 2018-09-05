from model.imports import *
import matplotlib.pyplot as plt
from model.activations import sigmoid

def get_train_val(X,y,val_ratio=0.2,shuffle=False):
	n = X.shape[0]
	if shuffle:
		idx = np.random.permutation(n)
		X,y = X[idx],y[idx]
	val_size = int(n*val_ratio)
	return X[:n-val_size,:],y[:n-val_size], X[-val_size:,:],y[-val_size:]

def batch_iterator(X,y=None,bs=64):
	for i in range((X.shape[0]-1) // bs + 1):
		i_start= i*bs
		i_end = i_start + bs
		if y is not None:
			yield X[i_start:i_end],y[i_start:i_end]
		else:
			yield X[i_start:i_end]
def generate_linear_dataset(n,dim,noise_bound=0.5,is_reg=True):
	'''
	Generate a linear dataset with uniform random noise within noise_bound
	'''
	W = np.random.randn(dim+1,1) # including bias W0   
	X = np.random.randn(n,dim)
	X0 = np.array([[1]*n]).T # nx1
	X = np.concatenate((X0,X),axis=1) # including 1s column to simplify linear function

	#add uniform random noise between -0.5 and 0.5
	y = np.squeeze(X @ W + np.random.rand(n,1) * noise_bound*2 -noise_bound)
	if not is_reg:
		y = (sigmoid(y) >=0.5).astype(np.uint8)
	return X,y,W

def plot_learning_curve(train_losses,val_losses):    
	plt.plot(range(len(train_losses)),train_losses,'o-',color='r',label='Training loss',markersize=1)
	plt.plot(range(len(train_losses)),val_losses,'o-',color='g',label='Validation loss',markersize=1)
	plt.legend(loc="best")
	plt.show()


