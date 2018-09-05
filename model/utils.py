import numpy as np
import pandas as pd

def get_train_val(X,y,val_ratio=0.2):
    val_size = int(X.shape[0]*val_ratio)
    return X[:n-val_size,:],y[:n-val_size], X[-val_size:,:],y[-val_size:]

def generate_linear_dataset(n,dim,noise_bound=0.5):
	'''
	Generate a linear dataset with uniform random noise within noise_bound
	'''
	W = np.random.randn(dim+1,1) # including bias W0   
    X = np.random.randn(n,dim)
    X0 = np.array([[1]*n]).T # nx1
    X = np.concatenate((X0,X),axis=1) # including 1s column to simplify linear function
    
    #add uniform random noise between -0.5 and 0.5
    y = X @ W + np.random.rand(n,1) * noise_bound*2 -noise_bound
    return X,y,W