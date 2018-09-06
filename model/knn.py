import numpy as np
class CustomNearestNeighbor():
	def __init__(self,k):
		self.k = k
		self.eps=1e-6
	def fit(self,X,y=None):
		self.X_train = np.array(X)
		if y is not None:
			self.y_train = y
			self.n_classes = np.unique(y).shape[0]
			
	def kneighbors(self,X):
		'''
		Return sorted k distance and k indices of input X
		'''
		X = np.array(X)
		dist = np.zeros([len(X),self.k])
		idxs = np.zeros([len(X),self.k],dtype=int)
		#euclidian distance
		for i,x in enumerate(X):
			temp_dist = np.linalg.norm(x - self.X_train,axis=1)
			idxs[i] = (np.argsort(temp_dist)[:self.k])
			dist[i] = temp_dist[idxs[i]]      
		return [dist,idxs]
	def predict_classification(self,X_test,weighted=False):
		if not hasattr(self,'y_train'):
			raise ValueError('y and n_class are undefined')
			
		dist,idxs = self.kneighbors(X_test)
		inv_dist = 1/(dist+ self.eps) # eps to avoid divided by 0
		
		y_pred = np.zeros(idxs.shape[0])
		wc = np.zeros([idxs.shape[0],self.n_classes])
		class_sorted = np.zeros([idxs.shape[0],self.n_classes],dtype=int)     
		for i,idx in enumerate(idxs):
			# calculate weighted count (wc)
			class_counter = np.bincount(self.y_train[idx],weights=inv_dist[i] if weighted else None,minlength=self.n_classes)
			class_sorted[i] = (np.argsort(class_counter)[::-1])
			wc[i] = class_counter
			y_pred[i]=class_sorted[i][0]
		return y_pred,wc,class_sorted