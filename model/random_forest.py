import numpy as np
from model.metrics import logloss, MSE

def var_agg(n,s,s_squared): return (s_squared/n) - (s/n)**2 
	
class RandomForest():
	def __init__(self, X, y, n_trees, sample_sz,is_reg=True, min_leaf=3,max_features=1):
		np.random.seed(42)
		if hasattr(y,'values'): y = y.values
		if hasattr(X,'values'): X = X.values
		self.X,self.y,self.sample_sz,self.min_leaf = X,y,sample_sz,min_leaf
		self.trees = [self.create_tree(is_reg,min_leaf,max_features) for i in range(n_trees)] # store roots of n_trees decision trees
		self.is_reg = is_reg

	def create_tree(self,is_reg,min_leaf,max_features):
		# generate random idxs with size sample_sz
		sample_idxs = np.random.permutation(len(self.y))[:self.sample_sz]
		return DecisionTreeNode(self.X[sample_idxs,:], self.y[sample_idxs], is_reg,min_leaf,max_features)
	def predict(self, X,thres=0.5):
		if hasattr(X,'values'): X = X.values
		y_pred = np.mean([t.predict(X) for t in self.trees], axis=0)
		if not self.is_reg:
			return (y_pred >= thres).astype(np.uint8)
		return y_pred
	def predict_proba(self,X):
		if self.is_reg:
			raise Exception('Cannot predict probability for regression')
		if hasattr(X,'values'): X = X.values
		return np.mean([t.predict(X) for t in self.trees], axis=0)


class DecisionTreeNode():
	def __init__(self, X, y, is_reg,min_leaf,max_features):
		self.X,self.y,self.min_leaf,self.max_features,self.is_reg = X,y,min_leaf,max_features,is_reg
		self.n,self.c = len(y), X.shape[1]
		if self.X.shape[0] != self.n:
			raise ValueError('X and y don\'t have the same size')            
		self.val = np.mean(y)
		
		# Metric (loss score)
		self.score = float('inf') # initialize to infinity for a leaf
		
		self.col_idx= -1 # index of column chosen to split
		self.split_value = None # chosen split value from col with col_idx
		
		self.lhs_tree_node = None
		self.rhs_tree_node = None
		
		self.find_varsplit() # find best split and populate lhs + rhs tree

		
	def find_varsplit(self):
		# Assuming max_feature = self.c, as we consider all features for splitting
		n_col = int(self.c*self.max_features)
		for i in np.random.permutation(n_col): 
			self.find_best_split_reg(i) if self.is_reg else self.find_best_split_clas(i)
		if self.is_leaf: return
		split_col = self.split_col
		lhs_idx = np.nonzero(split_col<=self.split_value)[0]
		rhs_idx = np.nonzero(split_col>self.split_value)[0]
		
		self.lhs_tree = DecisionTreeNode(self.X[lhs_idx,:], self.y[lhs_idx],self.is_reg,self.min_leaf,self.max_features)
		self.rhs_tree = DecisionTreeNode(self.X[rhs_idx,:], self.y[rhs_idx],self.is_reg,self.min_leaf,self.max_features)

	def find_best_split_reg(self, col_idx): 
		x = self.X[:,col_idx]
		y = self.y

		sort_idx = np.argsort(x)
		sort_x,sort_y = x[sort_idx],y[sort_idx]
		rhs_cnt,rhs_sum,rhs_sum2 = self.n,sort_y.sum(), (sort_y**2).sum()
		lhs_cnt,lhs_sum,lhs_sum2=0,0.0,0.0
		for i in range(0,self.n- self.min_leaf):
			xi,yi = sort_x[i],sort_y[i]
			lhs_cnt += 1; rhs_cnt -= 1
			lhs_sum += yi; rhs_sum -= yi
			lhs_sum2 += yi**2; rhs_sum2 -= yi**2
			if i<self.min_leaf-1 or xi==sort_x[i+1]: # cannot split at a duplicate. ALl dups should be in 1 side
				continue
			
			# the idea is to find the split with LOWEST VARIANCE. Variance (ddof = 0) is equivalent to MSE score against mean
			# That means standard Deviation is equivalent to RMSE score against mean
			lhs_var = var_agg(lhs_cnt, lhs_sum, lhs_sum2)
			rhs_var = var_agg(rhs_cnt, rhs_sum, rhs_sum2)
			curr_score = (lhs_var*lhs_cnt + rhs_var*rhs_cnt) # equivalent to MSE score in comparison
			if curr_score<self.score: 
				self.col_idx,self.score,self.split_value = col_idx,curr_score,xi
	
	def find_best_split_clas(self, col_idx): 
		x = self.X[:,col_idx]
		y = self.y

		sort_idx = np.argsort(x)
		sort_x,sort_y = x[sort_idx],y[sort_idx]
		rhs_cnt,rhs_sum = self.n,sort_y.sum()
		lhs_cnt,lhs_sum=0,0.0
		
		for i in range(0,self.n- self.min_leaf):
			xi,yi = sort_x[i],sort_y[i]
			lhs_cnt += 1; rhs_cnt -= 1
			lhs_sum += yi; rhs_sum -= yi
			if i<self.min_leaf-1 or xi==sort_x[i+1]: # cannot split at a duplicate. ALl dups should be in 1 side
				continue
			lhs_pred = np.array([lhs_sum/lhs_cnt] * lhs_cnt)
			rhs_pred = np.array([rhs_sum/rhs_cnt] * rhs_cnt)
			pred_at_split = np.concatenate([lhs_pred,rhs_pred])
			curr_score = logloss(sort_y,pred_at_split) #binary log loss score
			if curr_score<self.score:
				self.col_idx,self.score,self.split_value = col_idx,curr_score,xi

	@property
	def split_col(self): 
		return self.X[:,self.col_idx]

	@property
	def is_leaf(self): return self.score == float('inf')
	
	#prediction
	def predict(self,X):
		return np.array([self.predict_row(xi) for xi in X])
	def predict_row(self,xi):
		if self.is_leaf: return self.val
		subtree = self.lhs_tree if xi[self.col_idx]<=self.split_value else self.rhs_tree
		return subtree.predict_row(xi)
	
	def __repr__(self):
		loss_fn = MSE if self.is_reg else logloss
		loss = loss_fn(self.y, [self.val]*self.n)
		s = f'Sample size: {self.n}. Pred value: {self.val:0.2f}. Loss: {loss:0.3f}\n'
		if not self.is_leaf:
			s += f'Best split from feature {self.col_idx} at value {self.split_value}'
		return s