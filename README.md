# Machine Learning from scratch!

Update: Code implementations have been moved to python module. Notebook will only show results and model comparison

To refresh my knowledge, I will attempt to implement some basic machine learning algorithms from scratch using only python and limited numpy/pandas function.
My model implementations will be compared to existing models from popular ML library (sklearn)
- [Linear Regression with weight decay (L2 regularization)](https://github.com/anhquan0412/basic_model_scratch/blob/master/linear_regression.ipynb)
- [Logistic Regression with weight decay](https://github.com/anhquan0412/basic_model_scratch/blob/master/logistic_regression.ipynb)
- Random Forest with Permutation Feature Importances
	- [Random Forest Regressor](https://github.com/anhquan0412/basic_model_scratch/blob/master/random_forest_regressor.ipynb)
	- [Random Forest Classifier](https://github.com/anhquan0412/basic_model_scratch/blob/master/random_forest_classifier.ipynb)
- [K Nearest Neighbors: supervised and unsupervised](https://github.com/anhquan0412/basic_model_scratch/blob/master/knn.ipynb)
- [Neural network for classification](https://github.com/anhquan0412/basic_model_scratch/blob/master/scratch_neural_net.ipynb)
	- Stochastic Gradient Descent
	- Multiple hidden layers
	- Variety of activation functions + gradients (Sigmoid, Softmax, ReLU ...) customized for each hidden layer
	- L2 regularization
	- Dropout
	- [Dynamic learning rate optimizer](https://github.com/anhquan0412/basic_model_scratch/blob/master/neural_net_optimizers.ipynb) (momentum, RMSProp and Adam)
    - TODO: batchnorm

The following notebooks uses Pytorch libraries so they are not implemented from scratch. However, I try not to use any high level Pytorch function
- [Pytorch Neural Network](https://github.com/anhquan0412/basic_model_scratch/blob/master/NN_pytorch.ipynb) with: 
	- Custom Data Loader
	- Data Augmentation on 1 channel image: torchvision vs fastai
	- Shallow NN with batchnorm and dropout
	- Learning rate finder
- [Auto Encoding](https://github.com/anhquan0412/basic_model_scratch/blob/master/autoencoder.ipynb)
- [Collaborative Filtering](https://github.com/anhquan0412/basic_model_scratch/blob/master/collab_filtering.ipynb)
- [Char RNN in Vietnamese (Fast.ai)](https://github.com/anhquan0412/basic_model_scratch/blob/master/rnn-vietnamese.ipynb)
