import numpy as np

def MSE(y,y_pred):
    return np.mean((y_pred -y)**2)