import numpy as np

def init_exp_grad(shape):
    '''
    Initialize value for exponential weight average
    '''
    return [np.zeros(shape), np.zeros((1,shape[1]))]
def bias_correction(t,beta):
    temp= max(beta**t,10e-6)
    return 1/(1-temp)
class GradientDescent():
    def __init__(self,layers):
        pass
    def step(self,grad_w,grad_wbias,layer,**kwargs):
        return grad_w,grad_wbias

class Momentum():
    def __init__(self,layers):
        self.exp_grad = [init_exp_grad((layers[i],layers[i+1])) for i in range(len(layers)-1)]
        self.t=[0]*(len(layers)-1)
    def step(self,grad_w,grad_wbias,layer,**kwargs):
        beta1 = kwargs['beta1']
        self.t[layer]+=1
        self.exp_grad[layer][0] = beta1*self.exp_grad[layer][0] + (1-beta1)*grad_w
        self.exp_grad[layer][1] = beta1*self.exp_grad[layer][1] + (1-beta1)*grad_wbias
        bias_corr = bias_correction(self.t[layer],beta1)
        return bias_corr*self.exp_grad[layer][0],bias_corr*self.exp_grad[layer][1]

class RMSProp():
    def __init__(self,layers):
        self.exp_grad_sqr = [init_exp_grad((layers[i],layers[i+1])) for i in range(len(layers)-1)]
        self.t=[0]*(len(layers)-1)
    def step(self,grad_w,grad_wbias,layer,**kwargs):
        beta2 = kwargs['beta2']
        eps = 10e-8
        self.t[layer]+=1
        self.exp_grad_sqr[layer][0] = beta2*self.exp_grad_sqr[layer][0] + (1-beta2)* grad_w**2
        self.exp_grad_sqr[layer][1] = beta2*self.exp_grad_sqr[layer][1] + (1-beta2)* grad_wbias**2
        bias_corr = bias_correction(self.t[layer],beta2)
        new_gradw =  grad_w / (np.sqrt(bias_corr*self.exp_grad_sqr[layer][0]) + eps)
        new_gradwb =  grad_wbias / (np.sqrt(bias_corr*self.exp_grad_sqr[layer][1]) + eps)
        return new_gradw,new_gradwb

class Adam():
    def __init__(self,layers):
        self.exp_grad = [init_exp_grad((layers[i],layers[i+1])) for i in range(len(layers)-1)]
        self.exp_grad_sqr = [init_exp_grad((layers[i],layers[i+1])) for i in range(len(layers)-1)]
        self.t=[0]*(len(layers)-1)
    def step(self,grad_w,grad_wbias,layer,**kwargs):
        beta1,beta2 = kwargs['beta1'],kwargs['beta2']
        eps = 10e-8
        self.t[layer]+=1

        self.exp_grad[layer][0] = beta1*self.exp_grad[layer][0] + (1-beta1)*grad_w
        self.exp_grad[layer][1] = beta1*self.exp_grad[layer][1] + (1-beta1)*grad_wbias

        self.exp_grad_sqr[layer][0] = beta2*self.exp_grad_sqr[layer][0] + (1-beta2)* grad_w**2
        self.exp_grad_sqr[layer][1] = beta2*self.exp_grad_sqr[layer][1] + (1-beta2)* grad_wbias**2

        bias_corr1 = bias_correction(self.t[layer],beta1)
        bias_corr2 = bias_correction(self.t[layer],beta2)

        new_gradw =  (bias_corr1*self.exp_grad[layer][0]) / (np.sqrt(bias_corr2*self.exp_grad_sqr[layer][0]) + eps)
        new_gradwb =  (bias_corr1*self.exp_grad[layer][1]) / (np.sqrt(bias_corr2*self.exp_grad_sqr[layer][1]) + eps)
        return new_gradw,new_gradwb
