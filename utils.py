import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

def sharedX(x, name=None) :
	if name is None:
		return theano.shared( theano._asarray(x, dtype=theano.config.floatX) ) 
	else:
		return theano.shared( theano._asarray(x, dtype=theano.config.floatX), name=name ) 

def rand_ortho(r,c):
    irange = np.sqrt(0.6/(r + c))
    w = 2*irange*np.random.rand(r,c)-irange
    U,s,V = np.linalg.svd(w,full_matrices=True)
    w = np.dot(U, np.dot( np.eye(U.shape[1], V.shape[0]), V ))
    return w
	
def rand(*size):
	return np.random.rand(*size)
#activations
def sfmx(x) : return T.nnet.softmax(x)
def sig(x): return (1+np.exp(-x))**-1    
def tsig(x): return T.nnet.sigmoid(x)
def ttanh(x): return T.tanh(x)
def thsig(x, th=0): return T.switch(x < th, 0, T.switch(x > 1, 1, x))	
def relu(x):
    return np.maximum(x, 0)
def rest(x, n):
	return T.switch(x>n, x, 0)
def st(x, n):
	return T.switch(x<-n, x, T.switch(x>n, x, 0))
def lin(x):
	return x
	

#cost and err
def NLL(x, y): return -T.sum(T.log(x).dot(y))#one hot cod
def NLL_batch(x,y): return -T.mean(T.log(x).T.dot(y))#one hot cod
def predict(probs) : return T.argmax(probs, axis=0) # predict labels from probs
def predict_batch(probs) : return T.argmax(probs, axis=1) # predict labels from probs batch setting
def error(pred_label,label) : return T.mean(T.eq(pred_label, label))*100 # get error (%)
#def ortho_l4(x):
def KLD(x,y): return T.sum((x+10**-4)*(T.log(x+10**-4)-T.log(y+10**-4)))
def Ldiff(x,y): return T.sum((x-y)**2)
def cross_ent(x, y): return T.sum(-y*T.log(x) - (1-y)*T.log(1-x))

	
def onehot(x):
	xoh = np.zeros((x.shape[0], x.max() + 1))
	xoh[np.arange(x.shape[0]), x] = 1
	return xoh

def whiten(x):
	U, s, V = np.linalg.svd(x.T.dot(x)/x.shape[0], full_matrices=True)
	return x.dot(U).dot(np.diag(s**-0.5)).dot(U.T)

#iteration methods
def mov_avg(param_dict, fr):
	updates = OrderedDict()
	for param in param_dict.keys():
		updates[param] = (1-fr)*param + fr*param_dict[param]
	return updates

def step(param_grad_dict, learning_rate):
	updates = OrderedDict()
	for param in param_grad_dict.keys():
		updates[param] = param + learning_rate*param_grad_dict[param]
	return updates
	
def norm_update(param_grad_dict, learning_rate):
	updates = OrderedDict()
	for param in param_grad_dict.keys():
		uparam = param + learning_rate*param_grad_dict[param]
		updates[param] = uparam/(uparam.norm(2, axis=0)+10**-8)
	return updates
	
def adam(param_grad_dict, learning_rate=0.001, b1=.9, b2=0.999, e=10**-8):
	updates = OrderedDict()
	for param in param_grad_dict.keys():
		inc = theano.shared(np.zeros_like(param.get_value()))
		m = theano.shared(np.zeros_like(param.get_value()))
		v = theano.shared(np.zeros_like(param.get_value()))
		
		new_m = b1*m + (1-b1)*param_grad_dict[param]
		new_v = b2*v + (1-b2)*param_grad_dict[param]**2
		
		new_mc = new_m/(1-b1)
		new_vc = new_v/(1-b2)
		
		updates[m] = new_m
		updates[v] = new_v
		updates[param] = param + learning_rate*new_mc/(T.sqrt(new_vc) + e)
	return updates

def adam_norm(param_grad_dict, learning_rate=0.001, b1=.9, b2=0.999, e=10**-8):
	updates = OrderedDict()
	for param in param_grad_dict.keys():
		inc = theano.shared(np.zeros_like(param.get_value()))
		m = theano.shared(np.zeros_like(param.get_value()))
		v = theano.shared(np.zeros_like(param.get_value()))
		
		new_m = b1*m + (1-b1)*param_grad_dict[param]
		new_v = b2*v + (1-b2)*param_grad_dict[param]**2
		
		new_mc = new_m/(1-b1)
		new_vc = new_v/(1-b2)
		
		updates[m] = new_m
		updates[v] = new_v
		uparam = param + learning_rate*new_mc/(T.sqrt(new_vc) + e)
		updates[param] = uparam/uparam.norm(2, axis=0)
	return updates

def rms_prop( param_grad_dict, learning_rate, 
				momentum=.9, averaging_coeff=.95, stabilizer=.001) :
	updates = OrderedDict()
	for param in param_grad_dict.keys() :

		inc = sharedX(param.get_value() * 0.)
		avg_grad = sharedX(np.zeros_like(param.get_value()))
		avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))

		new_avg_grad = averaging_coeff * avg_grad \
			+ (1 - averaging_coeff) * param_grad_dict[param]
		new_avg_grad_sqr = averaging_coeff * avg_grad_sqr \
			+ (1 - averaging_coeff) * param_grad_dict[param]**2

		normalized_grad = param_grad_dict[param] / \
				T.sqrt(new_avg_grad_sqr - new_avg_grad**2 + stabilizer)
		updated_inc = momentum * inc - learning_rate * normalized_grad

		updates[avg_grad] = new_avg_grad
		updates[avg_grad_sqr] = new_avg_grad_sqr
		updates[inc] = updated_inc
		updates[param] = param + inc

	return updates
	
def rms_prop_norm( param_grad_dict, learning_rate, 
			momentum=.9, averaging_coeff=.95, stabilizer=.001) :
	updates = OrderedDict()
	for param in param_grad_dict.keys() :

		inc = sharedX(param.get_value() * 0.)
		avg_grad = sharedX(np.zeros_like(param.get_value()))
		avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))

		new_avg_grad = averaging_coeff * avg_grad \
			+ (1 - averaging_coeff) * param_grad_dict[param]
		new_avg_grad_sqr = averaging_coeff * avg_grad_sqr \
			+ (1 - averaging_coeff) * param_grad_dict[param]**2

		normalized_grad = param_grad_dict[param] / \
				T.sqrt(new_avg_grad_sqr - new_avg_grad**2 + stabilizer)
		updated_inc = momentum * inc - learning_rate * normalized_grad

		updates[avg_grad] = new_avg_grad
		updates[avg_grad_sqr] = new_avg_grad_sqr
		updates[inc] = updated_inc
		unparam = param + inc
		updates[param] = unparam/unparam.norm(2, axis=0)
	return updates

#classes
