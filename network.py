import random

import numpy as np

class Network(object):
	def __init__(self, sizes):
		self.layers=len(sizes)
		self.sizes=sizes
		# sizes[:-1] means input side ,from first layer to the second-last 
		# sizes[1:] means output side, from second layer to the last layer
		self.bias=[np.random.randn(b,1) for b in sizes[1:]]
		# all weights are refer to second layer to the last layer.
		self.weights=[np.random.randn(x,y) for x, y in zip(sizes[1:], sizes[:-1])]

	def feedforward(self, x):
		for w, b in zip(self.weights, self.bias):
			x = activateFunc(np.dot(w,x) + b)
		return x	

	def evaluate(self, test_data):
		cal = 0
		for (x, y) in test_data:
			if np.argmax(self.feedforward(x)) == y:
				cal+=1
		return cal

	def backprop(self, x, y):
		nabla_b = [np.zeros(b.shape) for b in self.bias]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# feedforward
		feedin = x
		activates = [x]
		zs = []
		for b, w in zip(self.bias, self.weights):
			z = np.dot(w, feedin) + b
			zs.append(z)
			activated = activateFunc(z)
			feedin=activated
			activates.append(activated)
		# backward pass 
		delta = (activates[-1] - y) * activateFunc_der(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activates[-2].transpose())
		
		for l in range(2, self.layers):
			z = zs[-l]
			der = activateFunc_der(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * der
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, activates[-l-1].transpose())
		return (nabla_b, nabla_w)

	def update_wb_by_batch(self, batch_size, rate):
		nabla_b = [np.zeros(b.shape) for b in self.bias]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in batch_size:
			delta_b, delta_w =self.backprop(x,y)		
		# accumalte
			nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_b)]
			nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_w)]
		self.bias=[b-(rate/len(batch_size)) * del_b for b, del_b in zip(self.bias, nabla_b)]
		self.weights=[w-(rate/len(batch_size)) * del_w for w, del_w in zip(self.weights, nabla_w)]
		
	def SGD(self, tr_data, epochs, batch_size, rate, te_data=None):
		tr_data = list(tr_data)
		n = len(tr_data)
		
		if te_data:
			te_data = list(te_data)
			n_te = len(te_data)
		
		for j in range(epochs):
			random.shuffle(tr_data)
			batches=[ tr_data[k:k+batch_size] for k in range(0, n, batch_size)]
			for batch in batches:
				self.update_wb_by_batch(batch,rate)
			if te_data:
				print("Epoch {} : {} /{}".format(j, self.evaluate(te_data), n_te))
			else:
				print("Epoch {} complete".format(j))
		


# this activateFunc is sigmod function 
def activateFunc(z):
	return 1.0 / (1+np.exp(-z))

def activateFunc_der(z):
	return activateFunc(z)*(1-activateFunc(z))
	
