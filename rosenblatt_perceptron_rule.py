import numpy as np

class Perceptron(object):

	def __init__(self, learningRate = 0.01, numIterations = 10):
		self.learningRate = learningRate
		self.numIterations = numIterations

	def fit(self, X, y):
		self.weight_ = np.zeros(1 + X.shape[1])
		self.errors = []

		for _ in range(self.numIterations):
			errors = 0
			for xi, target in zip(X, y):
				update = self.learningRate * (target - self.predict(xi))
				self.weight_[1:] += update * xi
				self.weight_[0] += update
				errors += int(update != 0.0)
			self.errors_.append(errors)
		return self

	# EFFECTS: Calculates the net input
	def net_input(self, X):
		return np.dot(X, self.weight_[1:] + self.w_[0]

	# EFFECTS: Returns class label after unit step
	def predict(self, X):
		return np.where(self.net_input(X) >= 0.0, 1, -1)

