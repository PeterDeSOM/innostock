import numpy as np
from lstm.dataset import Controller

class Activities(object):
	def __init__(self, dataset, in_dim, input_length, out_dim):
		self._DATASET_CONTROLLER_ = Controller(dataset, in_dim, input_length, out_dim)
		self._DIM_INPUT_ = in_dim
		self._DIM_OUTPUT_ = out_dim
		self.input_length = input_length


	def softmax(self, x):
		e = np.exp(x - np.max(x))  # prevent overflow
		if e.ndim == 1:
			return e / np.sum(e, axis=0)
		else:
			return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2


	def step(self, action_probability, y_value):
		action = np.argmax(action_probability)
		done = ~(action == y_value)

		reward = 1.
		if done: reward = .0

		return reward, done


	def get_observation(self):
		return self._DATASET_CONTROLLER_.next()
