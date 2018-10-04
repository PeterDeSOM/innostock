import numpy as np
from isin_episod.dataset import Controller

class Activities(object):
	def __init__(self, dataset, in_dim, out_dim):
		self._DATASET_CONTROLLER_ = Controller(dataset)
		self._DIM_INPUT_ = in_dim
		self._DIM_OUTPUT_ = out_dim


	def _parse_current_observation(self):
		state = np.nan_to_num(self._STATE_[0:-2].astype('float'))

		if np.sum(state) == 0:
			self._STATE_[-1] = 0

		return state, self._STATE_[-1]


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
		self._STATE_, rate_applied = self._DATASET_CONTROLLER_.next()

		if type(self._STATE_) is int:
			return self._STATE_, -1,  rate_applied

		state, y_value = self._parse_current_observation()

		return state, y_value, rate_applied
