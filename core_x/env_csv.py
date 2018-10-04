import numpy as np
import pandas as pd


class Activities(object):
	def __init__(self, config):
		self._CONFIG_ = config

		self._ENV_ORSEVATIONS_ = Observations(self._CONFIG_)
		self._COUNT_TOTAL_RUNNING_ = 0.
		self._DIM_INPUT_ = self._CONFIG_['DIM_INPUT']
		self._DIM_OUTPUT_ = self._CONFIG_['DIM_OUTPUT']

	def _parse_current_observation(self):
		y_value = self._STATE_[-1]
		state = np.nan_to_num(self._STATE_[2:-1].astype('float'))

		if np.sum(state) == 0:
			y_value = 0
			self._STATE_[-1] = 0

		return state, y_value

	def reset(self):
		a, b, self._SIZE_OF_CURRENT_DATESET_, epoch, _ = self._ENV_ORSEVATIONS_.next_source()
		return a, b, self._SIZE_OF_CURRENT_DATESET_, epoch

	def reset_episod(self):
		obs_size = self._ENV_ORSEVATIONS_.get_size_of_current_datasource()
		start_index = int(np.random.uniform(0, obs_size-2))

		self._ENV_ORSEVATIONS_.set_index_of_current_datasource(start_index)

	def softmax(self, x):
		e = np.exp(x - np.max(x))  # prevent overflow
		if e.ndim == 1:
			return e / np.sum(e, axis=0)
		else:
			return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

	def step(self, action_probability, action_type):
		action = np.argmax(action_probability)

		_, y_value = self._parse_current_observation()
		done = ~(action == y_value)

		sending_massage = ' # PREDICTION: %s / %s, REAL: %s%s' % (
			'{0:2d}'.format(action),
			action_type,
			'{0:2d}'.format(y_value),
			' # CORRECT #' if not done else ''
		)

		i = -1
		reward = 1.
		real_action_stochastic_policy = None
		if done:
			reward = .0

			while i != y_value:
				real_action_stochastic_policy = np.random.uniform(0, 1, self._DIM_OUTPUT_)
				i = np.argmax(real_action_stochastic_policy)

			real_action_stochastic_policy = self.softmax(real_action_stochastic_policy)
			real_action_stochastic_policy = real_action_stochastic_policy.reshape((1, self._DIM_OUTPUT_))
		else:
			real_action_stochastic_policy = action_probability

		state, y_value, rate_applied = self.get_observation()
		return state, y_value, rate_applied , reward, done, real_action_stochastic_policy, sending_massage


	def get_observation(self):
		self._STATE_, rate_applied = self._ENV_ORSEVATIONS_.next_observation()

		if type(self._STATE_) is int:
			return self._STATE_, -1,  rate_applied

		self._COUNT_TOTAL_RUNNING_ += 1.
		state, y_value = self._parse_current_observation()

		return state, y_value, rate_applied


	def get_size_of_an_epoch(self):
		return self._ENV_ORSEVATIONS_.get_size_of_an_epoch()


	def run_times(self):
		return self._COUNT_TOTAL_RUNNING_


class Observations(object):
	def __init__(self, config):
		self._CONFIG_ = config

		self._SIZE_OBSERVATIONS_ = None
		self._SIZE_OBSERVATIONS_IN_A_EPOCH_ = None
		self._CURRENT_SOURCE_OBSERVATIONS_ = None
		self._CURRENT_SOURCE_OBSERVATIONS_STATUS_ = None

		self._SIZE_EPOCH_ = self._CONFIG_['AGENTS']['EPOCH']
		self._RATE_SOURCE_TRAINING_ = self._CONFIG_['AGENTS']['OBSERVATION_ENV']['RATE_SOURCE_TRAINING']
		self._RATE_OBSERVATION_APPLIED_RATE_ = 1.

		self._SOURCE_SYMBOLS_ = pd.read_csv('%s/%s/symbols.csv' % (
			self._CONFIG_['JOB_DIR'],
			self._CONFIG_['AGENTS']['OBSERVATION_ENV']['DATASOURCE_DIR_NAME']
		), sep=',')
		self._SIZE_SYMBOLS_ = len(self._SOURCE_SYMBOLS_)
		self._CURRENT_SOURCE_SYMBOL_INDEX_ = -1
		self._CURRENT_OBSERVATION_INDEX_ = -1
		self._CURRENT_EPOCH_INDEX_ = 0

	def get_symbols(self):
		return self._SOURCE_SYMBOLS_

	def get_current_datasource(self):
		return self._CURRENT_SOURCE_OBSERVATIONS_

	def next_source(self):
		self._CURRENT_SOURCE_SYMBOL_INDEX_ += 1

		if self._CURRENT_SOURCE_SYMBOL_INDEX_ == len(self._SOURCE_SYMBOLS_):
			self._CURRENT_EPOCH_INDEX_ += 1
			self._CURRENT_SOURCE_SYMBOL_INDEX_ = 0

			if self._CURRENT_EPOCH_INDEX_ == self._SIZE_EPOCH_:
				self._CURRENT_EPOCH_INDEX_ = -1
				return -1, self._SIZE_EPOCH_

		isin = self._SOURCE_SYMBOLS_.loc[self._CURRENT_SOURCE_SYMBOL_INDEX_, 'isin']
		self._CURRENT_SOURCE_OBSERVATIONS_ = pd.read_csv('%s/%s/%s.csv' % (
			self._CONFIG_['JOB_DIR'],
			self._CONFIG_['AGENTS']['OBSERVATION_ENV']['DATASOURCE_DIR_NAME'],
			isin
		), sep=',')
		self._SIZE_OBSERVATIONS_ = len(self._CURRENT_SOURCE_OBSERVATIONS_)
		self._CURRENT_SOURCE_OBSERVATIONS_STATUS_ = pd.DataFrame({'applied': [0] * self._SIZE_OBSERVATIONS_})
		return self._CURRENT_SOURCE_SYMBOL_INDEX_, len(self._SOURCE_SYMBOLS_), self._SIZE_OBSERVATIONS_, self._CURRENT_EPOCH_INDEX_, isin

	def next_observation(self):
		rate_applied = self._CURRENT_SOURCE_OBSERVATIONS_STATUS_['applied'].sum() / self._SIZE_OBSERVATIONS_
		self._CURRENT_OBSERVATION_INDEX_ += 1

		if (self._CURRENT_OBSERVATION_INDEX_ == self._SIZE_OBSERVATIONS_):
			self._CURRENT_OBSERVATION_INDEX_ = -1
			return 1, 0.

		self._CURRENT_SOURCE_OBSERVATIONS_STATUS_.loc[self._CURRENT_OBSERVATION_INDEX_, 'applied'] = 1
		observation = self._CURRENT_SOURCE_OBSERVATIONS_.loc[self._CURRENT_OBSERVATION_INDEX_,]
		return np.array(observation.values), rate_applied

	def get_size_of_an_epoch(self):
		return self._SIZE_OBSERVATIONS_IN_A_EPOCH_

	def get_size_of_current_datasource(self):
		return self._SIZE_OBSERVATIONS_

	def get_index_of_current_datasource(self):
		return self._CURRENT_OBSERVATION_INDEX_

	def set_index_of_current_datasource(self, idx):
		self._CURRENT_OBSERVATION_INDEX_ = idx
