import numpy as np
import sys

import pandas as pd
from databases import maria


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
	formatStr = "{0:." + str(decimals) + "f}"
	percent = formatStr.format(100 * (iteration / float(total)))
	filledLength = int(round(barLength * iteration / float(total)))
	bar = '#' * filledLength + '-' * (barLength - filledLength)
	sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()


""" ##### Reinforcement Learning by Keractoritic (Keras&Actor&Critic) ###################

01. Action : 'a'
	- Agent's actions are determined by stochastic policy

02. Stochastic/Parameterized policy (Action space) : π(s)
	- Distribution of probabilities over actions, which sum to 1.0
	- Notation π(a|s) : Probability of taking action 'a' in state 's'
	- Policy 'π' does not maximize any value.
	  It is a simply a function of a state 's', returning probabilites for all possible actions.

	- Output policy - Expresstion stochastic policy
	  a. Softmax Policy : Categorial type of policy
	  b. Gaussian Policy: Continueus type of policy

03. Value function
	A. Action-Value function
	B. State-Value function

	미래의 state을 가지고 어떠한 이득(값)을 알고리즘에 적용되게 하는가 확인.

04. Policy objective/Performance/loss/cost function - DNN을 업데이트하기 위한 기준(값) 제시/리턴
	-  With gradient descent, this returned value will be used for optimizing stochastic policy

	A. In episodic case     : Start value function
	B. In continuing task   : Average value function
	                        : Average reward per time-step function ☆ (With stationary distribution of Markov chain for π_θ)
	                          ??? How to stationary distribution ???

05. Gradient descent
	- Find gradient 'θ' to maximize the Policy objective function's value 
	- θ : Gradient of the objective/performance function (Weights for STATE & ACTION & REWARD)
		: Parameter Vector to maximise objective/performance function
		: Policy Gradient Vector OR ???Policy weight???

	A. Finite Difference Policy Gradient
	B. Monte-Carlo Policy Gradient
	C. Actor-Critic Policy Gradient

06. Actor-Critic Method - Learning method both of Policy and Value
	- Solve/improve the high variation problem of Monte-Carlo policy gradient
	- Critic (Learned state-value)  : State???Action-Value function - Approximate Q function
									: Updates action-value function parameters 'w'

	- Actor (Learned policy)        : Stochastic/Parameterized policy function - Approximate policy
									: Updates policy parameters 'θ', in direction suggested by critic

	- Baseline  : Reducing variance using Advantage function based state-value function
				: A good baseline is the state-value function
				: Critic estimates advantage function with;
				  a. Monte-Carlo policy evaluation
				  b. Temporal-Difference learning
				  c. TD(λ)

	- Off-Policy Actor-Critic

##### Reinforcement Learning by Keractoritic (Keras&Actor&Critic) ################### """


class Activities(object):
	def __init__(self, running_mode, source_type, epoch, out_dim, in_dim):
		# running_mode = {0: Training mode, 1: Testing mode}
		self._ENV_ORSEVATIONS_ = Observations(running_mode, source_type, epoch)
		self._COUNT_TOTAL_RUNNING_ = 0.
		self._DIM_INPUT_ = in_dim
		self._DIM_OUTPUT_ = out_dim

	def _parse_current_observation(self):
		y_value = self._STATE_[-1]
		state = np.nan_to_num(self._STATE_[2:-1].astype('float'))

		if np.sum(state) == 0:
			y_value = 0
			self._STATE_[-1] = 0

		return state, y_value

	def reset(self):
		a, b, self._SIZE_OF_CURRENT_DATESET_, epoch = self._ENV_ORSEVATIONS_.next_source()
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
	def __init__(self, running_mode=0, source_type=0, epoch=1):
		self._SIZE_OBSERVATIONS_ = None
		self._SIZE_OBSERVATIONS_IN_A_EPOCH_ = None
		self._CURRENT_SOURCE_OBSERVATIONS_ = None
		self._CURRENT_SOURCE_OBSERVATIONS_STATUS_ = None

		self._RUNNING_MODE_ = running_mode
		self._SOURCE_TYPE_ = source_type
		self._SIZE_EPOCH_ = epoch
		self._RATE_SOURCE_TRAINING_ = 0.85
		self._RATE_OBSERVATION_APPLIED_RATE_ = 1.
		self._SOURCE_CONNECTION_ = maria()

		if source_type == 0: self._TABLE_NAME_ = 'drl_1d_pure'
		elif source_type == 1: self._TABLE_NAME_ = 'drl_1d_foreign_x'
		elif source_type == 2: self._TABLE_NAME_ = 'drl_1d_full'

		query_string = 'SELECT  COUNT(isin) count_symbols ' + \
		               'FROM    (SELECT isin FROM ' + self._TABLE_NAME_ + ' WHERE isin = isin AND trans_date = trans_date GROUP BY isin) R_'
		df_result = self._SOURCE_CONNECTION_.select(query_string)
		count_symbols = df_result.get_value(0, 'count_symbols')

		division_num = int(count_symbols * self._RATE_SOURCE_TRAINING_)

		query_string = 'SELECT  A.isin isin, 0 applied ' + \
		               'FROM    krx_symbols A INNER JOIN ' + self._TABLE_NAME_ + ' B ON (A.isin = B.isin AND B.trans_date = B.trans_date) '+ \
		               'GROUP BY A.isin ' + \
		               'ORDER BY symb_name LIMIT %s, %s'
		values = [0 if running_mode == 0 else division_num, division_num if running_mode == 0 else count_symbols - division_num + 1]

		self._SOURCE_SYMBOLS_ = self._SOURCE_CONNECTION_.select(query_string, values)
		self._SIZE_SYMBOLS_ = len(self._SOURCE_SYMBOLS_)
		self._CURRENT_SOURCE_SYMBOL_INDEX_ = -1
		self._CURRENT_OBSERVATION_INDEX_ = -1
		self._CURRENT_EPOCH_INDEX_ = 0

		# self._get_size_observations_in_a_epoch_()

	def _get_size_observations_in_a_epoch_(self):
		total_symbol_size_in_a_epoch = 0

		i = 1
		for isin, _ in self._SOURCE_SYMBOLS_.values:
			printProgress(i, self._SIZE_SYMBOLS_, '', ' Initialized.', 2, 50)

			query_string = 'SELECT COUNT(*) count_symbols FROM ' + self._TABLE_NAME_ + ' WHERE isin = %s AND trans_date = trans_date'
			values = [isin]
			df_result = self._SOURCE_CONNECTION_.select(query_string, values)

			total_symbol_size_in_a_epoch += df_result.get_value(0, 'count_symbols')

			i += 1

		self._SIZE_OBSERVATIONS_IN_A_EPOCH_ = total_symbol_size_in_a_epoch

	def _standardization(self):
		for i in range(2, len(self._CURRENT_SOURCE_OBSERVATIONS_.columns) - 1):
			self._CURRENT_SOURCE_OBSERVATIONS_[self._CURRENT_SOURCE_OBSERVATIONS_.columns[i:i + 1]] = \
				self._CURRENT_SOURCE_OBSERVATIONS_[self._CURRENT_SOURCE_OBSERVATIONS_.columns[i:i + 1]].astype('float')

			mean = np.mean(self._CURRENT_SOURCE_OBSERVATIONS_[self._CURRENT_SOURCE_OBSERVATIONS_.columns[i:i + 1]])
			std_ = np.std(self._CURRENT_SOURCE_OBSERVATIONS_[self._CURRENT_SOURCE_OBSERVATIONS_.columns[i:i + 1]])

			self._CURRENT_SOURCE_OBSERVATIONS_[self._CURRENT_SOURCE_OBSERVATIONS_.columns[i:i + 1]] = \
				(self._CURRENT_SOURCE_OBSERVATIONS_[self._CURRENT_SOURCE_OBSERVATIONS_.columns[i:i + 1]] - mean) / std_

	def next_source(self):
		self._CURRENT_SOURCE_SYMBOL_INDEX_ += 1

		if self._CURRENT_SOURCE_SYMBOL_INDEX_ == len(self._SOURCE_SYMBOLS_):
			self._CURRENT_EPOCH_INDEX_ += 1
			self._CURRENT_SOURCE_SYMBOL_INDEX_ = 0

			if self._CURRENT_EPOCH_INDEX_ == self._SIZE_EPOCH_:
				self._CURRENT_EPOCH_INDEX_ = -1
				return -1, self._SIZE_EPOCH_

		isin = self._SOURCE_SYMBOLS_.loc[self._CURRENT_SOURCE_SYMBOL_INDEX_, 'isin']

		query_string = 'SELECT  * ' + \
		               'FROM    ' + self._TABLE_NAME_ + ' ' + \
		               'WHERE   isin = %s AND trans_date = trans_date ' + \
		               'ORDER BY trans_date'
		values = [isin]
		self._CURRENT_SOURCE_OBSERVATIONS_ = self._SOURCE_CONNECTION_.select(query_string, values)
		self._standardization()

		self._SIZE_OBSERVATIONS_ = len(self._CURRENT_SOURCE_OBSERVATIONS_)
		self._CURRENT_SOURCE_OBSERVATIONS_STATUS_ = pd.DataFrame({'applied': [0] * self._SIZE_OBSERVATIONS_})
		return self._CURRENT_SOURCE_SYMBOL_INDEX_, len(self._SOURCE_SYMBOLS_), self._SIZE_OBSERVATIONS_, self._CURRENT_EPOCH_INDEX_

	def next_observation(self):
		rate_applied = self._CURRENT_SOURCE_OBSERVATIONS_STATUS_['applied'].sum() / self._SIZE_OBSERVATIONS_
		# if rate_applied > self._RATE_OBSERVATION_APPLIED_RATE_:
		#	self._CURRENT_OBSERVATION_INDEX_ = -1
		#	return 2, 0.

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
