import tensorflow as tf
import keras.backend as kb
import numpy as np
import random
import sys

from keras import utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Activation
from keras.layers.merge import Add, Multiply
from collections import deque
from keras.optimizers import Adam
from keras.models import load_model


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

04. Policy objective/Performance function - DNN을 업데이트하기 위한 기준(값) 제시/리턴
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
	def __init__(self, running_mode, epoch, out_dim, in_dim):
		# running_mode = {0: Training mode, 1: Testing mode}
		self._ENV_ORSEVATIONS_ = Observations(running_mode, epoch)
		self._COUNT_TOTAL_RUNNING_ = 0.
		self._DIM_INPUT_ = in_dim
		self._DIM_OUTPUT_ = out_dim

		self.monitor = Monitor()

	def _parse_current_observation(self):
		y_value = self._STATE_[-1]
		state = np.nan_to_num(self._STATE_[2:-1].astype('float'))

		if np.sum(state) == 0:
			self._STATE_[-1] = 0
			y_value = self._STATE_[-1]
		else:
			state = (state - np.mean(state)) / np.std(state)

		return state, y_value

	def reset(self):
		a, b, self._SIZE_OF_CURRENT_DATESET_, epoch = self._ENV_ORSEVATIONS_.next_source()
		return a, b, self._SIZE_OF_CURRENT_DATESET_, epoch

	def reset_episod(self):
		obs_size = self._ENV_ORSEVATIONS_.get_size_of_current_datasource()
		start_index = int(np.random.uniform(0, int(obs_size / 1.1)))

		self._ENV_ORSEVATIONS_.set_index_of_current_datasource(start_index)

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

		if done:
			reward = 1.
		else:
			reward = 10.

		i = -1
		real_action_stochastic_policy = None
		while i != y_value:
			real_action_stochastic_policy = np.random.uniform(0, 1, self._DIM_OUTPUT_)
			i = np.argmax(real_action_stochastic_policy)

		real_action_stochastic_policy = real_action_stochastic_policy.reshape((1, self._DIM_OUTPUT_))

		state, rate_applied = self.get_observation()

		return state, rate_applied , reward, done, real_action_stochastic_policy, sending_massage


	def get_observation(self):
		self._STATE_, rate_applied = self._ENV_ORSEVATIONS_.next_observation()

		if type(self._STATE_) is int:
			return self._STATE_, rate_applied

		self._COUNT_TOTAL_RUNNING_ += 1.
		state, _ = self._parse_current_observation()

		return state, rate_applied


	def get_size_of_an_epoch(self):
		return self._ENV_ORSEVATIONS_.get_size_of_an_epoch()


	def run_times(self):
		return self._COUNT_TOTAL_RUNNING_


class Keractoritic(object):
	_MODEL_GRAPH_DIR_ = 'graduated/'
	_ACCURACY_CURRENT_SAVED_ = 40
	_ACCURACY_RASE_TO_SAVE_ = 0
	_ACCURACY_ACCEPTABLE_LEVEL_ = 0

	def __init__(self, sess, out_dim, in_dim):
		self._SESSION_ = sess
		self._DIM_INPUT_ = in_dim
		self._DIM_OUTPUT_ = out_dim
		self._RATE_LEARNING_ = 0.0001
		self._EPSILON_ = 1.
		self._EPSILON_DECAY_ = .99995
		self._EPSILON_RANDOM_MIN_ = .05
		self._EPSILON_RANDOM_MAX_ = .5
		self._GAMMA_DISCOUNT_FACTOR_ = .95
		self._TAU_ = .095
		self._COUNT_TOTAL_RUNNING_ = 0.
		self._COST_ACTOR_ = .0
		self._COST_CIRITIC_ = .0
		self._ACCURACY_ACTOR_ = .0
		self._ACCURACY_CIRITIC_ = .0
		self._ACTOR_TRAINING_STARTED_ = False
		self._STACK_ACTION_RESULT_ACCURACY_HOLDER_SIZE = 30
		self._STACK_ACTION_RESULT_ACCURACY_HOLDER_ = deque(maxlen=self._STACK_ACTION_RESULT_ACCURACY_HOLDER_SIZE)
		self._STACK_HISTORY_HOLDER_EVALUATE_STATE_ = deque(maxlen=self._STACK_ACTION_RESULT_ACCURACY_HOLDER_SIZE)
		self._STACK_HISTORY_HOLDER_EVALUATE_ACTION_ = deque(maxlen=self._STACK_ACTION_RESULT_ACCURACY_HOLDER_SIZE)
		self._STACK_HISTORY_HOLDER_ = []

		# ===================================================================== #
		#                               Actor Model                             #
		# Chain rule: find the gradient of chaging the actor network params in  #
		# getting closest to the final value network predictions, i.e. de/dA    #
		# Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
		# ===================================================================== #
		self._MODEL_ACTOR_, self._STATE_INPUT_ACTOR_ = self._actor()
		self._MODEL_ACTOR_TARGET_, _ = self._actor()

		# where we will feed de/dC (from critic)
		self._GRADIENT_ACTOR_CRITIC_ = tf.placeholder(tf.float32, [None, self._DIM_OUTPUT_])

		trainable_weights_actor = self._MODEL_ACTOR_.trainable_weights
		# dC/dA (from actor)
		self._GRADIENT_ACTOR_ = tf.gradients(self._MODEL_ACTOR_.outputs[0], trainable_weights_actor,
		                                     -self._GRADIENT_ACTOR_CRITIC_)

		gradients_actor = zip(self._GRADIENT_ACTOR_, trainable_weights_actor)
		self._COST_OPTIMIZED_ = tf.train.AdamOptimizer(self._RATE_LEARNING_).apply_gradients(gradients_actor)

		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #
		self._MODEL_CRITIC_, self._STATE_INPUT_CRITIC_, self._ACTION_INPUT_CRITIC_ = self._critic()
		self._MODEL_CRITIC_TARGET_, _, _ = self._critic()

		# where we calcaulte de/dC for feeding above
		self._GRADIENT_CRITIC_ = tf.gradients(self._MODEL_CRITIC_.outputs[0], self._ACTION_INPUT_CRITIC_)

		# Initialize for later gradient calculations
		self._SESSION_.run(tf.global_variables_initializer())

	"""
	def _actor(self):
		model = load_model('%s_MODEL_ACTOR_v_87.h5' % self._MODEL_GRAPH_DIR_)
		return model, model.input

	def _critic(self):
		model = load_model('%s_MODEL_CRITIC_v_87.h5' % self._MODEL_GRAPH_DIR_)
		return model, model.inputs[0], model.inputs[1]
	"""

	def _actor(self):
		input_state = Input(shape=(self._DIM_INPUT_,))
		layer1 = Dense(self._DIM_INPUT_ ** 2, activation='relu')(input_state)
		layer2 = Dense(self._DIM_INPUT_ ** 2, activation='relu')(layer1)
		layer3 = Dense(self._DIM_INPUT_ ** 2, activation='relu')(layer2)
		layer4 = Dense(self._DIM_INPUT_, activation='relu')(layer3)
		output = Dense(self._DIM_OUTPUT_, activation='softmax')(layer4)

		model = Model(inputs=input_state, outputs=output)
		model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self._RATE_LEARNING_), metrics=['accuracy'])

		return model, input_state

	def _critic(self):
		input_state = Input(shape=(self._DIM_INPUT_,))
		layer1 = Dense(self._DIM_INPUT_ ** 2, activation='relu')(input_state)
		layer2 = Dense(self._DIM_INPUT_ ** 2)(layer1)

		input_action = Input(shape=(self._DIM_OUTPUT_,))
		layer_action = Dense(self._DIM_INPUT_ ** 2)(input_action)

		merged = Add()([layer2, layer_action])
		layer_merged = Dense(self._DIM_INPUT_ ** 2, activation='relu')(merged)

		layer5 = Dense(self._DIM_INPUT_, activation='relu')(layer_merged)
		output = Dense(self._DIM_OUTPUT_, activation='relu')(layer5)

		model = Model(inputs=[input_state, input_action], outputs=output)
		model.compile(loss='mse', optimizer=Adam(lr=self._RATE_LEARNING_), metrics=['accuracy'])

		return model, input_state, input_action

	def set_running_times(self, running_time):
		self._COUNT_TOTAL_RUNNING_ = running_time

	def get_action(self, state):
		action_type = 1

		### OFF-POLICY ACTOR-CRITIC #############################################
		# Exploration problem will be solved through this random value
		# The outcome from this is a new policy
		if np.random.random() < (self._EPSILON_ if self._EPSILON_ > 0.45 else np.random.random()):
			action_probability = self.softmax(np.random.randn(self._DIM_OUTPUT_))
			action_type = 0
		else:
			action_probability = self._MODEL_ACTOR_.predict(state)
		#########################################################################

		if self._EPSILON_ > 0.45:
			self._EPSILON_ *= self._EPSILON_DECAY_

		return action_probability, action_type

	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #
	def keep_history(self, state_current, action_probability, reward, done, state_new, real_action_stochastic_policy):
		self._STACK_HISTORY_HOLDER_.append([state_current, action_probability, reward, done, state_new, real_action_stochastic_policy])


	def train(self):
		if len(self._STACK_HISTORY_HOLDER_) == 0:
			return

		self._train_critic()
		self._train_actor()

		self._STACK_HISTORY_HOLDER_ = []


	def _train_critic(self):
		accuracy_ = []

		arr_state, arr_reward, arr_action_label, arr_posicygradient_errors = [], [], [], []

		for sample in self._STACK_HISTORY_HOLDER_:
			state_current, action_probability, reward, done, state_new, real_action_stochastic_policy = sample

			# Convert rewards to categorical one-hot encoding - very important ***
			# ********************************************************************
			action_label = self.onthot(action_probability).reshape(1, self._DIM_OUTPUT_)
			# ********************************************************************

			arr_state.append(state_current)
			arr_reward.append(reward)
			arr_action_label.append(action_label)
			arr_posicygradient_errors.append(1 - action_probability)

			self._STACK_HISTORY_HOLDER_EVALUATE_STATE_.append(state_current)
			self._STACK_HISTORY_HOLDER_EVALUATE_ACTION_.append(real_action_stochastic_policy)
			accuracy_.append(int(~done))

		arr_state = np.vstack(arr_state)
		arr_reward = np.vstack(arr_reward)
		arr_action_label = np.vstack(arr_action_label)
		arr_posicygradient_errors = np.vstack(arr_posicygradient_errors)

		discounted_stack_rewards = self._disount_rewards(arr_reward)
		discounted_stack_rewards -= np.mean(discounted_stack_rewards)

		if np.std(discounted_stack_rewards) != 0:
			discounted_stack_rewards /= np.std(discounted_stack_rewards)
		else:
			discounted_stack_rewards = np.zeros_like(discounted_stack_rewards)

		arr_posicygradient_errors *= discounted_stack_rewards

		##### Q-Function - ADVANTAGE FUNCTION --------------------------------
		### Apply policy gradient to reward
		""" reward = reward * (1. - action_error) """

		### Reducing variance using a BASELINE
		# Reward standardization
		# 1. Encourage right action, Discourage wrong action
		# 2. Minimize the variance to be convergent well
		""" reward = ((reward - np.mean(reward)) / np.std(reward)) """
		##### ----------------------------------------------------------------

		self._MODEL_CRITIC_.fit([arr_state, arr_posicygradient_errors], arr_action_label, verbose=0)

		self._COST_CIRITIC_	= self._MODEL_ACTOR_.evaluate(
			np.vstack(self._STACK_HISTORY_HOLDER_EVALUATE_STATE_),
			np.vstack(self._STACK_HISTORY_HOLDER_EVALUATE_ACTION_),
			verbose=0
		)
		self._ACCURACY_CIRITIC_ = np.sum(accuracy_) / self._STACK_ACTION_RESULT_ACCURACY_HOLDER_SIZE


	def _train_actor(self):
		arr_state, arr_reward, arr_action_label, arr_posicygradient_errors = [], [], [], []

		for sample in self._STACK_HISTORY_HOLDER_:
			state_current, _, reward, _, _, _ = sample

			arr_state.append(state_current)

		arr_state = np.vstack(arr_state)

		predicted_action = self._MODEL_ACTOR_.predict(arr_state)
		gradients = self._SESSION_.run(self._GRADIENT_CRITIC_, feed_dict={
			self._STATE_INPUT_CRITIC_: arr_state,
			self._ACTION_INPUT_CRITIC_: predicted_action
		})[0]
		self._SESSION_.run(self._COST_OPTIMIZED_, feed_dict={
			self._STATE_INPUT_ACTOR_: arr_state,
			self._GRADIENT_ACTOR_CRITIC_: gradients
		})

		self._COST_ACTOR_ = self._MODEL_ACTOR_.evaluate(
			np.vstack(self._STACK_HISTORY_HOLDER_EVALUATE_STATE_),
			np.vstack(self._STACK_HISTORY_HOLDER_EVALUATE_ACTION_),
			verbose=0
		)
		self._STACK_ACTION_RESULT_ACCURACY_HOLDER_.append(self._COST_ACTOR_[1])

		if len(self._STACK_ACTION_RESULT_ACCURACY_HOLDER_) >= self._STACK_ACTION_RESULT_ACCURACY_HOLDER_SIZE:
			self._save_model()

	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #
	def _update_target(self):
		self._update_actor_target()
		self._update_critic_target()

	def _update_critic_target(self):
		critic_model_weights = self._MODEL_CRITIC_.get_weights()
		critic_target_weights = self._MODEL_CRITIC_TARGET_.get_weights()

		for i in range(len(critic_target_weights)):
			# critic_target_weights[i] = critic_model_weights[i]
			critic_target_weights[i] = critic_model_weights[i] * self._TAU_ + critic_target_weights[i] * (
			1 - self._TAU_)

		self._MODEL_CRITIC_TARGET_.set_weights(critic_target_weights)

	def _update_actor_target(self):
		actor_model_weights = self._MODEL_ACTOR_.get_weights()
		actor_target_weights = self._MODEL_ACTOR_TARGET_.get_weights()

		for i in range(len(actor_target_weights)):
			# actor_target_weights[i] = actor_model_weights[i]
			actor_target_weights[i] = actor_model_weights[i] * self._TAU_ + actor_target_weights[i] * (1 - self._TAU_)

		self._MODEL_ACTOR_TARGET_.set_weights(actor_target_weights)

	def _save_model(self):
		mean_acc = int(np.mean(self._STACK_ACTION_RESULT_ACCURACY_HOLDER_) * 100)
		min_to_save = self._ACCURACY_CURRENT_SAVED_ + self._ACCURACY_RASE_TO_SAVE_ - self._ACCURACY_ACCEPTABLE_LEVEL_

		if mean_acc < min_to_save:
			return

		from datetime import datetime
		from pathlib import Path

		datetoday = datetime.today()

		model_actor_located = '%s_%s_MODEL_ACTOR_v_%s.h5' % (
		self._MODEL_GRAPH_DIR_, datetoday.strftime('%Y%m%d%H'), mean_acc)
		model_critic_located = '%s_%s_MODEL_CRITIC_v_%s.h5' % (
		self._MODEL_GRAPH_DIR_, datetoday.strftime('%Y%m%d%H'), mean_acc)
		model_descriptor = Path(model_actor_located)

		if not model_descriptor.exists():
			self._MODEL_ACTOR_.save(model_actor_located)
			self._MODEL_CRITIC_.save(model_critic_located)

	def _disount_rewards(self, r):
		discounted_r = np.zeros_like(r)
		running_add = 0
		for t in reversed(range(0, r.size)):
			running_add = running_add * self._GAMMA_DISCOUNT_FACTOR_ + r[t]
			discounted_r[t] = running_add

		return discounted_r

	def learning_info(self):
		return self._COST_CIRITIC_, self._COST_ACTOR_, self._ACCURACY_CIRITIC_

	def softmax(self, x):
		e = np.exp(x - np.max(x))  # prevent overflow
		if e.ndim == 1:
			return e / np.sum(e, axis=0)
		else:
			return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2

	def onthot(self, x):
		x = x.reshape(-1)
		return np.eye(len(x))[np.argmax(x)]

	def dim_input(self):
		return self._DIM_INPUT_

	def dim_output(self):
		return self._DIM_OUTPUT_


class Observations(object):
	import pandas as pd
	from databases import maria

	_RUNNING_MODE_ = None

	_RATE_SOURCE_TRAINING_ = None

	_SOURCE_SYMBOLS_ = None
	_SOURCE_CONNECTION_ = None

	_SIZE_SYMBOLS_ = None
	_SIZE_OBSERVATIONS_ = None
	_SIZE_OBSERVATIONS_IN_A_EPOCH_ = None
	_SIZE_EPOCH_ = None

	_CURRENT_SOURCE_OBSERVATIONS_ = None
	_CURRENT_SOURCE_OBSERVATIONS_STATUS_ = None
	_CURRENT_SOURCE_SYMBOL_INDEX_ = None
	_CURRENT_OBSERVATION_INDEX_ = None
	_CURRENT_EPOCH_INDEX_ = None

	def __init__(self, running_mode=0, epoch=1):
		self._RUNNING_MODE_ = running_mode
		self._SIZE_EPOCH_ = epoch
		self._RATE_SOURCE_TRAINING_ = 0.7
		self._RATE_OBSERVATION_APPLIED_RATE_ = 0.9
		self._SOURCE_CONNECTION_ = self.maria()

		query_string = 'SELECT  COUNT(isin) count_symbols ' + \
		               'FROM    (SELECT isin FROM drl_proportion_source_1d WHERE isin = isin GROUP BY isin HAVING COUNT(isin) >= 30) R_'
		df_result = self._SOURCE_CONNECTION_.select(query_string)
		count_symbols = df_result.get_value(0, 'count_symbols')

		division_num = int(count_symbols * self._RATE_SOURCE_TRAINING_)

		query_string = 'SELECT  isin, 0 applied ' + \
		               'FROM    krx_symbols ' + \
		               'WHERE   isin NOT IN (SELECT isin FROM drl_proportion_source_1d WHERE isin = isin GROUP BY isin HAVING COUNT(isin) < 30) ' + \
		               'ORDER BY symb_name LIMIT %s, %s'
		values = [0 if running_mode == 0 else division_num,
		          division_num if running_mode == 0 else count_symbols - division_num + 1]

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

			query_string = 'SELECT COUNT(*) count_symbols FROM drl_proportion_source_1d WHERE isin = %s AND trans_date = trans_date'
			values = [isin]
			df_result = self._SOURCE_CONNECTION_.select(query_string, values)

			total_symbol_size_in_a_epoch += df_result.get_value(0, 'count_symbols')

			i += 1

		self._SIZE_OBSERVATIONS_IN_A_EPOCH_ = total_symbol_size_in_a_epoch

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
		               'FROM    drl_proportion_source_1d ' + \
		               'WHERE   isin = %s AND trans_date = trans_date ' + \
		               'ORDER BY trans_date'
		values = [isin]
		self._CURRENT_SOURCE_OBSERVATIONS_ = self._SOURCE_CONNECTION_.select(query_string, values)
		self._SIZE_OBSERVATIONS_ = len(self._CURRENT_SOURCE_OBSERVATIONS_)
		self._CURRENT_SOURCE_OBSERVATIONS_STATUS_ = self.pd.DataFrame({'applied': [0] * self._SIZE_OBSERVATIONS_})
		return self._CURRENT_SOURCE_SYMBOL_INDEX_, len(self._SOURCE_SYMBOLS_), self._SIZE_OBSERVATIONS_, self._CURRENT_EPOCH_INDEX_

	def next_observation(self):
		rate_applied = self._CURRENT_SOURCE_OBSERVATIONS_STATUS_['applied'].sum() / self._SIZE_OBSERVATIONS_
		if rate_applied > self._RATE_OBSERVATION_APPLIED_RATE_:
			self._CURRENT_OBSERVATION_INDEX_ = -1
			return 2, 0.

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


class Monitor(object):
	def __init__(self, on=True):
		self.on = on
		self._HISTORY_ = {'ACTION': {'RESULT': []}}

	def history(self, action, result):
		# self._HISTORY_['ACTION']['RESULT'].append(int(action == result))
		pass

	def show(self, request, reset=True):
		if request not in ['episod', 'anobservationset', 'anepoch', 'oneview']:
			print('Invalid request type. Please check the option to request history.')
			exit()

		if request == 'episod':
			success = np.sum(self._HISTORY_['ACTION']['RESULT'])
			if reset: self._HISTORY_['ACTION']['RESULT'] = []
			return success
		elif request == 'anobservationset':
			pass
		elif request == 'anepoch':
			pass
		else:
			pass




def main():
	sess = tf.Session()
	kb.set_session(sess)

	epoch = 5
	training_size = 117
	episode = 0.
	episode_comp = 0.
	in_dim = 41
	out_dim = 9
	done = False

	env_act = Activities(0, epoch, out_dim, in_dim)
	env_kac = Keractoritic(sess, out_dim, in_dim)

	# total_running = env_act.get_size_of_an_epoch()
	total_running = 40394827
	total_running *= epoch

	symb_idx, symb_size, observations_size_current, epoch_current = env_act.reset()
	state_current, rate_applied = env_act.get_observation()
	# Reshape (axis=0:41, axis=1:?) to (axis=0:1, axis=1:41)
	state_current = state_current.reshape((1, env_kac.dim_input()))
	sending_massage = ''

	while epoch_current < epoch:
		cost1, cost2, accuracy = env_kac.learning_info()
		print('(%s / %s / %s) EPOCH: %s, EPISOD: %s # COST: %s / %s, ACCURACY: %s / %s / %s %s' % (
						'{0:4,}'.format(symb_idx),
						'{0:4,}'.format(symb_size),
						'{0:.2f}'.format(rate_applied),
						'{0:2d}'.format(epoch_current),
						'{0:9,.0f}'.format(episode) if episode_comp == episode else ''.rjust(9),
						'{0:.7f}'.format(cost1[0] if type(cost1) is list else 0.),
						'{0:.7f}'.format(cost2[0] if type(cost2) is list else 0.),
						'{0:.2f}'.format(cost1[1] if type(cost1) is list else 0.),
						'{0:.2f}'.format(cost2[1] if type(cost2) is list else 0.),
						'{0:.2f}'.format(accuracy),
						sending_massage
					)
		      )

		if done:
			# ACTOR&CRITIC Models will be trained when the episod is done, not by every step
			env_kac.train()
			env_act.reset_episod()
			episode += 1.
			episode_comp = episode
		else:
			episode_comp += 1

		# [I]. 'action_probability' as a prediction from the ACTOR MODEL is the kind of new Policy
		action_probability, action_type = env_kac.get_action(state_current)
		# Reshape (axis=0:9, axis=1:?) to (axis=0:1, axis=1:9)
		# The output shape from the ACTOR model is already (1, 9), but the outcome from randomly pulled is not...
		action_probability = action_probability.reshape((1, out_dim))

		state_new, rate_applied, reward, done, real_action_stochastic_policy, sending_massage = env_act.step(action_probability, action_type)
		if type(state_new) is int:
			done = True

			if state_new == 2:
				symb_idx, symb_size, observations_size_current, epoch_current = env_act.reset()
				if epoch_current == epoch:
					break

				state_new, rate_applied = env_act.get_observation()

			continue

		state_new = state_new.reshape((1, in_dim))

		### [II]. 'action_error' is right the difference between real action and action_probability from the ACTOR MODEL
		### That means this difference is the Gradient of this STATE(state_current)'s Policy
		### In this case, real action could be 1. in probability
		# action_error = .9995
		# if not done:
		#	action_error = 1 - action_probability[0, real_action_position]

		env_kac.keep_history(state_current, action_probability, reward, done, state_new, real_action_stochastic_policy)

		state_current = state_new


if __name__ == '__main__':
	main()

