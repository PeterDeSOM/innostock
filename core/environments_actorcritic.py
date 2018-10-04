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


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()



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
		state = self._STATE_[2:-1].astype('float')
		state = (state - np.mean(state)) / np.std(state)

		return state, y_value

	def reset(self):
		self._SIZE_OF_CURRENT_DATESET_, epoch = self._ENV_ORSEVATIONS_.next_source()
		return self._SIZE_OF_CURRENT_DATESET_, epoch

	def step(self, action_probability, action_type):
		action = np.argmax(action_probability)

		_, y_value = self._parse_current_observation()
		done = ~(action == y_value)

		print('   ### PREDICTION: %s / %s, REAL: %s%s' % (
			'{0:2d}'.format(action),
			action_type,
			'{0:2d}'.format(y_value),
			' ### CORRECT ###' if not done else ''))

		if not done: reward = 1.
		else: reward = -1.

		i = -1
		while i != y_value:
			action_probability = np.random.normal(reward, 1., self._DIM_OUTPUT_)
			i = np.argmax(action_probability)

		action_probability = action_probability.reshape((1, self._DIM_OUTPUT_))
		reward = action_probability

		# self.monitor.history(action, y_value)

		return self.get_observation(), reward, done, y_value


	def get_observation(self):
		self._STATE_ = self._ENV_ORSEVATIONS_.next_observation()

		if self._STATE_ is None:
			return None

		self._COUNT_TOTAL_RUNNING_ += 1.
		state, _ = self._parse_current_observation()

		return state


	def get_size_of_an_epoch(self):
		return self._ENV_ORSEVATIONS_.get_size_of_an_epoch()


	def run_times(self):
		return self._COUNT_TOTAL_RUNNING_



# Reinforcement Learning by Keractoritic (Keras&Actor&Critic)
class Keractoritic(object):
	def __init__(self, sess, out_dim, in_dim):
		self._SESSION_ = sess
		self._DIM_INPUT_ = in_dim
		self._DIM_OUTPUT_ = out_dim
		self._SESSION_ = sess
		self._RATE_LEARNING_ = 0.002
		self._EPSILON_ = 1.
		self._EPSILON_DECAY_ = .995
		self._EPSILON_RANDOM_MIN_ = .05
		self._EPSILON_RANDOM_MAX_ = .35
		self._GAMMA_DISCOUNT_FACTOR_ = .95
		self._TAU_ = .095
		self._COUNT_TOTAL_RUNNING_ = 0.
		self._COST_ACTOR_ = .0
		self._COST_CIRITIC_ = .0
		self._ACCURACY_ACTOR_ = .0
		self._ACCURACY_CIRITIC_ = .0
		self._STACK_HISTORY_HOLDER_ = deque(maxlen=2000)

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
		self._GRADIENT_ACTOR_ = tf.gradients(self._MODEL_ACTOR_.outputs[0], trainable_weights_actor, -self._GRADIENT_ACTOR_CRITIC_)

		gradients_actor = zip(self._GRADIENT_ACTOR_, trainable_weights_actor)
		self._COST_OPTIMIZED_ = tf.train.AdamOptimizer(self._RATE_LEARNING_).apply_gradients(gradients_actor)

		# ===================================================================== #
		#                              Critic Model                             #
		# ===================================================================== #
		self._MODEL_CRITIC_, self._STATE_INPUT_CRITIC_, self._ACTION_INPUT_CRITIC_  = self._critic()
		self._MODEL_CRITIC_TARGET_, _, _ = self._critic()

		# where we calcaulte de/dC for feeding above
		self._GRADIENT_CRITIC_ = tf.gradients(self._MODEL_CRITIC_.outputs[0], self._ACTION_INPUT_CRITIC_)

		# Initialize for later gradient calculations
		self._SESSION_.run(tf.global_variables_initializer())


	def _actor(self):
		input_state = Input(shape=(self._DIM_INPUT_,))
		layer1 = Dense(self._DIM_INPUT_**2, activation='relu')(input_state)
		layer2 = Dense(self._DIM_INPUT_**2, activation='relu')(layer1)
		layer3 = Dense(self._DIM_INPUT_**2, activation='relu')(layer2)
		output = Dense(self._DIM_OUTPUT_, activation='softmax')(layer3)

		model = Model(inputs=input_state, outputs=output)
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		# model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, epsilon=1., decay=0.995), metrics=['accuracy'])

		return model, input_state


	def _critic(self):
		input_state = Input(shape=(self._DIM_INPUT_,))
		layer1 = Dense(self._DIM_INPUT_**2, activation='relu')(input_state)
		layer2 = Dense(self._DIM_INPUT_**2)(layer1)

		input_action = Input(shape=(self._DIM_OUTPUT_,))
		layer_action = Dense(self._DIM_INPUT_**2)(input_action)

		merged = Add()([layer2, layer_action])
		layer_merged = Dense(self._DIM_INPUT_**2, activation='relu')(merged)

		output = Dense(self._DIM_OUTPUT_, activation='softmax')(layer_merged)

		model = Model(inputs=[input_state, input_action], outputs=output)
		model.compile(loss='categorical_crossentropy', optimizer='adam')

		return model, input_state, input_action


	def get_action(self, state):
		action_type = 1

		### OFF-POLICY ACTOR-CRITIC #############################################
		# Exploration problem will be solved through this random value
		# The outcome from this is a new policy
		if np.random.random() < self._EPSILON_:
			action_probability = self.softmax(np.random.randn(self._DIM_OUTPUT_))
			action_type = 0
		else:
			action_probability = self._MODEL_ACTOR_.predict(state)
		#########################################################################

		return action_probability, action_type
	

	# ========================================================================= #
	#                               Model Training                              #
	# ========================================================================= #
	def keep_history(self, state_current, action_probability, reward, done, state_new, action_error):
		self._STACK_HISTORY_HOLDER_.append([state_current, action_probability, reward, done, state_new, action_error])

		if not done:
			if self._EPSILON_ > self._EPSILON_RANDOM_MIN_:
				self._EPSILON_ *= self._EPSILON_DECAY_
			else:
				self._EPSILON_ = np.random.uniform(self._EPSILON_RANDOM_MIN_, self._EPSILON_RANDOM_MAX_)


	def train(self, batch_size):
		if len(self._STACK_HISTORY_HOLDER_) < batch_size:
			return

		samples = random.sample(self._STACK_HISTORY_HOLDER_, batch_size)
		self._train_critic(samples)
		self._train_actor(samples)

		self._update_target()


	def _train_critic(self, samples):
		cost = []
		accuracy = []

		for sample in samples:
			state_current, action, reward, done, state_new, action_error = sample

			if not done:
				target_action = self._MODEL_ACTOR_TARGET_.predict(state_new)
				future_reward = self._MODEL_CRITIC_TARGET_.predict([state_new, target_action])[0][0]
				reward += self._GAMMA_DISCOUNT_FACTOR_ * future_reward

			### Reducing variance using a BASELINE - ADVANTAGE FUNCTION ##########
			# Reward standardization
			# 1. Encourage right action, Discourage wrong action
			# 2. Minimize the variance to be convergent well
			reward = ((reward - np.mean(reward)) / np.std(reward)) * action_error
			######################################################################

			# Convert rewards to categorical one-hot encoding - very important ***
			# ********************************************************************
			action = self.onthot(action)
			# ********************************************************************

			learning_history = self._MODEL_CRITIC_.fit([state_current, reward], action, verbose=0)

			cost.append(learning_history.history['loss'])
			accuracy.append(int(~done))

		self._COST_CIRITIC_ = np.mean(cost)
		self._ACCURACY_CIRITIC_ = np.sum(accuracy) / len(samples)


	def _train_actor(self, samples):
		for sample in samples:
			state_current, action, reward, done, state_new, action_error = sample

			predicted_action = self._MODEL_ACTOR_.predict(state_current)
			gradients = self._SESSION_.run(self._GRADIENT_CRITIC_, feed_dict={
				self._STATE_INPUT_CRITIC_: state_current,
				self._ACTION_INPUT_CRITIC_: predicted_action
			})[0]
			self._SESSION_.run(self._COST_OPTIMIZED_, feed_dict={
				self._STATE_INPUT_ACTOR_: state_current,
				self._GRADIENT_ACTOR_CRITIC_: gradients
			})


	# ========================================================================= #
	#                         Target Model Updating                             #
	# ========================================================================= #
	def _update_target(self):
		self._update_actor_target()
		self._update_critic_target()


	def _update_critic_target(self):
		critic_model_weights  = self._MODEL_CRITIC_.get_weights()
		critic_target_weights = self._MODEL_CRITIC_TARGET_.get_weights()

		for i in range(len(critic_target_weights)):
			# critic_target_weights[i] = critic_model_weights[i]
			critic_target_weights[i] = critic_model_weights[i] * self._TAU_ + critic_target_weights[i] * (1 - self._TAU_)

		self._MODEL_CRITIC_TARGET_.set_weights(critic_target_weights)


	def _update_actor_target(self):
		actor_model_weights  = self._MODEL_ACTOR_.get_weights()
		actor_target_weights = self._MODEL_ACTOR_TARGET_.get_weights()

		for i in range(len(actor_target_weights)):
			# actor_target_weights[i] = actor_model_weights[i]
			actor_target_weights[i] = actor_model_weights[i] * self._TAU_ + actor_target_weights[i] * (1 - self._TAU_)

		self._MODEL_ACTOR_TARGET_.set_weights(actor_target_weights)


	def learning_info(self):
		return self._COST_CIRITIC_, self._ACCURACY_CIRITIC_

	def softmax(self, x):
		e = np.exp(x - np.max(x)) # prevent overflow
		if e.ndim == 1:
			return e / np.sum(e, axis=0)
		else:
			return e / np.array([np.sum(e, axis=1)]).T # ndim = 2


	def onthot(self, x):
		# x = x.reshape(-1)
		# return np.eye(len(x))[np.argmax(x)]
		index = np.argmax(x)
		x *= 0.
		np.put(x, index, [1.])
		return x


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
	_CURRENT_SOURCE_SYMBOL_INDEX_ = None
	_CURRENT_OBSERVATION_INDEX_ = None
	_CURRENT_EPOCH_INDEX_ = None

	def __init__(self, running_mode=0, epoch=1):
		self._RUNNING_MODE_ = running_mode
		self._SIZE_EPOCH_ = epoch
		self._RATE_SOURCE_TRAINING_ = 0.7
		self._SOURCE_CONNECTION_ = self.maria()

		query_string = 'SELECT COUNT(*) count_symbols FROM krx_symbols'
		df_result = self._SOURCE_CONNECTION_.select(query_string)
		count_symbols = df_result.get_value(0, 'count_symbols')

		division_num = int(count_symbols * self._RATE_SOURCE_TRAINING_)

		query_string = 'SELECT isin, 1 FROM krx_symbols WHERE isin = isin ORDER BY symb_name LIMIT %s, %s'
		values = [0 if running_mode == 0 else division_num,
		          division_num if running_mode == 0 else count_symbols - division_num + 1]

		self._SOURCE_SYMBOLS_ = self._SOURCE_CONNECTION_.select(query_string, values)
		self._SIZE_SYMBOLS_ = len(self._SOURCE_SYMBOLS_)
		self._CURRENT_SOURCE_SYMBOL_INDEX_ = -1
		self._CURRENT_OBSERVATION_INDEX_ = -1
		self._CURRENT_EPOCH_INDEX_ = 0

		self._get_size_observations_in_a_epoch_()

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
		return self._SIZE_OBSERVATIONS_, self._CURRENT_EPOCH_INDEX_

	def next_observation(self):
		self._CURRENT_OBSERVATION_INDEX_ += 1

		if (self._CURRENT_OBSERVATION_INDEX_ == len(self._CURRENT_SOURCE_OBSERVATIONS_)):
			self._CURRENT_OBSERVATION_INDEX_ = -1
			return None

		observation = self._CURRENT_SOURCE_OBSERVATIONS_.loc[self._CURRENT_OBSERVATION_INDEX_,]
		return np.array(observation.values)

	def get_size_of_an_epoch(self):
		return self._SIZE_OBSERVATIONS_IN_A_EPOCH_

	def get_index_of_current_datasource(self):
		return self._CURRENT_OBSERVATION_INDEX_



class Monitor(object):
	def __init__(self, on=True):
		self.on = on
		self._HISTORY_ = {'ACTION':{'RESULT':[]}}


	def history(self, action, result):
		self._HISTORY_['ACTION']['RESULT'].append(int(action == result))
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



