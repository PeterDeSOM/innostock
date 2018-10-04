from keras.layers import Dense, Add, Input
from keras.optimizers import RMSprop
from keras import backend as K
from keras import initializers
from keras.models import Model
from keras.models import model_from_json
from core_x.env_x import Activities
from collections import deque

import tensorflow as tf
import numpy as np
import threading
import random
import time


class A3CAgent:
	def __init__(self, source_type, in_dim, out_dim):
		self._STACK_HISTORY_HOLDER_ACCURACY_ = deque(maxlen=111)
		self._SOURCE_TYPE_ = source_type
		self.AGENTS = []

		self.ciritic_loss = .0

		self.state_size = in_dim
		self.action_size = out_dim

		self.discount_factor = 0.99
		self.actor_lr = .0000015
		self.critic_lr = .000015

		self.threads = 5

		self.actor, self.critic = self.build_model()
		self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]
		# self.load_weight()

		self.sess = tf.InteractiveSession()
		K.set_session(self.sess)
		self.sess.run(tf.global_variables_initializer())

	def build_model(self):
		input_state = Input(shape=(self.state_size,))
		layer1 = Dense(self.state_size ** 2,
		               activation='relu',
		               kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1))(input_state)
		layer2 = Dense(self.state_size ** 2,
		               activation='relu',
		               kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1))(layer1)
		layer3 = Dense(self.state_size ** 2,
		               activation='relu',
		               kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1))(layer2)
		layer4 = Dense(self.state_size ** 2,
		               activation='relu',
		               kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1))(layer3)
		layer5 = Dense(self.state_size,
		               activation='relu',
		               kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1))(layer4)

		policy = Dense(self.action_size,
		               activation='softmax',
		               kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1))(layer5)
		value = Dense(1, activation='linear',
		               kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.1, seed=1))(layer5)

		actor = Model(inputs=input_state, outputs=policy)
		critic = Model(inputs=input_state, outputs=value)

		actor._make_predict_function()
		critic._make_predict_function()

		return actor, critic

	def build_model_complex(self):
		input_state = Input(shape=(self.state_size,))

		layer1_1 = Dense(self.state_size ** 2, activation='relu')(input_state)
		layer2_1 = Dense(self.state_size ** 2, activation='relu')(layer1_1)
		layer3_1 = Dense(self.state_size)(layer2_1)

		layer1_2 = Dense(self.state_size ** 2, activation='relu')(input_state)
		layer2_2 = Dense(self.state_size ** 2, activation='relu')(layer1_2)
		layer3_2 = Dense(self.state_size)(layer2_2)

		merge1 = Add()([layer3_1, layer3_2])

		layer4 = Dense(self.state_size ** 2, activation='relu')(merge1)
		layer5 = Dense(self.state_size ** 2, activation='relu')(layer4)

		layer6_1 = Dense(self.state_size ** 2, activation='relu')(layer5)
		layer7_1 = Dense(self.state_size ** 2, activation='relu')(layer6_1)
		layer8_1 = Dense(self.state_size)(layer7_1)

		layer6_2 = Dense(self.state_size ** 2, activation='relu')(layer5)
		layer7_2 = Dense(self.state_size ** 2, activation='relu')(layer6_2)
		layer8_2 = Dense(self.state_size)(layer7_2)

		merge2 = Add()([layer8_1, layer8_2])

		policy = Dense(self.action_size, activation='softmax')(merge2)
		value = Dense(1, activation='linear')(merge2)

		actor = Model(inputs=input_state, outputs=policy)
		critic = Model(inputs=input_state, outputs=value)

		actor._make_predict_function()
		critic._make_predict_function()

		return actor, critic

	def actor_optimizer(self):
		action = K.placeholder(shape=[None, self.action_size])
		advantages = K.placeholder(shape=[None, 1])

		policy = self.actor.output

		action_prob = K.sum(action * policy, axis=1)
		cross_entropy = K.log(action_prob + 1e-10) * advantages
		cross_entropy = -K.sum(cross_entropy)

		entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
		entropy = K.sum(entropy)

		loss = cross_entropy + 0.01 * entropy

		optimizer = RMSprop(lr=self.actor_lr, rho=0.99, epsilon=0.01)
		updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
		train = K.function([self.actor.input, action, advantages], [loss], updates=updates)

		return train

	def critic_optimizer(self):
		discounted_prediction = K.placeholder(shape=(None, 1))

		value = self.critic.output

		loss = K.mean(K.square(discounted_prediction - value))

		optimizer = RMSprop(lr=self.critic_lr, rho=0.99, epsilon=0.01)
		updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
		train = K.function([self.critic.input, discounted_prediction], [loss], updates=updates)

		return train

	def train(self):
		self.AGENTS = [
			Agent(self,
			      self._SOURCE_TYPE_,
			      self.action_size,
			      self.state_size,
			      [self.actor, self.critic],
			      self.sess,
			      self.optimizer,
			      self.discount_factor
			      ) for _ in range(self.threads)
		]
		for agent in self.AGENTS:
			time.sleep(2)
			agent.start()

		monitor = Monitor(self, self._SOURCE_TYPE_)
		monitor.start()

		time.sleep(60 * 5)
		while True:
			self.save()
			time.sleep(20)

	def save(self):
		mean_acc = int(np.mean(self.get_accuracy()) * 100)

		if mean_acc < 75:
			return

		#from datetime import datetime
		from pathlib import Path

		#datetoday = datetime.today()

		_MODEL_GRAPH_DIR_ = ''
		if self._SOURCE_TYPE_ == 0: _MODEL_GRAPH_DIR_ = 'graduated/'
		elif self._SOURCE_TYPE_ == 1: _MODEL_GRAPH_DIR_ = 'graduated_foreign_x/'
		elif self._SOURCE_TYPE_ == 2: _MODEL_GRAPH_DIR_ = 'graduated_full/'

		#model_actor_located = '%s_%s_MODEL_W_ACTOR_v_%s.h5' % (_MODEL_GRAPH_DIR_, mean_acc, datetoday.strftime('%Y%m%d%H%M%S'))
		#model_critic_located = '%s_%s_MODEL_W_CRITIC_v_%s.h5' % (_MODEL_GRAPH_DIR_, mean_acc, datetoday.strftime('%Y%m%d%H%M%S'))
		model_actor_located = '%sV%s_MODEL_W_ACTOR_.h5' % (_MODEL_GRAPH_DIR_, mean_acc)
		model_critic_located = '%sV%s_MODEL_W_CRITIC_.h5' % (_MODEL_GRAPH_DIR_, mean_acc)
		model_descriptor = Path(model_actor_located)

		#if not model_descriptor.exists():
		#	self.actor.save_weights(model_actor_located)
		#	self.critic.save_weights(model_critic_located)
		self.actor.save_weights(model_actor_located)
		self.critic.save_weights(model_critic_located)

	def get_accuracy(self):
		accuracys = []

		for agent in self.AGENTS:
			accuracys.append(agent.get_accuracy())

		return np.mean(accuracys)


class Agent(threading.Thread):
	def __init__(self, creator, source_type, action_size, state_size, model, sess, optimizer, discount_factor):
		threading.Thread.__init__(self)

		self._SOURCE_TYPE_ = source_type

		self.STACK_HISTORY_HOLDER_ACTOR_LOSS = deque(maxlen=10)
		self.STACK_HISTORY_HOLDER_CRITIC_LOSS = deque(maxlen=10)
		self.STACK_HISTORY_HOLDER_ACCURACY = deque(maxlen=111)

		self.CACHE_PROGRESS_CURRENT = []

		self.creator = creator
		self.action_size = action_size
		self.state_size = state_size
		self.actor, self.critic = model
		self.sess = sess
		self.optimizer = optimizer
		self.discount_factor = discount_factor
		self.states, self.actions, self.rewards, self.dones = [], [], [], []

		self.local_actor, self.local_critic = self.creator.build_model()
		self.local_actor.set_weights(self.actor.get_weights())
		self.local_critic.set_weights(self.critic.get_weights())

	def get_action(self, state):
		return self.local_actor.predict(state)

	def get_accuracy(self):
		return np.mean(self.STACK_HISTORY_HOLDER_ACCURACY)

	def get_progressing(self):
		mean_actor = np.mean(self.STACK_HISTORY_HOLDER_ACTOR_LOSS)
		mean_critic = np.mean(self.STACK_HISTORY_HOLDER_CRITIC_LOSS)
		return {
			'ACTOR_LOSS': '%s%s' % (' ' if mean_actor >= 0 else '', '{0:3.8f}'.format(mean_actor)),
			'CRITIC_LOSS': '%s%s' % (' ' if mean_critic >= 0 else '', '{0:3.8f}'.format(mean_critic)),
			'ACCURACY': '{0:.4f}'.format(np.mean(self.STACK_HISTORY_HOLDER_ACCURACY)),
			'PROGRESS_CURRENT': self.CACHE_PROGRESS_CURRENT
		}

	def get_progress_value(self):
		return {
			'ACTOR_LOSS': np.mean(self.STACK_HISTORY_HOLDER_ACTOR_LOSS),
			'CRITIC_LOSS': np.mean(self.STACK_HISTORY_HOLDER_CRITIC_LOSS),
			'ACCURACY': np.mean(self.STACK_HISTORY_HOLDER_ACCURACY)
		}

	def append_sample(self, state, action_label, reward, done):
		self.states.append(state)
		self.actions.append(action_label)
		self.rewards.append(reward)
		self.dones.append(done)

	def run(self):
		print('##### [SOURCE TYPE: %s] RUN : ' % self._SOURCE_TYPE_, self)

		epoch = 100
		steps_an_episode = 100
		steps_current = 1
		failure_significant_rate = .05
		failure_limit = int(steps_an_episode * failure_significant_rate)
		failure_current = 0
		episode_current = 1
		stop_running = False
		done = False

		env_act = Activities(running_mode=0, source_type=self._SOURCE_TYPE_, epoch=epoch, out_dim=self.action_size, in_dim=self.state_size)
		idx_current_symb, size_all_symbs, observations_size_current, epoch_current = env_act.reset()

		state_current, y_value, rate_applied = env_act.get_observation()
		state_current = state_current.reshape((1, self.state_size))

		while epoch_current < epoch and not stop_running:
			while failure_current <= failure_limit and steps_current <= steps_an_episode and not stop_running:
				time.sleep(np.round(random.uniform(.0, .005), 5))

				action_stochastic_policy = self.get_action(state_current)
				state_new, y_value, rate_applied, reward, done, real_action_stochastic_policy, sending_massage = env_act.step(action_stochastic_policy, 1)

				if type(state_new) is int:
					if state_new == 1:
						idx_current_symb, size_all_symbs, observations_size_current, epoch_current = env_act.reset()

						self.append_sample(
							state_current,
							self.onthot(action_stochastic_policy).reshape((1, self.action_size)),
							reward,
							done
						)

						if epoch_current == epoch:
							break

					state_new, y_value, rate_applied = env_act.get_observation()
					state_current = state_new.reshape((1, self.state_size))

					break

				if done:
					failure_current += 1

				self.append_sample(
					state_current,
					self.onthot(action_stochastic_policy).reshape((1, self.action_size)),
					reward,
					done
				)
				state_current = state_new.reshape((1, self.state_size))
				steps_current += 1

				self.STACK_HISTORY_HOLDER_ACCURACY.append(int(~done))
				self.CACHE_PROGRESS_CURRENT = [
					epoch,
					epoch_current,
					'{:4,}'.format(size_all_symbs),
					'{:4,}'.format(idx_current_symb),
					'{:4,}'.format(observations_size_current),
					'{0:.4f}'.format(rate_applied)
				]

			self.train_model()
			self.update_local_model()

			steps_current = 1
			failure_current = 0
			episode_current += 1

	def discounted_prediction(self):
		discounted_rewards = np.zeros_like(self.rewards)
		running_add = self.critic.predict(self.states[-1].reshape((1, self.state_size)))

		for reversed_idx in reversed(range(0, self.rewards.size)):
			if not self.dones[reversed_idx]:
				running_add = running_add * self.discount_factor + self.rewards[reversed_idx]
				discounted_rewards[reversed_idx] = running_add
			else:
				discounted_rewards[reversed_idx] = self.rewards[reversed_idx]

		if np.sum(discounted_rewards) > 0:
			discounted_rewards = (discounted_rewards - np.mean(discounted_rewards)) / np.std(discounted_rewards)

		return discounted_rewards

	def train_model(self):
		self.states = np.vstack(self.states)
		self.actions = np.vstack(self.actions)
		self.rewards = np.vstack(self.rewards)
		self.dones = np.vstack(self.dones)

		discounted_prediction = self.discounted_prediction()

		values = self.critic.predict(self.states)
		advantages = discounted_prediction - values

		if np.sum(advantages) > 0:
			advantages = (advantages - np.mean(advantages)) / np.std(advantages)

		actor_loss = self.optimizer[0]([self.states, self.actions, advantages])
		critic_loss = self.optimizer[1]([self.states, discounted_prediction])

		self.STACK_HISTORY_HOLDER_ACTOR_LOSS.append(actor_loss[0])
		self.STACK_HISTORY_HOLDER_CRITIC_LOSS.append(critic_loss[0])
		self.states, self.actions, self.rewards, self.dones = [], [], [], []

	def update_local_model(self):
		self.local_actor.set_weights(self.actor.get_weights())
		self.local_critic.set_weights(self.critic.get_weights())

	def onthot(self, x):
		x = x.reshape(-1)
		return np.eye(len(x))[np.argmax(x)]


class Monitor(threading.Thread):
	def __init__(self, creator, source_type):
		threading.Thread.__init__(self)

		self.creator = creator
		self.source_type = source_type

	def run(self):
		time.sleep(60 * 1)

		while True:
			print('---', '[SOURCE TYPE:', self.source_type, ']', '-' * 120)
			i = 0
			for agent in self.creator.AGENTS:
				print('[', i, ']', agent.get_progressing())
				i += 1
			print('##### ACCURACY OF THE CREATOR''s ACOTR: %s #####' % '{0:.4f}'.format(self.creator.get_accuracy()))

			time.sleep(10)

if __name__ == "__main__":
	agent_original = A3CAgent(0, 38, 7)
	agent_original.train()
