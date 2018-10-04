import os
import random
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import tensorflow as tf
from isin_episod.dataset import Source
from isin_episod.env import Activities
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import RMSprop


class A3CAgent:
	def __init__(self, name, dataset, input_dim, output_dim):
		self._STACK_HISTORY_HOLDER_ACCURACY_ = deque(maxlen=111)
		self._DATASOURCE_NAME_ = name
		self._DATASOURCE_ = dataset
		self.AGENTS = []

		self.state_size = input_dim
		self.action_size = output_dim

		self.discount_factor = 0.99
		self.actor_lr = .0001
		self.critic_lr = .0001

		self.threads = 7

		self.actor, self.critic = self.build_model()
		self.optimizer = [self.actor_optimizer(), self.critic_optimizer()]

		self.sess = tf.InteractiveSession()
		K.set_session(self.sess)
		self.sess.run(tf.global_variables_initializer())

	def build_model(self):
		input_state = Input(shape=(self.state_size,))
		layer1 = Dense(self.state_size, activation='relu')(input_state)
		layer2 = Dense(self.state_size, activation='relu')(layer1)
		layer3 = Dense(self.state_size, activation='relu')(layer2)
		layer4 = Dense(self.state_size, activation='relu')(layer3)
		layer5 = Dense(self.state_size, activation='relu')(layer4)
		layer6 = Dense(self.state_size, activation='relu')(layer5)
		layer7 = Dense(self.state_size, activation='relu')(layer6)

		policy = Dense(self.action_size, activation='softmax')(layer7)
		value = Dense(1, activation='linear')(layer7)

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

		optimizer = RMSprop(lr=self.actor_lr)
		updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
		train = K.function([self.actor.input, action, advantages], [loss], updates=updates)

		return train

	def critic_optimizer(self):
		discounted_prediction = K.placeholder(shape=(None, 1))

		value = self.critic.output
		loss = K.mean(K.square(discounted_prediction - value))

		optimizer = RMSprop(lr=self.critic_lr)
		updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
		train = K.function([self.critic.input, discounted_prediction], [loss], updates=updates)

		return train

	def train(self):
		print('-' * 100)
		self.AGENTS = [
			Agent(self,
			      self._DATASOURCE_NAME_,
			      self._DATASOURCE_.copy(),
			      self.state_size,
			      self.action_size,
			      [self.actor, self.critic],
			      self.sess,
			      self.optimizer,
			      self.discount_factor
			      ) for _ in range(self.threads)
		]
		for agent in self.AGENTS:
			time.sleep(.1)
			agent.start()

		monitor = Monitor(self)
		monitor.start()

		agnet_state = []
		time.sleep(60 * 15)

		while True:
			self.save()
			
			for agent in self.AGENTS:
				if agent.stopped: agnet_state.append(1)

			if len(agnet_state) == self.threads:
				for agent in self.AGENTS: agent.stop()

				monitor.stopped = True
				while True:
					time.sleep(5)
					if monitor.terminated:
						monitor.stop()
						break

				break

			time.sleep(20)

	def save(self):
		mean_acc = int(np.mean(self.get_accuracy()) * 100)
		if mean_acc < 90: return

		_MODEL_WEIGHT_DIR_ = 'graduated/%s/' % self._DATASOURCE_NAME_
		if not os.path.exists(_MODEL_WEIGHT_DIR_): os.makedirs(_MODEL_WEIGHT_DIR_)

		model_actor_located = '%s_MODEL_W_ACTOR_v%s.h5' % (_MODEL_WEIGHT_DIR_, mean_acc)
		model_critic_located = '%s_MODEL_W_CRITIC_v%s.h5' % (_MODEL_WEIGHT_DIR_, mean_acc)

		self.actor.save_weights(model_actor_located)
		self.critic.save_weights(model_critic_located)

		import logging
		for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
		logging.basicConfig(filename='%s_%s_build_log.log' % (_MODEL_WEIGHT_DIR_, datetime.today().strftime('%Y%m%d')), level=logging.DEBUG)
		logging.info(' ### Logging datatime : %s, Percentages of the process : 0.%s' % (datetime.today().strftime('%H:%M:%S'), mean_acc))

	def get_accuracy(self):
		accuracys = []
		for agent in self.AGENTS: accuracys.append(agent.get_accuracy())
		return np.mean(accuracys)


class Agent(threading.Thread):
	def __init__(self, creator, datasource_name, datasource, state_size, action_size, model, sess, optimizer, discount_factor):
		threading.Thread.__init__(self)

		self._DATASOURCE_NAME_ = datasource_name
		self._DATASOURCE_ = datasource
		self._EPSILON_ = 1.
		self._EPSILON_DECAY_ = .99995
		# self._EPSILON_DECAY_ = .995

		self.STACK_HISTORY_HOLDER_ACTOR_LOSS = deque(maxlen=10)
		self.STACK_HISTORY_HOLDER_CRITIC_LOSS = deque(maxlen=10)
		self.STACK_HISTORY_HOLDER_ACCURACY = deque(maxlen=111)
		self.STACK_HISTORY_HOLDER_MODEL_PREDICT = deque(maxlen=111)
		self.STACK_HISTORY_HOLDER_MODEL_ACTION_RANGE = deque(maxlen=111)

		self.CACHE_PROGRESS_CURRENT = []

		self.creator = creator
		self.action_size = action_size
		self.state_size = state_size
		self.actor, self.critic = model
		self.sess = sess
		self.optimizer = optimizer
		self.discount_factor = discount_factor
		self.states, self.actions, self.rewards, self.dones, self.y_values = [], [], [], [], []

		self.local_actor, self.local_critic = self.creator.build_model()
		self.local_actor.set_weights(self.actor.get_weights())
		self.local_critic.set_weights(self.critic.get_weights())

		self.stopped = False
		self._stop_event = threading.Event()

	def softmax(self, x):
		e = np.exp(x - np.max(x))
		if e.ndim == 1:
			return e / np.sum(e, axis=0)
		else:
			return e / np.array([np.sum(e, axis=1)]).T

	def onehot(self, x):
		x = x.reshape(-1)
		return np.eye(len(x))[np.argmax(x)]

	def get_action(self, state):
		action_type = 1

		if np.random.random() < self._EPSILON_:
			action_probability = self.softmax(np.random.uniform(0., 1., self.action_size)).reshape((1, self.action_size))
			action_type = 0
		else:
			action_probability = self.local_actor.predict(state)

		self.STACK_HISTORY_HOLDER_MODEL_ACTION_RANGE.append(self.onehot(action_probability).reshape((1, self.action_size)))

		if self._EPSILON_ > 0.0001:
			self._EPSILON_ *= self._EPSILON_DECAY_

		return action_probability, action_type

	def get_accuracy(self):
		return np.mean(self.STACK_HISTORY_HOLDER_ACCURACY)

	def get_progressing(self):
		actor_loss = list(self.STACK_HISTORY_HOLDER_ACTOR_LOSS)
		critic_loss = list(self.STACK_HISTORY_HOLDER_CRITIC_LOSS)
		pridict = list(self.STACK_HISTORY_HOLDER_MODEL_PREDICT)
		action_range = list(self.STACK_HISTORY_HOLDER_MODEL_ACTION_RANGE)
		accuracy = list(self.STACK_HISTORY_HOLDER_ACCURACY)
		data_progress = list(self.CACHE_PROGRESS_CURRENT)

		mean_actor = 0 if len(actor_loss) == 0 else np.nan_to_num(np.mean(actor_loss))
		mean_critic = 0 if len(critic_loss) == 0 else np.nan_to_num(np.mean(critic_loss))
		mean_model_predict = np.mean(pridict)
		action_range = np.round(np.mean(np.vstack(action_range), axis=0), 2) if len(action_range) > 0 else np.zeros(7)

		return {
			'LOSS': '%s%s %s%s %s' % (
				' ' if mean_actor >= 0 else '',
				'{0:3.7f}'.format(mean_actor),
				' ' if mean_critic >= 0 else '',
				'{0:3.7f}'.format(mean_critic),
				'{0:.4f}'.format(np.mean(accuracy))
			),
			'PREDICT': '%s %s' % (
				'{0:.4f}'.format(mean_model_predict),
				action_range
			),
			'PROGRESS': data_progress
		}

	def append_sample(self, state, action_label, reward, done, y_value):
		self.states.append(state)
		self.actions.append(action_label)
		self.rewards.append(reward)
		self.dones.append(done)
		self.y_values.append(y_value)

	def run(self):
		print('##### RUNNING : ', self)

		epoch = 3000
		epoch_current = 1
		steps_an_episode = 15
		steps_current = 1
		# failure_significant_rate = .05
		# failure_limit = int(steps_an_episode * failure_significant_rate)
		failure_limit = 5
		failure_current = 0
		episode_current = 1
		stop_running = False
		done = False

		env_act = Activities(self._DATASOURCE_, self.state_size, self.action_size)

		while epoch_current <= epoch and not stop_running:
			while failure_current <= failure_limit and steps_current <= steps_an_episode and not stop_running:
			# while steps_current <= steps_an_episode and not stop_running:
				time.sleep(np.round(random.uniform(.0, .05), 3))

				state, y_value, rate_applied = env_act.get_observation()
				if type(state) is int:
					epoch_current += 1
					break

				state = state.reshape((1, self.state_size))

				action_stochastic_policy, action_type = self.get_action(state)
				reward, done = env_act.step(action_stochastic_policy, y_value)

				self.append_sample(
					state,
					self.onehot(action_stochastic_policy).reshape((1, self.action_size)),
					reward,
					done,
					y_value
				)
				self.STACK_HISTORY_HOLDER_MODEL_PREDICT.append(action_type)
				self.STACK_HISTORY_HOLDER_ACCURACY.append(int(~done))
				self.CACHE_PROGRESS_CURRENT = [
					epoch,
					epoch_current,
					'{0:.4f}'.format(rate_applied)
				]

				if done: failure_current += 1
				steps_current += 1

			if len(self.states) > 0:
				self.train_model()
				self.update_local_model()

			steps_current = 1
			failure_current = 0
			episode_current += 1

		self.stopped = True

	def stop(self):
		self._stop_event.set()

		print('##### TERMINATING : %s' % self)

		try: self._stop()
		except: pass

	def is_stopped(self):
		return self._stop_event.is_set()

	def discounted_prediction(self):
		discounted_rewards = np.zeros_like(self.rewards)
		running_add = self.critic.predict(self.states[-1].reshape((1, self.state_size)))

		for reversed_idx in reversed(range(0, self.rewards.size)):
			if not self.dones[reversed_idx]:
				#print('(%s) %s: %s - %s' % (self, self.dones[reversed_idx], self.actions[reversed_idx], self.y_values[reversed_idx]))
				running_add = running_add * self.discount_factor + self.rewards[reversed_idx]
				discounted_rewards[reversed_idx] = running_add
			else:
				discounted_rewards[reversed_idx] = self.rewards[reversed_idx]

		return discounted_rewards

	def train_model(self):
		self.states = np.vstack(self.states)
		self.actions = np.vstack(self.actions)
		self.rewards = np.vstack(self.rewards)
		self.dones = np.vstack(self.dones)

		discounted_prediction = self.discounted_prediction()
		advantages = discounted_prediction - self.critic.predict(self.states)

		actor_loss = self.optimizer[0]([self.states, self.actions, advantages])
		critic_loss = self.optimizer[1]([self.states, discounted_prediction])

		self.STACK_HISTORY_HOLDER_ACTOR_LOSS.append(actor_loss[0])
		self.STACK_HISTORY_HOLDER_CRITIC_LOSS.append(critic_loss[0])

		self.states, self.actions, self.rewards, self.dones, self.y_values = [], [], [], [], []

	def update_local_model(self):
		self.local_actor.set_weights(self.actor.get_weights())
		self.local_critic.set_weights(self.critic.get_weights())


class Monitor(threading.Thread):
	def __init__(self, creator):
		threading.Thread.__init__(self)

		self.creator = creator

		self.terminated = False
		self.stopped = False
		self._stop_event = threading.Event()

	def run(self):
		while not self.stopped:
			time.sleep(60)

			print('-' * 100)
			i = 0
			for agent in self.creator.AGENTS:
				print(i, agent.get_progressing())
				i += 1
			print('##### ACCURACY OF THE CREATOR''s ACOTR: %s #####' % '{0:.4f}'.format(self.creator.get_accuracy()))

		self.terminated = True

	def stop(self):
		self._stop_event.set()

		print('##### TERMINATING : %s' % self)

		try: self._stop()
		except: pass

	def is_stopped(self):
		return self._stop_event.is_set()


def main():
	datasource = Source()
	source_size = datasource.source_size()

	for i in range(source_size):
		if len(datasource.next()) == 0: break

		name, dataset, input_dim, output_dim = datasource.get_dateset()
		dataset_info = datasource.dataset_info()
		print('##### [ISIN: %s, SYMBOL NAME: %s, SCALE: %s] started. (%s of %s)' % (dataset_info[1], dataset_info[2], dataset_info[3], i+1, source_size))

		agent = A3CAgent(name, dataset, input_dim, output_dim)
		agent.train()

	print('##### All the trainings are finished. -----------------------------------------------------------------------')



if __name__ == "__main__":
	main()