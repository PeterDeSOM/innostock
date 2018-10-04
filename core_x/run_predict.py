from keras.layers import Dense, Add, Input
from keras.optimizers import RMSprop
from keras import backend as K
from keras.models import Model
from keras.models import model_from_json
from core_x.env_x import Activities
from collections import deque

import tensorflow as tf
import numpy as np
import threading
import random
import time


class Prediction:
	def __init__(self, out_dim, in_dim):
		self.state_size = in_dim
		self.action_size = out_dim
		self.actor, self.critic = self.build_model()

	def build_model(self):
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

		layer9 = Dense(self.state_size, activation='relu')(merge2)

		policy = Dense(self.action_size, activation='softmax')(layer9)
		value = Dense(1, activation='linear')(layer9)

		actor = Model(inputs=input_state, outputs=policy)
		critic = Model(inputs=input_state, outputs=value)

		return actor, critic

	def load_model(self):
		_MODEL_GRAPH_DIR_ = 'graduated_pure/'
		self.actor.load_weights('%sV100_MODEL_W_ACTOR_.h5' % _MODEL_GRAPH_DIR_)
		self.critic.load_weights('%sV100_MODEL_W_CRITIC_.h5' % _MODEL_GRAPH_DIR_)

	def get_action(self, state):
		return self.actor.predict(state)


if __name__ == "__main__":
	action = Prediction(7, 35)
	action.load_model()

	env_act = Activities(0, 0, 1, 7, 35)
	idx_current_symb, size_all_symbs, observations_size_current, epoch_current = env_act.reset()

	for i in range(20):
		state_current, y_value, rate_applied = env_act.get_observation()
		state_current = state_current.reshape((1, 35))

		print(y_value==np.argmax(action.get_action(state_current)))