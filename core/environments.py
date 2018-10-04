import numpy as np
import sys

# np.seterr(divide='ignore', invalid='ignore')


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()



class Monitor(object):
	def __init__(self, on=True):
		self.on = on
		self._HISTORY_ = {'ACTION':{'RESULT':[]}}


	def history(self, action, result):
		self._HISTORY_['ACTION']['RESULT'].append(int(action == result))


	def show(self, request):
		if request not in ['episod', 'anobservationset', 'anepoch', 'oneview']:
			print('Invalid request type. Please check the option to request history.')
			exit()

		if request == 'episod':
			success = np.sum(self._HISTORY_['ACTION']['RESULT'])
			self._HISTORY_['ACTION']['RESULT'] = []
			return success

		elif request == 'anobservationset':
			pass
		elif request == 'anepoch':
			pass
		else:
			pass



class Activities(object):
	_DISCOUNT_FACTOR_ = 0.995

	def __init__(self, running_mode=0, epoch=1):
		self._COUNT_TOTAL_RUNNING_ = 0

		# running_type = 0: Training mode
		# running_type = 1: Testing mode
		self._ENV_ORSEVATIONS_ = Observations(running_mode, epoch)
		self.monitor = Monitor()


	def _parse_current_observation(self):
		y_value = self._STATE_[-1]
		state = self._STATE_[2:-1].astype('float')
		# state = (state - np.mean(state)) / np.std(state)

		return state, y_value


	def reset(self):
		self._SIZE_OF_CURRENT_DATESET_, epoch = self._ENV_ORSEVATIONS_.next_source()
		return self._SIZE_OF_CURRENT_DATESET_, epoch


	def step(self, preaction, action):
		_, y_value = self._parse_current_observation()
		done = ~(action == y_value)

		print('   ### PREDICTION: %s / %s, REAL: %s' % ('{0:2d}'.format(preaction), '{0:2d}'.format(action), '{0:2d}'.format(y_value)))

		if not done: reward = 1.
		else: reward = 0.1

		self.monitor.history(action, y_value)

		return reward, done


	def get_observation(self):
		self._STATE_ = self._ENV_ORSEVATIONS_.next_observation()

		if self._STATE_ is None:
			return None

		self._COUNT_TOTAL_RUNNING_ += 1
		state, _ = self._parse_current_observation()
		return state


	def get_size_of_an_epoch(self):
		return self._ENV_ORSEVATIONS_.get_size_of_an_epoch()


	def disount_rewards(self, rewards):
		discounted_rewards = np.zeros_like(rewards)
		running_add = 0
		for reversed_idx in reversed(range(0, rewards.size)):
			running_add = running_add * self._DISCOUNT_FACTOR_ + rewards[reversed_idx]
			discounted_rewards[reversed_idx] = running_add

		return discounted_rewards


	def run_times(self):
		return self._COUNT_TOTAL_RUNNING_



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

		query_string =  'SELECT COUNT(*) count_symbols FROM krx_symbols'
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

		query_string =  'SELECT  * ' + \
						'FROM    drl_proportion_source_1d ' + \
						'WHERE   isin = %s AND trans_date = trans_date ' + \
						'ORDER BY trans_date'
		values = [isin]
		self._CURRENT_SOURCE_OBSERVATIONS_ = self._SOURCE_CONNECTION_.select(query_string, values)
		self._SIZE_OBSERVATIONS_ = len(self._CURRENT_SOURCE_OBSERVATIONS_)
		return self._SIZE_OBSERVATIONS_, self._CURRENT_EPOCH_INDEX_


	def next_observation(self):
		self._CURRENT_OBSERVATION_INDEX_ += 1

		if(self._CURRENT_OBSERVATION_INDEX_ == len(self._CURRENT_SOURCE_OBSERVATIONS_)):
			self._CURRENT_OBSERVATION_INDEX_ = -1
			return None

		observation = self._CURRENT_SOURCE_OBSERVATIONS_.loc[self._CURRENT_OBSERVATION_INDEX_, ]
		return np.array(observation.values)


	def get_size_of_an_epoch(self):
		return self._SIZE_OBSERVATIONS_IN_A_EPOCH_


	def get_index_of_current_datasource(self):
		return self._CURRENT_OBSERVATION_INDEX_



class DeepNeuralNetwork(object):
	_BUFFER_RMS_BACK_PROPAGATION_ = []
	_BUFFER_COST_GRADIENTDESCENT_ = []
	_BUFFER_LAYER1_ = None
	_BUFFER_LAYER2_ = None
	_BUFFER_LAYER3_ = None
	_BUFFER_LAYER4_ = None
	_BUFFER_LAYER5_ = None
	_BUFFER_LAYER6_ = None

	_MODEL_ = {}
	
	_SIZE_BATCH_ = 10
	_INDEX_CURRENT_LAYER_BUFFER_ = 0

	_RATE_DECAY_ = 0.995
	_RATE_LEARNING_ = 0.001

	def __init__(self, out_dim, input_dim):
		np.random.seed(1)

		self._MODEL_['W1'] = self.xavier(input_dim**2, input_dim)
		self._MODEL_['W2'] = self.xavier(input_dim**2, input_dim**2)
		self._MODEL_['W3'] = self.xavier(input_dim**2, input_dim**2)
		self._MODEL_['W4'] = self.xavier(input_dim**2, input_dim**2)
		self._MODEL_['W5'] = self.xavier(input_dim**2, input_dim**2)
		self._MODEL_['W6'] = self.xavier(input_dim, input_dim**2)
		self._MODEL_['W7'] = self.xavier(out_dim, input_dim)

		for _ in range(self._SIZE_BATCH_):
			self._BUFFER_COST_GRADIENTDESCENT_.append({key: np.zeros_like(values) for key, values in self._MODEL_.items()})

		self._BUFFER_RMS_BACK_PROPAGATION_ = {key: np.zeros_like(values) for key, values in self._MODEL_.items()}
		self._init_layer_buf()

	def _init_layer_buf(self):
		self._BUFFER_LAYER1_ = []
		self._BUFFER_LAYER2_ = []
		self._BUFFER_LAYER3_ = []
		self._BUFFER_LAYER4_ = []
		self._BUFFER_LAYER5_ = []
		self._BUFFER_LAYER6_ = []
		self._INDEX_CURRENT_LAYER_BUFFER_ = 0


	def _release_layer_buf(self):
		self._BUFFER_LAYER1_ = np.vstack(self._BUFFER_LAYER1_)
		self._BUFFER_LAYER2_ = np.vstack(self._BUFFER_LAYER2_)
		self._BUFFER_LAYER3_ = np.vstack(self._BUFFER_LAYER3_)
		self._BUFFER_LAYER4_ = np.vstack(self._BUFFER_LAYER4_)
		self._BUFFER_LAYER5_ = np.vstack(self._BUFFER_LAYER5_)
		self._BUFFER_LAYER6_ = np.vstack(self._BUFFER_LAYER6_)


	# Policy Forward
	def feed_forward(self, x):
		self._BUFFER_LAYER1_.append(self.ReLU(np.dot(self._MODEL_['W1'], x)))                                                       # 1st Layer
		self._BUFFER_LAYER2_.append(self.ReLU(np.dot(self._MODEL_['W2'], self._BUFFER_LAYER1_[self._INDEX_CURRENT_LAYER_BUFFER_]))) # 2nd Layer
		self._BUFFER_LAYER3_.append(self.ReLU(np.dot(self._MODEL_['W3'], self._BUFFER_LAYER2_[self._INDEX_CURRENT_LAYER_BUFFER_]))) # 3rd Layer
		self._BUFFER_LAYER4_.append(self.ReLU(np.dot(self._MODEL_['W4'], self._BUFFER_LAYER3_[self._INDEX_CURRENT_LAYER_BUFFER_]))) # 4th Layer
		self._BUFFER_LAYER5_.append(self.ReLU(np.dot(self._MODEL_['W5'], self._BUFFER_LAYER4_[self._INDEX_CURRENT_LAYER_BUFFER_]))) # 5rd Layer
		self._BUFFER_LAYER6_.append(self.ReLU(np.dot(self._MODEL_['W6'], self._BUFFER_LAYER5_[self._INDEX_CURRENT_LAYER_BUFFER_]))) # 6th Layer
		hypothesis = self.softmax(np.dot(self._MODEL_['W7'], self._BUFFER_LAYER6_[self._INDEX_CURRENT_LAYER_BUFFER_]))              # 7th Layer / Hypothesis Layer

		self._INDEX_CURRENT_LAYER_BUFFER_ += 1
		return hypothesis


	# Poliby Backward - Back Propagation
	def feed_backward(self, states, policygradient_errs, episode):
		self._release_layer_buf()

		H_rmsprotagation_Layer6 = np.dot(policygradient_errs, self._MODEL_['W7'])
		D_rmsprotagation_Layer6 = self._BUFFER_LAYER6_ * (1 - self._BUFFER_LAYER6_)
		W_rmsprotagation_Layer6 = H_rmsprotagation_Layer6 * D_rmsprotagation_Layer6

		H_rmsprotagation_Layer5 = np.dot(W_rmsprotagation_Layer6, self._MODEL_['W6'])
		D_rmsprotagation_Layer5 = self._BUFFER_LAYER5_ * (1 - self._BUFFER_LAYER5_)
		W_rmsprotagation_Layer5 = H_rmsprotagation_Layer5 * D_rmsprotagation_Layer5

		H_rmsprotagation_Layer4 = np.dot(W_rmsprotagation_Layer5, self._MODEL_['W5'])
		D_rmsprotagation_Layer4 = self._BUFFER_LAYER4_ * (1 - self._BUFFER_LAYER4_)
		W_rmsprotagation_Layer4 = H_rmsprotagation_Layer4 * D_rmsprotagation_Layer4

		H_rmsprotagation_Layer3 = np.dot(W_rmsprotagation_Layer4, self._MODEL_['W4'])
		D_rmsprotagation_Layer3 = self._BUFFER_LAYER3_ * (1 - self._BUFFER_LAYER3_)
		W_rmsprotagation_Layer3 = H_rmsprotagation_Layer3 * D_rmsprotagation_Layer3

		H_rmsprotagation_Layer2 = np.dot(W_rmsprotagation_Layer3, self._MODEL_['W3'])
		D_rmsprotagation_Layer2 = self._BUFFER_LAYER2_ * (1 - self._BUFFER_LAYER2_)
		W_rmsprotagation_Layer2 = H_rmsprotagation_Layer2 * D_rmsprotagation_Layer2

		H_rmsprotagation_Layer1 = np.dot(W_rmsprotagation_Layer2, self._MODEL_['W2'])
		D_rmsprotagation_Layer1 = self._BUFFER_LAYER1_ * (1 - self._BUFFER_LAYER1_)
		W_rmsprotagation_Layer1 = H_rmsprotagation_Layer1 * D_rmsprotagation_Layer1

		W_rmsprotagation_Layer7 = np.dot(self._BUFFER_LAYER6_.T, policygradient_errs).T
		W_rmsprotagation_Layer6 = np.dot(W_rmsprotagation_Layer6.T, self._BUFFER_LAYER5_)
		W_rmsprotagation_Layer5 = np.dot(W_rmsprotagation_Layer5.T, self._BUFFER_LAYER4_)
		W_rmsprotagation_Layer4 = np.dot(W_rmsprotagation_Layer4.T, self._BUFFER_LAYER3_)
		W_rmsprotagation_Layer3 = np.dot(W_rmsprotagation_Layer3.T, self._BUFFER_LAYER2_)
		W_rmsprotagation_Layer2 = np.dot(W_rmsprotagation_Layer2.T, self._BUFFER_LAYER1_)
		W_rmsprotagation_Layer1 = np.dot(W_rmsprotagation_Layer1.T, states)

		self._BUFFER_COST_GRADIENTDESCENT_[episode % self._SIZE_BATCH_] = {
			'W1': W_rmsprotagation_Layer1,
			'W2': W_rmsprotagation_Layer2,
			'W3': W_rmsprotagation_Layer3,
			'W4': W_rmsprotagation_Layer4,
			'W5': W_rmsprotagation_Layer5,
			'W6': W_rmsprotagation_Layer6,
			'W7': W_rmsprotagation_Layer7
		}
		self._init_layer_buf()


	def deep_learning(self):
		# Skip the zeros buffer, it was assigned zeros to add the learned values new assigning to...
		# So, the process will be stated from 1.
		cost_optimized_layers = self._BUFFER_COST_GRADIENTDESCENT_[0]

		for i in range(1, self._SIZE_BATCH_):
			for key, values in self._MODEL_.items():
				cost_optimized_layers[key] += self._BUFFER_COST_GRADIENTDESCENT_[i][key]

		for key, values in self._MODEL_.items():
			cost_optimized_layer = cost_optimized_layers[key]
			self._BUFFER_RMS_BACK_PROPAGATION_[key] = self._RATE_DECAY_ * self._BUFFER_RMS_BACK_PROPAGATION_[key] + \
			                                          (1 - self._RATE_DECAY_) * cost_optimized_layer**2
			self._MODEL_[key] += self._RATE_LEARNING_ * cost_optimized_layer / (np.sqrt(self._BUFFER_RMS_BACK_PROPAGATION_[key]) + 1e-5)


	def model(self):
		return self._MODEL_


	def xavier(self, out_dim, in_dim, prime=False):
		return np.random.randn(out_dim, in_dim) / np.sqrt(in_dim / 2 if prime else 1)


	def sigmoid(self, x):
		return 1.0 / (1.0 + np.exp(-x))


	def softmax(self, x):
		e = np.exp(x - np.max(x)) # prevent overflow
		if e.ndim == 1:
			return e / np.sum(e, axis=0)
		else:
			return e / np.array([np.sum(e, axis=1)]).T # ndim = 2


	def ReLU(self, x):
		return x * (x > 0)

	def onthot(self, x):
		x = x.reshape(-1)
		return np.eye(len(x))[np.argmax(x)]



class ReinforcementLearning(object):
	_DISCOUNT_FACTOR_ = 0.995

	def __init__(self):
		pass

	def disount_rewards(self, rewards):
		discounted_rewards = np.zeros_like(rewards)
		running_add = 0
		for reversed_idx in reversed(range(0, rewards.size)):
			running_add = running_add * self._DISCOUNT_FACTOR_ + rewards[reversed_idx]
			discounted_rewards[reversed_idx] = running_add

		return discounted_rewards


