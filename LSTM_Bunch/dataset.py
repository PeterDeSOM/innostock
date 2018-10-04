import pandas as pd
import numpy as np

from keras.utils import np_utils
from databases import maria

class Source(object):
	def __init__(self):
		self._SOURCE_CONNECTION_ = maria()
		
		self._SET_ENV_ = False
		self._SOURCE_INDEX_ = -1
		self._SOURCE_SIZE_ = 0

		query_string = 'SELECT COUNT(*) col_count FROM information_schema.columns WHERE table_name = %s'
		self._SOURCE_COLS_ = self._SOURCE_CONNECTION_.select(query_string, ['drl_1d']).get_value(0, 'col_count')

	def set_env(self):
		query_string = 'SELECT  A.isin isin, A.symb_name symb_name, 0 applied ' + \
					   'FROM    krx_symbols A INNER JOIN drl_1d B ON (A.isin = B.isin) ' + \
					   'GROUP BY A.isin, A.symb_name ORDER BY symb_name'
		self._SOURCE_INFO_ = self._SOURCE_CONNECTION_.select(query_string)
		self._SOURCE_SIZE_ = len(self._SOURCE_INFO_)
		self._SET_ENV_ = True

	def predictable_col_len(self):
		return int(self._SOURCE_COLS_ - 3)

	def source_info(self):
		return [
			self._SOURCE_INDEX_,
			self._SOURCE_SIZE_,
			self._SOURCE_COLS_ - 3
		]
	
	def next(self):
		self._SOURCE_INDEX_ += 1

		if self._SOURCE_INDEX_ == self._SOURCE_SIZE_:
			self._SOURCE_INDEX_ = -1
			return pd.DataFrame()

		query_string = 'SELECT  * ' + \
					   'FROM    drl_1d ' + \
					   'WHERE	isin = %s AND trans_date > %s ' + \
					   'ORDER BY isin, trans_date'
		self._SYMBOL_DATASET_ = self._SOURCE_CONNECTION_.select(query_string, [
			self._SOURCE_INFO_.loc[self._SOURCE_INDEX_, 'isin'], '1900-01-01'
		])
		return self._SYMBOL_DATASET_

	def dataset_info(self):
		return [
			self._SOURCE_INFO_.loc[self._SOURCE_INDEX_, 'isin'] if self._SOURCE_INDEX_ >= 0 else '',
			self._SOURCE_INFO_.loc[self._SOURCE_INDEX_, 'symb_name'] if self._SOURCE_INDEX_ >= 0 else ''
		]

	def source_size(self):
		return self._SOURCE_SIZE_

	def get_dateset(self):
		return self._SYMBOL_DATASET_


class DatasetController:
	def __init__(self, in_dim, input_length, out_dim):
		self._DIM_INPUT_ = in_dim
		self._WINDOW_SIZE_ = input_length
		self._DIM_OUTPUT_ = out_dim

	def set_dataenv(self, dataset):
		self._DATASET_ = dataset

		self.dataset_X, self.dataset_Y = self._convert_to_predictable()

		self._SIZE_OBSERVATIONS_ = len(self.dataset_X)
		self._DATASET_DATA_STATUS = pd.DataFrame({'applied': [0] * self._SIZE_OBSERVATIONS_})
		self._CURRENT_OBSERVATION_INDEX_ = -1

	def _convert_to_predictable(self):
		col_len = len(self._DATASET_.columns)
		df_target = self._DATASET_[self._DATASET_.columns[2:col_len]]

		df_target = df_target.reset_index(drop=True)

		dataset_X = []
		dataset_Y = []

		for i in range(len(df_target) - self._WINDOW_SIZE_ + 1):
			df_subset = df_target.iloc[i:(i + self._WINDOW_SIZE_), :]

			for subset in df_subset.values:
				dataset_X.append(subset[0:-1])

			dataset_Y.append([df_subset.loc[i + self._WINDOW_SIZE_ - 1, 'target_value']])

		dataset_X = np.vstack(dataset_X)
		dataset_Y = np.vstack(dataset_Y)

		dataset_X = np.reshape(dataset_X, (np.shape(dataset_Y)[0], self._WINDOW_SIZE_, np.shape(dataset_X)[1]))
		dataset_Y = np_utils.to_categorical(dataset_Y, num_classes=self._DIM_OUTPUT_)

		return dataset_X, dataset_Y

	def size(self):
		return self._SIZE_OBSERVATIONS_

	def id(self):
		return self._CURRENT_OBSERVATION_INDEX_

	def next(self):
		rate_applied = self._DATASET_DATA_STATUS['applied'].sum() / self._SIZE_OBSERVATIONS_

		self._CURRENT_OBSERVATION_INDEX_ += 1

		if self._CURRENT_OBSERVATION_INDEX_ == self._SIZE_OBSERVATIONS_:
			self._DATASET_DATA_STATUS = pd.DataFrame({'applied': [0] * self._SIZE_OBSERVATIONS_})
			self._CURRENT_OBSERVATION_INDEX_ = -1
			return 1, -1, .0

		self._DATASET_DATA_STATUS.loc[self._CURRENT_OBSERVATION_INDEX_, 'applied'] = 1

		return self.dataset_X[self._CURRENT_OBSERVATION_INDEX_], \
			   np.argmax(self.dataset_Y[self._CURRENT_OBSERVATION_INDEX_]), \
			   rate_applied

class PredictSource:
	def __init__(self, input_length):
		self._SOURCE_CONNECTION_ = maria()
		self.input_length = input_length

		query_string = 'SELECT COUNT(*) col_count FROM information_schema.columns WHERE table_name = %s'
		self._SOURCE_COLS_ = self._SOURCE_CONNECTION_.select(query_string, ['drl_1d_testing']).get_value(0, 'col_count')

	def predictable_col_len(self):
		return int(self._SOURCE_COLS_ - 4)

	def get_symbols_header(self):
		query_string = 'SELECT isin, symb_name, symb_code FROM krx_symbols WHERE symb_status = %s'
		return self._SOURCE_CONNECTION_.select(query_string, ['A'])

	def is_predictable(self, allow_repredict, predict_length):
		where_clause = ''
		if not allow_repredict: where_clause = 'WHERE predict_value IS NULL '

		query_string = 'SELECT COUNT(trans_date) count_ FROM (SELECT trans_date FROM drl_1d_testing ' + where_clause + 'GROUP BY trans_date) R'
		targets = int(self._SOURCE_CONNECTION_.select(query_string).get_value(0, 'count_')) - (self.input_length - 1)

		if targets < 1:
			return False
		else:
			if predict_length != 0 and predict_length > targets:
				print('##### ERROR: predict_length (%s) must be below %s.' % (predict_length, targets))
				return False

			return True

	def get_dataset(self, allow_repredict, predict_length):
		where_clause = ''
		if not allow_repredict: where_clause = 'WHERE predict_value IS NULL '

		df = pd.DataFrame()

		if predict_length == 0:
			query_string = 'SELECT * FROM drl_1d_testing ' + where_clause + 'ORDER BY isin, trans_date'
			df = self._SOURCE_CONNECTION_.select(query_string)
		else:
			query_string = 'SELECT trans_date FROM drl_1d_testing ' + where_clause + 'GROUP BY trans_date DESC LIMIT %s, %s'
			start_date = self._SOURCE_CONNECTION_.select(query_string, [self.input_length + predict_length - 1, 1]).get_value(0, 'trans_date')

			query_string = 'SELECT * FROM drl_1d_testing ' + \
						   ('WHERE trans_date > %s ' if where_clause == '' else where_clause + ' AND trans_date > %s ') + \
						   'ORDER BY isin, trans_date'
			df = self._SOURCE_CONNECTION_.select(query_string, [start_date])

		return df

	def update_predictables(self, update_values):
		for values in update_values.values:
			query_string = 'UPDATE drl_1d_testing SET predict_value = %s WHERE isin = %s AND trans_date = %s'
			self._SOURCE_CONNECTION_.execute(query_string, [values[2], values[0], values[1]])


class PredictController:
	def __init__(self, dataset, state_size, input_length, action_size):
		self._SOURCE_CONNECTION_ = maria()
		self._DATASET_ = dataset

		self._DIM_INPUT_ = state_size
		self._WINDOW_SIZE_ = input_length
		self._DIM_OUTPUT_ = action_size

		self._SOURCE_INFO_ = dataset['isin'].unique()
		self._SOURCE_SIZE_ = len(self._SOURCE_INFO_)
		self._SOURCE_INDEX_ = -1

		self._CURRENT_PREDICTABLE_INDEX_ = -1
		self._CURRENT_PREDICTABLE_SIZE_ = 0

	def next(self):
		self._SOURCE_INDEX_ += 1

		if self._SOURCE_INDEX_ == self._SOURCE_SIZE_:
			return -1

		self._SYMBOL_DATASET_ = self._DATASET_[self._DATASET_['isin'] == self._SOURCE_INFO_[self._SOURCE_INDEX_]]
		self._convert_to_predictable()

		return self._SOURCE_INDEX_

	def _convert_to_predictable(self):
		col_len = len(self._SYMBOL_DATASET_.columns)
		df_target = self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[:col_len-2]].reset_index(drop=True)

		dataset_X = []
		dataset_HEADER = []

		for i in range(len(df_target) - self._WINDOW_SIZE_ + 1):
			df_subset = df_target.iloc[i:(i + self._WINDOW_SIZE_), 2:].astype('float')

			for subset in df_subset.values:
				dataset_X.append(subset[0:])

			dataset_HEADER.append([df_target.iloc[i + self._WINDOW_SIZE_ - 1, 0:2]])

		self.dataset_HEADER = np.vstack(dataset_HEADER)
		self.dataset_X = np.reshape(np.vstack(dataset_X), (np.shape(self.dataset_HEADER)[0], self._WINDOW_SIZE_, self._DIM_INPUT_))

		self._CURRENT_PREDICTABLE_INDEX_ = -1
		self._CURRENT_PREDICTABLE_SIZE_ = len(self.dataset_X)

	def next_predictable(self):
		self._CURRENT_PREDICTABLE_INDEX_ += 1

		if self._CURRENT_PREDICTABLE_INDEX_ == self._CURRENT_PREDICTABLE_SIZE_:
			self._CURRENT_PREDICTABLE_INDEX_ = -1
			return -1, np.array([])

		return self.dataset_X[self._CURRENT_PREDICTABLE_INDEX_], \
			   self.dataset_HEADER[self._CURRENT_PREDICTABLE_INDEX_]

