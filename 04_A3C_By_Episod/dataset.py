import pandas as pd
import numpy as np

from datetime import datetime
from databases import maria

class Source(object):
	def __init__(self):
		self._SOURCE_CONNECTION_ = maria()

		query_string = 'SELECT  A.isin isin, A.symb_name symb_name, 0 applied ' + \
		               'FROM    krx_symbols A INNER JOIN drl_1d B ON (A.isin = B.isin AND B.trans_date = B.trans_date) '+ \
		               'GROUP BY A.isin ORDER BY symb_name'
		self._SYMBOLS_SOURCE_ = self._SOURCE_CONNECTION_.select(query_string)
		self._SYMBOLS_SIZE_ = len(self._SYMBOLS_SOURCE_)
		self._SYMBOL_INDEX_ = -1

	def _standardization(self, isin):
		data_cols = len(self._SYMBOL_DATASET_.columns)
		stds_ = np.std(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols-2]].astype('float'), axis=0)

		if len(stds_[stds_ == 0]) > 0:
			self._SYMBOL_DATASET_ = self._SYMBOL_DATASET_.drop(stds_[stds_ == 0].index, axis=1)
			data_cols = len(self._SYMBOL_DATASET_.columns)

		means = np.round(np.mean(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols-2]].astype('float'), axis=0), 19)
		stds_ = np.round(np.std(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols-2]].astype('float'), axis=0), 19)

		self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols - 2]] = \
			(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols-2]].astype('float') - means) / stds_

		df_scaler = pd.concat([means, stds_], axis=1, join='inner').T

		scaler_size = len(df_scaler)
		state_size = data_cols-4
		applied_date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

		df_scaler = df_scaler.assign(**{'isin': [isin] * scaler_size})
		df_scaler = df_scaler.assign(**{'day_type': [1] * scaler_size})
		df_scaler = df_scaler.assign(**{'value_type': ['M', 'S']})
		df_scaler = df_scaler.assign(**{'applied_date': [applied_date] * scaler_size})
		df_scaler = df_scaler.assign(**{'applied_input_size': [state_size] * scaler_size})

		self._SOURCE_CONNECTION_.insert("drl_scalers", df_scaler)
		self.scale = state_size

	def next(self):
		self._SYMBOL_INDEX_ += 1

		if self._SYMBOL_INDEX_ == self._SYMBOLS_SIZE_:
			self._SYMBOL_INDEX_ = -1
			return pd.DataFrame()

		isin = self._SYMBOLS_SOURCE_.loc[self._SYMBOL_INDEX_, 'isin']

		query_string = 'SELECT  * ' + \
		               'FROM    drl_1d ' + \
		               'WHERE   isin = %s AND trans_date = trans_date ' + \
		               'ORDER BY trans_date'
		values = [isin]
		self._SYMBOL_DATASET_ = self._SOURCE_CONNECTION_.select(query_string, values)
		self._DATASET_SIZE_ = len(self._SYMBOL_DATASET_)
		self._standardization(isin)

		return self._SYMBOL_DATASET_

	def dataset_info(self):
		if self._SYMBOL_INDEX_ < 0:
			return [self._SYMBOL_INDEX_, '', '']

		return [
			self._SYMBOL_INDEX_,
			self._SYMBOLS_SOURCE_.loc[self._SYMBOL_INDEX_, 'isin'],
			self._SYMBOLS_SOURCE_.loc[self._SYMBOL_INDEX_, 'symb_name'],
			self.scale
		]

	def source_size(self):
		return self._SYMBOLS_SIZE_

	def get_dateset(self):
		col_num = len(self._SYMBOL_DATASET_.columns)
		return self._SYMBOLS_SOURCE_.loc[self._SYMBOL_INDEX_, 'isin'], \
		       self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:col_num]], \
		       col_num-4, \
		       7


class Controller:
	def __init__(self, dataset):
		self._DATASET_ = dataset
		self._SIZE_OBSERVATIONS_ = len(self._DATASET_)
		self._DATASET_DATA_STATUS = pd.DataFrame({'applied': [0] * self._SIZE_OBSERVATIONS_})
		self._CURRENT_OBSERVATION_INDEX_ = -1

	def size(self):
		return self._SIZE_OBSERVATIONS_

	def id(self):
		return self._CURRENT_OBSERVATION_INDEX_

	def next(self):
		rate_applied = self._DATASET_DATA_STATUS['applied'].sum() / self._SIZE_OBSERVATIONS_

		self._CURRENT_OBSERVATION_INDEX_ += 1

		if (self._CURRENT_OBSERVATION_INDEX_ == self._SIZE_OBSERVATIONS_):
			self._DATASET_DATA_STATUS = pd.DataFrame({'applied': [0] * self._SIZE_OBSERVATIONS_})
			self._CURRENT_OBSERVATION_INDEX_ = -1
			return 1, 0.

		self._DATASET_DATA_STATUS.loc[self._CURRENT_OBSERVATION_INDEX_, 'applied'] = 1
		observation = self._DATASET_.loc[self._CURRENT_OBSERVATION_INDEX_,]
		return np.array(observation.values), rate_applied


