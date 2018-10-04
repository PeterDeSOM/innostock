import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime
from databases import maria

class Source(object):
	def __init__(self):
		self._SOURCE_CONNECTION_ = maria()

		query_string = 'SELECT  A.isin isin, 0 applied ' + \
		               'FROM    krx_symbols A INNER JOIN drl_1d B ON (A.isin = B.isin AND B.trans_date = B.trans_date) '+ \
		               'GROUP BY A.isin ORDER BY symb_name'
		self._SYMBOLS_SOURCE_ = self._SOURCE_CONNECTION_.select(query_string)
		self._SYMBOLS_SIZE_ = len(self._SYMBOLS_SOURCE_)
		self._SYMBOL_INDEX_ = -1

	def _standardization(self, isin):
		drop_colnames = []
		col_len = len(self._SYMBOL_DATASET_.columns)

		scaler = MinMaxScaler(feature_range=(0, 1))
		# scaler = StandardScaler()

		self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[col_len-1:col_len]] = \
			scaler.fit_transform(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[col_len-1:col_len]]).round(2)

		for i in range(2, col_len-1):
			self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[i:i + 1]] = \
				scaler.fit_transform(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[i:i + 1]].astype('float'))

			sum_ = np.sum(np.abs(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[i]]))
			if sum_ < 1e-9:
				drop_colnames.append(self._SYMBOL_DATASET_.columns[i])
				continue

		if len(drop_colnames) > 0:
			self._SYMBOL_DATASET_ = self._SYMBOL_DATASET_.drop(drop_colnames, axis=1)

		"""
		means, std_s, drop_colnames = [], [], []

		for i in range(2, len(self._SYMBOL_DATASET_.columns) - 1):
			self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[i:i + 1]] = \
				self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[i:i + 1]].astype('float')

			sum_ = np.sum(np.abs(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[i]]))
			if sum_ < 1e-9:
				drop_colnames.append(self._SYMBOL_DATASET_.columns[i])
				continue

			std_ = np.round(np.std(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[i]]), 15)
			mean = np.round(np.mean(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[i]]), 15)

			means.append(mean)
			std_s.append(std_)

			if std_ > 0:
				self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[i:i + 1]] = \
					(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[i:i + 1]] - mean) / std_

		if len(drop_colnames) > 0:
			self._SYMBOL_DATASET_ = self._SYMBOL_DATASET_.drop(drop_colnames, axis=1)

		col_names = list(self._SYMBOL_DATASET_.columns[2:len(self._SYMBOL_DATASET_.columns)-1])
		today_date = datetime.today().strftime('%Y-%m-%d')

		query_string = 'SELECT COUNT(*) count_ FROM drl_1d_mean_std WHERE isin = %s AND compiled_date = %s AND compiled_type = %s'
		values = [isin, today_date, 'M']
		df_result = self._SOURCE_CONNECTION_.select(query_string, values)

		if df_result.get_value(0, 'count_') > 0:
			col_vals = list(zip(col_names, means))
			query_string = 'UPDATE drl_1d_mean_std SET '
			for k, v in col_vals: query_string += '%s = %s' % (k, v) + ', '
			query_string = query_string[:-2] + ' '
			query_string += 'WHERE isin = %s AND compiled_date = %s AND compiled_type = %s'

			values = [isin, today_date, 'M']
			self._SOURCE_CONNECTION_.execute(query_string, values)

			col_vals = list(zip(col_names, std_s))
			query_string = 'UPDATE drl_1d_mean_std SET '
			for k, v in col_vals: query_string += '%s = %s' % (k, v) + ', '
			query_string = query_string[:-2] + ' '
			query_string += 'WHERE isin = %s AND compiled_date = %s AND compiled_type = %s'
			values = [isin, today_date, 'S']
			self._SOURCE_CONNECTION_.execute(query_string, values)

		else:
			query_string =  'INSERT INTO drl_1d_mean_std (isin, compiled_date, compiled_type, applied_inputs, ' + \
			                ', '.join(["%s"] * len(col_names)) % tuple(col_names) + ') VALUES (%s, %s, %s, %s, '

			values = [isin, today_date, 'M', len(col_names)]
			self._SOURCE_CONNECTION_.execute(query_string + ', '.join(["%s"] * len(means)) % tuple(means) + ')', values)

			values = [isin, today_date, 'S', len(col_names)]
			self._SOURCE_CONNECTION_.execute(query_string + ', '.join(["%s"] * len(std_s)) % tuple(std_s) + ')', values)
		"""


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

	def source_size(self):
		return self._SYMBOLS_SIZE_

	def get_dateset(self):
		col_num = len(self._SYMBOL_DATASET_.columns)
		return self._SYMBOLS_SOURCE_.loc[self._SYMBOL_INDEX_, 'isin'], \
		       self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:col_num]], \
		       col_num-3, \
		       1
	
	
class Controller:
	def __init__(self, dataset):
		self._DATASET_ = dataset
		self._SIZE_OBSERVATIONS_ = len(self._DATASET_)
		self._DATASET_DATA_STATUS = pd.DataFrame({'applied': [0] * self._SIZE_OBSERVATIONS_})
		self._CURRENT_OBSERVATION_INDEX_ = -1

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


