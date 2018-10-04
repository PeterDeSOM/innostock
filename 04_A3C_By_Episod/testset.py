import pandas as pd
import numpy as np

from datetime import datetime
from databases import maria

class Controller:
	def __init__(self, dataset):
		self._DATASET_ = dataset
		self._CONNECTION_ = maria()

	def convert_to_predictable(self):
		isin = self._DATASET_.get_value(0, 'isin')

		query_string = 'SELECT COUNT(*) count_ FROM drl_scalers WHERE isin = %s'
		df = self._CONNECTION_.select(query_string, [isin])
		if df.get_value(0, 'count_') == 0:
			return pd.DataFrame({}), -1, -1

		query_string = 'SELECT MAX(applied_date) scaled_date FROM drl_scalers WHERE isin = %s'
		df = self._CONNECTION_.select(query_string, [isin])
		scaled_date = df.get_value(0, 'scaled_date')

		query_string = 'SELECT * FROM drl_scalers WHERE isin = %s AND applied_date = %s'
		df = self._CONNECTION_.select(query_string, [isin, scaled_date])

		df = self._remove_null_cols(df)
		self._standardization(df)

		return self._DATASET_, df.get_value(0, 'applied_input_size'), 7

	def _remove_null_cols(self, df_scaler):
		drop_cols = df_scaler.columns[df_scaler.isnull().any()].tolist()
		df_scaler = df_scaler.drop(drop_cols, axis=1)
		self._DATASET_ = self._DATASET_.drop(drop_cols, axis=1)
		return df_scaler

	def _standardization(self, df_scaler):
		if len(df_scaler.columns)-5 != df_scaler.get_value(0, 'applied_input_size'):
			print('##### Can''t not process of the prediction with unmatched input(column) size for %s(%s).' % (
				self._DATASET_.get_value(0, 'isin'),
				self._DATASET_.get_value(0, 'trans_date')
			))
			return -1
		
		means = df_scaler.iloc[0, 3:-2].astype('float')
		stds_ = df_scaler.iloc[1, 3:-2].astype('float')

		self._DATASET_[self._DATASET_.columns[2:-3]] = (self._DATASET_[self._DATASET_.columns[2:-3]].astype('float') - means) / stds_

