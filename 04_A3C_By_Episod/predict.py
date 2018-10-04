import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from isin_episod.testset import Controller
from keras.layers import Dense, Input
from keras.models import Model

from databases import maria


class Tomorrow:
	def __init__(self, dataset):
		self.controller = Controller(dataset)
		self.dataset = dataset

	def _parse_current_observation(self):
		return self.observation[self.observation.columns[2:-3]].values

	def predict(self):
		self.observation, self.state_size, self.action_size = self.controller.convert_to_predictable()

		if len(self.observation) == 0:
			print('##### ERROR : There is no scalable data for ''%s(%s)''.' % (
				self.dataset.get_value(0, 'isin'),
				self.dataset.get_value(0, 'trans_date')
			))
			return []

		self.state = self._parse_current_observation()
		self.actor, self.critic = self.build_model()
		self.load_model()
		return self.run()

	def build_model(self):
		input_state = Input(shape=(self.state_size,))
		layer1 = Dense(self.state_size * 2, activation='relu')(input_state)
		layer2 = Dense(self.state_size * 2, activation='relu')(layer1)
		layer3 = Dense(self.state_size * 2, activation='relu')(layer2)
		layer4 = Dense(self.state_size * 2, activation='relu')(layer3)
		layer5 = Dense(self.state_size * 2, activation='relu')(layer4)
		layer6 = Dense(self.state_size, activation='relu')(layer5)

		policy = Dense(self.action_size, activation='softmax')(layer6)
		value = Dense(1, activation='linear')(layer6)

		actor = Model(inputs=input_state, outputs=policy)
		critic = Model(inputs=input_state, outputs=value)

		actor._make_predict_function()
		critic._make_predict_function()

		return actor, critic

	def get_action(self):
		return self.actor.predict(self.state)

	def load_model(self):
		self.actor.load_weights('graduated/%s/_MODEL_W_ACTOR_.h5' % self.dataset.get_value(0, 'isin'))

	def onehot(self, x):
		x = x.reshape(-1)
		return np.eye(len(x))[np.argmax(x)]

	def run(self):
		epoch = 1000
		actions = []

		for i in range(epoch):
			action_stochastic_policy = self.get_action()
			action = self.onehot(action_stochastic_policy).reshape((1, self.action_size))
			actions.append(action)

		result = list(np.mean(np.vstack(actions), axis=0) * 100)
		return result



def get_strength(target):
	value = np.max(target)

	if value < 80.: strength_name = 'Very weak'
	elif value >= 80. and value < 85.: strength_name = 'Weak'
	elif value >= 85. and value < 90.: strength_name = 'Good'
	elif value >= 90. and value < 95.: strength_name = 'Strong'
	else: strength_name = 'Very strong'

	return strength_name

def get_target_date(date):
	stop = False
	increase = 1
	target_date = ''

	while not stop:
		target_date = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=increase)

		if target_date.weekday() not in (5, 6):
			stop = True
			target_date = target_date.strftime('%Y-%m-%d')

		increase += 1

	return target_date

if __name__ == "__main__":
	connection = maria()

	query_string = 'SELECT isin, symb_code, symb_name FROM krx_symbols WHERE symb_status = %s'
	df_symbol_info = connection.select(query_string, ['A'])

	# Build flat transaction data to be preditable data (state)
	# connection.callproc('TESTING_DATA_UPDATE')

	# predict_mode -----------------------------------------------------
	# 0: All untested data (default)
	# 1: Only tomorrow data (last date of transaction will be applied in prediction)
	predict_type = 0
	# allow_repredict --------------------------------------------------
	# True  : Both of prected and unpredicted data will be applied in prediction (default)
	# Flase : Only the unprected data will be applied in prediction
	allow_repredict = True

	query_string = 'SELECT trans_date, 0 pred_status FROM drl_1d_testing '
	if not allow_repredict: query_string += 'WHERE predict_value IS NULL '
	query_string += 'GROUP BY trans_date'
	df = connection.select(query_string)

	if len(df) == 0:
		print('##### There is totally no predictable data. #####')
		exit()

	if predict_type == 1: df = df[df['trans_date']==df.get_value(len(df)-1, 'trans_date')]

	report_data = []

	for trans_date in df['trans_date']:
		query_string = 'SELECT isin FROM drl_1d_testing WHERE trans_date = %s '
		if not allow_repredict: query_string += 'AND predict_value IS NULL '
		df_isin = connection.select(query_string, [trans_date])

		target_isins = []

		# If the model does not exist, then skip prediction
		for isin in df_isin['isin']:
			_MODEL_WEIGHT_DIR_ = 'graduated/%s/_MODEL_W_ACTOR_.h5' % isin
			if os.path.isfile(_MODEL_WEIGHT_DIR_): target_isins.append(isin)

		if len(target_isins) == 0: continue

		df_isin = pd.DataFrame(target_isins, columns=['isin'])

		for isin in df_isin['isin']:
			query_string = 'SELECT * FROM drl_1d_testing WHERE isin = %s AND trans_date = %s'
			df_predictable = connection.select(query_string, [isin, trans_date])

			tomorrow = Tomorrow(df_predictable)
			predict_result = tomorrow.predict()

			if len(predict_result) == 0: continue

			action_value = int(np.argmax(predict_result))

			symbol_info = df_symbol_info.loc[df_symbol_info['isin']==isin, ['symb_name', 'symb_code']].values.tolist()[0]
			strength_name = get_strength(predict_result)
			target_date = get_target_date(trans_date)
			predict_result = sum([symbol_info, [trans_date, '%s or next working day' % target_date], predict_result, [strength_name]], [])
			report_data.append(predict_result)

			query_string = 'UPDATE drl_1d_testing SET predict_value = %s WHERE isin = %s AND trans_date = %s'
			df_predictable = connection.execute(query_string, [action_value, isin, trans_date])

	df_report = pd.DataFrame(data=report_data, columns=[
		'Symbol Name'    , 'Symbol Code' , 'Request Date', 'Target Date',
		'Less than -7.0%', '-7.0% ~ -3.0', '-3.0% ~ -1.0', '-1.0% ~ 1.0', '1.0% ~ 3.0', '3.0% ~ 7.0', '7.0% and over',
		'Predicted Strength'
	])

	_REPORT_DIR_ = 'report'
	df_report.to_csv('%s/%s.csv' % (_REPORT_DIR_, datetime.today().strftime('%Y%m%d%H%M%S')), index=False)

	print('##### All done. #####')