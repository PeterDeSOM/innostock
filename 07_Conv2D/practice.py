import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import keras
import sys
import _pickle as cPickle

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from datetime import datetime
from matplotlib.finance import candlestick_ohlc

from databases import maria

from pymongo import MongoClient
from urllib import parse
from bson.binary import Binary

mongoconn = MongoClient('mongodb://%s:%s@192.168.233.134:27017' % (parse.quote_plus('superclient'), parse.quote_plus('Infor3233!@#')))
db = mongoconn.innostock

ds_width = 256
ds_height = 256
output_dim = 7


class Source(object):
	def __init__(self):
		self._SOURCE_CONNECTION_ = maria()

		query_string = 'SELECT  A.isin isin, A.symb_name symb_name, 0 applied ' + \
		               'FROM    krx_symbols A INNER JOIN drl_1d B ON (A.isin = B.isin) ' + \
		               'GROUP BY A.isin, A.symb_name ORDER BY A.symb_name'
		self._SYMBOLS_SOURCE_ = self._SOURCE_CONNECTION_.select(query_string)
		self._SYMBOLS_SIZE_ = len(self._SYMBOLS_SOURCE_)
		self._SYMBOL_INDEX_ = -1

	def _standardization(self, isin):
		data_cols = len(self._SYMBOL_DATASET_.columns)

		"""
		stds_ = np.std(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols - 1]].astype('float'), axis=0)

		if len(stds_[stds_ == 0]) > 0:
			self._SYMBOL_DATASET_ = self._SYMBOL_DATASET_.drop(stds_[stds_ == 0].index, axis=1)
			data_cols = len(self._SYMBOL_DATASET_.columns)

		means = np.round(np.mean(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols - 1]].astype('float'), axis=0), 19)
		stds_ = np.round(np.std(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols - 1]].astype('float'), axis=0), 19)

		self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols - 1]] = \
			(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols - 1]].astype('float') - means) / stds_
		self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[data_cols - 1:data_cols]] = \
			self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[data_cols - 1:data_cols]].astype('int')
		"""

		self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:9]] = \
			self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:9]].astype('int')
		self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[9:data_cols - 1]] = \
			self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[9:data_cols - 1]].astype('float')
		self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[data_cols - 1:data_cols]] = \
			self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[data_cols - 1:data_cols]].astype('int')

		self.scale = data_cols - 3

	def next(self):
		self._SYMBOL_INDEX_ += 1

		if self._SYMBOL_INDEX_ == self._SYMBOLS_SIZE_:
			self._SYMBOL_INDEX_ = -1
			return pd.DataFrame()

		isin = self._SYMBOLS_SOURCE_.loc[self._SYMBOL_INDEX_, 'isin']

		query_string = 'SELECT * FROM drl_1d WHERE isin = %s ORDER BY trans_date'
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
		return self._SYMBOL_DATASET_, \
		       self._SYMBOLS_SOURCE_.loc[self._SYMBOL_INDEX_, 'isin'], \
		       len(self._SYMBOL_DATASET_), col_num - 3, 7


class LossHistory(keras.callbacks.Callback):
	def init(self):
		self.losses = []

	def on_epoch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))


matplotlib.rcParams.update({'font.size': 9})

def parse_dataset(df_source, window_size):
	col_len = len(df_source.columns)
	# df_target = df_source[df_source.columns[2:col_len]]
	df_target = df_source.reset_index(drop=True)

	dataset_X = []
	dataset_Y = []

	for i in range(len(df_target) - window_size):
		row_index = i + window_size - 1

		plot_image = convert_to_image(df_target.iloc[i:(i + window_size), 2:-1])
		isin = df_target.loc[row_index, 'isin']
		trans_date = df_target.loc[row_index, 'trans_date']
		target_value = df_target.loc[row_index, 'target_value']

		dataset_X.append([plot_image])
		dataset_Y.append([target_value])

		# print(cPickle.loads(Binary(cPickle.dumps(plot_image, protocol=2)))) - Use of binary data converted to mongodb... cPickle.loads(MongoDB Field data)
		# exit()

		"""
		db.ConvertedPlotImages.insert(
			{
				'isin': isin,
				'trans_date': trans_date,
				'plot_image': Binary(cPickle.dumps(plot_image, protocol=2)),
				'target_value': int(target_value)
			}
		)
		"""

		"""
		collection.insert({'cpickle': Binary(cPickle.dumps(np.random.rand(50,3), protocol=2))})
		-n 100 [cPickle.loads(x['cpickle']) for x in collection.find()]
		"""

		printProgress(i, len(df_target) - window_size,
		              '##### CONVERTING PLOT TO IMAGE-ARRAY:',
		              '%s of %s' % (i, len(df_target) - window_size), 2, 40)

	return np.vstack(dataset_X), np.vstack(dataset_Y)


def rsiFunc(prices, n=14):
	deltas = np.diff(prices)
	seed = deltas[:n + 1]
	up = seed[seed >= 0].sum() / n
	down = -seed[seed < 0].sum() / n
	rs = (up / down) if down != 0 else 0
	rsi = np.zeros_like(prices)
	rsi[:n] = 100. - 100. / (1. + rs)

	for i in range(n, len(prices)):
		delta = deltas[i - 1]  # cause the diff is 1 shorter

		if delta > 0:
			upval = delta
			downval = 0.
		else:
			upval = 0.
			downval = -delta

		up = (up * (n - 1) + upval) / n
		down = (down * (n - 1) + downval) / n

		rs = (up / down) if down != 0 else 0
		rsi[i] = 100. - 100. / (1. + rs)

	return rsi


def computeMACD(x, slow=26, fast=12):
	"""
	compute the MACD (Moving Average Convergence/Divergence) using a fast and slow exponential moving avg'
	return value is emaslow, emafast, macd which are len(x) arrays
	"""
	emaslow = ExpMovingAverage(x, slow)
	emafast = ExpMovingAverage(x, fast)
	return emaslow, emafast, emafast - emaslow


def ExpMovingAverage(values, window):
	weights = np.exp(np.linspace(-1., 0., window))
	weights /= weights.sum()
	a = np.convolve(values, weights, mode='full')[:len(values)]
	a[:window] = a[window]
	return a


def convert_to_image(df):
	facebolor = '#07000d'
	colorup = '#53c156'
	colordown = '#ff1717'

	df['ma02'] = df['p_current'].rolling(2).mean()
	df['ma03'] = df['p_current'].rolling(3).mean()
	df['ma05'] = df['p_current'].rolling(5).mean()
	df['ma10'] = df['p_current'].rolling(10).mean()
	df['rsi'] = rsiFunc(df['p_current'])
	df['emaslow'], df['emafast'], df['macd'] = computeMACD(df['p_current'])

	vol_colors = []
	old_volume = 0

	for volume in df['volume'].values:
		vol_colors.append(colorup if volume >= old_volume else colordown)
		old_volume = volume

	df['vol_colors'] = vol_colors

	df = df.iloc[10:, :]
	df = df.reset_index(drop=True)
	df['ma02'] = df['ma02'].astype('float')
	df['ma03'] = df['ma03'].astype('float')
	df['ma05'] = df['ma05'].astype('float')
	df['ma10'] = df['ma10'].astype('float')
	df['rsi'] = df['rsi'].astype('int')

	df['ma02'] = round(df['ma02'] - df['ma02'].min() + int(df['ma02'].min() * .05))
	df['ma03'] = round((df['ma03'] - df['ma03'].min() + int(df['ma03'].min() * .05)) * -1)
	df['ma05'] = round(df['ma05'] - df['ma05'].min() + int(df['ma05'].min() * .05))
	df['ma10'] = round((df['ma10'] - df['ma10'].min() + int(df['ma10'].min() * .05)) * -1)

	moving_avg_max = max([max(df['ma02']), max(df['ma05']), max(abs(df['ma03'])), max(abs(df['ma10']))])
	moving_avg_ylim = [moving_avg_max * -1, moving_avg_max]

	date = []
	data = []
	pcls = []
	e_dt = []

	e_bottom = []
	e_height = []

	dods = []
	dod_colors = []
	vols = []

	i = 0

	for open, high, offer, bid, low, current, dod, volume, eopen, ehigh, elow, eclose in \
			df.loc[:, ['p_open', 'p_high', 'p_offer', 'p_bid', 'p_low', 'p_current', 'dod', 'volume',
			           'e_p_open', 'e_p_high', 'e_p_low', 'e_p_close']].values:
		date.append(i)
		data.append((i, open, high, low, current))
		pcls.append(current)

		vols.append(volume)
		dods.append(dod)
		dod_colors.append(colorup if dod >= 0 else colordown)

		e_dt.append((i, eopen, ehigh, elow, eclose))
		e_bottom.append(eopen if eopen <= eclose else eclose)
		e_height.append(abs(eopen - eclose))

		i += 1

	fig = plt.figure(facecolor=facebolor)

	""" ### LEFT """
	ax0 = plt.subplot2grid((8, 4), (0, 0), rowspan=1, colspan=2, facecolor=facebolor)
	ax0.plot(date, df['rsi'], '#c1f9f7', linewidth=1)
	ax0.xaxis.set_major_locator(mticker.MaxNLocator(10))
	ax0.axhline(70, color='#8f2020', linewidth=0.5)
	ax0.axhline(30, color='#386d13', linewidth=0.5)
	ax0.set_yticks([30, 70])
	ax0.set_ylim([0, 100])
	ax0.fill_between(date, df['rsi'], 70, where=(df['rsi'] >= 70), facecolor='#8f2020', edgecolor='#8f2020')
	ax0.fill_between(date, df['rsi'], 30, where=(df['rsi'] <= 30), facecolor='#386d13', edgecolor='#386d13')
	ax0.get_yaxis().set_visible(False)
	ax0.get_xaxis().set_visible(False)

	ax1 = plt.subplot2grid((8, 4), (1, 0), sharex=ax0, rowspan=2, colspan=2, facecolor=facebolor)
	ax1.set_frame_on(True)
	ax1.get_yaxis().set_visible(False)
	ax1.get_xaxis().set_visible(False)
	candlestick_ohlc(ax1, e_dt, width=.4, colorup=colorup, colordown=colordown)

	ax2 = plt.subplot2grid((8, 4), (3, 0), sharex=ax0, rowspan=3, colspan=2, facecolor=facebolor)
	ax2.get_yaxis().set_visible(False)
	ax2.get_xaxis().set_visible(False)
	candlestick_ohlc(ax2, data, width=.4, colorup=colorup, colordown=colordown)

	ax3 = plt.subplot2grid((8, 4), (6, 0), sharex=ax0, rowspan=2, colspan=2, facecolor=facebolor)
	ax3.bar(date, df['ma02'], width=.4, align='center', color=colorup)
	ax3.bar(date, df['ma03'], width=.4, align='center', color=colordown)
	ax3.set_ylim(moving_avg_ylim)
	ax3.get_yaxis().set_visible(False)
	ax3.get_xaxis().set_visible(False)

	""" ### RIGHT """
	nema = 9
	ax5 = plt.subplot2grid((8, 4), (0, 2), sharex=ax1, rowspan=1, colspan=2, facecolor=facebolor)
	emaslow, emafast, macd = df['emaslow'], df['emafast'], df['macd']
	ema9 = ExpMovingAverage(macd, nema)
	ax5.plot(date, macd, color='#4ee6fd', lw=1)
	ax5.plot(date, ema9, color='#e1edf9', lw=1)
	ax5.fill_between(date, macd - ema9, 0, facecolor='#00ffe8')
	ax5.get_yaxis().set_visible(False)
	ax5.get_xaxis().set_visible(False)

	ax8 = plt.subplot2grid((8, 4), (1, 2), sharex=ax0, rowspan=2, colspan=2, facecolor=facebolor)
	ax8.bar(date, dods, width=.4, align='center', color=dod_colors)
	ax8.get_yaxis().set_visible(False)
	ax8.get_xaxis().set_visible(False)

	ax4 = plt.subplot2grid((8, 4), (3, 2), sharex=ax0, rowspan=3, colspan=2, facecolor=facebolor)
	ax4.bar(date, vols, width=.4, align='center', color=df['vol_colors'])
	ax4.get_yaxis().set_visible(False)
	ax4.get_xaxis().set_visible(False)

	ax9 = plt.subplot2grid((8, 4), (6, 2), sharex=ax0, rowspan=2, colspan=2, facecolor=facebolor)
	ax9.bar(date, df['ma05'], width=.4, align='center', color=colorup)
	ax9.bar(date, df['ma10'], width=.4, align='center', color=colordown)
	ax9.set_ylim(moving_avg_ylim)
	ax9.get_yaxis().set_visible(False)
	ax9.get_xaxis().set_visible(False)

	plt.subplots_adjust(left=.0, bottom=.0, right=1., top=1., wspace=.0, hspace=.0)
	DPI = fig.get_dpi()
	fig.set_size_inches(ds_width / float(DPI), ds_height / float(DPI))

	fig.canvas.draw()
	imarr = np.array(fig.canvas.renderer._renderer)
	# plt.show()
	plt.close(fig)

	from skimage.color import rgb2gray
	from skimage import io as skiio
	from skimage.transform import resize

	img = np.uint8(resize(rgb2gray(imarr), (256, 256), mode='constant') * 255) / 255.
	# skiio.imshow(img)
	# plt.show()
	# print(img)

	return img


def seperator():
	print('-' * 100)


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
	formatStr = "{0:." + str(decimals) + "f}"
	percent = formatStr.format(100 * (iteration / float(total)))
	filledLength = int(round(barLength * iteration / float(total)))
	bar = '#' * filledLength + '-' * (barLength - filledLength)
	sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()


def main():
	datasource = Source()
	source_size = datasource.source_size()

	input_length = 30  # Window Size

	model = Sequential()
	model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='same', activation='relu', input_shape=(ds_height, ds_width, 1)))
	model.add(Conv2D(32, (4, 4), strides=(2, 2), padding='same', activation='relu'))
	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dense(output_dim, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	for i in range(source_size):
		if len(datasource.next()) == 0: break

		df_dataset, dataset_name, dataset_size, input_dim, _ = datasource.get_dateset()
		dataset_info = datasource.dataset_info()

		seperator()
		print('##### [ISIN: %s, SYMBOL NAME: %s, SCALE: %s] started. (%s of %s)' % (
		dataset_info[1], dataset_info[2], dataset_info[3], i + 1, source_size))

		dataset_X, dataset_Y = parse_dataset(df_dataset, input_length)
		dataset_X = np.reshape(dataset_X, (np.shape(dataset_X)[0], ds_height, ds_width, 1))

		# One-Hot encoding for the Y(Prediction) value
		# print(dataset_Y[np.r_[:5, :]])
		dataset_Y = np_utils.to_categorical(dataset_Y, num_classes=output_dim)

		history = LossHistory()
		history.init()

		num_epochs = 100

		acc_count = 0
		for epoch_idx in range(num_epochs):
			print('epochs : ' + str(epoch_idx))
			fit_history = model.fit(dataset_X, dataset_Y, epochs=1, batch_size=1, verbose=2, shuffle=False, callbacks=[history])
			# model.reset_states()

			if fit_history.history['acc'][0] > .9:
				acc_count += 1

			if acc_count > 4:
				break

	print('##### All the trainings are finished. --------------------------------------------------------------')


if __name__ == "__main__":
	main()