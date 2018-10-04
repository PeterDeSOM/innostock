import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import sys
import os

from skimage.color import rgb2gray
from skimage import io as skiio
from skimage.transform import resize
from matplotlib.finance import candlestick_ohlc
from databases import maria

ds_width = 256
ds_height = 256
output_dim = 7

_PLOT_IMAGE_DIR_ = 'plots/%s/'

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


matplotlib.rcParams.update({'font.size': 9})

def parse_dataset(df_source, window_size):
	df_target = df_source.reset_index(drop=True)

	for i in range(len(df_target) - window_size):
		row_index = i + window_size - 1

		isin = df_target.loc[row_index, 'isin']
		trans_date = df_target.loc[row_index, 'trans_date']
		target_value = df_target.loc[row_index, 'target_value']

		dir_path = _PLOT_IMAGE_DIR_ % target_value
		image_name = dir_path + '%s_%s_%s.png' % (target_value, trans_date.replace('-', ''), isin)

		if not os.path.exists(dir_path):
			os.makedirs(dir_path)

		convert_to_image(df_target.iloc[i:(i + window_size), 2:-1], image_name)

		printProgress(i, len(df_target) - window_size,
		              '##### CONVERTING PLOT TO IMAGE-ARRAY:',
		              '%s of %s' % (i, len(df_target) - window_size), 2, 40)


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


def convert_to_image(df, image_name):
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
	#plt.savefig(image_name)
	plt.close(fig)

	fig.canvas.draw()
	imarr = np.array(fig.canvas.renderer._renderer)
	plt.close(fig)

	img = np.uint8(resize(rgb2gray(imarr), (ds_height, ds_width), mode='constant') * 255)
	skiio.imsave(image_name, img)

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

	for i in range(source_size):
		if len(datasource.next()) == 0: break

		df_dataset, dataset_name, dataset_size, input_dim, _ = datasource.get_dateset()
		dataset_info = datasource.dataset_info()

		seperator()
		print('##### [ISIN: %s, SYMBOL NAME: %s, SCALE: %s] started. (%s of %s)' % (
		dataset_info[1], dataset_info[2], dataset_info[3], i + 1, source_size))

		parse_dataset(df_dataset, input_length)


	print('##### All the trainings are finished. --------------------------------------------------------------')


if __name__ == "__main__":
	main()