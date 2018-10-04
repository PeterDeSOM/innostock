import pandas as pd
import numpy as np

from databases import maria

from matplotlib.finance import candlestick_ohlc

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib.finance as matfin

# mpl_finance


def graphData(stock,MA1,MA2):
	fig = plt.figure(facecolor='#07000d')

	ax1 = plt.subplot2grid((6, 4), (1, 0), rowspan=4, colspan=4, axisbg='#07000d')
	candlestick_ohlc(ax1, stock.loc[:, ['trans_date', 'p_open', 'p_high', 'p_low', 'p_current', 'volume']], width=.6, colorup='#53c156', colordown='#ff1717')



mariaDB = maria()

query_string = 'SELECT * FROM drl_1d LIMIT 40'
df = mariaDB.select(query_string)

df['ma2'] = df['p_current'].rolling(2).mean() + 2500
df['ma3'] = df['p_current'].rolling(3).mean() + 2500
df['ma5'] = df['p_current'].rolling(5).mean() + 2500
df['ma10'] = df['p_current'].rolling(10).mean() + 2500

df = df.iloc[10:, :]
df = df.reset_index(drop=True)
df['ma2'] = df['ma2'].astype('int')
df['ma3'] = df['ma3'].astype('int')
df['ma5'] = df['ma5'].astype('int')
df['ma10'] = df['ma10'].astype('int')

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

day_list = range(len(df))

ax.xaxis.set_major_locator(ticker.FixedLocator(day_list))
ax.xaxis.set_major_formatter(ticker.FixedFormatter(df['trans_date']))

matfin.candlestick2_ohlc(ax, df['p_open'], df['p_high'], df['p_low'], df['p_current'], width=0.5, colorup='r', colordown='b')
df.ma10.plot(ax=ax)
df.ma5.plot(ax=ax)
df.ma3.plot(ax=ax)
df.ma2.plot(ax=ax)

# plt.grid()
plt.show()