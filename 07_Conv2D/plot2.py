import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.finance as matfin


from matplotlib.finance import candlestick_ohlc
from databases import maria

def rsiFunc(prices, n=14):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

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
    a =  np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a

matplotlib.rcParams.update({'font.size': 9})

mariaDB = maria()

query_string = 'SELECT * FROM drl_1d LIMIT 1280, 30'
df = mariaDB.select(query_string)

df.loc[:, ['p_open', 'p_high', 'p_offer', 'p_bid', 'p_low', 'p_current', 'volume']] = \
	df.loc[:, ['p_open', 'p_high', 'p_offer', 'p_bid', 'p_low', 'p_current', 'volume']].astype('int')
df.loc[:, ['dod', 'e_p_open', 'e_p_high', 'e_p_low', 'e_p_close']] = df.loc[:, ['dod', 'e_p_open', 'e_p_high', 'e_p_low', 'e_p_close']].astype('float')

df['ma02'] = df['p_current'].rolling(2).mean()
df['ma03'] = df['p_current'].rolling(3).mean()
df['ma05'] = df['p_current'].rolling(5).mean()
df['ma10'] = df['p_current'].rolling(10).mean()
df['rsi'] = rsiFunc(df['p_current'])
df['emaslow'], df['emafast'], df['macd'] = computeMACD(df['p_current'])

df = df.iloc[10:, :]
df = df.reset_index(drop=True)
df['ma02'] = df['ma02'].astype('float')
df['ma03'] = df['ma03'].astype('float')
df['ma05'] = df['ma05'].astype('float')
df['ma10'] = df['ma10'].astype('float')
df['rsi'] = df['rsi'].astype('int')

data = []
ohdt = []
date = []
pcls = []
e_dt = []

dods = []
dod_colors = []
vols = []
vol_colors = []


i = 0
old_volume = 0

for open, high, offer, bid, low, current, dod, volume, eopen, ehigh, elow, ecurrent in \
		df.loc[:, ['p_open'  , 'p_high'  , 'p_offer', 'p_bid', 'p_low', 'p_current', 'dod', 'volume',
		           'e_p_open', 'e_p_high', 'e_p_low', 'e_p_close']].values:
	data.append((i, open, high, low, current, volume))
	ohdt.append((i, open, offer, bid, current, volume))
	date.append(i)
	pcls.append(current)

	vols.append(volume)
	vol_colors.append('#333333' if volume >= old_volume else '#aaaaaa')

	dods.append(dod)
	dod_colors.append('#333333' if dod >= 0 else '#aaaaaa')

	e_dt.append((i, eopen, ehigh, elow, ecurrent, volume))

	old_volume = volume
	i += 1

rsiCol = '#c1f9f7'
posCol = '#386d13'
negCol = '#8f2020'

fig = plt.figure()

""" ### RIGHT """
ax0 = plt.subplot2grid((8,4), (0,0), rowspan=1, colspan=2)
ax0.plot(date, df['rsi'], '#333333', linewidth=1.5)
ax0.xaxis.set_major_locator(mticker.MaxNLocator(10))
# ax0.xaxis.set_major_formatter(mticker.FixedFormatter(df['trans_date']))
ax0.grid(True, color='#dddddd', linestyle='dotted')
# ax0.spines['bottom'].set_color("#5998ff")
# ax0.spines['top'].set_color("#5998ff")
# ax0.spines['left'].set_color("#5998ff")
# ax0.spines['right'].set_color("#5998ff")
ax0.axhline(70, color='#999999', linewidth=0.5)
ax0.axhline(30, color='#999999', linewidth=0.5)
ax0.set_yticks([30,70])
ax0.set_ylim([0, 100])
ax0.fill_between(date, df['rsi'], 70, where=(df['rsi']>=70), facecolor='#bbbbbb')
ax0.fill_between(date, df['rsi'], 30, where=(df['rsi']<=30), facecolor='#bbbbbb')
# plt.ylabel('RSI')
ax0.get_yaxis().set_visible(False)
ax0.get_xaxis().set_visible(False)

ax1 = plt.subplot2grid((8,4), (1, 0), sharex=ax0, rowspan=2, colspan=2)
ax1.grid(True, color='#dddddd', linestyle='dotted')
ax1.get_yaxis().set_visible(False)
ax1.get_xaxis().set_visible(False)
candlestick_ohlc(ax1, e_dt, width=.7, colorup='#000000', colordown='#aaaaaa')
plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
# plt.ylabel('KRX Exchange')

ax2 = plt.subplot2grid((8,4), (3, 0), sharex=ax0, rowspan=5, colspan=2)
ax2.grid(True, color='#dddddd', linestyle='dotted')
ax2.plot(date, list(df['ma02'].values + (df['ma02'].mean() * 0.05)), '#999999', label='MA2', linestyle='-.', linewidth=0.5)
ax2.plot(date, list(df['ma03'].values + (df['ma03'].mean() * 0.05)), '#999999', label='MA3', linewidth=0.5)
ax2.plot(date, list(df['ma05'].values + (df['ma05'].mean() * 0.05)), '#555555', label='MA5', linewidth=0.5)
ax2.plot(date, list(df['ma10'].values + (df['ma10'].mean() * 0.05)), '#555555', label='MA10', linestyle='-.', linewidth=0.5)
ax2.get_yaxis().set_visible(False)
ax2.get_xaxis().set_visible(False)
candlestick_ohlc(ax2, data, width=.7, colorup='#000000', colordown='#aaaaaa')
# matfin.candlestick2_ohlc(ax2, df['p_open'], df['p_high'], df['p_low'], df['p_current'], width=0.5, colorup='r', colordown='b')
plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
# plt.ylabel('Stock price and Volume')

# ax3 = ax2.twinx()
# ax3.set_ylim(ax2.get_ylim())
# ax3.scatter(date, df['p_offer'], marker="x")
# ax3.scatter(date, df['p_bid'], marker="^")

""" ### LEFT """
ax8 = plt.subplot2grid((8,4), (0,2), sharex=ax0, rowspan=3, colspan=2)
ax8.grid(True, color='#dddddd', linestyle='dotted')
# ax8.axhline(0, color='#aaaaaa', linewidth='0.5')
ax8.bar(date, dods, align='center', color=dod_colors)
ax8.get_yaxis().set_visible(False)
ax8.get_xaxis().set_visible(False)
plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))

ax4 = plt.subplot2grid((8,4), (3,2), sharex=ax0, rowspan=4, colspan=2)
ax4.grid(True, color='#eeeeee', linestyle='dotted')
# ax4.fill_between(date, 0, vols, facecolor='#00ffe8', alpha=.4)
ax4.bar(date, vols, align='center', color=vol_colors)
ax4.get_yaxis().set_visible(False)
ax4.get_xaxis().set_visible(False)
plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))

fillcolor = '#00ffe8'
nslow = 26
nfast = 12
nema = 9

ax5 = plt.subplot2grid((8,4), (7,2), sharex=ax1, rowspan=1, colspan=2)
ax5.grid(True, color='#dddddd', linestyle='dotted')
emaslow, emafast, macd = df['emaslow'], df['emafast'], df['macd']
ema9 = ExpMovingAverage(macd, nema)
ax5.plot(date, macd, color='#555555', lw=1.5)
ax5.plot(date, ema9, color='#999999', lw=1.5)
ax5.fill_between(date, macd-ema9, 0, facecolor='#bbbbbb')
# ax5.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, prune='upper'))
ax5.get_yaxis().set_visible(False)
ax5.get_xaxis().set_visible(False)
# plt.gca().yaxis.set_major_locator(mticker.MaxNLocator(prune='upper'))
# plt.ylabel('MACD')

plt.subplots_adjust(left=.0, bottom=.0, right=1., top=1., wspace=.0, hspace=.0)
DPI = fig.get_dpi()
fig.set_size_inches(360.0/float(DPI), 360.0/float(DPI))

# plt.gray()
fig.canvas.draw()
imarr = np.array(fig.canvas.renderer._renderer)
# print(np.shape(imarr))
# fig.savefig('images/example.jpg')
# plt.show()
plt.close(fig)

from skimage.transform import resize
from skimage.color import rgb2gray
from skimage import data, io

img = rgb2gray(imarr)
# img = np.uint8(resize(rgb2gray(imarr), (360, 360), mode='constant') * 255)
# img = np.reshape(rgb2gray(imarr), (360, 360, 3))

print(img)
# print(np.shape(imarr), np.shape(img))
io.imshow(img)
plt.show()

"""
# from PIL import Image

# basewidth = 500
# im = Image.open('images/example.jpg')
# im = im.crop((72, 40, 603, 714))
# wpercent = (basewidth/float(im.size[0]))
# hsize = int((float(im.size[1])*float(wpercent)))
# im = im.resize((basewidth,hsize), Image.ANTIALIAS)
# im.save('images/example.jpg')

from skimage.transform import resize

img = io.imread('images/example.jpg', as_grey=True)

print(img)


"""
