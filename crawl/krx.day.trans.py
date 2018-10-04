import pandas as pd
import numpy as np
import requests
import random
import time
import sys
import io

from datetime import datetime, timedelta
from databases import maria


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
	formatStr = "{0:." + str(decimals) + "f}"
	percent = formatStr.format(100 * (iteration / float(total)))
	filledLength = int(round(barLength * iteration / float(total)))
	bar = '#' * filledLength + '-' * (barLength - filledLength)
	sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
	if iteration == total:
		sys.stdout.write('\n')
	sys.stdout.flush()


def from_symbolstoday_all(transdate=datetime.today()):
	# STEP 01: Generate OTP
	gen_otp_url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
	gen_otp_data = {
		'name': 'fileDown',
		'filetype': 'xls',
		'url': 'MKD/04/0406/04060200/mkd04060200',
		'market_gubun': 'ALL',
		'sect_tp_cd': 'ALL',
		'isu_cdnm': '전체',
		'secugrp': 'ST',
		'stock_gubun': 'on',
		'schdate': '%s' % transdate.strftime('%Y%m%d'),
		'pagePath': '/contents/MKD/04/0406/04060200/MKD04060200.jsp',
	}  # Query String Parameters

	r = requests.post(gen_otp_url, gen_otp_data)
	code = r.content

	# STEP 02: download
	down_url = 'http://file.krx.co.kr/download.jspx'
	down_data = {
		'code': code,
	}

	r = requests.post(down_url, down_data)
	f = io.BytesIO(r.content)

	usecols = ['종목코드',
	           '종목명',
	           '현재가',
	           '대비',
	           '등락률(%)',
	           '매도호가',
	           '매수호가',
	           '거래량(주)',
	           '거래대금(원)',
	           '시가',
	           '고가',
	           '저가',
	           '액면가',
	           '통화구분',
	           '상장주식수(주)',
	           '상장시가총액(원)']
	df = pd.read_excel(f, converters={'종목코드': str,
	                                  '현재가': str,
	                                  '대비': str,
	                                  '등락률(%)': str,
	                                  '매도호가': str,
	                                  '매수호가': str,
	                                  '거래량(주)': str,
	                                  '거래대금(원)': str,
	                                  '시가': str,
	                                  '고가': str,
	                                  '저가': str,
	                                  '액면가': str,
	                                  '상장주식수(주)': str,
	                                  '상장시가총액(원)': str},
	                   usecols=usecols)
	df = df.fillna('')
	df.columns = ['symb_code',
	              'symb_name',
	              'p_current',
	              'p_diff',
	              'dod',
	              'p_offer',
	              'p_bid',
	              'volume',
	              'p_total',
	              'p_open',
	              'p_high',
	              'p_low',
	              'par_value',
	              'currency_code',
	              'listed_shares',
	              'market_cap']
	df['p_current'] = df['p_current'].str.replace(r'[^0-9-.]', '')
	df['p_diff'] = df['p_diff'].str.replace(r'[^0-9-.]', '')
	df['dod'] = df['dod'].str.replace(r'[^0-9-.]', '')
	df['p_offer'] = df['p_offer'].str.replace(r'[^0-9-.]', '')
	df['p_bid'] = df['p_bid'].str.replace(r'[^0-9-.]', '')
	df['volume'] = df['volume'].str.replace(r'[^0-9-.]', '')
	df['p_total'] = df['p_total'].str.replace(r'[^0-9-.]', '')
	df['p_open'] = df['p_open'].str.replace(r'[^0-9-.]', '')
	df['p_high'] = df['p_high'].str.replace(r'[^0-9-.]', '')
	df['p_low'] = df['p_low'].str.replace(r'[^0-9-.]', '')
	df['par_value'] = df['par_value'].str.replace(r'[^0-9-.]', '')
	df['listed_shares'] = df['listed_shares'].str.replace(r'[^0-9-.]', '')
	df['market_cap'] = df['market_cap'].str.replace(r'[^0-9-.]', '')
	df['currency_code'] = df['currency_code'].str.replace(r'[^a-zA-Z]', '')
	df['symb_type_code'] = df['symb_code'].apply(lambda x: x[len(x) - 1:len(x)])
	df = df.assign(**{'trans_date': [transdate.strftime('%Y-%m-%d')] * len(df)})

	df['dod'] = df['dod'].astype('float') * df['p_diff'].apply(lambda x: -1 if int(x) < 0 else 1)

	return df


# End def from_symbolstoday

mariadb = maria()

# Get current symbols information
querystring = 'SELECT isin, symb_code, market_type FROM krx_symbols WHERE symb_status <> %s'
values = ['I']
df_symbols = mariadb.select(querystring, values)

querystring = 'SELECT MAX(trans_date) trans_date FROM krx_trans_daily'
df_result = mariadb.select(querystring)
last_trans_date = df_result.get_value(0, 'trans_date')

prevday = datetime.today() - timedelta(days=1)
# daysbefore = (prevday - datetime.strptime('2017-10-23', '%Y-%m-%d')).days
daysbefore = (prevday - datetime.strptime(last_trans_date, '%Y-%m-%d')).days - 1

for i in range(daysbefore, -1, -1):
	transdate = prevday - timedelta(days=i)

	if transdate.weekday() in (5, 6):
		continue

	df_get_transtoday = from_symbolstoday_all(transdate)

	# If the data not corresponded is crawled, we have to see what it is...
	df_notexist = df_get_transtoday[~df_get_transtoday['symb_code'].isin(df_symbols['symb_code'])]
	if len(df_notexist) > 0:
		df_get_transtoday = df_get_transtoday[df_get_transtoday['symb_code'].isin(df_symbols['symb_code'])]
		print('### %d not corresponded transaction data with symbols are found. ###' % len(df_notexist))
		print(df_notexist.head())
		print('###\n')

	df_get_transtoday = pd.merge(df_get_transtoday, df_symbols, on='symb_code')

	# Get today's transaction data, if it exists
	querystring = 'SELECT isin FROM krx_trans_daily WHERE trans_date = %s'
	values = [prevday.strftime('%Y-%m-%d')]
	df_db_transdata = mariadb.select(querystring, values)

	df_get_transtoday = df_get_transtoday.drop(['symb_code', 'symb_name', 'symb_type_code', 'market_type'], axis=1)

	if len(df_db_transdata) == 0:
		mariadb.insert("krx_trans_daily", df_get_transtoday)
		print('%s New inserted...' % len(df_get_transtoday))
	else:
		proc_length = len(df_get_transtoday)
		print('%s data(s) will be processed.' % '{:,}'.format(proc_length))

		for idx, data in df_get_transtoday.iterrows():
			if len(df_db_transdata[df_db_transdata['isin'] == data['isin']]) == 0:
				print('%s th row inserted...' % idx)
				mariadb.insert("krx_trans_daily", df_get_transtoday[idx:idx + 1])
			else:
				where = "isin='%s' AND trans_date='%s'" % (data['isin'], data['trans_date'])
				mariadb.update_single('krx_trans_daily', where,
				                      p_offer=data['p_offer'],
				                      p_bid=data['p_bid'],
				                      p_open=data['p_open'],
				                      p_high=data['p_high'],
				                      p_low=data['p_low'],
				                      p_current=data['p_current'],
				                      p_diff=data['p_diff'],
				                      dod=data['dod'],
				                      volume=data['volume'],
				                      p_total=data['p_total'],
				                      par_value=data['par_value'],
				                      listed_shares=data['listed_shares'],
				                      market_cap=data['market_cap'])
			## End if

			printProgress(idx, proc_length, 'Progress:', 'Complete', 0, 80)

		## End for

		exec_string = "UPDATE krx_symbols A INNER JOIN krx_trans_daily B ON(A.isin = B.isin) " + \
		              "SET A.p_offer = B.p_offer, " + \
		              "A.p_bid = B.p_bid, " + \
		              "A.p_open = B.p_open, " + \
		              "A.p_high = B.p_high, " + \
		              "A.p_low = B.p_low, " + \
		              "A.p_current = B.p_current, " + \
		              "A.p_diff = B.p_diff, " + \
		              "A.dod = B.dod, " + \
		              "A.volume = B.volume, " + \
		              "A.p_total = B.p_total, " + \
		              "A.par_value = B.par_value, " + \
		              "A.listed_shares = B.listed_shares, " + \
		              "A.market_cap = B.market_cap " + \
		              "WHERE B.trans_date = %s"
		mariadb.execute(exec_string, [prevday.strftime('%Y-%m-%d')])
	## End if

print('Transaction data processed successfully.')



