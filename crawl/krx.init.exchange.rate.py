import pandas as pd
import numpy as np
import requests
import random
import time
import sys
import io

from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from databases import maria


def get_random_sleep_time():
    return round(random.uniform(0.25, 0.70), 2)

def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def get_daily_exhange_rate(trans_s_date, trans_e_date):
    # ** 70010 : 해외시장연계정보 > 외환시세 > 일별환율변동

    url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
    params = {'bld' : 'MKD/09/0904/09040300/mkd09040300_01',
              'name' : 'form',
              '_' : '1511247932335'}
    code = requests.get(url, params=params).text

    url = 'http://marketdata.krx.co.kr/contents/MKD/99/MKD99000001.jspx'
    params = {'isu_eng_symbl' : 'USDKRWCOMP',
              'uly_id' : 'USD',
              'fromdate' : '%s' % trans_s_date.strftime('%Y%m%d'),
              'todate' : '%s' % trans_e_date.strftime('%Y%m%d'),
              'pagePath' : '/contents/MKD/09/0904/09040300/MKD09040300.jsp',
              'code' : '%s' % code}
    response = requests.get(url, params=params)
    dailytrans = response.json()['block1']

    time.sleep(get_random_sleep_time())

    data = []
    for attr in dailytrans:
	    data.append(list(attr.values()))

    df = pd.DataFrame(data, columns=['trans_date',
                                     'p_close',
                                     'p_diff',
                                     'dod',
                                     'p_open',
                                     'p_high',
                                     'p_low',
                                     'p_diff_type']
                      )
    df['p_close'] = df['p_close'].str.replace(r'[^0-9-.]', '')
    df['p_diff'] = df['p_diff'].str.replace(r'[^0-9-.]', '')
    df['dod'] = df['dod'].str.replace(r'[^0-9-.]', '')
    df['p_open'] = df['p_open'].str.replace(r'[^0-9-.]', '')
    df['p_high'] = df['p_high'].str.replace(r'[^0-9-.]', '')
    df['p_low'] = df['p_low'].str.replace(r'[^0-9-.]', '')
    df['p_diff_type'] = df['p_diff_type'].astype('int')
    df = df.assign(**{'exchange_type': ['USD'] * len(df)})

    df['p_diff'] = df['p_diff'].astype('float') * df['p_diff_type'].apply(lambda x: -1 if x == 2 else 1)
    df['dod'] = df['dod'].astype('float') * df['p_diff_type'].apply(lambda x: -1 if x == 2 else 1)

    df = df.drop(['p_diff_type'], axis=1).fillna('')

    return df


# End def get_daily_exhange_rate(transdate):

mariadb = maria()

querystring = 'SELECT MAX(trans_date) trans_date FROM krx_trans_d_exchage_rate'
df_result = mariadb.select(querystring)
last_trans_date = df_result.get_value(0, 'trans_date')

## Start loop with dates...
datetoday = datetime.today() - timedelta(days=1)
datefrom = datetime.strptime(last_trans_date, '%Y-%m-%d') + timedelta(days=1)

rd_diff = relativedelta(datetoday, datefrom)
years = rd_diff.years

for i in range(years+1):
	trans_s_date = datefrom + relativedelta(years=i)
	trans_e_date = datefrom + relativedelta(years=i+1, days=-1)

	if trans_e_date > datetoday:
		trans_e_date = datetoday

	time.sleep(get_random_sleep_time())

	df_trans = get_daily_exhange_rate(trans_s_date, trans_e_date)

	mariadb.insert('krx_trans_d_exchage_rate', df_trans)

	#printProgress(i, years, '# Progress :', '(%s ~ %s) %s datas inserted' % (trans_s_date.strftime('%Y-%m-%d'), trans_e_date.strftime('%Y-%m-%d'), len(df_trans)), 2, 30)

## End for loop, for i in range(daysbefore, -1, -1):

print('All the transactions data are processed successfully.')

