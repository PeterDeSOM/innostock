import pandas as pd
import numpy as np
import requests
import random
import time
import sys
import io

from datetime import datetime, timedelta
from databases import maria


def get_random_sleep_time():
    return round(random.uniform(1.15, 3.40), 2)


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def get_investment_index(transdate=datetime.today()):
    # ** 30009 : 주식 > 투자참고 > 투자지표
    # 1. Crawl the daily transaction data from KRX
    url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
    params = {'bld': 'MKD/04/0403/04030800/mkd04030800',
              'name': 'form',
              '_': '1508498310426'}
    code = requests.get(url, params=params).text

    url = 'http://marketdata.krx.co.kr/contents/MKD/99/MKD99000001.jspx'
    params = {'market_gubun' : 'ALL',
              'gubun' : '1',
              'schdate': '%s' % transdate.strftime('%Y%m%d'),
              'pagePath': '/contents/MKD/04/0403/04030800/MKD04030800.jsp',
              'code': '%s' % code}
    response = requests.get(url, params=params)
    dailytrans = response.json()['result']

    data = []
    for attr in dailytrans:
        data.append(list(attr.values()))

    df = pd.DataFrame(data, columns=['dummy1',
                                     'symb_code',
                                     'dummy2',
                                     'dummy3',
                                     'dummy4',
                                     'eps',
                                     'per',
                                     'bps',
                                     'pbr',
                                     'dps',
                                     'pdr',
                                     'dummy5',
                                     'dummy6',
                                     'dummy7']
                      )
    df = df.drop(['dummy1', 'dummy2', 'dummy3', 'dummy4', 'dummy5', 'dummy6', 'dummy7'], axis=1).fillna('')
    df['eps'] = df['eps'].str.replace(r'[^0-9.-]', '')
    df['per'] = df['per'].str.replace(r'[^0-9.-]', '')
    df['bps'] = df['bps'].str.replace(r'[^0-9.-]', '')
    df['pbr'] = df['pbr'].str.replace(r'[^0-9.-]', '')
    df['dps'] = df['dps'].str.replace(r'[^0-9.-]', '')
    df['pdr'] = df['pdr'].str.replace(r'[^0-9.-]', '')
    df = df.assign(**{'trans_date': [transdate.strftime('%Y-%m-%d')] * len(df)})

    return df.replace('-', '0').replace('', '0')


## End def get_transactions_by_trader

mariadb = maria()

# Get current symbols information
querystring = 'SELECT isin, symb_code FROM krx_symbols'
df_symbols = mariadb.select(querystring)

# Get available transaction date
querystring = 'SELECT trans_date FROM krx_trans_daily GROUP BY trans_date'
df_avatrans = mariadb.select(querystring)

querystring = 'SELECT MAX(trans_date) trans_date FROM krx_trans_d_investment_index'
df_result = mariadb.select(querystring)
last_trans_date = df_result.get_value(0, 'trans_date')

prevdate = datetime.today() - timedelta(days=1)
# daysbefore = (prevdate - datetime(2017, 10, 20)).days
daysbefore = (prevdate - datetime.strptime(last_trans_date, '%Y-%m-%d')).days - 1

for i in range(daysbefore, -1, -1):
    transdate = prevdate - timedelta(days=i)
    str_transdate = transdate.strftime('%Y-%m-%d')

    if transdate.weekday() in (5, 6) or len(df_avatrans[df_avatrans['trans_date'] == str_transdate]) == 0:
        continue

    time.sleep(0.33)

    df_trans = get_investment_index(transdate)
    # continue, if the retrieved data is empty...
    if len(df_trans) == 0:
        continue

    # remove the row(s) which is empty data
    df_trans = df_trans[~((df_trans['eps'].astype('float') == 0) &
                          (df_trans['per'].astype('float') == 0) &
                          (df_trans['bps'].astype('float') == 0) &
                          (df_trans['pbr'].astype('float') == 0) &
                          (df_trans['dps'].astype('float') == 0) &
                          (df_trans['pdr'].astype('float') == 0))]
    # all the rows are empty, continue
    if len(df_trans) == 0:
        continue

    df_trans = pd.merge(df_symbols, df_trans, on='symb_code')
    # continues, all the rows are not corresponded with symbols
    if len(df_trans) == 0:
        continue

    df_trans = df_trans.drop(['symb_code'], axis=1)
    mariadb.insert("krx_trans_d_investment_index", df_trans)

    printProgress(daysbefore - i, daysbefore, '# Progress :',
                  ' (%s) %s of the Investment Indexes are stored.' % (str_transdate, '{:,}'.format(len(df_trans))), 2, 30)

## End for loop, for i in range(daysbefore, -1, -1):

print('All the transactions data are processed successfully.')


