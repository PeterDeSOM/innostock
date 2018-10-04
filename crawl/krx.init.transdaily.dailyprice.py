import re
import requests
import pandas as pd
import numpy as np
import sys
import time

from databases import maria
from datetime import datetime, timedelta

def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


colsed_limitdate = (datetime.today() - timedelta(days=12)).strftime('%Y-%m-%d')

mariadb = maria()

querystring = 'SELECT isin, symb_code, symb_name, day_opened, day_closed FROM krx_symbols WHERE p_close IS NULL'
df_symbols = mariadb.select(querystring)

print('%s datas will be processed.' % '{:,}'.format(len(df_symbols)))

for idx, symbols in df_symbols.iterrows():
    # 1. Crawl the daily transaction data from KRX
    url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
    params = {'bld' : 'MKD/04/0402/04020100/mkd04020100t3_02',
              'name' : 'form',
              '_' : '1507527937619'}
    code = requests.get(url, params=params).text

    url = 'http://marketdata.krx.co.kr/contents/MKD/99/MKD99000001.jspx'
    params = {'isu_cdnm':'A%s/%s' % (symbols['symb_code'], symbols['symb_name']),
              'isu_cd':'%s' % symbols['isin'],
              'isu_nm':'%s' % symbols['symb_name'],
              'isu_srt_cd':'A%s' % symbols['symb_code'],
              'fromdate':'1990101',
              'todate':'20170929',
              'pagePath':'/contents/MKD/04/0402/04020100/MKD04020100T3T2.jsp',
              'code' : '%s' % code}
    response = requests.get(url, params=params)
    dailytrans = response.json()['block1']

    crawled_count = len(dailytrans)
    if crawled_count == 0:
        print('%s-%s, No daily transaction data was found.' % (symbols['isin'], symbols['symb_name']))
        continue

    # 2. Extract the transaction date and closed price
    data_pclosed = []
    for attr in dailytrans:
        data_pclosed.append([attr['trd_dd'], attr['tdd_clsprc']])

    df_pclosed = pd.DataFrame(data_pclosed, columns=['trans_date', 'p_close'])
    df_pclosed['trans_date'] = df_pclosed['trans_date'].str.replace('/', '-')
    df_pclosed['p_close'] = df_pclosed['p_close'].str.replace(r'[^0-9-.]', '')
    df_pclosed = df_pclosed.assign(**{'isin': [symbols['isin']] * len(df_pclosed)})

    # 3. Update symbol's open date and close date if it is not defined yet.
    # Lastest transaction date is the top of rows
    p_close = df_pclosed.get_value(0, 'p_close')


    day_closed = df_pclosed.get_value(0, 'trans_date')
    # Very first transaction date is the bottom of rows
    day_opened = df_pclosed.get_value(len(df_pclosed) - 1, 'trans_date')

    df_update = pd.DataFrame([[symbols['isin'],
                               p_close,
                               day_opened if not symbols['day_opened'] else symbols['day_opened'],
                               day_closed if not symbols['day_closed'] and day_closed < colsed_limitdate else symbols['day_closed']
                               ]], columns=['isin', 'p_close', 'day_opened', 'day_closed'])
    mariadb.update('krx_symbols', df_update, 'isin')

    # 4. Update closed price to the daily transaction
    df_pclosed = df_pclosed[df_pclosed['trans_date'] >= '2001-01-01']
    proc_length = len(df_pclosed)
    print('%s, %s of %s data is processing. (%s)' % (symbols['isin'],
                                                     '{:,}'.format(proc_length),
                                                     '{:,}'.format(crawled_count),
                                                     symbols['symb_name']))
    for i, attr in df_pclosed.iterrows():
        where = "isin='%s' AND trans_date='%s'" % (attr['isin'], attr['trans_date'])
        mariadb.update_single('krx_trans_daily', where, p_close=attr['p_close'])

        printProgress(i, proc_length, 'Progress:', 'Complete', 0, 80)

    ## End for loop

    print('\n')




