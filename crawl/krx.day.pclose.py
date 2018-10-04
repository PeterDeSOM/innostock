import pandas as pd
import numpy as np
import requests
import random
import time
import sys
import io
import re

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

def get_closed_price(isin, symb_code, symb_name, transdate=datetime.today()):
    # 1. Crawl the daily transaction data from KRX
    url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
    params = {'bld': 'MKD/04/0402/04020100/mkd04020100t3_02',
              'name': 'form',
              '_': '1507527937619'}
    code = requests.get(url, params=params).text

    url = 'http://marketdata.krx.co.kr/contents/MKD/99/MKD99000001.jspx'
    params = {'isu_cdnm': 'A%s/%s' % (symb_code, symb_name),
              'isu_cd': '%s' % isin,
              'isu_nm': '%s' % symb_name,
              'isu_srt_cd': 'A%s' % symb_code,
              'fromdate': '%s' % transdate.strftime('%Y%m%d'),
              'todate': '%s' % transdate.strftime('%Y%m%d'),
              'pagePath': '/contents/MKD/04/0402/04020100/MKD04020100T3T2.jsp',
              'code': '%s' % code}
    response = requests.get(url, params=params)
    dailytrans = response.json()['block1']

    crawled_count = len(dailytrans)
    if crawled_count == 0:
        return ''

    # 2. Extract the transaction date and closed price
    return re.sub('[^0-9-.]', '', dailytrans[0]['tdd_clsprc'])


# End def get_closed_price

mariadb = maria()

# Get current symbols information
querystring = 'SELECT isin, symb_code, symb_name FROM krx_symbols WHERE symb_status <> %s'
values = ['I']
df_symbols = mariadb.select(querystring, values)
prevday = datetime.today() - timedelta(days=1)

proc_length = len(df_symbols)
print('%s. %s closed prices will be updated.' % (prevday.strftime('%Y-%m-%d'), '{:,}'.format(proc_length)))

for idx, symbols in df_symbols.iterrows():
    time.sleep(0.1)

    p_close = get_closed_price(symbols['isin'], symbols['symb_code'], symbols['symb_name'], prevday)

    if len(p_close) == 0:
        print('%s. %s\'s close price information for %s(%s, %s) does not exist.' % (str(idx + 1).zfill(4),
                                                                                    prevday.strftime('%Y-%m-%d'),
                                                                                    symbols['isin'],
                                                                                    symbols['symb_code'],
                                                                                    symbols['symb_name']))
        continue

    where = "isin='%s' AND trans_date='%s'" % (symbols['isin'], prevday.strftime('%Y-%m-%d'))
    mariadb.update_single('krx_trans_daily', where, p_close=p_close)

    printProgress(idx, proc_length, 'Progress:', 'Complete', 2, 80)

## End for loop

exec_string = "UPDATE krx_symbols A INNER JOIN krx_trans_daily B ON(A.isin = B.isin) " + \
              "SET A.p_close = B.p_close " + \
              "WHERE B.trans_date = %s"
mariadb.execute(exec_string, [prevday.strftime('%Y-%m-%d')])

# df_transtoday.ix[idx,'p_close'] = get_closed_price(symbols['isin'], symbols['symb_code'], symbols['symb_name'])

print('\nTransaction data processed successfully.')



