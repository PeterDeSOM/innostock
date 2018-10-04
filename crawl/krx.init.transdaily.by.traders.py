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


def get_transactions_by_trader(mem_code, transdate=datetime.today()):
    ### 80020 : 통계 > 주식 > 종목 > 회원사별 거래실적
    # ** 80036 : 통계 > 주식 > 시장 > 회원사별 거래실적
    # 1. Crawl the daily transaction data from KRX
    url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
    params = {'bld': 'MKD/10/1002/10020308/mkd10020308',
              'name': 'form',
              '_': '1508384556871'}
    code = requests.get(url, params=params).text

    url = 'http://marketdata.krx.co.kr/contents/MKD/99/MKD99000001.jspx'
    params = {'ind_tp': 'ALL',
              'mbr_tp_cd': '%s' % mem_code,
              'period_strt_dd': '%s' % transdate.strftime('%Y%m%d'),
              'period_end_dd': '%s' % transdate.strftime('%Y%m%d'),
              'etctype': 'ST',
              'pagePath': '/contents/MKD/10/1002/10020308/MKD10020308.jsp',
              'code': '%s' % code}
    response = requests.get(url, params=params)
    dailytrans = response.json()['block1']

    data = []
    for attr in dailytrans:
        data.append(list(attr.values()))

    df = pd.DataFrame(data, columns=['dummy1',
                                     'symb_code',
                                     'dummy11',
                                     'vol_selling',
                                     'amt_selling',
                                     'vol_buying',
                                     'amt_buying',
                                     'dummy2',
                                     'dummy3']
                      )
    df = df.drop(['dummy1', 'dummy11', 'dummy2', 'dummy3'], axis=1).fillna('')
    df['vol_buying'] = df['vol_buying'].str.replace(r'[^0-9-.]', '')
    df['vol_selling'] = df['vol_selling'].str.replace(r'[^0-9-.]', '')
    df['amt_buying'] = df['amt_buying'].str.replace(r'[^0-9-.]', '')
    df['amt_selling'] = df['amt_selling'].str.replace(r'[^0-9-.]', '')
    df = df.assign(**{'trans_date': [transdate.strftime('%Y-%m-%d')] * len(df)})
    df = df.assign(**{'mem_code': [mem_code] * len(df)})

    return df


## End def get_transactions_by_trader

mariadb = maria()

# Get the memebers information
querystring = 'SELECT mem_code, mem_name FROM krx_members'
df_members = mariadb.select(querystring)

# Get current symbols information
querystring = 'SELECT isin, symb_code FROM krx_symbols'
df_symbols = mariadb.select(querystring)
# df_symbols['symb_name'] = df_symbols['symb_name'].str.replace(r'\s+', '')

# Get available transaction date
querystring = 'SELECT trans_date FROM krx_trans_daily GROUP BY trans_date'
df_avatrans = mariadb.select(querystring)

datetoday = datetime.today()
daysbefore = (datetoday - datetime(2017, 10, 20)).days

for i in range(daysbefore, -1, -1):
    transdate = datetoday - timedelta(days=i)
    str_transdate = transdate.strftime('%Y-%m-%d')

    if transdate.weekday() in (5, 6) or len(df_avatrans[df_avatrans['trans_date'] == str_transdate]) == 0:
        continue

    printProgress(daysbefore - i, daysbefore, '# Progress :', ' Date of %s is processing.\n' % str_transdate, 2, 30)

    i = 1
    proc_length = len(df_members)
    df_trans = pd.DataFrame()

    for mem_code, mem_name in df_members.values:
        printProgress(i, proc_length, '# Progress :', 'Member \'%s\' transactions are processing.' % mem_name, 2, 30)
        i += 1

        # continue, if the retrieved data is empty...
        df_received = get_transactions_by_trader(mem_code, transdate)
        if len(df_received) == 0:
            continue

        df_trans = pd.concat([df_trans, df_received], ignore_index=True)

        # time.sleep(get_random_sleep_time())
        time.sleep(0.33)

    ## End for loop, for mem_code, mem_name in df_members.values:

    if len(df_trans) == 0:
        print('# Progress : Date of %s, The trader\'s transaction data does not exist.\n' % str_transdate)
        continue

    # print('1. ', '{:,}'.format(len(df_trans)), ' : 크롤링 데이터 수.')

    # remove the row(s) which is empty data
    df_trans = df_trans[~((df_trans['vol_buying'].astype('float') == 0) &
                          (df_trans['vol_selling'].astype('float') == 0) &
                          (df_trans['amt_buying'].astype('float') == 0) &
                          (df_trans['amt_selling'].astype('float') == 0))]
    # all the rows are empty, continue
    # print('2. ', '{:,}'.format(len(df_trans)), ' : 모든 값이 0인 데이터를 제거한 후의 자료 수.')
    if len(df_trans) == 0:
        print('# Progress : Date of %s, The trader\'s transaction data does not exist.\n' % str_transdate)
        continue

    # df_trans['symb_name'] = df_trans['symb_name'].str.replace(r'\s+', '')
    df_trans = pd.merge(df_symbols, df_trans, on='symb_code')
    # print('5. ', '{:,}'.format(len(df_trans)), ' : Symbols 테이블과 조인된 후 일치하는 자료 수')
    # continues, all the rows are not corresponded with symbols
    # print('3. ', '{:,}'.format(len(df_trans)), ' : Symbol과 일치하는 자료와의 조인 후의 자료 수.')
    if len(df_trans) == 0:
        print('# Progress : Date of %s, The trader\'s transaction data does not exist.\n' % str_transdate)
        continue

    df_trans = df_trans.drop(['symb_code'], axis=1)
    mariadb.insert("krx_trans_d_traders", df_trans)

    print('# Progress : Date of %s, %s of the trader\'s transaction data are stored.\n' % (
    str_transdate, '{:,}'.format(len(df_trans))))

## End for loop, for i in range(daysbefore, -1, -1):

print('All the transactions data are processed successfully.')


