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


def get_fornlimit_transactions(transdate):
    #** 30019 : 주식 > 순위정보 > 외국인한도소진상위 (외국인보유현황)
    # STEP 01: Generate OTP
    gen_otp_url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
    gen_otp_data = {
        'name': 'fileDown',
        'filetype': 'xls',
        'url': 'MKD/04/0404/04040600/mkd04040600',
        'market_gubun': 'ALL',
        'lmt_tp': '1',
        'sect_tp_cd': 'ALL',
        'schdate': '%s' % transdate.strftime('%Y%m%d'),
        'pagePath': '/contents/MKD/04/0404/04040600/MKD04040600.jsp'
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

    usecols = ['순위',
               '종목코드',
               '종목명',
               '종가',
               '대비',
               '상장주식수',
               '외국인한도수량',
               '외국인보유수량',
               '외국인한도소진률']
    df = pd.read_excel(f, converters={'순위': str,
                                      '종목코드': str,
                                      '종목명': str,
                                      '종가': str,
                                      '대비': str,
                                      '상장주식수': str,
                                      '외국인한도수량': str,
                                      '외국인보유수량': str,
                                      '외국인한도소진률': str},
                       usecols=usecols)
    df.columns = ['dummy1',
                  'symb_code',
                  'symb_name',
                  'dummy2',
                  'dummy3',
                  'listed_shares',
                  'forn_limit',
                  'forn_possession',
                  'dummy4']
    df = df.drop(['dummy1', 'dummy2', 'dummy3', 'dummy4'], axis=1).fillna('')
    df['symb_name'] = df['symb_name'].str.replace(r'\s+', '')
    df['listed_shares'] = df['listed_shares'].str.replace(r'[^0-9-.]', '')
    df['forn_limit'] = df['forn_limit'].str.replace(r'[^0-9-.]', '')
    df['forn_possession'] = df['forn_possession'].str.replace(r'[^0-9-.]', '')
    df = df.assign(**{'trans_date': [transdate.strftime('%Y-%m-%d')] * len(df)})

    return df


# End def from_symbolstoday


mariadb = maria()

## Prepare the comparison & parameter data
# 1. Get symbols information
querystring = 'SELECT isin, symb_code FROM krx_symbols'
df_symbols = mariadb.select(querystring)
# df_symbols['symb_name'] = df_symbols['symb_name'].str.replace(r'\s+', '')

# 2. Get available transaction date
querystring = 'SELECT trans_date FROM krx_trans_daily GROUP BY trans_date'
df_avatrans = mariadb.select(querystring)

## Start loop with dates...
datetoday = datetime.today()
daysbefore = (datetoday - datetime(2006, 11, 28)).days

# 1. Dates loop process
for i in range(daysbefore, -1, -1):
    time.sleep(0.12)

    transdate = datetoday - timedelta(days=i)

    if transdate.weekday() in (5, 6) or len(df_avatrans[df_avatrans['trans_date']==transdate.strftime('%Y-%m-%d')]) == 0:
        continue

    printProgress(daysbefore - i, daysbefore, '# Progress :',
                  ' Date of %s is processing.' % transdate.strftime('%Y-%m-%d'), 2, 50)

    df_trans = pd.DataFrame()

    # continue, if the retrieved data is empty...
    df_received = get_fornlimit_transactions(transdate)
    # print('\n')
    # print('1. ', '{:,}'.format(len(df_received)), ' : 크롤링한 데이터')
    if len(df_received) == 0:
        continue

    # remove the row(s) which is empty data
    df_received = df_received[~((df_received['forn_limit'].astype('float') == 0) &
                                (df_received['forn_possession'].astype('float') == 0))]
    # all the rows are empty, continue
    # print('2. ', '{:,}'.format(len(df_received)), ' : 모든값이 0인 데이터 제거 후...')
    if len(df_received) == 0:
        continue

    # continues, all the rows are not corresponded with symbols...
    # df_received = pd.merge(df_symbols, df_received, on=['symb_code', 'symb_name'])
    df_received = pd.merge(df_symbols, df_received, on='symb_code')
    # print('3. ', '{:,}'.format(len(df_received)), ' : Symbols 테이블과 조인 후...')
    if len(df_received) == 0:
        continue

    df_trans = pd.concat([df_trans, df_received], ignore_index=True)

    if len(df_trans) == 0:
        print('# Progress : Date of %s, The foreign limit transaction data does not exist.\n' % transdate.strftime(
            '%Y-%m-%d'))
        continue

    df_trans = df_trans.drop(['symb_code'], axis=1)

    mariadb.insert("krx_trans_d_foreign_possession", df_trans)

    # print('# Progress : Date of %s, %s of the foreign limit transaction data are stored.\n' % (transdate.strftime('%Y-%m-%d'), '{:,}'.format(len(df_trans))))

    # print(df_trans.head())

## End for loop, for i in range(daysbefore, -1, -1):

print('All the transactions data are processed successfully.')

