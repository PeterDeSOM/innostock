import pandas as pd
import numpy as np
import requests
import random
import time
import sys
import io

from dateutil.relativedelta import relativedelta
from datetime import datetime
from databases import maria


def get_random_sleep_time():
    return round(random.uniform(1.20, 4.90), 2)


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def get_shorting_transactions(trans_s_date, trans_e_date):
    # STEP 01: Generate OTP
    gen_otp_url = 'http://short.krx.co.kr/contents/COM/GenerateOTP.jspx'

    df_trans = pd.DataFrame()

    for marketype in ('1', '3', '4'):
        # marketype (1: KOSPI, 3: KOSDAQ, 4:KONEX)
        gen_otp_data = {
            'name': 'fileDown',
            'filetype': 'xls',
            'url': 'SRT/02/02020100/srt02020100',
            'mkt_tp_cd': '%s' % marketype,
            'isu_cdnm': '전체',
            'strt_dd': '%s' % trans_s_date.strftime('%Y%m%d'),
            'end_dd': '%s' % trans_e_date.strftime('%Y%m%d'),
            'pagePath': '/contents/SRT/02/02020100/SRT02020100.jsp'
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

        usecols = ['일자',
                   '종목코드',
                   '종목명',
                   '공매도거래량',
                   '총거래량',
                   '비중',
                   '공매도거래대금',
                   '총거래대금',
                   '비중.1']
        df = pd.read_excel(f, converters={'일자': str,
                                          '종목코드': str,
                                          '종목명': str,
                                          '공매도거래량': str,
                                          '총거래량': str,
                                          '비중': str,
                                          '공매도거래대금': str,
                                          '총거래대금': str,
                                          '비중.1': str},
                           usecols=usecols)
        df.columns = ['trans_date',
                      'isin',
                      'dummy1',
                      'vol_trans',
                      'dummy2',
                      'dummy3',
                      'amt_trans',
                      'dummy4',
                      'dummy5']
        df = df.drop(['dummy1', 'dummy2', 'dummy3', 'dummy4', 'dummy5'], axis=1).fillna('')
        df['trans_date'] = df['trans_date'].str.replace('/', '-')
        df['vol_trans'] = df['vol_trans'].str.replace(r'[^0-9-.]', '')
        df['amt_trans'] = df['amt_trans'].str.replace(r'[^0-9-.]', '')

        df_trans = pd.concat([df_trans, df], ignore_index=True)

        time.sleep(get_random_sleep_time())

    ## End for loop

    return df_trans

# End def get_shorting_transactions

def get_shorting_balances(trans_s_date, trans_e_date):
    # STEP 01: Generate OTP
    gen_otp_url = 'http://short.krx.co.kr/contents/COM/GenerateOTP.jspx'

    df_trans = pd.DataFrame()

    for marketype in ('1', '2', '6'):
        # marketype (1: KOSPI, 2: KOSDAQ, 6:KONEX)
        gen_otp_data = {
            'name': 'fileDown',
            'filetype': 'xls',
            'url': 'SRT/02/02030100/srt02030100',
            'mkt_tp_cd': '%s' % marketype,
            'isu_cdnm': '전체',
            'strt_dd': '%s' % trans_s_date.strftime('%Y%m%d'),
            'end_dd': '%s' % trans_e_date.strftime('%Y%m%d'),
            'pagePath': '/contents/SRT/02/02030100/SRT02030100.jsp'
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

        usecols = ['공시의무발생일',
                   '종목코드',
                   '종목명',
                   '공매도잔고수량',
                   '상장주식수',
                   '공매도잔고금액',
                   '시가총액',
                   '비중']
        df = pd.read_excel(f, converters={'공시의무발생일': str,
                                          '종목코드': str,
                                          '종목명': str,
                                          '공매도잔고수량': str,
                                          '상장주식수': str,
                                          '공매도잔고금액': str,
                                          '시가총액': str,
                                          '비중': str},
                           usecols=usecols)
        df.columns = ['trans_date',
                      'isin',
                      'dummy1',
                      'vol_balance',
                      'dummy2',
                      'amt_balance',
                      'dummy3',
                      'dummy4']
        df = df.drop(['dummy1', 'dummy2', 'dummy3', 'dummy4'], axis=1).fillna('')
        df['trans_date'] = df['trans_date'].str.replace('/', '-')
        df['vol_balance'] = df['vol_balance'].str.replace(r'[^0-9-.]', '')
        df['amt_balance'] = df['amt_balance'].str.replace(r'[^0-9-.]', '')

        df_trans = pd.concat([df_trans, df], ignore_index=True)

        time.sleep(get_random_sleep_time())

    ## End for loop

    return df_trans

# End def get_shorting_balances


mariadb = maria()

## Prepare the comparison & parameter data
# 1. Get symbols information
querystring = 'SELECT isin FROM krx_symbols'
df_symbols = mariadb.select(querystring)

## Start loop with dates...
datetoday = datetime.today()
datefrom = datetime(2002, 3, 1)

rd_diff = relativedelta(datetoday, datefrom)
t_months = (12 * rd_diff.years + rd_diff.months) + 1

# 1. Dates loop process
for i in range(t_months):
    trans_s_date = datefrom + relativedelta(months=i)
    trans_e_date = datefrom + relativedelta(months=i + 1, days=-1)

    if trans_e_date > datetoday:
        trans_e_date = datetoday

    printProgress(i, t_months, '# Progress :',
                  'Date range of %s to %s is processing...\n' % (trans_s_date.strftime('%Y-%m-%d'),
                                                                 trans_e_date.strftime('%Y-%m-%d')),
                  2, 30)

    df_trans = get_shorting_transactions(trans_s_date, trans_e_date)
    print('1. ', '{:,}'.format(len(df_trans)), ' : 공매도 거래내역 자료 수.')
    print(df_trans.head(), '\n')

    if trans_s_date.strftime('%Y-%m-%d') >= '2016-06-01':
        df_balance = get_shorting_balances(trans_s_date, trans_e_date)
        print('2. ', '{:,}'.format(len(df_balance)), ' : 공매도 잔고현황 자료 수.')
        print(df_balance.head(), '\n')
    else:
        df_balance = pd.DataFrame({'isin': [], 'trans_date': [], 'vol_balance': [], 'amt_balance': []})
        print('2. 공매도 잔고현황  조회 무시')

    if len(df_trans) == 0 and len(df_balance) == 0:
        print('# Progress : Date range of %s to %s, the short selling transaction data does not exist.\n' %
              (trans_s_date.strftime('%Y-%m-%d'), trans_e_date.strftime('%Y-%m-%d')))
        continue

    df_trans = pd.merge(df_trans, df_balance, how='outer', on=['trans_date', 'isin']).fillna(0)
    print('3. ', '{:,}'.format(len(df_trans)), ' : 공매도 거래내역과 잔고현황을 통합한 후의 자료 수.')
    print(df_trans.head(), '\n')

    # remove the row(s) which is empty data
    df_trans = df_trans[~((df_trans['vol_balance'].astype('float') == 0) &
                          (df_trans['amt_balance'].astype('float') == 0) &
                          (df_trans['vol_trans'].astype('float') == 0) &
                          (df_trans['amt_trans'].astype('float') == 0))]
    # all the rows are empty, continue
    if len(df_trans) == 0:
        print('# Progress : Date range of %s to %s, the short selling transaction data does not exist.\n' %
              (trans_s_date.strftime('%Y-%m-%d'), trans_e_date.strftime('%Y-%m-%d')))
        continue
    print('4. ', '{:,}'.format(len(df_trans)), ' : 모든 값이 0인 데이터를 제거한 후 자료 수.')
    print(df_trans.head(), '\n')

    # continues, all the rows are not corresponded with symbols
    df_trans = pd.merge(df_symbols, df_trans, on='isin')
    print('5. ', '{:,}'.format(len(df_trans)), ' : Symbols 테이블과 조인된 후 일치하는 자료 수')
    if len(df_trans) == 0:
        print('# Progress : Date range of %s to %s, the short selling transaction data does not exist.\n' %
              (trans_s_date.strftime('%Y-%m-%d'), trans_e_date.strftime('%Y-%m-%d')))
        continue
    print(df_trans.head(), '\n')

    mariadb.insert("krx_trans_d_shorting", df_trans)

    print('# Progress : Date range of %s to %s, the short selling transaction data is stored.\n' %
          (trans_s_date.strftime('%Y-%m-%d'), trans_e_date.strftime('%Y-%m-%d')))

## End for loop, for i in range(daysbefore, -1, -1):

print('All the transactions data are processed successfully.')

