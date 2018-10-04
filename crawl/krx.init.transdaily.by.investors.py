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
    return round(random.uniform(2.35, 9.10), 2)


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


def get_investor_transactions(inv_code, transdate):
    # ** 30017 : 주식 > 순위정보 > 투자자별순위(미거래종목포함)
    ### 80019 : 통계 > 주식 > 종목 > 투자자별 거래실적
    # STEP 01: Generate OTP
    gen_otp_url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
    gen_otp_data = {
        'name': 'fileDown',
        'filetype': 'xls',
        'url': 'MKD/04/0404/04040400/mkd04040400',
        'stctype': 'ALL',
        'var_invr_cd': '%s' % inv_code,
        'schdate': '%s' % transdate.strftime('%Y%m%d'),
        'etctype': 'ST',
        'pagePath': '/contents/MKD/04/0404/04040400/MKD04040400.jsp'
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
               '매수거래량',
               '매도거래량',
               '순매수거래량',
               '매수거래대금',
               '매도거래대금',
               '순매수거래대금']

    df = pd.DataFrame()
    try:
        df = pd.read_excel(f, converters={'종목코드': str,
                                          '종목명': str,
                                          '매수거래량': str,
                                          '매도거래량': str,
                                          '순매수거래량': str,
                                          '매수거래대금': str,
                                          '매도거래대금': str,
                                          '순매수거래대금': str},
                           usecols=usecols)
    except:
        return 0

    df.columns = ['symb_code',
                  'symb_name',
                  'vol_buying',
                  'vol_selling',
                  'dummy1',
                  'amt_buying',
                  'amt_selling',
                  'dummy2']
    df = df.drop(['dummy1', 'dummy2'], axis=1).fillna('')
    # df['symb_name'] = df['symb_name'].str.replace(r'\s+', '')
    df['vol_buying'] = df['vol_buying'].str.replace(r'[^0-9-.]', '')
    df['vol_selling'] = df['vol_selling'].str.replace(r'[^0-9-.]', '')
    df['amt_buying'] = df['amt_buying'].str.replace(r'[^0-9-.]', '')
    df['amt_selling'] = df['amt_selling'].str.replace(r'[^0-9-.]', '')
    df = df.assign(**{'inv_code': [inv_code] * len(df)})
    df = df.assign(**{'trans_date': [transdate.strftime('%Y-%m-%d')] * len(df)})

    return df


# End def from_symbolstoday


mariadb = maria()

## Prepare the comparison & parameter data
# 1. Get investors information
querystring = 'SELECT inv_code, inv_name FROM krx_investors ORDER BY inv_code'
df_investors = mariadb.select(querystring)

# 2. Get symbols information
querystring = 'SELECT isin, symb_code FROM krx_symbols'
df_symbols = mariadb.select(querystring)
# df_symbols['symb_name'] = df_symbols['symb_name'].str.replace(r'\s+', '')

# 3. Get available transaction date
querystring = 'SELECT trans_date FROM krx_trans_daily GROUP BY trans_date'
df_avatrans = mariadb.select(querystring)

## Start loop with dates...
datetoday = datetime.today()
daysbefore = (datetoday - datetime(2016, 3, 7)).days

proc_stop = False

# 1. Dates loop process
for i in range(daysbefore, -1, -1):
    if proc_stop:
        break

    transdate = datetoday - timedelta(days=i)

    if transdate.weekday() in (5, 6) or len(
            df_avatrans[df_avatrans['trans_date'] == transdate.strftime('%Y-%m-%d')]) == 0:
        continue

    printProgress(daysbefore - i, daysbefore, '# Progress :',
                  ' Date of %s is processing...\n' % transdate.strftime('%Y-%m-%d'), 2, 50)

    # 2. Investors loop process
    df_trans = pd.DataFrame()

    i = 1
    for inv_code, _ in df_investors.values:
        if proc_stop:
            break

        printProgress(i, len(df_investors), '# Progress :', 'Investor %s is processing...' % inv_code, 2, 50)
        i += 1

        # continue, if the retrieved data is empty...
        error_count = 0
        df_received = 0

        while isinstance(df_received, int):
            df_received = get_investor_transactions(inv_code, transdate)

            if error_count == 4:
                proc_stop = True
                df_received = []
                break

            error_count += 1
            time.sleep(0.33)
            # time.sleep(get_random_sleep_time())

        # print('\n', '1. ', '{:,}'.format(len(df_received)), ' : 크롤링한 데이터')
        if len(df_received) == 0:
            continue

        # remove the row(s) which is empty data
        df_received = df_received[~((df_received['vol_buying'].astype('float') == 0) &
                                    (df_received['vol_selling'].astype('float') == 0) &
                                    (df_received['amt_buying'].astype('float') == 0) &
                                    (df_received['amt_selling'].astype('float') == 0))]
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

    ## End for loop...

    if len(df_trans) == 0:
        print('# Progress : Date of %s, The investor\'s transaction data does not exist.\n' % transdate.strftime(
            '%Y-%m-%d'))
        continue

    # df_trans = df_trans.drop(['symb_code', 'symb_name'], axis=1)
    df_trans = df_trans.drop(['symb_code'], axis=1)

    mariadb.insert("krx_trans_d_investors", df_trans)

    print('# Progress : Date of %s, %s of the investor\'s transaction data are inserted.\n' % (
    transdate.strftime('%Y-%m-%d'), '{:,}'.format(len(df_trans))))

## End for loop, for i in range(daysbefore, -1, -1):

print('All the transactions data are processed successfully.')

