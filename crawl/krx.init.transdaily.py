import pandas as pd
import numpy as np
import requests
import random
import time
import io

from datetime import datetime, timedelta
from databases import maria


def from_symbolstoday(marketype='I', transdate=datetime.today()):
    # STEP 01: Generate OTP
    gen_otp_url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
    gen_otp_data = {
        'name': 'fileDown',
        'filetype': 'xls',
        'url': 'MKD/04/0406/04060200/mkd04060200',
        'market_gubun': '%s' % ("STK" if marketype == 'I' else ("KSQ" if marketype == 'Q' else "KNX")),
        'indx_ind_cd': '%s' % ("1001" if marketype == 'I' else ("2001" if marketype == 'Q' else "N001")),
        'sect_tp_cd': 'ALL',
        'isu_cdnm': '전체',
        'secugrp': 'ST',
        'stock_gubun': 'on',
        'schdate': '%s' % transdate.strftime('%Y%m%d'),
        'pagePath': '/contents/MKD/04/0406/04060200/MKD04060200.jsp'
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
    df = df.assign(**{'market_type': np.full(len(df), marketype)})
    df = df.assign(**{'trans_date': np.full(len(df), transdate.strftime('%Y-%m-%d'))})

    return df


# End def from_symbolstoday

datetoday = datetime.today()
daysbefore = (datetoday - datetime(2017, 10, 19)).days
for i in range(daysbefore, -1, -1):
    transdate = datetoday - timedelta(days=i)

    if transdate.weekday() in (5, 6):
        continue

    df_daytrans = pd.DataFrame()
    for j in range(1, 4):
        marketype = "I" if j == 1 else ("Q" if j == 2 else "X")
        df_daytrans = pd.concat([df_daytrans, from_symbolstoday(marketype, transdate)], ignore_index=True)

    # End for loop to Market type

    mariadb = maria()
    mariadb.insert("krx_trans_daily", df_daytrans)

    procedate = transdate.strftime('%Y-%m-%d')
    print('%s daily tradding data of %s rows are processed.' % (procedate, '{:,}'.format(len(df_daytrans))))

    time.sleep(0.1)

# End for loop to Target date

print("symbols information inserted successfully.")

