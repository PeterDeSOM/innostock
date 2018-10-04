import pandas as pd
import numpy as np
import requests
import random
import time
import sys
import io

from datetime import datetime, timedelta
from databases import maria


def get_investor_arbitrage(transdate):
    # ** 80035 : 통계 > 주식 > 시장 > 프로그램 거래실적
    # 1. Crawl the daily transaction data from KRX

    # STK=KOSPI, KSQ=KOSDAQ
    marketypes = ['STK', 'KSQ']
    data_arbitrages = []

    for marketype in marketypes:
        url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
        params = {'bld' : 'MKD/10/1002/10020307/mkd10020307_03',
                  'name' : 'form',
                  '_' : '1508311432156'}
        code = requests.get(url, params=params).text

        url = 'http://marketdata.krx.co.kr/contents/MKD/99/MKD99000001.jspx'
        params = {'ind_tp' : '%s' % marketype,
                  'state_type' : 'C',
                  'period_strt_dd' : '%s' % transdate.strftime('%Y%m%d'),
                  'period_end_dd' : '%s' % transdate.strftime('%Y%m%d'),
                  'pagePath' : '/contents/MKD/10/1002/10020307/MKD10020307.jsp',
                  'code' : '%s' % code}
        response = requests.get(url, params=params)
        dailytrans = response.json()['block1']

        time.sleep(0.34)

        crawled_count = len(dailytrans)
        if crawled_count == 0:
            continue

        # 2. Extract the transaction date and closed price
        for attr in dailytrans:
            data_arbitrages.append([attr['invst_tp_nm'],
                                    attr['ofr_prf_tr_vl'],
                                    attr['ofr_prf_tr_amt'],
                                    attr['bid_prf_tr_vl'],
                                    attr['bid_prf_tr_amt'],
                                    'I' if marketype == 'STK' else 'Q',
                                    transdate.strftime('%Y-%m-%d')])
        ## End for loop
    ## End for loop, for marketype in marketypes:

    df = pd.DataFrame(data_arbitrages, columns=['inv_name',
                                                'vol_selling',
                                                'amt_selling',
                                                'vol_buying',
                                                'amt_buying',
                                                'market_type_code',
                                                'trans_date'])
    df['vol_selling'] = df['vol_selling'].str.replace(r'[^0-9-.]', '')
    df['amt_selling'] = df['amt_selling'].str.replace(r'[^0-9-.]', '')
    df['vol_buying'] = df['vol_buying'].str.replace(r'[^0-9-.]', '')
    df['amt_buying'] = df['amt_buying'].str.replace(r'[^0-9-.]', '')

    return df

# End def get_investor_arbitrage(transdate):


mariadb = maria()

## Prepare the comparison & parameter data
# 1. Get investors information
querystring = 'SELECT inv_code, inv_name FROM krx_investors'
df_investors = mariadb.select(querystring)

# 2. Get available transaction date
querystring = 'SELECT MAX(trans_date) trans_date FROM krx_trans_d_arbitrages_by_investor'
df_result = mariadb.select(querystring)
transdate = df_result.get_value(0, 'trans_date')

## Start loop with dates...
prevdate = datetime.today() - timedelta(days=1)
daysbefore = (prevdate - (datetime.strptime(transdate, '%Y-%m-%d') + timedelta(days=1))).days

if daysbefore < 0:
    print('# Progress : No updatable transaction found.')
    exit()

for i in range(daysbefore, -1, -1):
    transdate = prevdate - timedelta(days=i)

    if transdate.weekday() in (5, 6):
        continue

    df_received = get_investor_arbitrage(transdate)

    time.sleep(0.33)

    if len(df_received) == 0:
        continue

    # remove the row(s) which is empty data
    df_received = df_received[~((df_received['vol_buying'].astype('float') == 0) &
                                (df_received['vol_selling'].astype('float') == 0) &
                                (df_received['amt_buying'].astype('float') == 0) &
                                (df_received['amt_selling'].astype('float') == 0))]
    # all the rows are empty, continue
    if len(df_received) == 0:
        print('# Progress : Date of %s, The investor\'s arbitrage transaction data does not exist.' %
              transdate.strftime('%Y-%m-%d'))
        continue

    # continues, all the rows are not corresponded with symbols...
    df_received = pd.merge(df_investors, df_received, on='inv_name')
    df_received = df_received.drop(['inv_name'], axis=1)

    mariadb.insert("krx_trans_d_arbitrages_by_investor", df_received)
    print('# Progress : (%s) %s row(s) are stored' % (transdate.strftime('%Y-%m-%d'), '{:,}'.format(len(df_received))))

## End for loop, for i in range(daysbefore, -1, -1):
