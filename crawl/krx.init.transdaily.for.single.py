import pandas as pd
import numpy as np
import requests
import random
import time
import io

from datetime import datetime, timedelta
from databases import maria


def from_symbolstoday(isin, symb_code, symb_name, marketype, transdate=datetime.today()):
    # 거래소 > 주식 > 상장현황 > 상장폐지종목검색(x) > 상장폐지종목검색기
    url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
    params = {'bld': 'MKD/04/0406/04060200/mkd04060200',
              'name': 'form',
              '_': '1507618195030'}
    code = requests.get(url, params=params).text

    url = 'http://marketdata.krx.co.kr/contents/MKD/99/MKD99000001.jspx'
    params = {'market_gubun': 'ALL',
              'sect_tp_cd': 'ALL',
              'isu_cdnm': 'A%s/%s' % (symb_code, symb_name),
              'isu_cd': '%s' % isin,
              'isu_nm': '%s' % symb_name,
              'isu_srt_cd': 'A%s' % symb_code,
              'secugrp': 'ST',
              'stock_gubun': 'on',
              'schdate': '%s' % transdate.strftime('%Y%m%d'),
              'pagePath': '/contents/MKD/04/0406/04060200/MKD04060200.jsp',
              'code': '%s' % code}
    response = requests.get(url, params=params)
    listsymbs = response.json()['상장종목검색']

    if len(listsymbs) == 0:
        return pd.DataFrame()

    df = pd.DataFrame([list(listsymbs[0].values())],
                      columns=['symb_code',
                               'symb_name',
                               'p_current',
                               'sign',
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
                               'market_cap',
                               'total_count'])
    df['p_current'] = df['p_current'].str.replace(r'[^0-9-.]', '')
    df['p_diff'] = df['sign'].apply(lambda x: '-' if x == '2' else '') + df['p_diff'].str.replace(r'[^0-9-.]', '')
    df['dod'] = df['sign'].apply(lambda x: '-' if x == '2' else '') + df['dod'].str.replace(r'[^0-9-.]', '')
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
    df = df.assign(**{'isin': [isin] * len(df)})
    df = df.assign(**{'market_type': [marketype] * len(df)})
    df = df.assign(**{'trans_date': [transdate.strftime('%Y-%m-%d')] * len(df)})

    df = df.drop(['sign', 'total_count'], axis=1)

    return df


def get_start_end_date(isin, symb_code, symb_name):
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
              'fromdate': '1990101',
              'todate': '20170929',
              'pagePath': '/contents/MKD/04/0402/04020100/MKD04020100T3T2.jsp',
              'code': '%s' % code}
    response = requests.get(url, params=params)
    dailytrans = response.json()['block1']

    if len(dailytrans) == 0:
        return []

    # 2. Extract the transaction date and closed price
    data_pclosed = []
    for attr in dailytrans:
        data_pclosed.append([attr['trd_dd'], attr['tdd_clsprc']])

    df_pclosed = pd.DataFrame(data_pclosed, columns=['trans_date', 'p_close'])
    df_pclosed['trans_date'] = df_pclosed['trans_date'].str.replace('/', '-')

    day_closed = df_pclosed.get_value(0, 'trans_date')
    day_opened = df_pclosed.get_value(len(df_pclosed) - 1, 'trans_date')

    return [day_opened, day_closed]


##### Get current stored symbols to compare to the target symbols
querystring = 'SELECT isin, symb_code, symb_name, market_type FROM krx_symbols WHERE symb_code IN (%s, %s, %s)'
values = ['277410', '255440', '263920']

mariadb = maria()
df_targets = mariadb.select(querystring, values)

datetoday = datetime.today() - timedelta(days=11)
daysbefore = (datetoday - datetime(2017, 9, 28)).days

for isin, symb_code, symb_name, marketype in df_targets.values:
    print('%s, %s, %s, %s' % (isin, symb_code, symb_name, marketype))

    date_range = get_start_end_date(isin, symb_code, symb_name)
    if len(date_range) == 0:
        print('%s, %s, %s, %s - No transaction data.' % (isin, symb_code, symb_name, marketype))
        continue

    print(date_range)

    df_trans = pd.DataFrame()
    for i in range(daysbefore, -1, -1):
        transdate = datetoday - timedelta(days=i)

        if transdate.weekday() in (5, 6):
            continue

        df = from_symbolstoday(isin, symb_code, symb_name, marketype, transdate)

        if len(df) == 0:
            print('%s, %s, %s, %s : The data is not exist.' % (
            transdate.strftime('%Y-%m-%d'), isin, symb_code, symb_name))
            continue

        df_trans = pd.concat([df_trans, df], ignore_index=True)

        print('%s, %s, %s, %s : The data is crawled.' % (transdate.strftime('%Y-%m-%d'), isin, symb_code, symb_name))

        time.sleep(0.15)

    ## End for loop

    mariadb.insert("krx_trans_daily", df_trans)

    print('%s, %s, %s : %s datas are inserted.' % (isin, symb_code, symb_name, '{:,}'.format(len(df_trans))))



