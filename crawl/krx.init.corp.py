'''
Created on 2017. 9. 25.

@author: Peter Kim
'''

import pandas as pd
import numpy as np
import requests
import io

from databases import maria
from io import BytesIO
from datetime import datetime, timedelta


def form_general(marketType='KOSPI'):
    ## KIND / 상장법인목록 (주요정보 : 상장일)
    url = 'http://kind.krx.co.kr/corpgeneral/corpList.do'
    data = {
        'method': 'download',
        'orderMode': '1',  # 정렬컬럼
        'orderStat': 'D',  # 정렬 내림차순
        'searchType': '13',  # 검색유형: 상장법인
        'marketType' : '%s' % ("stockMkt" if marketType == 'KOSPI' else ("kosdaqMkt" if marketType == 'KOSDAQ' else "konexMkt")),
        'fiscalYearEnd': 'all',  # 결산월: 전체
        'location': 'all',  # 지역: 전체
    }

    r = requests.post(url, data=data)
    f = BytesIO(r.content)
    # dfs = pd.read_html(f, header=0, parse_dates=['상장일'], flavor='html5lib')
    dfs = pd.read_html(f, header=0, flavor='html5lib')
    df = dfs[0].copy()

    # 숫자를 앞자리가 0인 6자리 문자열로 변환
    df['종목코드'] = df['종목코드'].astype(np.str)
    df['종목코드'] = df['종목코드'].str.zfill(6)

    df = df.fillna("").loc[:, ['종목코드', '업종', '주요제품', '상장일', '결산월']]
    df.columns = ['symb_code', 'corp_type_name', 'prod_type_name', 'day_opened', 'month_of_settlement']
    df['month_of_settlement'] = df['month_of_settlement'].str.replace(r'[^0-9]', '')
    df['month_of_settlement'] = df['month_of_settlement'].str.zfill(2)

    print('%s %s data(s) was crawled from the KIND.' % ('{:,}'.format(len(df)), marketType))

    return df

def from_market(marketType='KOSPI'):
    ## MARKET / 주식 > 상장현황 > 상장회사검색 (주요정보 : 업종코드)
    # STEP 01: Generate OTP
    gen_otp_url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
    gen_otp_data = {
        'name': 'fileDown',
        'filetype': 'xls',
        'url': 'MKD/04/0406/04060100/mkd04060100_01',
        'market_gubun': '%s' % ("STK" if marketType == 'KOSPI' else ("KSQ" if marketType == 'KOSDAQ' else "KNX")),
        'isu_cdnm': '전체',
        'sort_type': 'A',  # 정렬 : A 기업명
        'std_ind_cd': '01',
        'cpt': '1',
        'in_cpt': '',
        'in_cpt2': '',
        'pagePath': '/contents/MKD/04/0406/04060100/MKD04060100.jsp',
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

    usecols = ['종목코드', '업종코드', '자본금(원)']
    df = pd.read_excel(f, converters={'종목코드': str, '업종코드': str}, usecols=usecols)
    df = df.fillna('')
    df.columns = ['symb_code', 'corp_type_code', 'corp_cap']
    df['corp_cap'] = df['corp_cap'].str.replace(r'[^0-9-]', '')

    print('%s %s data(s) was crawled from the Market.' % ('{:,}'.format(len(df)), marketType))

    return df

def from_symbolstoday(marketType='KOSPI', transdate=datetime.today()):
    ## MARKET / 주식 > 상장현황 > 상장종목검색 (주요정보 : 전일 전체종목 거래내역)
    # STEP 01: Generate OTP
    gen_otp_url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
    gen_otp_data = {
        'name': 'fileDown',
        'filetype': 'xls',
        'url': 'MKD/04/0406/04060200/mkd04060200',
        'market_gubun': '%s' % ("STK" if marketType == 'KOSPI' else ("KSQ" if marketType == 'KOSDAQ' else "KNX")),
        'indx_ind_cd' : '%s' % ("1001" if marketType == 'KOSPI' else ("2001" if marketType == 'KOSDAQ' else "N001")),
        'sect_tp_cd': 'ALL',
        'isu_cdnm': '전체',
        'secugrp' : 'ST',
        'stock_gubun' : 'on',
        'schdate' : '%s' % transdate.strftime('%Y%m%d'),
        'pagePath': '/contents/MKD/04/0406/04060200/MKD04060200.jsp',
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

    print('%s provious day\'s %s data(s) was crawled from the daily market.' % ('{:,}'.format(len(df)), marketType))

    return df

def form_active_symbol_search():
    ## MARKET / 종목검색기 (주요정보 : ISIN)
    url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
    params = {'bld': 'COM/finder_stkisu',
              'name': 'form',
              '_': '1507166558992'}
    code = requests.get(url, params=params).text

    url = 'http://marketdata.krx.co.kr/contents/MKD/99/MKD99000001.jspx'
    params = {'no': 'P1',
              'mktsel': 'ALL',
              'pagePath': '/contents/COM/FinderStkIsu.jsp',
              'code': '%s' % code}
    ## } 주식종목 검색기

    response = requests.get(url, params=params)
    listsymbs = response.json()['block1']

    df_krxsymbols = pd.DataFrame()
    for attr in listsymbs:
        df_attrs = pd.DataFrame([list(attr.values())], columns=['isin', 'symb_code', 'symb_name', 'market_type'])
        df_krxsymbols = pd.concat([df_krxsymbols, df_attrs], ignore_index=True)

    df_krxsymbols = df_krxsymbols.drop('symb_name', axis=1)
    df_krxsymbols = df_krxsymbols[df_krxsymbols['symb_code'].str.len() == 7]
    df_krxsymbols['symb_code'] = df_krxsymbols['symb_code'].apply(lambda x: x[1:])
    df_krxsymbols['market_type'] = df_krxsymbols['market_type'].apply(lambda x: ('I' if x == 'KOSPI' else ('Q' if x == 'KOSDAQ' else 'X')))
    df_krxsymbols = df_krxsymbols.assign(**{'symb_status': ['A']*len(df_krxsymbols)})

    print('%s active symbols was crawled from the search tool at KRX WEB.' % '{:,}'.format(len(df_krxsymbols)))

    return df_krxsymbols

## End def, form_active_symbol_search():


prevday = datetime.today() - timedelta(days=1)

print('As of %s, activated symblos will be crawled, if the information exists.' % prevday.strftime('%Y-%m-%d'))

df_combined = df_general = df_market = df_symbols = pd.DataFrame()
for i in range(1, 4):
    marketType = "KOSPI" if i == 1 else ("KOSDAQ" if i == 2 else "KONET")

    df_general = pd.concat([df_general, form_general(marketType)], ignore_index=True)
    df_market = pd.concat([df_market, from_market(marketType)], ignore_index=True)
    df_symbols = pd.concat([df_symbols, from_symbolstoday(marketType, prevday)], ignore_index=True)

# End for loop

df_combined = pd.merge(df_general, df_market, on='symb_code', how='left')
df_combined = pd.merge(df_symbols, df_combined, on='symb_code', how='left')
df_combined = pd.merge(df_combined, form_active_symbol_search(), on='symb_code', how='left').fillna('')
df_combined = df_combined[~df_combined['isin'].isin([''])]

##### Get current stored symbols to compare to the target symbols
querystring = 'SELECT isin FROM krx_symbols'

mariadb = maria()
df_currentsymbols = mariadb.select(querystring)

##### 1. Sift the new stock on the market
## If value(s) exist in the comparison values, isin() function returns Ture.
## Returns false if it is not in comparison values.
# print (df_krxsymbols['isin'].isin(df_reslut['isin']))

## Contional dataframe return only true data.
## If we want to see false conditional data, 'false' status must be turned into 'true' with '~'.
# The new datas will be inserted.
df_insert = df_combined[~df_combined['isin'].isin(df_currentsymbols['isin'])]
if len(df_insert):
    mariadb.insert('krx_symbols', df_insert)
    print(df_insert.head())
    print('...\n%d data(s) inserted successfully' % len(df_insert))
else:
    print('...\nThere is no stock newly listed on the market.')


