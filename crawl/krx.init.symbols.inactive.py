import re
import time
import requests
import pandas as pd
import numpy as np

from databases import maria

# 거래소 > 주식 > 상장현황 > 상장폐지종목검색(x) > 상장폐지종목검색기
url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
params = {'bld' : 'COM/finder_dellist_isu',
          'name' : 'form',
          '_' : '1507337165232'}
code = requests.get(url, params=params).text

url = 'http://marketdata.krx.co.kr/contents/MKD/99/MKD99000001.jspx'
params = {'mktsel' : 'ALL',
          'pagePath' : '/contents/COM/FinderDelListIsu.jsp',
          'code' : '%s' % code}
response = requests.get(url, params=params)
listsymbs = response.json()['result']

print('%d data(s) crawled.' % len(listsymbs))

data = []
for attr in listsymbs:
    data.append(list(attr.values()))

df_krxsymbols = pd.DataFrame(data, columns=['isin',
                                            'day_closed',
                                            'symb_code',
                                            'symb_name',
                                            'symb_name_long',
                                            'market_type']
                             )
df_krxsymbols = df_krxsymbols[df_krxsymbols['symb_code'].str.len() == 7]
df_krxsymbols = df_krxsymbols.reset_index(drop=True)
df_krxsymbols['symb_code'] = df_krxsymbols['symb_code'].apply(lambda x: x[1:])
df_krxsymbols['symb_type_code'] = df_krxsymbols['symb_code'].apply(lambda x: x[len(x) - 1:len(x)])
df_krxsymbols['day_closed'] = df_krxsymbols['day_closed'].apply(lambda x: '%s-%s-%s' % (x[0:4], x[4:6], x[6:8]))
df_krxsymbols['market_type'] = df_krxsymbols['market_type'].apply(lambda x: ('I' if x == 'KOSPI' else ('Q' if x == 'KOSDAQ' else 'X')))
# df_krxsymbols['symb_name'] = df_krxsymbols['symb_name'].str.replace(r"\s+", '')
df_krxsymbols = df_krxsymbols.assign(**{'symb_status': ['I'] * len(df_krxsymbols)})

### Get more information of closed symbols
# 거래소 > 주식 > 상장현황 > 상장폐지현황
url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
params = {'bld' : 'MKD/04/0406/04060600/mkd04060600',
          'name' : 'form',
          '_' : '1507527937606'}
code = requests.get(url, params=params).text

url = 'http://marketdata.krx.co.kr/contents/MKD/99/MKD99000001.jspx'
params = {'market_gubun':'ALL',
          'su_cdnm':'전체',
          'fromdate':'1990101',
          'todate':'20170929',
          'pagePath' : '/contents/MKD/04/0406/04060600/MKD04060600.jsp',
          'code' : '%s' % code}
response = requests.get(url, params=params)
listsymbs = response.json()['block1']

data = []
for attr in listsymbs:
    data.append(list(attr.values()))

df_more = pd.DataFrame(data,
                       columns=['symb_code',
                                'symb_name',
                                'day_closed',
                                'desc_close']
                       )
df_more['symb_code'] = df_more['symb_code'].apply(lambda x: x[1:])
# df_more['symb_name'] = df_more['symb_name'].str.replace(r"\s+", '')
df_krxsymbols = pd.merge(df_krxsymbols, df_more[['symb_code', 'symb_name', 'desc_close']], how='left', on=['symb_code', 'symb_name']).fillna('')
df_krxsymbols = df_krxsymbols[df_krxsymbols['day_closed'] > '2001-01-01']

mariadb = maria()

querystring = 'SELECT isin, symb_status FROM krx_symbols'
df_reslut = mariadb.select(querystring)

## If value(s) exist in the comparison values, isin() function returns Ture.
## Returns false if it is not in comparison values.
# print (df_krxsymbols['isin'].isin(df_reslut['isin']))

## Contional dataframe return only true data.
## If we want to see false conditional data, 'false' status must be turned into 'true' with '~'.
# The new datas will be inserted.
df_insert = df_krxsymbols[~df_krxsymbols['isin'].isin(df_reslut['isin'])]
if len(df_insert):
	df_insert = df_insert.assign(**{'day_opened': [''] * len(df_insert)})

	url = 'http://kind.krx.co.kr/corpdetail/totalinfo.do'

	for idx, r in df_insert.iterrows():
		source = r.to_dict()
		params = {'method': 'searchTotalInfo',
                  'isurCd': source['symb_code'][0:-1],
                  'repIsuCd': source['isin']}
		response = requests.get(url, params=params)
		dates = re.findall(r'\b[0-9]{4}-[0-9]{2}-[0-9]{2}', response.text)

		if len(dates) == 2:
			df_insert.loc[idx, 'day_opened'] = dates[1]

		print(str(idx).zfill(4), ', ', dates)
		time.sleep(0.2)

	df_insert = df_insert.fillna('')
	mariadb.insert('krx_symbols', df_insert)
	print(df_insert.head())
	print('...\nInit %d data(s) inserted successfully' % len(df_insert))
else:
	print('...\nNo newly closed symbol(s).')


# The data which was already in the database will be updated.
df_update = df_krxsymbols[df_krxsymbols['isin'].isin(df_reslut[df_reslut['symb_status']!='I']['isin'])]
# df_update = df_krxsymbols[df_krxsymbols['isin'].isin(df_reslut['isin'])]
if len(df_update):
    # 00: isin
    # 01: day_closed
    # 02: symb_code_long
    # 03: symb_name
    # 04: symb_name_long
    # 05: market_type
    # 06: symb_code
    # 07: symb_member_code
    # 08: symb_type_code
    # 09: symb_type_code1
    # 10: symb_status
    # df_update = df_update.drop(df_update.columns[[2, 3, 5, 6, 7, 8, 9]], axis=1)
    df_update = df_update[['isin', 'day_closed', 'desc_close', 'symb_status']]
    mariadb.update('krx_symbols', df_update, 'isin')
    print(df_update.head())
    print('...\n%d symbol\'s status changed to \'Inactive\'.' % len(df_update))
else:
    print('...\nNo symbol status changed to \'Inactive\' was found.')

