import pandas as pd
import requests
import re

from bs4 import BeautifulSoup
from databases import maria
from datetime import datetime, timedelta


mariadb = maria()

day_opened = datetime.today() - timedelta(days=1)

querystring = 'SELECT   symb_code '+ \
              'FROM     krx_symbols '+ \
              'WHERE    symb_type_code = %s AND '+ \
                       'day_opened >= %s AND '+ \
                       'symb_code NOT IN (SELECT symb_code FROM dart_corp_info)'
values = ['0', day_opened.strftime('%Y-%m-%d')]
df_symbols = mariadb.select(querystring, values)

if len(df_symbols) == 0:
    print('# No newly registered corporation.')
    exit()

data = []

for symb_code in df_symbols.values:
    symb_code = symb_code[0]
    url = 'http://dart.fss.or.kr/dsae001/search.ax'
    params = {'currentPage':'1',
              'maxResults':'45',
              'maxLinks' : '10',
              'autoSearch' : 'true',
              'textCrpNm' : symb_code,
              'typesOfBusiness' : 'all',
              'corporationType' : 'all'
              }
    response = requests.post(url, params=params)
    corp_code = re.findall('CikCDvalue=(\w{8})', re.sub('[ \t\r\n\v\f]|\'|\"', '', response.text))

    if len(corp_code) == 0:
        continue

    corp_code = corp_code[0]

    url = 'http://dart.fss.or.kr/dsae001/selectPopup.ax'
    params = {'selectKey' : corp_code}

    # corporationType: P(유가증권, KOSPI), A(KOSDAQ), N(KONEX), E(기타법인)
    response = requests.post(url, params=params)
    soup = BeautifulSoup(response.content, 'lxml')
    trs = soup.find_all('tr')

    if len(trs) == 0:
        continue

    data.append([re.findall('scrapFeed\((\w{8})\)', re.sub('[\t\r\n\v\f]|\'|\"', '', str(trs[0].td)))[0],
                 re.sub('[\t\r\n\v\f]|\'|\"', '', trs[0].td.text),
                 re.sub('[\t\r\n\v\f]|\'|\"', '', trs[1].td.text),
                 symb_code,
                 re.sub('[\t\r\n\v\f]|\'|\"', '', trs[5].td.text),
                 re.sub('[\t\r\n\v\f]|\'|\"', '', trs[13].td.text)]
                )

## End for loop

df_corpinfo = pd.DataFrame(data, columns=['corp_code',
                                          'corp_name',
                                          'corp_name_en',
                                          'symb_code',
                                          'market_type_code',
                                          'corp_type_name'])
df_corpinfo['market_type_code'] = df_corpinfo['market_type_code'].apply(
    lambda x: 'E' if x == '기타법인' else (
                 'I' if x == '유가증권시장' else (
                     'Q' if x == '코스닥시장' else (
                         'X' if x == '코넥스시장' else '')))
)

mariadb.insert("dart_corp_info", df_corpinfo)

print('# %s of new corporation information was saved.\n' % '{:,}'.format(len(df_corpinfo)))
print(df_corpinfo)

