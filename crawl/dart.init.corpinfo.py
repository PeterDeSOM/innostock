import pandas as pd
import requests
import time
import sys
import re

from urllib.request import urlopen
from bs4 import BeautifulSoup
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


url = 'http://dart.fss.or.kr/corp/searchCorpL.ax'

for i in range(8, 218):
    printProgress(i, 217, '# Progress :', 'processing page %s of 217.' % i, 2, 50)

    params = {'currentPage': '%s' % i}

    # corporationType: P(유가증권, KOSPI), A(KOSDAQ), N(KONEX), E(기타법인)
    response = requests.post(url, params=params)
    soup = BeautifulSoup(response.content, 'lxml')
    trs = soup.find_all('tr')

    data = []
    is_first = True

    for a in trs:
        if is_first:
            is_first = False
            continue

        corp_code = a.find('input', {'name': 'hiddenCikCD1'}).get('value')
        corp_name = a.find('input', {'name': 'hiddenCikNM1'}).get('value')
        market_type = a.find('img').get('alt')

        tr_string = re.sub('[\t\r\n\v\f]|\'|\"', '', str(a))
        symb_code = re.findall('<td style=padding:0 2px;text-align:center;>([A-Z0-9]{6})</td>', tr_string)
        corp_type_name = re.findall('종 :([\w.가-힣 ,;]*)>', tr_string)
        corp_name_both = re.findall('회  사  명 :(.*)영  문  명 :(.*)대표자', tr_string)

        data.append([corp_code,
                     corp_name,
                     corp_name_both[0][1] if len(corp_name_both) and len(corp_name_both[0]) > 1 else '',
                     corp_type_name[0] if len(corp_type_name) else '',
                     'E' if market_type == '기타법인' else (
                         'I' if market_type == '유가증권시장' else (
                             'Q' if market_type == '코스닥시장' else (
                                 'X' if market_type == '코넥스시장' else ''))),
                     symb_code[0] if len(symb_code) else ''])

    ## End for loop, for a in trs:

    df_corpinfo = pd.DataFrame(data, columns=['corp_code',
                                              'corp_name',
                                              'corp_name_en',
                                              'corp_type_name',
                                              'market_type_code',
                                              'symb_code'])
    mariadb = maria()
    mariadb.insert("dart_corp_info", df_corpinfo)

    print('\n# Progress : %s the corporation & symbol informations are stored successfully.' % '{:,}'.format(
        len(df_corpinfo)))

    time.sleep(0.15)

## End for loop, for i in range(1, 218):

print(df_corpinfo.head())


