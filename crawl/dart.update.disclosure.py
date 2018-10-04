import pandas as pd
import requests
import time
import sys
import re

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from databases import maria


###########################################################################
# 정(remark01) : 본 보고서 제출 후 정정신고가 있으니 관련 보고서를 참조하시기 바람
# 보고서 명: 공시구분+보고서명+기타정보
###########################################################################
# 01. (disc_type_code = 'A', rept_type_code = 'A001') : %사업보고서%
# 02. (disc_type_code = 'A', rept_type_code = 'A002') : %반기보고서%
# 03. (disc_type_code = 'A', rept_type_code = 'A003') : %분기보고서%
# 04. (disc_type_code = 'A', rept_type_code = 'A003') : %등록법인결산서류%
###########################################################################


mariadb = maria()

querystring = 'SELECT MAX(disc_date) disc_last_datetime FROM dart_disclosures'
df_result = mariadb.select(querystring)
disc_last_datetime = df_result.get_value(0, 'disc_last_datetime')

querystring = 'SELECT corp_code FROM dart_corp_info'
df_dartcorps = mariadb.select(querystring)

prevtoday = datetime.today() - timedelta(days=1)
daysbefore = (prevtoday - datetime.strptime(disc_last_datetime, '%Y-%m-%d %H:%M:%S')).days

url = 'http://dart.fss.or.kr/dsac001/mainAll.do'
# 1. Dates loop process
for i in range(daysbefore, -1, -1):
    transdate = prevtoday - timedelta(days=i)

    if transdate.weekday() in (5, 6):
        continue

    params = {'currentPage': '1',
              'mdayCnt': '0',
              'selectDate': '%s' % transdate.strftime('%Y.%m.%d')}
    headers = {'Cookie': 'DSAC001_MAXRESULTS=5000;'}
    response = requests.post(url, params=params, headers=headers, stream=True)

    soup = BeautifulSoup(response.content, 'lxml')
    pagenum_info = re.findall('\[([0-9]{1,2})/([0-9]{1,2})\] \[총 [0-9,]{1,5}건\]', response.text)

    # print('### pagenum_info : ', pagenum_info)

    trs = soup.find_all('tr')

    data = []
    is_first = True

    for tr in trs:
        if is_first:
            is_first = False
            continue

        tr_string = re.sub('[\t\r\n\v\f]|\'|\"', '', str(tr))
        disc_time = re.findall('\d{2}:\d{2}', tr_string)
        corp_code = re.findall('openCorpInfo\((\w{8})\)', tr_string)
        rept_numb = re.findall('openReportViewer\((\w{14})\)', tr_string)
        rept_name = re.sub('[ \t\r\n\v\f]|\'|\"', '', tr.find_all('td')[2].a.text)
        rept_smtd = re.sub('[\t\r\n\v\f]|\'|\"', '', tr.find_all('td')[3].text)
        rept_updt = tr_string.find('remark01')

        disc_time = disc_time[0] + ':00' if len(disc_time) else ''
        corp_code = corp_code[0] if len(corp_code) else ''
        rept_numb = rept_numb[0] if len(rept_numb) else ''
        rept_updt = 'Y' if rept_updt > 0 else 'N'

        data.append(['%s %s' % (transdate.strftime('%Y-%m-%d'), disc_time),
                     rept_numb,
                     rept_name,
                     corp_code,
                     rept_updt,
                     rept_smtd])

    ## End for loop, for tr in trs:

    df_disclosures = pd.DataFrame(data, columns=['disc_date',
                                                 'disc_no',
                                                 'disc_name',
                                                 'corp_code',
                                                 'rept_updated',
                                                 'submitted_by'])
    df_disclosures = df_disclosures[df_disclosures['disc_date'] > disc_last_datetime]
    df_disclosures = pd.merge(df_dartcorps, df_disclosures, on='corp_code')

    if len(df_disclosures) == 0:
        print('# Date of %s, No disclosure to save.' % transdate.strftime('%Y-%m-%d'))
        continue

    mariadb.insert("dart_disclosures", df_disclosures)

    print('# Date of %s, %s of the disclosures are stored successfully.' % (
        transdate.strftime('%Y-%m-%d'), '{:,}'.format(len(df_disclosures))))
    # print(df_disclosures)

    time.sleep(0.2)

## End for loop, for i in range(daysbefore, -1, -1):
