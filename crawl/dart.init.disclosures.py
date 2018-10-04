import pandas as pd
import requests
import time
import sys
import re

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
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

# Get available transaction date
querystring = 'SELECT trans_date FROM krx_trans_daily GROUP BY trans_date'
df_avatrans = mariadb.select(querystring)

## Start loop with dates...
datetoday = datetime.today()
daysbefore = (datetoday - datetime(2001, 1, 2)).days

url = 'http://dart.fss.or.kr/dsac001/mainAll.do'
# 1. Dates loop process
for i in range(daysbefore, -1, -1):
    transdate = datetoday - timedelta(days=i)

    if transdate.weekday() in (5, 6) or len(df_avatrans[df_avatrans['trans_date']==transdate.strftime('%Y-%m-%d')]) == 0:
        continue

    printProgress(daysbefore-i, daysbefore, '# Progress :', ' Date of %s is processing...\n' % transdate.strftime('%Y-%m-%d'), 2, 50)

    params = {'currentPage': '1',
              'mdayCnt' : '0',
              'selectDate' : '%s' % transdate.strftime('%Y.%m.%d')}
    headers = {'Cookie':'DSAC001_MAXRESULTS=5000;'}
    response = requests.post(url, params=params, headers=headers, stream=True)
    
    soup = BeautifulSoup(response.content, 'lxml')
    pagenum_info = re.findall('\[([0-9]{1,2})/([0-9]{1,2})\] \[총 [0-9,]{1,5}건\]', response.text)
    
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
        
        disc_time = disc_time[0]+':00' if len(disc_time) else ''
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
    mariadb.insert("dart_disclosures", df_disclosures)
    
    print('# Progress : Date of %s, %s of the disclosures are stored successfully.\n' % (
        transdate.strftime('%Y-%m-%d'), '{:,}'.format(len(df_disclosures))))

    time.sleep(0.2)
    
## End for loop, for i in range(daysbefore, -1, -1):

print('All the transactions data are processed successfully.')


