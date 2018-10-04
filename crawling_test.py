import requests
import pandas as pd
import numpy as np

from databases import maria
import re

'''
url = 'http://marketdata.krx.co.kr/contents/COM/GenerateOTP.jspx'
params = {'bld' : 'MKD/04/0402/04020100/mkd04020100t3_02',
          'name' : 'chart',
          '_' : '1507444903702'}
code = requests.get(url, params=params).text

url = 'http://marketdata.krx.co.kr/contents/MKD/99/MKD99000001.jspx'
params = {'isu_cdnm':'F722012HB/삼성베스트제3호',
          'isu_cd':'KR7052560000',
          'isu_nm':'삼성베스트제3호',
          'isu_srt_cd':'F722012HB',
          'fromdate':'20070901',
          'todate':'20080831',
          'pagePath':'/contents/MKD/04/0402/04020100/MKD04020100T3T2.jsp',
          'chartNo':'70efdf2ec9b086079795c442636b55fb',
          'code' : '%s' % code}
url = 'http://kind.krx.co.kr/common/searchcorpname.do'
params = {'method':'searchCorpNameJson',
          'tabMenu':'0',
          'companyNM':'3SOFT',
          'searchCorpName':'3SOFT',
          'spotIsuTrdMktTpCd':'2',
          'comAttrTpCd':'1',
          'comAbbrv':'3SOFT'}
code = requests.get(url, params=params).text
print('code: ', code)

'''

url = 'http://kind.krx.co.kr/corpdetail/totalinfo.do'
params = {'method':'searchTotalInfo',
          'isurCd':'03773',
          'repIsuCd':'KR7037730009'}
response = requests.get(url, params=params)
dates = re.findall(r'\b[0-9]{4}-[0-9]{2}-[0-9]{2}', response.text)

print(lambda x: x[3:], '12345')



#listsymbs = response.json()['상장종목검색']
#print('%d data(s) crawled.\n...' % len(listsymbs))

#df_krxsymbols = pd.DataFrame()
#for attr in listsymbs:
#    print(attr)

'''
             isin  day_closed symb_code      symb_name         symb_name_long  \
0    KR7037730009  2004-04-21    037730             3R                    쓰리알   
2    KR7036360006  2009-04-28    036360          3SOFT                  쓰리소프트   
3    KYG887121070  2013-06-04    900010         3노드디지탈           3노드디지탈그룹유한공사   
4    KR7038120002  2013-01-29    038120          AD모터스                 에이디모터스   
5    KR7013340005  2014-09-01    013340            AJS                 에이제이에스   
7    KR7015670003  2007-09-11    015670         AP우주통신                 AP우주통신   
8    KR7015671001  2007-09-11    015675    AP우주통신(1우B)            AP우주통신(1우B)   
10   KR7036820009  2005-04-20    036820            BET                    비이티   
11   KR7036821007  2005-04-20    036825           BET우                비이티1우선주   
12   KR7003990009  2009-04-29    003990            BHK                 BHK보통주   
100  KR7038710000  2011-05-08    038710        BRN사이언스                비알엔사이언스   
103  KR7000790006  2009-05-12    000790           C&상선                씨앤상선보통주   
104  KR7013200001  2009-05-12    013200           C&우방                씨앤우방보통주   
105  KR7008400004  2009-05-12    008400          C&중공업               씨앤중공업보통주   
106  KR7008401002  2009-05-12    008405         C&중공업우              씨앤중공업1우선주   
112  KR7001042001  2009-03-19    001047          CJ2우B             CJ2우선주(신형)   
113  KR7001043009  2010-01-12    001049          CJ3우B             CJ3우선주(신형)   
139  KR7049370000  2006-04-19    049370         CJ엔터테인              씨제이엔터테인먼트   
140  KR7037150000  2011-03-21    037150          CJ인터넷                 씨제이인터넷   
141  KR7097952006  2009-03-19    097957     CJ제일제당 2우B  씨제이제일제당 주식회사 2우선주(신형)   
142  KR7097953004  2010-01-12    097959     CJ제일제당 3우B  씨제이제일제당 주식회사 3우선주(신형)   
144  KR7039720008  2001-02-05    039720         CJ프론티어           씨제이프론티어전환형펀드   
145  KR7035710003  2010-05-02    035710             CL                  씨엘엘씨디   
147  KR7050470004  2012-05-01    050470           CT&T                   씨티앤티   
148  KR7056340003  2014-05-20    056340           CU전자                   씨유전자   
150  KR7001370006  2009-08-16    001370         FnC코오롱              FnC코오롱보통주   
151  KR7001371004  2009-08-16    001375        FnC코오롱우             FnC코오롱1우선주   
153  KR7076170000  2009-04-21    076170            GBS                 GBS보통주   
160  KR7054020003  2010-04-04    054020           GK파워                  지케이파워   
161  KR7014040000  2002-11-28    014040            GPS                    GPS   
'''