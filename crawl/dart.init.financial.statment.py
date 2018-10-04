import requests
import re

__REPORT_VIEWR_MAIN_URL = 'http://dart.fss.or.kr/dsaf001/main.do'
__REPORT_VIEWR_PAGE_URL = 'http://dart.fss.or.kr/report/viewer.do'
__REPORT_VIEWR_MAIN_PARAMS = {'rcpNo' : '20171011000196'}
response = requests.get(__REPORT_VIEWR_MAIN_URL, params=__REPORT_VIEWR_MAIN_PARAMS)

page_object = re.sub('\s|["\'\{\};]', '', response.text).replace('click:function()', '')
page_object = re.findall('text:[0-9]\.[가-힣]{0,2}재무제표[ 등]{0,2}[a-zA-Z0-9,:\(.]+\)', page_object)

if len(page_object) == 0:
    print('No report\'s property')

if len(page_object) > 2:
    print('Not expected property values, many values are extracted.')

for property_string in page_object:
    protperty_values = re.findall('\(\w{14},([\w,.]+)\)', property_string)[0].split(',')

    if len(protperty_values) != 5:
        print('Not expected property values are not corresponded.')
        continue

    protperty_values.append('연결' if property_string.find('연결') > 0 else '별도')

    params = {'rcpNo' : '20171011000196',
              'dcmNo' : '%s' % protperty_values[0],
              'eleId' : '%s' % protperty_values[1],
              'offset' : '%s' % protperty_values[2],
              'length' : '%s' % protperty_values[3],
              'dtd' : '%s' % protperty_values[4]}
    response = requests.get(__REPORT_VIEWR_PAGE_URL, params=params)

    print(response.text)
