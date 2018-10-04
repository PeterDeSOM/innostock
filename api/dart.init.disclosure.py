import pandas as pd
import numpy as np
import time
import sys
import io

from datetime import datetime, timedelta
from databases import maria
from urllib.request import urlopen
from bs4 import BeautifulSoup
from time import sleep


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


__DART_URL = 'http://dart.fss.or.kr/api/search.xml'
__DART_KEY = '349cd46358442c7a0d9ec05b9334bad126aedc07'
__DART_DISCLOSURE_START_DATE = '20010102'  # (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')
__DART_REQUEST_URL_PAST_DISCLOSURES = "%s?auth=%s&start_dt=%s&crp_cd=%s&page_set=%s&page_no=%d&sort=%s&series=%s"

querystring = 'SELECT * FROM (' +\
                    'SELECT (@row_number:=@row_number+1) r_num, ' +\
                            'isin, ' +\
                            'symb_code, ' +\
                            'IF(isin = %s, 3, -1) total_pages, ' +\
                            'IF(isin = %s, 3, 1) page_no ' +\
                    'FROM   krx_symbols, (SELECT @row_number:=-1) r ' +\
                    'WHERE  isin NOT IN (SELECT isin FROM dart_disclosures GROUP BY isin) OR ' +\
                           'isin = %s) k'
wherelements = ['KR7089980007', 'KR7089980007', 'KR7089980007']
mariadb = maria()
result = mariadb.select(querystring, wherelements)

# 1. result['symb_code'].apply(lambda x: x)
# for i in range(0, rows-1):
#   2. result.get_value(i, 'symb_code')
#   3. result.loc[[i], ['symb_code'].index.values]
# 4. Finally, "for symbcode in result['symb_code'].values:"

stop = False
proc_total = len(result)

for i, isin, symbcode, totalpages, pageno in result.values:
    printProgress(i, proc_total, '# Progress :', ' (%s) Disclosures are processing.\n' % isin, 2, 50)

    if stop:
        break

    sleep(0.12)

    df_disclog = disclog = None
    while (pageno != totalpages + 1):
        url = __DART_REQUEST_URL_PAST_DISCLOSURES % (__DART_URL,
                                                     __DART_KEY,
                                                     __DART_DISCLOSURE_START_DATE,
                                                     symbcode,
                                                     '100',
                                                     pageno,
                                                     'date',
                                                     'asc')
        http_received = urlopen(url)
        bytes_read = http_received.read()
        xml_result = BeautifulSoup(bytes_read, 'html.parser', from_encoding='utf-8')

        # Never use '.string' like 'errcode.string' to get value, it makes most terrible 'bs4.element.navigablestring' problem.
        # Instead, must be used '.get_text()' like 'errcode.get_text()', it returns pure text/string
        errcode = xml_result.find('err_code').get_text()
        if not errcode in ('000', '100'):
            print('# (%s) Error code %s, %s, Page no: %d' % (
            isin, errcode, xml_result.find('err_msg').get_text(), pageno))
            stop = True
            break

        totalpages = int(xml_result.find('total_page').get_text())
        if not totalpages:
            # print("# (%s)%s, No disclosure information to the page number %d. #" % (isin, symbname, pageno))
            break

        te = xml_result.findAll("list")

        data = []
        for t in te:
            data.append([t.crp_cls.get_text(),
                         t.crp_nm.get_text(),
                         t.crp_cd.get_text(),
                         t.rpt_nm.get_text(),
                         t.rcp_no.get_text(),
                         t.flr_nm.get_text(),
                         t.rcp_dt.get_text(),
                         t.rmk.get_text()]
                        )
        ## End for loop

        df_disclosures = pd.DataFrame(data, columns=['market_type',
                                                     'symb_name',
                                                     'symb_code',
                                                     'disc_name',
                                                     'disc_no',
                                                     'submitted_by',
                                                     'disc_date',
                                                     'cmt_code']
                                      )
        df_disclosures = pd.merge(result, df_disclosures, on='symb_code')
        # continues, all the rows are not corresponded with symbols...
        if len(df_disclosures) == 0:
            continue

        df_disclosures['disc_date'] = df_disclosures['disc_date'].apply(lambda x: '%s-%s-%s 00:00:00' % (x[0:4], x[4:6], x[6:8]))
        df_disclosures = df_disclosures.drop(['r_num', 'symb_code', 'market_type', 'total_pages', 'page_no'], axis=1).fillna('')

        if mariadb.insert("dart_disclosures", df_disclosures) < 0:
           stop = True
           break

        disclog = "# (%s) disclosures on %s ~ %s, to %d page of %d pages, are processed. #" % (isin,
                                                                                               df_disclosures.get_value(
                                                                                                      0, 'disc_date'),
                                                                                               df_disclosures.get_value(
                                                                                                      len(
                                                                                                          df_disclosures) - 1,
                                                                                                      'disc_date'),
                                                                                               pageno,
                                                                                               totalpages)
        print(disclog)
        pageno += 1

        ## End While loop, (pageno != totalpages)
## End for loop, symbcode in result['symb_code'].values
