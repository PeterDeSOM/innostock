import re
import sys
import time
import random
import pandas as pd
import numpy as np

from PyQt5.QtWidgets import *
from PyQt5.QAxContainer import *
from PyQt5.QtCore import *
from databases import maria

TR_REQ_TIME_INTERVAL = 0.50


class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()

        # Create the Kiwoom OpenAPI+ instance
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

        ### Set the signal(event) to the slot(execution function) ###
        # 1. The 'OnEventConnect' signal will be occurred and execute the '_on_event_connect' slot after the user's Login action
        self.OnEventConnect.connect(self._on_event_connect)
        # 2. The 'OnReceiveTrData' signal will be occurred and execute the '_on_receive_tr_data' slot after transaction data received
        self.OnReceiveTrData.connect(self._on_receive_tr_data)

        # Call login process(window) to make a connection to the Kiwoom OpenAPI+
        self.dynamicCall("CommConnect()")

        # Create event loop instance and run it to keep waiting for the user action 'logIn'
        self.eventloop_login = QEventLoop()
        self.eventloop_login.exec_()

        self.event_result = None
        self.retrievable = 0

    def _on_event_connect(self, err_code):
        self.event_result = None

        if err_code == 0: print('### Connected... ###\n')
        else: print('### Not connected... ###\n')

        self.eventloop_login.exit()

    def _on_receive_tr_data(self, screen_no, rqname, trcode, record_name, next, unused1, unused2, unused3, unused4):
        self.event_result = None
        self.retrievable = int('0' if next == '' else next)

        attr = [attr for attr in dir(self) if attr.endswith(rqname.split(':')[1].lower())]
        attrlen = len(attr)
        if attrlen == 0: print('Transaction process function for %s is not defined.' % rqname)
        elif attrlen > 1: print('Transaction process function for %s is duplicated.' % rqname)
        else:
            self._transaction_event_handler = getattr(self, attr[0])
            self._transaction_event_handler(screen_no, rqname, trcode, record_name, next, unused1, unused2, unused3, unused4)

        try: self.eventloop_tr.exit()
        except AttributeError: pass

    def _opt10015_transaction_process_daily_transaction_details(self, *args):
        rqname = args[1]
        trcode = args[2]

        data_cnt = self._get_repeat_cnt(trcode, rqname)
        for i in range(data_cnt):
            date = self._comm_get_data(trcode, "", rqname, i, "일자")
            print('Data: %s' % date)

    def _opt10081_transaction_process_daily_candlestick_chart(self, *args):
        rqname = args[1]
        trcode = args[2]

        data_cnt = self._get_repeat_cnt(trcode, rqname)
        for i in range(data_cnt):
            date = self._comm_get_data(trcode, "", rqname, i, "일자")
            open = self._comm_get_data(trcode, "", rqname, i, "시가")
            high = self._comm_get_data(trcode, "", rqname, i, "고가")
            low = self._comm_get_data(trcode, "", rqname, i, "저가")
            close = self._comm_get_data(trcode, "", rqname, i, "현재가")
            volume = self._comm_get_data(trcode, "", rqname, i, "거래량")
            print(date, open, high, low, close, volume)

    def _opt10086_transaction_process_daily_stock_price(self, *args):
        rqname = args[1]
        trcode = args[2]

        data_cnt = int(self._get_repeat_cnt(trcode, rqname))
        if not data_cnt:
            print('There is no retrieved data. Check the data existence.')

        else:
            df_result = pd.DataFrame()

            for i in range(data_cnt):
                trans_date = self._comm_get_data(trcode, '', rqname, i, '날짜')

                if trans_date < '20000101':
                    self.retrievable = 0
                    break

                data = {
                    'p_open' : [re.sub(r'[^0-9-.]|-{1}-', '', self._comm_get_data(trcode, '', rqname, i, '시가')).replace('--', '-')],
                    'p_hight' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '고가')).replace('--', '-')],
                    'p_low' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '저가')).replace('--', '-')],
                    'p_close' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '종가')).replace('--', '-')],
                    'p_diff' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '전일비')).replace('--', '-')],
                    'dod' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '등락률')).replace('--', '-')],
                    'volumn' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '거래량')).replace('--', '-')],
                    'p_total' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '금액(백만)')).replace('--', '-')],
                    'credit_diff' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '신용비')).replace('--', '-')],
                    'personal' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '개인')).replace('--', '-')],
                    'institution' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '기관')).replace('--', '-')],
                    'foreign_volumn' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '외인수량')).replace('--', '-')],
                    'foreign_amount' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '외국계')).replace('--', '-')],
                    'program' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '프로그램')).replace('--', '-')],
                    'foreign_diff' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '외인비')).replace('--', '-')],
                    'conclusion_strength' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '체결강도')).replace('--', '-')],
                    'foreign_possession' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '외인보유')).replace('--', '-')],
                    'foreign_proportion' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '외인비중')).replace('--', '-')],
                    'foreign_net_buying' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '외인순매수')).replace('--', '-')],
                    'institution_net_buying' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '기관순매수')).replace('--', '-')],
                    'personal_net_buying' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '개인순매수')).replace('--', '-')],
                    'credit_balance_rate' : [re.sub(r'[^0-9-.]', '', self._comm_get_data(trcode, '', rqname, i, '신용잔고율')).replace('--', '-')],
                    'trans_date' : [trans_date]
                }
                df_result = pd.concat([df_result, pd.DataFrame(data)], ignore_index=True)

            ## End of for loop, i in range(data_cnt)

            self.event_result = df_result

    def _get_repeat_cnt(self, trcode, rqname):
        ret = self.dynamicCall("GetRepeatCnt(QString, QString)", trcode, rqname)
        return ret

    def _comm_get_data(self, code, real_type, field_name, index, item_name):
        ret = self.dynamicCall("CommGetData(QString, QString, QString, int, QString", code, real_type, field_name, index, item_name)
        return ret.strip()

    def set_input_value(self, id, value):
        self.dynamicCall("SetInputValue(QString, QString)", id, value)

    def comm_rq_data(self, rqname, trcode, next, screen_no):
        self.dynamicCall("CommRqData(QString, QString, int, QString", '%s:%s' % (trcode.upper(), rqname.upper()), trcode, next, screen_no)
        self.eventloop_tr = QEventLoop()
        self.eventloop_tr.exec_()

    def get_branch_code(self):
        return [x.split('|') for x in self.dynamicCall("GetBranchCodeName()").split(';')]

    def get_code_list_by_market(self, market):
        code_list = self.dynamicCall("GetCodeListByMarket(QString)", market)
        code_list = code_list.split(';')
        return code_list[:-1]

    def get_master_code_name(self, code):
        code_name = self.dynamicCall("GetMasterCodeName(QString)", code)
        return code_name

    def get_random_sleep_time(self):
        return round(random.uniform(1.70, 4.90), 2)

### End class Kiwoom


def init_symbols(kiwoom):
    df_symbols = pd.DataFrame(kiwoom.get_branch_code(), columns=['symb_code', 'symb_name'])

    mariadb = maria()
    mariadb.insert("kiwoom_branch_code", df_symbols)

    print('Symbol codes are initialized successfully.')


def init_branches(kiwoom):
    df_branchcode = pd.DataFrame(kiwoom.get_branch_code(), columns=['branch_code', 'branch_name'])

    mariadb = maria()
    mariadb.insert("kiwoom_branch_code", df_branchcode)

    print('Branch codes are initialized successfully.')


def init_daily_transaction_detail(kiwoom):
    querystring = 'SELECT   symb_code, IF(symb_code = %s, %s, %s) trans_date ' +\
                  'FROM     innostock.trans_daily ' +\
                  'WHERE    symb_code >= %s ' +\
                  'GROUP BY symb_code'
    values = ['000100', '20151119', '20170929', '000100']
    mariadb = maria()
    reslut = mariadb.select(querystring, values)

    for symb_code, trans_date in reslut.values:
        print('\n### Start to retrieve data for %s.' % symb_code)
        while True:
            # 일별주가요청
            kiwoom.set_input_value("종목코드", symb_code)
            kiwoom.set_input_value("조회일자", trans_date)
            kiwoom.set_input_value("표시구분", "1")
            kiwoom.comm_rq_data("DAILY_STOCK_PRICE", "opt10086", kiwoom.retrievable, "0101")

            df_result = pd.DataFrame(kiwoom.event_result)
            if len(df_result):
                df_result.fillna('')
                df_result = df_result.assign(**{'symb_code': np.full(len(df_result), symb_code)})
                df_result['trans_date'] = df_result['trans_date'].apply(lambda x: '%s-%s-%s 23:59:59' % (x[0:4], x[4:6], x[6:8]))

                mariadb = maria()
                mariadb.insert("kiwoom_daily_stock_price", df_result)

                print('(%s ~ %s) datas are processed.' % (
                    df_result.get_value(0, 'trans_date'),
                    df_result.get_value(len(df_result)-1, 'trans_date')
                ))

            time.sleep(kiwoom.get_random_sleep_time())

            if not kiwoom.retrievable:
                break

        ## End of While loop, while True:
    ## End of for loop, for symb_code in reslut.values:

    print('### End of init_daily_stock_price(kiwoom).')


def init_daily_candlestick_chart(kiwoom):
    while True:
        # 종목별일봉차트요청
        # kiwoom.set_input_value("종목코드", "039490")
        # kiwoom.set_input_value("기준일자", "20170929")
        # kiwoom.set_input_value("수정주가구분", 1)
        # kiwoom.comm_rq_data("DAILY_CANDLESTICK_CHART", "opt10081", kiwoom.retrievable, "0101")

        if not kiwoom.retrievable:
            break
        else:
            time.sleep(kiwoom.get_random_sleep_time())

def init_daily_stock_price(kiwoom):
    querystring = 'SELECT   symb_code, IF(symb_code = %s, %s, %s) trans_date ' +\
                  'FROM     innostock.trans_daily ' +\
                  'WHERE    symb_code >= %s ' +\
                  'GROUP BY symb_code LIMIT 3'
    values = ['000210', '20091005', '20170929', '000210']
    mariadb = maria()
    reslut = mariadb.select(querystring, values)

    for symb_code, trans_date in reslut.values:
        print('\n### Start to retrieve data for %s.' % symb_code)
        while True:
            # 일별주가요청
            kiwoom.set_input_value("종목코드", symb_code)
            kiwoom.set_input_value("조회일자", trans_date)
            kiwoom.set_input_value("표시구분", "1")
            kiwoom.comm_rq_data("DAILY_STOCK_PRICE", "opt10086", kiwoom.retrievable, "0101")

            df_result = pd.DataFrame(kiwoom.event_result)
            if len(df_result):
                df_result.fillna('')
                df_result = df_result.assign(**{'symb_code': np.full(len(df_result), symb_code)})
                df_result['trans_date'] = df_result['trans_date'].apply(lambda x: '%s-%s-%s 23:59:59' % (x[0:4], x[4:6], x[6:8]))

                mariadb = maria()
                mariadb.insert("kiwoom_daily_stock_price", df_result)

                print('(%s ~ %s) datas are processed.' % (
                    df_result.get_value(0, 'trans_date'),
                    df_result.get_value(len(df_result)-1, 'trans_date')
                ))

            time.sleep(kiwoom.get_random_sleep_time())

            if not kiwoom.retrievable:
                break

        ## End of While loop, while True:
    ## End of for loop, for symb_code in reslut.values:

    print('### End of init_daily_stock_price(kiwoom).')


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # init_branches(Kiwoom())
    # init_daily_transaction_detail(Kiwoom())
    # init_daily_stock_price(Kiwoom())
