import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from keras.models import model_from_json
from lstm_bunch.dataset import PredictSource, PredictController

from databases import maria

_MODEL_WEIGHT_DIR_ = 'graduated/appling/'


class Tomorrow:
    def __init__(self):
        self.STACK_REPORT_HOLDER = []

        self.AGENTS = []

        self.day_depth = 3
        self.input_length = 5
        self.action_size = 7
        self.threads = 9

        self._SOURCE_ = PredictSource(self.input_length)
        self.state_size = self._SOURCE_.predictable_col_len()

        print('##### PREDICTABLE DATA SCALE: %s, APPLIED LAYER WIDTH: %s' % (
            self.state_size,
            '{:,}'.format(self.state_size * self.input_length * 2)
        ))

        self.actor = self.load_model()
        self.load_weights()

    def load_model(self):
        # load json and create model
        json_file = open('%s_MODEL_ACTOR_.json' % _MODEL_WEIGHT_DIR_, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        actor = model_from_json(loaded_model_json)

        actor._make_predict_function()
        return actor

    def load_weights(self):
        model_actor_located = '%s_MODEL_W_ACTOR_.h5' % _MODEL_WEIGHT_DIR_
        self.actor.load_weights(model_actor_located)

    def predict(self):
        # Build flat transaction data to be preditable data (state)
        connection = maria()
        connection.callproc('proc_TESTING_DATA_UPDATE', [self.input_length, self.day_depth])

        # predict_mode -----------------------------------------------------
        # 0              : Targets of the predicting are the all the testing data stored in 'drl_1d_testing' (default)
        # 1              : Only the data for the tomorrow's prediction (last date of transaction will be applied in prediction)
        # 2, 3, 4, ... n : The days of the data to be targeted for prediction
        predict_length = 0
        # allow_repredict --------------------------------------------------
        # True  : Both of predicted and unpredicted data will be applied in prediction (default)
        # False : Only the unpredicted data will be applied in prediction
        allow_repredict = True

        if not self._SOURCE_.is_predictable(allow_repredict, predict_length):
            print('##### There is no predictable data. #####')
            exit()

        dataset = self._SOURCE_.get_dataset(allow_repredict, predict_length)

        self.AGENTS = [Agent(self, dataset) for _ in range(self.threads)]
        for agent in self.AGENTS:
            agent.start()

        while True:
            agnet_state = []
            for agent in self.AGENTS: agnet_state.append(int(agent.stopped))
            if sum(agnet_state) == self.threads: break

            time.sleep(1)

        for agent in self.AGENTS:
            agent.stop()

        self._report()

    def _report(self):
        df_reports = pd.DataFrame({})

        for df_report in self.STACK_REPORT_HOLDER:
            if len(df_reports) == 0:
                df_reports = df_report
            else:
                df_reports[df_reports.columns[2:]] += df_report[df_report.columns[2:]]

        df_reports[df_reports.columns[2:]] = np.round((df_reports[df_reports.columns[2:]].astype('float') * 100.) / len(self.STACK_REPORT_HOLDER), 2)
        df_reports['predicted_strength'] = self.get_strength_name(df_reports[df_reports.columns[2:]])
        df_reports['predict_value'] = self.get_predict_value(df_reports[df_reports.columns[2:-1]])
        df_reports['target_date'] = self.get_target_date(df_reports['trans_date'])

        self._SOURCE_.update_predictables(df_reports[['isin', 'trans_date', 'predict_value']])
        df_symbols = self._SOURCE_.get_symbols_header()
        df_reports = pd.merge(df_symbols, df_reports, on=['isin'])
        df_reports = df_reports[['symb_name', 'symb_code', 'trans_date', 'target_date',
                                 'Less than -7.0%', '-7.0% ~ -3.0%', '-3.0% ~ -1.0%', '-1.0% ~ 1.0%', '1.0% ~ 3.0%', '3.0% ~ 7.0%', '7.0% and over',
                                 'predicted_strength']]

        df_reports['Less than -7.0%'] = df_reports['Less than -7.0%'].apply(lambda x: '{:.2f}'.format(x))
        df_reports['-7.0% ~ -3.0%'] = df_reports['-7.0% ~ -3.0%'].apply(lambda x: '{:.2f}'.format(x))
        df_reports['-3.0% ~ -1.0%'] = df_reports['-3.0% ~ -1.0%'].apply(lambda x: '{:.2f}'.format(x))
        df_reports['-1.0% ~ 1.0%'] = df_reports['-1.0% ~ 1.0%'].apply(lambda x: '{:.2f}'.format(x))
        df_reports['1.0% ~ 3.0%'] = df_reports['1.0% ~ 3.0%'].apply(lambda x: '{:.2f}'.format(x))
        df_reports['3.0% ~ 7.0%'] = df_reports['3.0% ~ 7.0%'].apply(lambda x: '{:.2f}'.format(x))
        df_reports['7.0% and over'] = df_reports['7.0% and over'].apply(lambda x: '{:.2f}'.format(x))

        _PREDICTION_REPORT_DIR_ = 'report/'
        df_reports.to_csv('%s%s.csv' % (_PREDICTION_REPORT_DIR_, datetime.today().strftime('%Y%m%d%H%M%S')), float_format='%.2f')

    def get_predict_value(self, predicted_values):
        values_ = []

        for values in predicted_values.values:
            values_.append(np.argmax(values))

        return values_

    def get_target_date(self, date):
        target_dates = []

        for trans_date in date:
            stop = False
            increase = 1

            while not stop:
                target_date = datetime.strptime(trans_date, '%Y-%m-%d') + timedelta(days=increase)

                if target_date.weekday() not in (5, 6):
                    target_dates.append('%s or next working day' % target_date.strftime('%Y-%m-%d'))
                    stop = True

                increase += 1

        return target_dates

    def get_strength_name(self, predicted_values):
        strength_names = []

        for values in predicted_values.values:
            value = np.max(values)

            if value < 80.: strength_name = 'Very weak'
            elif value >= 80. and value < 85.: strength_name = 'Weak'
            elif value >= 85. and value < 90.: strength_name = 'Good'
            elif value >= 90. and value < 95.: strength_name = 'Strong'
            else: strength_name = 'Very strong'

            strength_names.append(strength_name)

        return strength_names


class Agent(threading.Thread):
    def __init__(self, creator, dataset):
        threading.Thread.__init__(self)

        self._CONTROLLER_ = PredictController(dataset, creator.state_size, creator.input_length, creator.action_size)
        self._CREATOR_ = creator

        self.state_size = creator.state_size
        self.input_length = creator.input_length
        self.action_size = creator.action_size

        self.actor = creator.actor
        self.local_actor = self._CREATOR_.load_model()
        self.local_actor.set_weights(self.actor.get_weights())

        self.stopped = False
        self._stop_event = threading.Event()

    def run(self):
        print('##### RUNNING : ', self)

        report_data = []

        while self._CONTROLLER_.next() >= 0:
            while True:
                state, state_HEADER = self._CONTROLLER_.next_predictable()
                if type(state) is int:
                    break

                action_stochastic_policy = self.get_action(state)
                report_data.append(sum([list(state_HEADER), list(action_stochastic_policy[0])], []))

        df = pd.DataFrame(data=report_data, columns=[
            'isin', 'trans_date',
            'Less than -7.0%', '-7.0% ~ -3.0%', '-3.0% ~ -1.0%', '-1.0% ~ 1.0%', '1.0% ~ 3.0%', '3.0% ~ 7.0%', '7.0% and over'
        ])

        self._CREATOR_.STACK_REPORT_HOLDER.append(df)
        self.stopped = True

    def get_action(self, state):
        action_probability = self.local_actor.predict(state.reshape((1, self.input_length, self.state_size)))
        return action_probability

    def stop(self):
        self._stop_event.set()

        print('##### TERMINATING : %s' % self)

        try:
            self._stop()
        except:
            pass

    def is_stopped(self):
        return self._stop_event.is_set()

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        if e.ndim == 1:
            return e / np.sum(e, axis=0)
        else:
            return e / np.array([np.sum(e, axis=1)]).T

    def onehot(self, x):
        x = x.reshape(-1)
        return np.eye(len(x))[np.argmax(x)]


def seperator():
    print('-' * 150)

if __name__ == "__main__":
    print('##### START TRAINING MODEL.')
    agent = Tomorrow()
    agent.predict()
    print('##### ALL THE TRAINING PROCESS IS FINISHED.')
