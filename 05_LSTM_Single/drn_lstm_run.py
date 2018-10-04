import pandas as pd
import numpy as np
import keras

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.utils import np_utils
from datetime import datetime
from databases import maria

class Source(object):
    def __init__(self):
        self._SOURCE_CONNECTION_ = maria()

        query_string = 'SELECT  A.isin isin, A.symb_name symb_name, 0 applied ' + \
                       'FROM    krx_symbols A INNER JOIN drl_1d B ON (A.isin = B.isin AND B.trans_date = B.trans_date) WHERE A.isin IN (%s, %s, %s) ' + \
                       'GROUP BY A.isin, A.symb_name ORDER BY symb_name'
        self._SYMBOLS_SOURCE_ = self._SOURCE_CONNECTION_.select(query_string, ['KR7084990001', 'KR7030000004', 'KR7027410000'])
        self._SYMBOLS_SIZE_ = len(self._SYMBOLS_SOURCE_)
        self._SYMBOL_INDEX_ = -1

    def _standardization(self, isin):
        data_cols = len(self._SYMBOL_DATASET_.columns)

        """
        stds_ = np.std(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols - 1]].astype('float'), axis=0)

        if len(stds_[stds_ == 0]) > 0:
            self._SYMBOL_DATASET_ = self._SYMBOL_DATASET_.drop(stds_[stds_ == 0].index, axis=1)
            data_cols = len(self._SYMBOL_DATASET_.columns)

        means = np.round(np.mean(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols - 1]].astype('float'), axis=0), 19)
        stds_ = np.round(np.std(self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols - 1]].astype('float'), axis=0), 19)

        self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols - 1]] = \
            (self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols - 1]].astype('float') - means) / stds_
        self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[data_cols - 1:data_cols]] = \
            self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[data_cols - 1:data_cols]].astype('int')
        """

        self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols - 1]] = \
            self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[2:data_cols - 1]].astype('float')
        self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[data_cols - 1:data_cols]] = \
            self._SYMBOL_DATASET_[self._SYMBOL_DATASET_.columns[data_cols - 1:data_cols]].astype('int')

        self.scale = data_cols - 3

    def next(self):
        self._SYMBOL_INDEX_ += 1

        if self._SYMBOL_INDEX_ == self._SYMBOLS_SIZE_:
            self._SYMBOL_INDEX_ = -1
            return pd.DataFrame()

        isin = self._SYMBOLS_SOURCE_.loc[self._SYMBOL_INDEX_, 'isin']

        query_string = 'SELECT  * ' + \
                       'FROM    drl_1d ' + \
                       'WHERE   isin = %s AND trans_date = trans_date ' + \
                       'ORDER BY trans_date'
        values = [isin]
        self._SYMBOL_DATASET_ = self._SOURCE_CONNECTION_.select(query_string, values)
        self._DATASET_SIZE_ = len(self._SYMBOL_DATASET_)
        self._standardization(isin)

        return self._SYMBOL_DATASET_

    def dataset_info(self):
        if self._SYMBOL_INDEX_ < 0:
            return [self._SYMBOL_INDEX_, '', '']

        return [
            self._SYMBOL_INDEX_,
            self._SYMBOLS_SOURCE_.loc[self._SYMBOL_INDEX_, 'isin'],
            self._SYMBOLS_SOURCE_.loc[self._SYMBOL_INDEX_, 'symb_name'],
            self.scale
        ]

    def source_size(self):
        return self._SYMBOLS_SIZE_

    def get_dateset(self):
        col_num = len(self._SYMBOL_DATASET_.columns)
        return self._SYMBOL_DATASET_, \
               self._SYMBOLS_SOURCE_.loc[self._SYMBOL_INDEX_, 'isin'], \
               len(self._SYMBOL_DATASET_), col_num-3, 7


class LossHistory(keras.callbacks.Callback):
    def init(self):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def parse_dataset(df_source, window_size):
    col_len = len(df_source.columns)
    df_target = df_source[df_source.columns[2:col_len]]
    df_target = df_target.reset_index(drop=True)

    dataset_X = []
    dataset_Y = []

    for i in range(len(df_target) - window_size):
        df_subset = df_target.iloc[i:(i + window_size), :]

        for subset in df_subset.values:
            dataset_X.append(subset[0:-1])

        dataset_Y.append([df_subset.loc[i + window_size - 1, 'target_value']])

    return np.vstack(dataset_X), np.vstack(dataset_Y)

def seperator():
    print('-' * 100)

def main():
    datasource = Source()
    source_size = datasource.source_size()

    input_dim = 41
    input_length = 5 # Window Size
    output_dim = 7

    model = Sequential()
    model.add(GRU(input_dim * input_length, input_shape=(input_length, input_dim), return_sequences=True))
    # model.add(GRU(input_dim * input_length * 2, return_sequences=True))
    model.add(GRU(input_dim))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = LossHistory()
    history.init()

    num_epochs = 200
    for epoch_idx in range(num_epochs):
	    while len(datasource.next()) > 0:
	        df_dataset, dataset_name, dataset_size, _, _ = datasource.get_dateset()
	        dataset_info = datasource.dataset_info()

	        seperator()
	        print('##### [EPOCHS : %s][ISIN: %s, SYMBOL NAME: %s, SCALE: %s] started.' % (
		        epoch_idx, dataset_info[1], dataset_info[2], dataset_info[3])
	        )
	        dataset_X, dataset_Y = parse_dataset(df_dataset, input_length)
	        dataset_X = np.reshape(dataset_X, (np.shape(dataset_Y)[0], input_length, np.shape(dataset_X)[1]))
	        dataset_Y = np_utils.to_categorical(dataset_Y, num_classes=output_dim)
	        model.fit(dataset_X, dataset_Y, epochs=1, batch_size=1, verbose=2, shuffle=False, callbacks=[history])

    print('##### All the trainings are finished. --------------------------------------------------------------')


if __name__ == "__main__":
    main()