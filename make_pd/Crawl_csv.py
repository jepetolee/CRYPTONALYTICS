'''
csv 값 수집, 해당 csv 파일에 로그 차트 값 추가
'''

import pandas as pd


def future_symbol_daily_data(symbol):
    url = 'D:/CRYPTONALYTICS/TradeAlgorithm/csv/future/1day/daily_data_' + symbol + ".csv"
    try:
        data = pd.read_csv(url)
        data = data.rename(columns=data['0'])
        data_time = data['0'].copy().to_numpy().astype('datetime64[ms]')
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
        data = data[['1', '2', '3', '4', '5']]

        data = data.set_index(data_time)

        return data
    except FileNotFoundError:
        return


def future_symbol_1hour_data(symbol):
    url = 'D:/CRYPTONALYTICS/TradeAlgorithm/csv/future/1hour/1hour_data_' + symbol + ".csv"
    try:
        data = pd.read_csv(url)
        data_time = data['0'].copy().to_numpy().astype('datetime64[ms]')
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
        data = data[['1', '2', '3', '4', '5']]

        data = data.set_index(data_time)
        return data
    except FileNotFoundError:
        return


def future_symbol_15min_data(symbol):
    url = 'D:/CRYPTONALYTICS/TradeAlgorithm/csv/future/15min/15min_data_' + symbol + ".csv"
    try:
        data = pd.read_csv(url)
        data_time = data['0'].copy().to_numpy().astype('datetime64[ms]')
        data.drop(['Unnamed: 0'], axis=1, inplace=True)
        data = data[['1', '2', '3', '4', '5']]

        data = data.set_index(data_time)
        return data
    except FileNotFoundError:
        return


def future_symbol_5min_data(symbol):
    url = 'D:/CRYPTONALYTICS/TradeAlgorithm/csv/future/5min/5min_data_' + symbol + ".csv"
    try:
        data = pd.read_csv(url)
        data = data.rename(columns=data['0'], index=data.iloc[0])
        data.drop('11', axis=1, inplace=True)
        return data
    except FileNotFoundError:
        return


def future_symbol_1min_data(symbol):
    url = 'D:/CRYPTONALYTICS/TradeAlgorithm/csv/future/1min/1min_data_' + symbol + ".csv"
    try:
        data = pd.read_csv(url)

        data = data.rename(columns=data['0'], index=data.iloc[0])

        data.drop('11', axis=1, inplace=True)
        return data
    except FileNotFoundError:
        return