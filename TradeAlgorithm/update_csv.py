import pandas as pd
from binance.client import Client
import TradeAlgorithm.trace_current_price as tp
from tqdm import trange
from multiprocessing import Process


def update_future_1hour_csv():
    client = Client(api_key="", api_secret="")
    symbol = ['BTCUSDT',
              'ETHUSDT',
              'BNBUSDT',
              'SOLUSDT',
              'XRPUSDT',
              'LUNAUSDT',
              'BCHUSDT',
              'WAVESUSDT',
              'TRXUSDT',
              'AVAXUSDT',
              'LTCUSDT',
              'NEARUSDT']
    for i in trange(len(symbol)):
        url = 'D:/CRYPTONALYTICS/TradeAlgorithm/csv/future/1hour/1hour_data_' + symbol[i] + ".csv"
        try:
            data = pd.read_csv(url)
            data_time = data['0'].copy().to_numpy().astype('datetime64[ms]')
            day = pd.DataFrame(
                client.futures_historical_klines(symbol=symbol[i], interval='1h', start_str=str(data_time[-1]),
                                                 limit=1000)).drop(index=0, axis=0)
            day.to_csv('./temp.csv')
            day = pd.read_csv('./temp.csv')
            data.drop(['Unnamed: 0'], axis=1, inplace=True)
            day.drop(['Unnamed: 0'], axis=1, inplace=True)
            data = pd.concat([data, day], ignore_index=True)
            data.to_csv(url)

        except FileNotFoundError:
            pass


def update_future_15min_csv():
    client = Client(api_key="", api_secret="")
    symbol = ['BTCUSDT',
              'ETHUSDT',
              'BNBUSDT',
              'SOLUSDT',
              'XRPUSDT',
              'LUNAUSDT',
              'BCHUSDT',
              'WAVESUSDT',
              'TRXUSDT',
              'AVAXUSDT',
              'LTCUSDT',
              'NEARUSDT']
    for i in trange(len(symbol)):
        url = 'D:/CRYPTONALYTICS/TradeAlgorithm/csv/future/15min/15min_data_' + symbol[i] + ".csv"
        try:
            data = pd.read_csv(url)
            data_time = data['0'].copy().to_numpy().astype('datetime64[ms]')
            day = pd.DataFrame(
                client.futures_historical_klines(symbol=symbol[i], interval='15m', start_str=str(data_time[-1]),
                                                 limit=1000)).drop(index=0, axis=0)
            day.to_csv('./temp.csv')
            day = pd.read_csv('./temp.csv')
            data.drop(['Unnamed: 0'], axis=1, inplace=True)
            day.drop(['Unnamed: 0'], axis=1, inplace=True)
            data = pd.concat([data, day], ignore_index=True)
            data.to_csv(url)

        except FileNotFoundError:
            pass


def update_future_1min_csv():
    client = Client(api_key="", api_secret="")
    symbol = ['BTCUSDT',
              'ETHUSDT',
              'BNBUSDT',
              'SOLUSDT',
              'XRPUSDT',
              'LUNAUSDT',
              'BCHUSDT',
              'WAVESUSDT',
              'TRXUSDT',
              'AVAXUSDT',
              'LTCUSDT',
              'NEARUSDT']
    for i in trange(len(symbol)):
        url = 'D:/CRYPTONALYTICS/TradeAlgorithm/csv/future/1min/1min_data_' + symbol[i] + ".csv"
        try:
            data = pd.read_csv(url)
            data_time = data['0'].copy().to_numpy().astype('datetime64[ms]')

            day = pd.DataFrame(
                client.futures_historical_klines(symbol=symbol[i], interval='1m', start_str=str(data_time[-1]),
                                                 limit=1000)).drop(index=0, axis=0)

            day.to_csv('./temp.csv')
            day = pd.read_csv('./temp.csv')
            data.drop(['Unnamed: 0'], axis=1, inplace=True)
            day.drop(['Unnamed: 0'], axis=1, inplace=True)
            data = pd.concat([data, day], ignore_index=True)

            data.to_csv(url)

        except FileNotFoundError:
            pass
