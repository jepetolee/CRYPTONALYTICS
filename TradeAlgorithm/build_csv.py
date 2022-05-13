import pandas as pd
from binance.client import Client
import trace_current_price as tp
from tqdm import trange
from multiprocessing import Process


def build_future_1day_csv():
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
        day = pd.DataFrame(
            client.futures_historical_klines(symbol=symbol[i], interval='1d', start_str="2022-02-01", limit=1000))
        day.to_csv('./csv/future/1day/daily_data_' + symbol[i] + '.csv')


def build_future_1hour_csv():
    client = Client(api_key="", api_secret="")
    symbol = ['BTCUSDT']
    for i in trange(len(symbol)):
        day = pd.DataFrame(
            client.futures_historical_klines(symbol=symbol[i], interval='1h', start_str="2019-01-01", limit=1000))
        day.to_csv('./csv/future/1hour/1hour_data_' + symbol[i] + '.csv')


def build_future_15min_csv():
    client = Client(api_key="", api_secret="")
    symbol = ['BTCUSDT']
    for i in trange(len(symbol)):
        day = pd.DataFrame(
            client.futures_historical_klines(symbol=symbol[i], interval='15m', start_str="2019-01-01", limit=1000))
        day.to_csv('./csv/future/15min/15min_data_' + symbol[i] + '.csv')


def build_future_5min_csv():
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
        day = pd.DataFrame(
            client.futures_historical_klines(symbol=symbol[i], interval='5m', start_str="2022-02-01", limit=5000))
        day.to_csv('./csv/future/5min/5min_data_' + symbol[i] + '.csv')


def build_future_1min_csv():
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
        day = pd.DataFrame(
            client.futures_historical_klines(symbol=symbol[i], interval='1m', start_str="2019-01-01", limit=1000))
        day.to_csv('./csv/future/1min/1min_data_' + symbol[i] + '.csv')


def build_future_4hour_csv():
    client = Client(api_key="", api_secret="")
    symbol = ['BTCUSDT']
    for i in trange(len(symbol)):
        day = pd.DataFrame(
            client.futures_historical_klines(symbol=symbol[i], interval='4h', start_str="2019-01-01", limit=1000))
        day.to_csv('./csv/future/4hour/4hour_data_' + symbol[i] + '.csv')


def build_future_package():
    if __name__ == "__main__":
        th1 = Process(target=build_future_4hour_csv)
        th3 = Process(target=build_future_1hour_csv)
        th5 = Process(target=build_future_15min_csv)
        th7 = Process(target=build_future_1min_csv)

        #th1.start()

        th3.start()

        th5.start()

        # th7.start()

        #th1.join()

        th3.join()

        th5.join()

        # th7.join()
        print("FINISHED BUILDING DATA SETS NEED TO CHECK DATAS")


if __name__ == "__main__":
    build_future_package()
    print("FNINSHED")
