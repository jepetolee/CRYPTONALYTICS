import pandas as pd
from binance.client import Client
import trace_current_price as tp
from tqdm import trange
from multiprocessing import Process, Queue


def build_1day_csv():
    client = Client(api_key="", api_secret="")
    tickers = client.get_all_tickers()
    symbol = tp.make_symbol_list(tickers)
    for i in trange(len(symbol)):
        day = pd.DataFrame(
            client.get_historical_klines(symbol=symbol[i], interval='1d', start_str="2017-01-01", limit=1000))
        day.to_csv('./csv/1day/daily_data_' + symbol[i] + '.csv')


def build_1hour_csv():
    client = Client(api_key="", api_secret="")
    tickers = client.get_all_tickers()
    symbol = tp.make_symbol_list(tickers)
    for i in trange(len(symbol)):
        day = pd.DataFrame(
            client.get_historical_klines(symbol=symbol[i], interval='1h', start_str="2017-01-01", limit=1000))
        day.to_csv('./csv/1hour/1hour_data_' + symbol[i] + '.csv')


def build_4hour_csv():
    client = Client(api_key="", api_secret="")
    tickers = client.get_all_tickers()
    symbol = tp.make_symbol_list(tickers)
    for i in trange(len(symbol)):
        day = pd.DataFrame(
            client.get_historical_klines(symbol=symbol[i], interval='4h', start_str="2017-01-01", limit=1000))
        day.to_csv('./csv/4hour/4hour_data_' + symbol[i] + '.csv')


def build_package():
    if __name__ == "__main__":
        th1 = Process(target=build_1day_csv)
        th2 = Process(target=build_4hour_csv)
        th3 = Process(target=build_1hour_csv)
        th1.start()
        th2.start()
        th3.start()

        th1.join()
        th2.join()
        th3.join()
        print("FINISHED BUILDING DATA SETS NEED TO CHECK DATAS")


if __name__ == "__main__":
    build_package()
