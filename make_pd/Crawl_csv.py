'''
csv 값 수집, 해당 csv 파일에 로그 차트 값 추가
'''

import pandas as pd
from TradeAlgorithm import make_symbol_list
from binance.client import Client


def market_symbol_daily_data(symbol):
    url = 'D:/CRYPTONALYTICS/TradeAlgorithm/csv/market/1day/daily_data_' + symbol + ".csv"
    try:
        data = pd.read_csv(url)
        return data
    except FileNotFoundError:
        print("오류:" + url + "을 찾을 수 없습니다!")
        return


def future_symbol_daily_data(symbol):
    url = 'D:/CRYPTONALYTICS/TradeAlgorithm/csv/future/1day/daily_data_' + symbol + ".csv"
    try:
        data = pd.read_csv(url)
        return data
    except FileNotFoundError:
        print("오류:" + url + "을 찾을 수 없습니다!")
        return
