from binance.client import Client
from TradeAlgorithm import make_symbol_list
from calcul import cal_varience_percent
from make_pd import *


def MarketFiveMinuteVariancePercent():
    client = Client(api_key="", api_secret="")
    tickers = client.get_all_tickers()
    symbol = make_symbol_list(tickers)
    array = []
    symbolic = []
    for i in range(len(symbol)):
        data = market_symbol_5min_data(symbol[i])
        if data is not None:
            data = cal_varience_percent(data['2'].to_numpy())
            array.append(data)
            symbolic.append(symbol[i])
    return array, symbolic


def ProphetFiveMinute():
    print("1분봉 예측값은 다음과 같습니다.")
