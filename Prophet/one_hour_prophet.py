from binance.client import Client
from neuralprophet import NeuralProphet
from numpy import log
from pandas import DataFrame

from TradeAlgorithm import make_symbol_list
from calcul import cal_varience_percent
from make_pd import market_symbol_1hour_data


def MarketOneHourVariancePercent():
    client = Client(api_key="", api_secret="")
    tickers = client.get_all_tickers()
    symbol = make_symbol_list(tickers)
    array = []
    symbolic = []
    for i in range(len(symbol)):
        data = market_symbol_1hour_data(symbol[i])
        if data is not None:
            data = cal_varience_percent(data['2'].to_numpy())
            array.append(data)
            symbolic.append(symbol[i])
    return array, symbolic


def ProphetOneHour():
    client = Client(api_key="", api_secret="")
    tickers = client.get_all_tickers()
    symbols = make_symbol_list(tickers)
    arrays = []
    symbolic = []
    for i in range(len(symbols)):
        data = market_symbol_1hour_data(symbols[i])
        if data is not None:
            prophet = NeuralProphet(daily_seasonality=True, n_forecasts=24, epochs=30, n_changepoints=600)
            data = DataFrame({'ds': data.index, 'y': data['2'].copy()})

            prophet.fit(data, validation_df=data, freq="H")
            future = prophet.make_future_dataframe(data, periods=24)
            df = prophet._prepare_dataframe_to_predict(future)
            dates, predicted, components = prophet._predict_raw(df, include_components=True)
            print(predicted[-24:])
            print(dates[-24:])
            arrays.append(predicted[-24])
            symbolic.append(symbols[i])
    return arrays, symbolic


import time

ts = time.time()
array, symbol = ProphetOneHour()
print(str(time.time() - ts) + "초")
for i in range(len(array)):
    if symbol[i] == "BTCUSDT":
        for j in range(len(array[i])):
            print(array[i][j])