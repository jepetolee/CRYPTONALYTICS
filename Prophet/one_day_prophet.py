import talib.abstract as ta
from talib import MA_Type
from neuralprophet import NeuralProphet
from numpy import log
from pandas import DataFrame

from calcul import cal_varience_percent
from make_pd import future_symbol_daily_data


def FutureOneDayVariancePercent():
    array = []
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
    for _i in range(len(symbol)):

        data = future_symbol_daily_data(symbol[_i])
        if data is not None:
            data = cal_varience_percent(data['4'].to_numpy())

            array.append(data)
    return array


def FutureOneDayData():
    array = []
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
    for _i in range(len(symbol)):

        data = future_symbol_daily_data(symbol[_i])
        if data is not None:
            array.append(data.to_numpy())
    return array


def FutureOneDaylog():
    arrays = []
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
    for _i in range(len(symbol)):
        data = future_symbol_daily_data(symbol[_i])
        if data is not None:
            data = log(data['4'].to_numpy())
            arrays.append(data)
    return arrays


def FutureProphetOneDay():
    arrays = []
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
    for _i in range(len(symbol)):
        data = future_symbol_daily_data(symbol[_i])
        if data is not None:
            prophet = NeuralProphet(seasonality_mode='multiplicative', n_forecasts=14, epochs=300, n_changepoints=200)
            data = DataFrame({'ds': data.index, 'y': data['4'].copy()})

            prophet.fit(data, validation_df=data, freq="D")
            future = prophet.make_future_dataframe(data, periods=14)
            df = prophet._prepare_dataframe_to_predict(future)
            dates, predicted, components = prophet._predict_raw(df, include_components=True)
            arrays.append(predicted[-14])
    return arrays


def FutureOneDayRsi():
    arrays = []
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
    for _i in range(len(symbol)):
        data = future_symbol_daily_data(symbol[_i])
        if data is not None:
            data = ta._ta_lib.RSI(data['4'].to_numpy(), 14)

            arrays.append(data)
    return arrays


def FutureOneDayMacd():
    upper, middles = list(), list()
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
    for _i in range(len(symbol)):
        data = future_symbol_daily_data(symbol[_i])
        if data is not None:
            up = ta._ta_lib.SMA(data['4'].to_numpy(), 3)
            middle = ta._ta_lib.SMA(data['4'].to_numpy(), 14)
            upper.append(up)
            middles.append(middle)

    return upper, middles


def FutureOneDayBBands():
    upper, middles, lows = list(), list(), list()
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
    for _i in range(len(symbol)):
        data = future_symbol_daily_data(symbol[_i])
        if data is not None:
            up, middle, low = ta._ta_lib.BBANDS(data['4'].to_numpy(), 14, 2, 2, matype=MA_Type.SMA)
            upper.append(up)
            middles.append(middle)
            lows.append(low)
    return upper, middles, lows

