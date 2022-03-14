import talib.abstract as ta
from talib import MA_Type
from neuralprophet import NeuralProphet
from numpy import log
from pandas import DataFrame

from calcul import cal_varience_percent
from make_pd import future_symbol_daily_data


def MarketOneDayVariancePercent():
    array = []
    symbolic = []
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
            data = cal_varience_percent(data['2'].to_numpy())
            array.append(data)
            symbolic.append(symbol[_i])
    return array, symbolic


def MarketOneDaylog():
    arrays = []
    symbolic = []
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
            data = log(data['2'].to_numpy())
            arrays.append(data)
            symbolic.append(symbol[_i])
    return arrays, symbolic


def ProphetOneDay():
    arrays = []
    symbolic = []
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
            data = DataFrame({'ds': data.index, 'y': data['2'].copy()})

            prophet.fit(data, validation_df=data, freq="D")
            future = prophet.make_future_dataframe(data, periods=14)
            df = prophet._prepare_dataframe_to_predict(future)
            dates, predicted, components = prophet._predict_raw(df, include_components=True)

            arrays.append(predicted[-14])
            symbolic.append(symbol[_i])
    return arrays, symbolic


def FutureOneDayRsi():
    arrays = []
    symbolic = []
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
            data = ta._ta_lib.RSI(data['3'].to_numpy(), 14)
            arrays.append(data)
            symbolic.append(symbol[_i])
    return arrays, symbolic


def FutureOneDayMacd():
    arrays = []
    symbolic = []
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
            data = ta._ta_lib.MACD(data['3'].to_numpy(), 14)
            arrays.append(data)
            symbolic.append(symbol[_i])
    return arrays, symbolic


def FutureOneDayBBands():
    upper, middles, lows, symbolic = list(), list(), list(), list()
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
            up, middle, low = ta._ta_lib.BBANDS(data['3'].to_numpy(), 20, 2, 2, matype=MA_Type.SMA)
            upper.append(up)
            middles.append(middle)
            lows.append(low)
            symbolic.append(symbol[_i])
    return upper, middles, lows, symbolic


