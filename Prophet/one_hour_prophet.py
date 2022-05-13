import talib.abstract as ta
from talib import MA_Type
from neuralprophet import NeuralProphet
from numpy import log
from pandas import DataFrame
import numpy as np
from calcul import cal_varience_percent
from make_pd import future_symbol_1hour_data
from sklearn.cluster import KMeans


def FutureOneHourVariancePercent():
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

        data = future_symbol_1hour_data(symbol[_i])
        if data is not None:
            data = cal_varience_percent(data['4'].to_numpy())

            array.append(data)
    return array


def FutureOneHourData():
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

        data = future_symbol_1hour_data(symbol[_i])
        if data is not None:
            array.append(data.to_numpy())
    return array



def FutureOneHourlog():
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
        data = future_symbol_1hour_data(symbol[_i])
        if data is not None:
            data = log(data['4'].to_numpy())
            arrays.append(data)
    return arrays


def FutureProphetOneHour():
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

    f = open('D:/CRYPTONALYTICS/TrainingModel/predict/onehour/predict.txt', "w")

    for _i in range(len(symbol)):
        data = future_symbol_1hour_data(symbol[_i])
        if data is not None:
            prophet = NeuralProphet(seasonality_mode='multiplicative', n_forecasts=12, epochs=30)
            datas = DataFrame({'ds': data.index, 'y': data['4'].copy()})
            prophet.fit(datas, validation_df=datas, freq="H")
            future = prophet.make_future_dataframe(datas, periods=14 * 24)
            df = prophet._prepare_dataframe_to_predict(future)
            dates, predicted, components = prophet._predict_raw(df, include_components=True)
            arrays.append((predicted[-12]))

            f.writelines(str(predicted[-12]) + '\n')
    f.close()
    return arrays


def FutureOneHourRsi():
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
        data = future_symbol_1hour_data(symbol[_i])
        if data is not None:
            data = ta._ta_lib.RSI(data['4'].to_numpy(), 14)

            arrays.append(data)
    return arrays


def FutureOneHourMacd():
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
        data = future_symbol_1hour_data(symbol[_i])
        if data is not None:
            up = ta._ta_lib.SMA(data['4'].to_numpy(), 3)
            middle = ta._ta_lib.SMA(data['4'].to_numpy(), 14)
            upper.append(up)
            middles.append(middle)

    return upper, middles


def FutureOneHourBBands():
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
        data = future_symbol_1hour_data(symbol[_i])
        if data is not None:
            up, middle, low = ta._ta_lib.BBANDS(data['4'].to_numpy(), 14, 2, 2, matype=MA_Type.SMA)
            upper.append(up)
            middles.append(middle)
            lows.append(low)
    return upper, middles, lows


def FutureOneHourDerivative():
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

        data = future_symbol_1hour_data(symbol[_i])
        if data is not None:
            array.append(np.gradient(data['4'].to_numpy()))

    return array
