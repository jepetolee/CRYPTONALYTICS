import talib.abstract as ta
from talib import MA_Type
from neuralprophet import NeuralProphet
from numpy import log
from pandas import DataFrame
import numpy as np
from calcul import cal_varience_percent
from make_pd import future_symbol_15min_data
from sklearn.cluster import KMeans


def FutureProphetFifteenMinute():
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

    f = open('D:/CRYPTONALYTICS/TrainingModel/predict/onehour/predict.txt',"w")
    for _i in range(len(symbol)):
        data = future_symbol_15min_data(symbol[_i])

        if data is not None:
            prophet = NeuralProphet(seasonality_mode='multiplicative', n_forecasts=4 * 2, epochs=30)
            datas = DataFrame({'ds': data.index, 'y': data['4'].copy()})
            prophet.fit(datas, validation_df=datas, freq="15min")
            future = prophet.make_future_dataframe(datas, periods=4 * 24)
            df = prophet._prepare_dataframe_to_predict(future)
            dates, predicted, components = prophet._predict_raw(df, include_components=True)
            arrays.append((predicted[-8:]))
            f.writelines(str(predicted[-8:]))
    f.close()
    return arrays


