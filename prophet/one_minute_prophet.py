import talib.abstract as ta
from talib import MA_Type
from neuralprophet import NeuralProphet
from numpy import log
from pandas import DataFrame
import numpy as np
from calcul import cal_varience_percent
from make_pd import future_symbol_1min_data
from sklearn.cluster import KMeans


def FutureOneMinuteData(symbol):
    return future_symbol_1min_data(symbol).to_numpy()


def ProphetOneMinute():
    print("1분봉 예측값은 다음과 같습니다.")
