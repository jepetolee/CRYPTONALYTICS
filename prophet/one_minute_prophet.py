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


def FutureOneMinuteDerivative(symbol):
    return np.gradient(future_symbol_1min_data(symbol).to_numpy())
