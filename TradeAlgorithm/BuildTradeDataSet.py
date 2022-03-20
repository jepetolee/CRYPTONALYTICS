import numpy as np
from Prophet import *


class DatasetBuilder:

    def __init__(self, data, input_data=60, output=20, stride=3):
        label = data.shape[0]
        samples = int((label - input_data - output) // stride) + 1

        X = np.zeros(([samples, input_data, 13]))
        Y = np.zeros([samples, output])
        for i in range(samples):
            # build data set fot X, Y
            startx = stride * i
            endx = startx + input_data

            X[i] = data[startx:endx]

            starty = stride * i + input_data
            endy = starty + output
            Y[i] = data.T[3][starty:endy]
        self.x = X
        self.y = Y

    def pop(self):
        return self.x, self.y


def OneDayDataSetOut():
    # prophet = FutureProphetOneDay()
    data = FutureOneDayData()
    mupper, mmiddles = FutureOneDayMacd()

    data_bundleX, data_bundleY = list(), list()
    rsi = FutureOneDayRsi()
    upper, middles, lows = FutureOneDayBBands()
    logs, variance = FutureOneDaylog(), FutureOneDayVariancePercent()
    for i in range(len(data)):
        dataset = np.vstack([data[i][14:].T, rsi[i][14:]])
        dataset = np.vstack([dataset, mupper[i][14:]])
        dataset = np.vstack([dataset, mmiddles[i][14:]])
        dataset = np.vstack([dataset, logs[i][14:]])
        dataset = np.vstack([dataset, upper[i][14:]])
        dataset = np.vstack([dataset, middles[i][14:]])
        dataset = np.vstack([dataset, lows[i][14:]])
        dataset = np.vstack([dataset, variance[i][13:]]).T

        datasetX, datasetY = DatasetBuilder(dataset, 14, 3, 4).pop()
        data_bundleX.append(datasetX)
        data_bundleY.append(datasetY)

    return data_bundleX,data_bundleY
