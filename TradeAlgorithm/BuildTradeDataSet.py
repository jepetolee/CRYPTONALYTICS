import numpy as np
from Prophet import *
import torch
from tqdm import trange
from pathlib import Path


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


class DatasetTrainBuilder:

    def __init__(self, data, input_data=60, output=20, stride=3):
        label = data.shape[0]
        samples = int((label - input_data - output) // stride)

        X = np.zeros(([samples, input_data, 14]))
        X_prime = np.zeros(([samples, input_data, 14]))
        Y = np.zeros([samples, 1])
        for i in range(samples):
            # build data set fot X, Y
            startx = stride * i + 1
            endx = startx + input_data

            X[i] = data[startx:endx]
            X_prime[i] = data[startx - 1:endx - 1]

            starty = stride * i + input_data + 1

            Y[i] = data.T[12][starty]
        self.x = X
        self.y = Y
        self.x_prime = X_prime

    def pop(self):
        return self.x, self.y, self.x_prime


def OneDayDataSetOut():
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

        datasetX, datasetY = DatasetBuilder(dataset, 14, 7, 4).pop()
        data_bundleX.append(datasetX)
        data_bundleY.append(datasetY)

    return data_bundleX, data_bundleY


def CurrentDataOut():
    data = FutureOneDayData()
    mupper, mmiddles = FutureOneDayMacd()

    datasets = list()
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
        datasets.append(dataset[-14:])
    return datasets


def OneDayTrainDataSetOut():
    data = FutureOneDayData()
    mupper, mmiddles = FutureOneDayMacd()

    data_bundleX, data_bundleY, data_bundleX_prime = list(), list(), list()
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
        dataset = np.vstack([dataset, gradients[i][14:]]).T

        datasetX, datasetY, datasetX_prime = DatasetTrainBuilder(dataset, 14, 1, 4).pop()
        data_bundleX.append(datasetX)
        data_bundleY.append(datasetY)
        data_bundleX_prime.append(datasetX_prime)

    return data_bundleX, data_bundleY, data_bundleX_prime


def OneHourTrainDataSetOut():
    data = FutureOneHourData()
    mupper, mmiddles = FutureOneHourMacd()

    data_bundleX, data_bundleY, data_bundleX_prime = list(), list(), list()
    rsi = FutureOneHourRsi()
    upper, middles, lows = FutureOneHourBBands()
    logs, variance = FutureOneHourlog(), FutureOneHourVariancePercent()
    gradients = FutureOneHourDerivative()
    for i in range(len(data)):
        dataset = np.vstack([data[i][14:].T, rsi[i][14:]])
        dataset = np.vstack([dataset, mupper[i][14:]])
        dataset = np.vstack([dataset, mmiddles[i][14:]])
        dataset = np.vstack([dataset, logs[i][14:]])
        dataset = np.vstack([dataset, upper[i][14:]])
        dataset = np.vstack([dataset, middles[i][14:]])
        dataset = np.vstack([dataset, lows[i][14:]])
        dataset = np.vstack([dataset, variance[i][13:]])
        dataset = np.vstack([dataset, gradients[i][14:]]).T

        datasetX, datasetY, datasetX_prime = DatasetTrainBuilder(dataset, 14 * 24, 24, 1).pop()

        data_bundleX.append(datasetX)
        data_bundleY.append(datasetY)
        data_bundleX_prime.append(datasetX_prime)

    return data_bundleX, data_bundleY, data_bundleX_prime


def OneHourDataSetOut():
    data = FutureOneHourData()
    mupper, mmiddles = FutureOneHourMacd()

    data_bundleX, data_bundleY = list(), list()
    rsi = FutureOneHourRsi()
    upper, middles, lows = FutureOneHourBBands()
    logs, variance = FutureOneHourlog(), FutureOneHourVariancePercent()

    for i in range(len(data)):
        dataset = np.vstack([data[i][14:].T, rsi[i][14:]])
        dataset = np.vstack([dataset, mupper[i][14:]])
        dataset = np.vstack([dataset, mmiddles[i][14:]])
        dataset = np.vstack([dataset, logs[i][14:]])
        dataset = np.vstack([dataset, upper[i][14:]])
        dataset = np.vstack([dataset, middles[i][14:]])
        dataset = np.vstack([dataset, lows[i][14:]])
        dataset = np.vstack([dataset, variance[i][13:]]).T

        datasetX, datasetY = DatasetBuilder(dataset, 14 * 24, 24, 1).pop()

        data_bundleX.append(datasetX)
        data_bundleY.append(datasetY)

    return data_bundleX, data_bundleY


def BuildBatchTrainDataset(batchsize=16, built=False):
    path = 'D:/CRYPTONALYTICS/TradeAlgorithm/datasets/hour/'
    dataX, dataY, dataX_prime = OneHourTrainDataSetOut()
    biggest = 0
    first = [True, True, True, True, True, True, True, True, True, True, True, True]
    locker = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    parse = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(dataX)):
        dat = dataX[i].shape[0]
        parse[i] = dat % batchsize
        if biggest < dataX[i].shape[0]:
            biggest = dataX[i].shape[0]

    size = int(biggest / batchsize)
    left = biggest % batchsize

    if left > 0:
        size += 1

    if built:
        for t in trange(size):
            builded, out, builded_prime = list(), list(), list()
            for i in range(len(dataX)):
                if dataX[i].shape[0] < biggest - batchsize * t:
                    input = torch.zeros([batchsize, 14 * 24, 14])
                    input_prime = input.detach()
                    out.append(torch.zeros([batchsize, 1])-99)

                else:
                    if first[i]:
                        first[i] = False
                        locker[i] = t + 1
                        input = torch.zeros([batchsize, 14 * 24, 14])
                        input_prime = input.detach()
                        out_frame = torch.zeros([batchsize, 1])-99

                        for j in range(parse[i]):
                            input[batchsize - parse[i] + j] = torch.from_numpy(dataX[i][j]).float()
                            input_prime[batchsize - parse[i] + j] = torch.from_numpy(dataX_prime[i][j]).float()
                            out_frame[batchsize - parse[i] + j] = torch.from_numpy(dataY[i][j]).float()

                        out.append(out_frame)
                    else:
                        input = torch.from_numpy(
                            dataX[i][
                            batchsize * (t - locker[i]) + parse[i]:batchsize * (t + 1 - locker[i]) + parse[i]]).float()

                        input_prime = torch.from_numpy(
                            dataX_prime[i][
                            batchsize * (t - locker[i]) + parse[i]:batchsize * (t + 1 - locker[i]) + parse[i]]).float()

                        out.append(
                            torch.from_numpy(dataY[i][batchsize * (t - locker[i]) + parse[i]:batchsize * (
                                    t + 1 - locker[i]) + parse[i]]).float())

                builded.append(input)
                builded_prime.append(input_prime)

            output = torch.argmax(torch.stack(out).permute(1, 0, 2), dim=1)
            builts = torch.stack(builded).permute(1, 2, 0, 3)
            builts_prime = torch.stack(builded_prime).permute(1, 2, 0, 3)
            PT = str(t + 1) + '.pt'
            Path(path + 'Y/' + PT).touch(exist_ok=True)
            Path(path + 'X/' + PT).touch(exist_ok=True)
            Path(path + 'X_prime/' + PT).touch(exist_ok=True)
            torch.save(output, path + 'Y/' + str(t + 1) + '.pt')
            torch.save(builts, path + 'X/' + str(t + 1) + '.pt')
            torch.save(builts_prime, path + 'X_prime/' + str(t + 1) + '.pt')

    return size
