import gc
import numpy as np
import sys

sys.path.append('..')
from Prophet import *
import torch
from tqdm import trange
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_finance import candlestick2_ohlc
from TradeAlgorithm.update_csv import *
from make_pd import *


def TradeDatasetBuilder(data, input_data=60, stride=3):
    label = data.shape[0]
    data = data.T[:3].T


    samples = int((label - input_data) // stride) + 1

    X = np.zeros(([samples, input_data, 3]))
    for i in range(samples):
        # build data set fot X, Y
        startx = stride * i
        endx = startx + input_data

        X[i] = data[startx:endx]
    return X


def FinalDatasetBuilder(data, symbol, date, passer=0, stop=0, input_data=60, stride=3):
    label = data.shape[0]
    samples = int((label - input_data) // stride)
    if date == "1h":
        time = '1hour/'
    elif date == '15m':
        time = '15min/'
    elif date == '1m':
        time = '1min/'
    elif date == '15m2':
        time = '15min2/'
    elif date == '4h':
        time = '4hour/'
    elif date == '1d':
        time = '1day/'
    if stop == 0:
        stop = samples
    for i in trange(stop - passer + 1):
        # build data set fot X, Y
        i += passer
        startx = stride * i
        endx = startx + input_data

        temp = data.iloc[startx:endx]

        temp.reset_index(inplace=True)
        temp.index += 1

        fig = plt.figure(figsize=(20, 20))
        gs = gridspec.GridSpec(2, 4)
        fig.subplots_adjust(wspace=0.2, hspace=0.2)

        axes = list()
        axes.append(plt.subplot(gs[0, :]))
        axes.append(plt.subplot(gs[1, :], sharex=axes[0]))

        candlestick2_ohlc(axes[0], temp['1'], temp['2'], temp['3'], temp['4'], width=1, colorup='r', colordown='b')
        axes[1].bar(temp.index, temp['5'], color='k', width=0.8, align='center')
        copies = temp['4'].copy()
        axes[0].plot(temp.index, copies.rolling(window=3).mean(), label='Ma3')
        axes[0].plot(temp.index, copies.rolling(window=14).mean(), label='Ma14')

        plt.savefig('D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/' + time + str(i) + '.png', dpi=50)
        plt.close('all')
        del axes, temp, gs, fig
        gc.collect()


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


def TradeDataSetOut(symbol):
    return TradeDatasetBuilder(FutureOneMinuteData(symbol), 60, 1)


def TradeData2SetOut(symbol):
    return TradeDatasetBuilder(FutureFifteenMinuteData(symbol), 120, 1)


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
                    out.append(torch.zeros([batchsize, 1]) - 99)

                else:
                    if first[i]:
                        first[i] = False
                        locker[i] = t + 1
                        input = torch.zeros([batchsize, 14 * 24, 14])
                        input_prime = input.detach()
                        out_frame = torch.zeros([batchsize, 1]) - 99

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


def DatasetFinal(symbol):
    update_future_15min_csv(symbol)
    update_future_1hour_csv(symbol)
    update_future_1min_csv(symbol)

    onehour = future_symbol_1hour_data(symbol)
    fifteen_data = future_symbol_15min_data(symbol)
    oneminute_data = future_symbol_1min_data(symbol)
    FinalDatasetBuilder(oneminute_data, symbol, '1m', passer=7140 + 72000, input_data=60, stride=1)
    FinalDatasetBuilder(fifteen_data, symbol, '15m', passer=5180, input_data=4 * 24, stride=1)
    FinalDatasetBuilder(onehour, symbol, '1h', passer=1200, input_data=24 * 5, stride=1)


def DatasetFinal2(symbol):
    # update_future_15min_csv(symbol)
    # update_future_4hour_csv(symbol)
    # update_future_1hour_csv(symbol)
    oneday_data = future_symbol_1hour_data(symbol)
    fourhour_data = future_symbol_4hour_data(symbol)
    fifteenminute_data = future_symbol_15min_data(symbol)
    FinalDatasetBuilder(fifteenminute_data, symbol, '15m2', passer=90387, stop=90388, input_data=120, stride=1)


#  FinalDatasetBuilder(oneday_data, symbol, '1h', passer=19249,stop =0, input_data=60, stride=1)
#   FinalDatasetBuilder(fourhour_data, symbol, '4h', passer=0, input_data=45, stride=1)

'''
DatasetFinal2('BTCUSDT')
import PIL
for i in range(92477-90388):
    print(i)
    sprime_FiftMP = PIL.Image.open(
        'D:/CRYPTONALYTICS/TradeAlgorithm/dataset/BTCUSDT/15min2/' + str(i+90388) + '.png').convert("L")
    sprime_FiftMP.close()'''