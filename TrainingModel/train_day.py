from ValueCrypto import TrendReader, InvestmentSelect
from TradeAlgorithm import OneDayDataSetOut, CurrentDataOut, OneDayTrainDataSetOut
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
from torch.distributions import Categorical


def pretrain_day(device, saved=False, grad_lock=False):
    epoch = 1000
    dataX, dataY = OneDayDataSetOut()

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
    for i in range(len(dataX)):

        model = TrendReader(14, 13, 3).to(device)
        progress = tqdm(range(epoch))

        if saved:
            model.load_state_dict(torch.load('./model/oneday' + symbol[i] + '.pt'))
        size = int(dataX[i].shape[0] / 64)
        left = dataX[i].shape[0] % 64
        if left > 0:
            size += 1
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        for _i in progress:
            batch_loss = 0.0
            for t in range(size):
                if t == size - 1 and left > 0:
                    input = torch.from_numpy(dataX[i][64 * t:64 * t + left]).float().to(device).reshape(-1, 14, 13)
                    output = torch.from_numpy(dataY[i][64 * t:64 * t + left]).float().to(device).reshape(-1, 3)
                    result = model.value(input, device)
                    loss = F.mse_loss(result, output)

                    if not torch.isfinite(loss):
                        print(symbol[i] + "loss has outed none finited")

                    else:
                        loss.backward()
                        if grad_lock:
                            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()
                        batch_loss += loss
                else:

                    input = torch.from_numpy(dataX[i][64 * t:64 * t + 64]).float().to(device).reshape(-1, 14, 13)
                    output = torch.from_numpy(dataY[i][64 * t:64 * t + 64]).float().to(device).reshape(-1, 3)
                    result = model.value(input, device)

                    loss = F.mse_loss(result, output)

                    if not torch.isfinite(loss):
                        print(symbol[i] + "loss has outed none finited")

                    else:
                        loss.backward()
                        if grad_lock:
                            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                        optimizer.step()
                        batch_loss += loss
            if _i == 0:
                smallest_loss = batch_loss / dataX[i].shape[0]

            progress.set_description("loss: {:0.6f}".format(batch_loss / dataX[i].shape[0]))
            if batch_loss != 0.0:
                if smallest_loss > batch_loss / dataX[i].shape[0]:
                    smallest_loss = batch_loss / dataX[i].shape[0]
                    torch.save(model.state_dict(), './model/oneday' + symbol[i] + '.pt')


def eval_day(device):
    dataX = CurrentDataOut()
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
    builded = list()
    for i in range(len(dataX)):
        model = TrendReader(14, 13, 3).to(device)
        model.load_state_dict(torch.load('./model/oneday' + symbol[i] + '.pt'))
        input = torch.from_numpy(dataX[i]).float().to(device).reshape(-1, 14, 13)
        result = model.value(input, device)
        builded.append(result)

    built = torch.cat(builded, dim=0).reshape(-1, 12, 3)
    del dataX, symbol, model, builded
    gc.collect()
    torch.cuda.empty_cache()

    model_inv = InvestmentSelect(12, 3).to(device)
    model_inv.load_state_dict(torch.load('./model/oneday_investment.pt'))
    out = model_inv.pi(built, device)
    prob = Categorical(out).sample()
    print(prob)


def pretrain_selecter_day(device, grad_lock=False):
    epoch = 1000
    dataX, dataY = OneDayTrainDataSetOut()
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

    smallest = 1e+10
    for i in range(len(dataX)):
        if smallest > dataX[i].shape[0]:
            smallest = dataX[i].shape[0]

    size = int(smallest / 64)
    left = smallest % 64

    builts, outputs = list(), list()
    if left > 0:
        size += 1
        for t in range(size):

            if t == size - 1 and left > 0:
                outputs.append(torch.from_numpy(dataY[:][-smallest + 64 * t:]).float() \
                               .to(device))
            else:
                outputs.append(torch.from_numpy(dataY[:][-smallest + 64 * t:-smallest + 64 + 64 * t]).float() \
                               .to(device))

            builded = list()
            for i in range(len(dataX)):
                if t == size - 1 and left > 0:
                    input = torch.from_numpy(dataX[i][-smallest + 64 * t:]).float() \
                        .to(device).reshape(-1, 14, 13)
                else:
                    input = torch.from_numpy(dataX[i][-smallest + 64 * t:-smallest + 64 + 64 * t]).float(). \
                        to(device).reshape(-1, 14, 13)
                model = TrendReader(14, 13, 3).to(device)
                model.load_state_dict(torch.load('./model/oneday' + symbol[i] + '.pt'))
                result = model.value(input, device)
                builded.append(result)

            builts.append(torch.cat(builded, dim=0).reshape(-1, 12, 3))
        print(outputs[0].shape)

    #progress = tqdm(range(epoch))
    # for _i in progress:


if __name__ == '__main__':
    pretrain_selecter_day('cuda')
