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

        model = TrendReader(14, 13, 7).to(device)
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
                    output = torch.from_numpy(dataY[i][64 * t:64 * t + left]).float().to(device).reshape(-1, 7)
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
                    output = torch.from_numpy(dataY[i][64 * t:64 * t + 64]).float().to(device).reshape(-1, 7)
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
        model = TrendReader(14, 13, 7).to(device)
        model.load_state_dict(torch.load('./model/oneday' + symbol[i] + '.pt'))
        input = torch.from_numpy(dataX[i]).float().to(device).reshape(-1, 14, 13)
        result = model.value(input, device)
        builded.append(result)

    built = torch.cat(builded, dim=0).reshape(-1, 12, 7)
    del dataX, symbol, model, builded
    gc.collect()
    torch.cuda.empty_cache()

    model_inv = InvestmentSelect(12, 7).to(device)
    model_inv.load_state_dict(torch.load('./model/oneday_investment.pt'))
    out = model_inv.pi(built, device)
    prob = Categorical(out).sample()
    return prob


def pretrain_selecter_day(device, saved=False, grad_lock=False):
    epoch, gamma = 1000, 0.98
    dataX, dataY, dataX_prime = OneDayTrainDataSetOut()
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

    size = int(smallest / 4)
    left = smallest % 4

    builts, outputs, builts_prime = list(), list(), list()
    if left > 0:
        size += 1

    for t in range(size):
        builded, out, builded_prime = list(), list(), list()
        for i in range(len(dataX)):
            if t == size - 1 and left > 0:
                input = torch.from_numpy(dataX[i][-smallest + 4 * t:]).float() \
                    .to(device).reshape(-1, 14, 13)
                input_prime = torch.from_numpy(dataX_prime[i][-smallest + 4 * t:]).float() \
                    .to(device).reshape(-1, 14, 13)
                out.append(torch.from_numpy(dataY[i][-smallest + 4 * t:]).float() \
                           .to(device))
            else:
                input = torch.from_numpy(dataX[i][-smallest + 4 * t:-smallest + 4 + 4 * t]).float(). \
                    to(device).reshape(-1, 14, 13)
                input_prime = torch.from_numpy(dataX_prime[i][-smallest + 4 * t:-smallest + 4 + 4 * t]).float(). \
                    to(device).reshape(-1, 14, 13)
                out.append(torch.from_numpy(dataY[i][-smallest + 4 * t:-smallest + 4 + 4 * t]).float() \
                           .to(device))

            model = TrendReader(14, 13, 7).to(device)
            model.load_state_dict(torch.load('./model/oneday' + symbol[i] + '.pt'))
            result = model.value(input, device)
            result_prime = model.value(input_prime, device)
            builded.append(result)
            builded_prime.append(result_prime)
        tensor = torch.stack(out).permute(1, 0, 2)
        outputs.append(torch.argmax(tensor, dim=1).reshape(-1))
        builts.append(torch.stack(builded).permute(1, 0, 2))
        builts_prime.append(torch.stack(builded_prime).permute(1, 0, 2))

    del model, out, builded, builded_prime, symbol, dataX, dataY, dataX_prime,input,input_prime,out,tensor
    gc.collect()
    torch.cuda.empty_cache()

    progress = tqdm(range(epoch))
    model_inv = InvestmentSelect(12, 7).to(device)

    if saved:
        model_inv.load_state_dict(torch.load('./model/oneday_investment.pt'))

    optimizer = torch.optim.Adam(model_inv.parameters(), lr=0.02)
    smallest_loss = 5e+11
    for _i in progress:
        batch_loss = 0.0
        for t in range(size):
            out = model_inv(builts[t].reshape(-1, 12, 7))
            out_prime = model_inv(builts_prime[t].reshape(-1, 12, 7))

            a_prime = Categorical(out_prime).sample()
            a = Categorical(out).sample()

            reward = 3 * torch.eq(a, outputs[t]) - 2

            target = reward + gamma * a_prime
            Q_out = out.gather(1, a.reshape(-1, 1))

            loss = F.smooth_l1_loss(Q_out.reshape(-1), target)
            batch_loss += loss
            if not torch.isfinite(loss):
                print("loss has not finited")
            else:
                if grad_lock:
                    nn.utils.clip_grad_norm_(model_inv.parameters(), 0.5)

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

        if batch_loss != 0.0:
            progress.set_description("loss: {:0.6f}".format(batch_loss))
            if smallest_loss > batch_loss:
                smallest_loss = batch_loss
                torch.save(model_inv.state_dict(), './model/oneday_investment.pt')


if __name__ == '__main__':
    #    pretrain_day('cuda', grad_lock=True)
    pretrain_selecter_day('cpu')