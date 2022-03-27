from ValueCrypto import TrendReader, InvestmentSelect
from TradeAlgorithm import CurrentDataOut, OneHourTrainDataSetOut
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
from torch.distributions import Categorical


def eval_hour(device):
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


def pretrain_hour(device, saved=False, grad_lock=False, batchsize=16):
    epoch, gamma = 1000, 0.98
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

    batchsize = 16

    size = int(biggest / batchsize)
    left = biggest % batchsize

    builts, outputs, builts_prime = list(), list(), list()
    if left > 0:
        size += 1
    for t in range(size):
        builded, out, builded_prime = list(), list(), list()
        for i in range(len(dataX)):
            if dataX[i].shape[0] < biggest - batchsize * t:
                input = torch.zeros([batchsize, 14 * 24, 14])
                input_prime = input.detach()
                out.append(torch.zeros([batchsize, 24]))

            else:
                if first[i]:
                    first[i] = False
                    locker[i] = t + 1
                    input = torch.zeros([batchsize, 14 * 24, 14])
                    input_prime = input.detach()
                    out_frame = torch.zeros([batchsize, 24])

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

        outputs.append(torch.argmax(torch.stack(out).permute(1, 0, 2), dim=1).reshape(-1))
        builts.append(torch.stack(builded).permute(1, 0, 2, 3))
        builts_prime.append(torch.stack(builded_prime).permute(1, 0, 2, 3))

    del  out, builded, builded_prime, dataX, dataY, dataX_prime, input, input_prime
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    #    pretrain_day('cuda', grad_lock=True)
    pretrain_hour('cpu')

'''
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
'''
