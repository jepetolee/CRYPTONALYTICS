from ValueCrypto import TrendReader
from TradeAlgorithm import OneDayDataSetOut, CurrentDataOut
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch


def pretrain_day(device, saved=False):
    epoch = 100
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

        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        for _i in progress:
            batch_loss = 0.0
            for t in range(dataX[i].shape[0]):
                input = torch.from_numpy(dataX[i][t]).float().to(device).reshape(-1, 14, 13)
                output = torch.from_numpy(dataY[i][t]).float().to(device)
                result = model.value(input, device)

                loss = F.smooth_l1_loss(result, output).mean()

                if not torch.isfinite(loss):
                    print(symbol[i] + "loss has outed none finited")

                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 0.2)
                    optimizer.step()
                    batch_loss += loss
            progress.set_description("loss: {:0.6f}".format(batch_loss / dataX[i].shape[0]))
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
    for i in range(len(dataX)):
        model = TrendReader(14, 13, 3).to(device)
        model.load_state_dict(torch.load('./model/oneday' + symbol[i] + '.pt'))
        input = torch.from_numpy(dataX[i]).float().to(device)
        result = model.value(input, device)


#        print(result)


if __name__ == "__main__":
    pretrain_day('cuda')
