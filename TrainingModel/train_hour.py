from ValueCrypto import TrendReader, InvestmentSelect
from TradeAlgorithm import CurrentDataOut, BuildBatchTrainDataset
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
from torch.distributions import Categorical


def pretrain_hour(device, saved=False, grad_lock=False, batchsize=16, builded=True):
    epoch, gamma = 1000, 0.98
    size = BuildBatchTrainDataset(batchsize, builded)

    progress = tqdm(range(epoch))
    model_inv = InvestmentSelect(14 * 24, 24, 14, device).to(device)

    if saved:
        model_inv.load_state_dict(torch.load('./model/oneday_investment.pt'))

    optimizer = torch.optim.Adam(model_inv.parameters(), lr=0.02)
    smallest_loss = 5e+11
    for _i in progress:
        batch_loss = 0.0
        for t in range(size):
            dataX = torch.load('D:/CRYPTONALYTICS/TradeAlgorithm/datasets/hour/X/' + str(t + 1) + '.pt')
            out = model_inv(dataX)
            del dataX
            gc.collect()

            dataX_prime = torch.load('D:/CRYPTONALYTICS/TradeAlgorithm/datasets/hour/X_prime/' + str(t + 1) + '.pt')
            out_prime = model_inv(dataX_prime)

            del dataX_prime
            gc.collect()

            Y = torch.load('D:/CRYPTONALYTICS/TradeAlgorithm/datasets/hour/Y/' + str(t + 1) + '.pt')
            print(Y.shape)
            a_prime = Categorical(out_prime).sample()
            a = Categorical(out).sample()

            reward = 3 * torch.eq(a, Y) - 2

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
    pretrain_hour('cpu')
