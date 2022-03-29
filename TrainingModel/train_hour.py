from ValueCrypto import TrendReader, InvestmentSelect
from TradeAlgorithm import CurrentDataOut, BuildBatchTrainDataset
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
import os
from torch.distributions import Categorical


def pretrain_hour(device, saved=False, grad_lock=False, batchsize=16, builded=False):
    epoch, gamma = 1000, 0.98
    size = BuildBatchTrainDataset(batchsize, builded)

    model_inv = InvestmentSelect(14 * 24, 24, 14, device).to(device)

    if saved:
        model_inv.load_state_dict(torch.load('./model/oneday_investment.pt'))

    optimizer = torch.optim.Adam(model_inv.parameters(), lr=0.02)

    smallest_loss = 5e+11
    for _i in range(epoch):
        batch_loss = 0.0
        length = tqdm(range(size))
        for t in length:

            dataX = torch.load('D:/CRYPTONALYTICS/TradeAlgorithm/datasets/hour/X/' + str(t + 1) + '.pt').to(device)
            out = model_inv(dataX)
            del dataX
            gc.collect()

            Y = torch.load('D:/CRYPTONALYTICS/TradeAlgorithm/datasets/hour/Y/' + str(t + 1) + '.pt').to(device)
            
            loss = F.cross_entropy(out.clone(), Y.reshape(-1))

            del Y
            gc.collect()

            batch_loss += loss
            if not torch.isfinite(loss):
                print("loss has not finited")
            else:
                if grad_lock:
                    nn.utils.clip_grad_norm_(model_inv.parameters(), 0.5)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                length.set_description("loss_min :{:0.6f}".format(loss))

        if batch_loss != 0.0:
            os.system('cls')
            print("{:.2f}".format(batch_loss / batchsize))
            if smallest_loss > batch_loss:
                smallest_loss = batch_loss
                torch.save(model_inv.state_dict(), './model/oneday_investment.pt')


if __name__ == '__main__':
    pretrain_hour('cuda', batchsize=16, saved=False,builded=True)
