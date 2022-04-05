from ValueCrypto import TrendReader, InvestmentSelect
from TradeAlgorithm import CurrentDataOut, BuildBatchTrainDataset
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
from torch.distributions import Categorical


def pretrain_hour(device, saved=False, grad_lock=False, batchsize=16, builded=False, passer=0, isolation=5):
    epoch, smallest_loss = 1000, 5e+11

    size = BuildBatchTrainDataset(batchsize, builded)

    model_inv = InvestmentSelect(14 * 24, 24, 14, device).to(device)
    if saved:
        model_inv.load_state_dict(torch.load('./model/oneday_investment.pt'))

    optimizer = torch.optim.Adam(model_inv.parameters(), lr=4e-6)


    for _i in range(epoch):
        batch_loss = 0.0
        length = tqdm(range(size - passer))
        for t in length:
            t += passer
            dataX = torch.load('D:/CRYPTONALYTICS/TradeAlgorithm/datasets/hour/X/' + str(t + 1) + '.pt').to(device)
            Y = torch.load('D:/CRYPTONALYTICS/TradeAlgorithm/datasets/hour/Y/' + str(t + 1) + '.pt').to(device)
            for _v in range(isolation):
                out = model_inv(dataX)
                loss = F.cross_entropy(out.clone(), Y.reshape(-1))
                if not torch.isfinite(loss):
                    print("loss has not finited")
                else:
                    if grad_lock:
                        nn.utils.clip_grad_norm_(model_inv.parameters(), 0.5)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.cuda.empty_cache()
                    length.set_description("loss_min :{:0.6f}".format(loss))
            batch_loss += loss

        del dataX, Y
        gc.collect()
        if batch_loss != 0.0:
            print("totalloss : {:.2f}".format(batch_loss))
            if smallest_loss > batch_loss:
                smallest_loss = batch_loss
                torch.save(model_inv.state_dict(), './model/oneday_investment.pt')
