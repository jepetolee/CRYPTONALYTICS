from ValueCrypto import PositionDecisioner, Determiner, Leverage
from TradeAlgorithm import CurrentDataOut, TradeDataSetOut,update_future_1min_csv
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
from torch.distributions import Categorical


def TrainingTrader(symbol, device, saved=False):
    leverage, determine, position = Leverage(50, device=device).to(device), Determiner(50, device=device).to(device), \
                                    PositionDecisioner(50, device=device).to(device)
    if saved:
        leverage.load_state_dict('./model/' + symbol + '_leverage.pt')
        determine.load_state_dict('./model/' + symbol + '_determine.pt')
        position.load_state_dict('./model/' + symbol + '_position.pt')
    update_future_1min_csv(symbol)
    dataset = TradeDataSetOut()
    dataset_train = dataset[number]
    del dataset
    gc.collect()
    epoch = 1000
    for _i in trange(epoch):
        for t in range(len(dataset_train)):
            print("도랄팍쥐 화이팅")


TrainingTrader('BTCUSDT','cpu')