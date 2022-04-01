from ValueCrypto import Trader
from TradeAlgorithm import CurrentDataOut, BuildBatchTrainDataset
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
from torch.distributions import Categorical


def TrainingTrader(symbol):
    print("훈련시작")
