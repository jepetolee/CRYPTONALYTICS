import torch

from ValueCrypto import TrendReader
from TradeAlgorithm import OneDayDataSetOut
from torch.utils.data import DataLoader
from tqdm import trange
import torch


def train(device):
    dataX,dataY = OneDayDataSetOut()

    for i in trange(len(dataX)):
        model = TrendReader(14,13, 3).to(device)
        for t in range(dataX[i].shape[0]):
            input = torch.from_numpy(dataX[i][t]).float().to(device)
            result = model.value(input, device)


if __name__ == "__main__":
    train('cpu')
