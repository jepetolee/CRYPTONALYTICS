import torch
import torch.nn as nn
import torch.nn.functional as func
from Model import Transformer


class InvestmentSelect(nn.Module):
    def __init__(self, insize):
        super(InvestmentSelect, self).__init__()
        self.encoder = Transformer(insize, 12 * 10 * 3, 256, 8, 4, 0.1)

        self.p = nn.Linear(self.encoder.h1, 12)
        self.v = nn.Linear(self.encoder.h1, 1)

    def pi(self, x, softmax_dim=1):
        x = func.leaky_relu(self.encoder(x))
        x = self.p(x)
        prob = func.log_softmax(x, dim=softmax_dim)
        return prob

    def value(self, x):
        x = func.leaky_relu(self.encoder(x))
        x = self.v(x)
        return x


class TrendReader(nn.Module):
    def __init__(self, insize,size, outsize):
        super(TrendReader, self).__init__()
        self.tencoder = Transformer(insize, 120,size, 256, 8, 4, 0.1)

        self.v = nn.Linear(120, outsize)

    def value(self, x, device):
        src_mask = self.tencoder.generate_square_subsequent_mask(x.shape[1]).to(device)
        x = func.leaky_relu(self.tencoder(x,src_mask))
        return self.v(x)


class PositionDecisioner(nn.Module):
    def __init__(self, insize, outsize):
        super(PositionDecisioner, self).__init__()
        self.encoder = TrendReader(insize, outsize)
        self.p = nn.Linear(outsize, 3)
        self.v = nn.Linear(outsize, 1)

    def pi(self, x, softmax_dim=1):
        x = func.leaky_relu(self.encoder(x))
        x = self.p(x)
        prob = func.log_softmax(x, dim=softmax_dim)
        return prob

    def value(self, x):
        x = func.leaky_relu(self.encoder(x))
        x = self.v(x)
        return x
