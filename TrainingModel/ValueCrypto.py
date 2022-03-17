import torch
import torch.nn as nn
import torch.nn.functional as func
from Model import TransformerEncoder


class InvestmentSelect(nn.Module):
    def __init__(self, insize):
        super(InvestmentSelect, self).__init__()
        self.encoder = TransformerEncoder(insize, 12 * 10 * 3, 256, 8, 4, 0.1)

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
    def __init__(self):
        super(TrendReader, self).__init__()
        self.t1hencoder = TransformerEncoder()
        self.t15mencoder = TransformerEncoder()
        self.t5mencoder = TransformerEncoder()
        self.t1mencoder = TransformerEncoder()

        self.v1h = nn.Linear(self.encoder.h1, 2)
        self.v15m = nn.Linear(self.encoder.h1, 8)
        self.v5m = nn.Linear(self.encoder.h1, 24)
        self.v1m = nn.Linear(self.encoder.h1, 120)

    def value(self, x):
        x = func.leaky_relu(self.t1hencoder(x))
        return self.v(x)


class PositionDecisioner(nn.Module):
    def __init__(self):
        super(PositionDecisioner, self).__init__()
        self.encoder = TrendReader()
        self.p = nn.Linear(self.encoder.h1, 3)
        self.v = nn.Linear(self.encoder.h1, 1)

    def pi(self, x, softmax_dim=1):
        x = func.leaky_relu(self.encoder(x))
        x = self.p(x)
        prob = func.log_softmax(x, dim=softmax_dim)
        return prob

    def value(self, x):
        x = func.leaky_relu(self.encoder(x))
        x = self.v(x)
