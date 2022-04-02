import gc

import torch
import torch.nn as nn
import torch.nn.functional as func
from Model import Transformer


class InvestmentSelect(nn.Module):
    def __init__(self, incode, hidden, size, device):
        super(InvestmentSelect, self).__init__()
        self.incode = incode
        self.device = device
        self.encoder1 = nn.Linear(size, 1)
        self.tencoder = Transformer(incode, 60, 12, 256, 8, 4, 0.1)
        self.q = nn.Linear(60, 12)

    def forward(self, x, softmax_dim=1):
        x = self.encoder1(x)[:, :, :, 0]
        src_mask = self.tencoder.generate_square_subsequent_mask(x.shape[1]).to(self.device)
        x = func.relu(self.tencoder(x, src_mask))
        del src_mask
        gc.collect()
        torch.cuda.empty_cache()
        x = self.q(x)
        return func.softmax(x, dim=softmax_dim)


class PositionDecisioner(nn.Module):
    def __init__(self, incode,device):
        super(PositionDecisioner, self).__init__()
        self.incode = incode
        self.device = device
        self.tencoder = Transformer(incode, 60, 1, 512, 8, 4, 0.1)
        self.q = nn.Linear(60, 3)
        self.v = nn.Linear(60, 1)

    def setposition(self, x, softmax_dim=1):
        src_mask = self.tencoder.generate_square_subsequent_mask(x.shape[1]).to(self.device)
        x = func.relu(self.tencoder(x, src_mask))
        del src_mask
        gc.collect()
        torch.cuda.empty_cache()
        x = self.q(x)
        return func.softmax(x, dim=softmax_dim)

    def value(self, x):
        src_mask = self.tencoder.generate_square_subsequent_mask(x.shape[1]).to(self.device)
        x = func.relu(self.tencoder(x, src_mask))
        del src_mask
        gc.collect()
        torch.cuda.empty_cache()
        return func.elu(self.v(x))


class Determiner(nn.Module):
    def __init__(self, incode, device):
        super(Determiner, self).__init__()
        self.incode = incode
        self.device = device
        self.tencoder = Transformer(incode, 60, 1, 512, 16, 8, 0.1)
        self.determiner = nn.Linear(60, 2)
        self.v = nn.Linear(60, 1)

    def determine(self, x, softmax_dim=1):
        src_mask = self.tencoder.generate_square_subsequent_mask(x.shape[1]).to(self.device)
        x = func.relu(self.tencoder(x, src_mask))
        del src_mask
        gc.collect()
        torch.cuda.empty_cache()
        x = self.determiner(x)
        return func.softmax(x, dim=softmax_dim)

    def value(self, x):
        src_mask = self.tencoder.generate_square_subsequent_mask(x.shape[1]).to(self.device)
        x = func.relu(self.tencoder(x, src_mask))
        del src_mask
        gc.collect()
        torch.cuda.empty_cache()
        return func.elu(self.v(x))


class Leverage(nn.Module):
    def __init__(self, incode,  device):
        super(Leverage, self).__init__()
        self.incode = incode
        self.device = device
        self.tencoder = Transformer(incode, 60, 1, 512, 16, 8, 0.1)
        self.determine = nn.Linear(60, 2)
        self.v = nn.Linear(60, 1)

    def setleverage(self, x, softmax_dim=1):
        src_mask = self.tencoder.generate_square_subsequent_mask(x.shape[1]).to(self.device)
        x = func.relu(self.tencoder(x, src_mask))
        del src_mask
        gc.collect()
        torch.cuda.empty_cache()
        x = self.determine(x)
        return func.softmax(x, dim=softmax_dim)

    def value(self, x):
        src_mask = self.tencoder.generate_square_subsequent_mask(x.shape[1]).to(self.device)
        x = func.relu(self.tencoder(x, src_mask))
        del src_mask
        gc.collect()
        torch.cuda.empty_cache()
        return func.elu(self.v(x))


class TrendReader(nn.Module):
    def __init__(self, insize, size, outsize):
        super(TrendReader, self).__init__()
        self.tencoder = Transformer(insize, 120, size, 256, 16, 8, 0.1)
        self.v = nn.Linear(120, outsize)

    def value(self, x, device):
        src_mask = self.tencoder.generate_square_subsequent_mask(x.shape[1]).to(device)
        x = func.leaky_relu(self.tencoder(x, src_mask))
        return self.v(x)
