import math
import torch
from torch import nn
from torch.functional import F


class Transformer(nn.Module):
    def __init__(self, iw, ow, size, d_model, nhead, nlayers, dropout=0.5):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.encoder = nn.Sequential(
            nn.Linear(size, d_model // 2),
            nn.ELU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.linear = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ELU(),
            nn.Linear(d_model // 2, 1)
        )

        self.linear2 = nn.Sequential(
            nn.Linear(iw, (iw + ow) // 2),
            nn.ELU(),
            nn.Linear((iw + ow) // 2, ow)
        )

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, srcmask):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src.transpose(0, 1), srcmask).transpose(0, 1)
        output = self.linear(output)[:, :, 0]
        output = self.linear2(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def gen_attention_mask(x):
    return torch.eq(x, 0)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, stride=1)
        self.max_pool1 = nn.MaxPool2d(kernel_size=5)
        self.conv2 = nn.Conv2d(5, 2, 3, stride=1)
        self.max_pool2 = nn.MaxPool2d(kernel_size=5)
        self.linear1 = nn.Linear(722, 64)
        self.lstm = nn.LSTM(64, 16)

    def forward(self, x,hidden):
        x = F.elu(self.conv1(x))
        x = F.elu(self.max_pool1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.max_pool2(x))
        x = x.view(-1,722)
        x = F.elu(self.linear1(x))
        return self.lstm(x.reshape(-1,1,64), hidden)

