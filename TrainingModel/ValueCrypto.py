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
    def __init__(self, incode, device):
        super(PositionDecisioner, self).__init__()
        self.incode = incode
        self.device = device
        self.tencoder = Transformer(incode, 60, 1, 512, 8, 4, 0.1)
        self.q = nn.Linear(60, 3)
        self.v = nn.Linear(60, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.03)

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

    def train_model(self, s, s_prime, a, prob_a, reward, gamma=0.98, tau=0.97, weight=0.87,
                    eps_clip=0.1):
        for k in range(10):
            td_target = reward + gamma * self.value(s_prime)
            value = self.value(s)
            delta = (td_target - value).detach()
            pi = self.setposition(s)
            pi_a = pi.gather(1, torch.tensor([a], dtype=torch.int64).reshape(-1, 1))
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))
            surr1 = ratio * delta
            surr2 = torch.clamp(ratio, 1 + eps_clip, 1 - eps_clip) * delta
            lossL = -torch.min(surr1, surr2).mean().item() + func.smooth_l1_loss(value, td_target) * weight
            if not torch.isfinite(lossL):
                print("loss has not finited")
            else:
                #    nn.utils.clip_grad_norm_(leverage.parameters(), 0.5)
                self.optimizer.zero_grad()
                lossL.backward()
                self.optimizer.step()


class Determiner(nn.Module):
    def __init__(self, incode, device):
        super(Determiner, self).__init__()
        self.incode = incode
        self.device = device
        self.tencoder = Transformer(incode, 60, 1, 512, 16, 8, 0.1)
        self.determiner = nn.Linear(60, 2)
        self.v = nn.Linear(60, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.03)

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

    def train_model(self, s, s_prime, determined, determined_prob, reward, gamma=0.98, tau=0.97, weight=0.87,
                    eps_clip=0.1):
        for k in range(3):
            td_target = reward + gamma * self.value(s_prime)
            value = self.value(s)
            delta = (td_target - value).detach()
            pi = self.determine(s)

            pi_a = pi.gather(1, torch.tensor([determined], dtype=torch.int64).reshape(-1, 1))
            ratio = torch.exp(torch.log(pi_a) - torch.log(determined_prob))
            surr1 = ratio * delta
            surr2 = torch.clamp(ratio, 1 + eps_clip, 1 - eps_clip) * delta
            lossD = -torch.min(surr1, surr2).mean().item() + func.smooth_l1_loss(value, td_target) * weight
            if not torch.isfinite(lossD):
                print("loss has not finited")
            else:
                self.optimizer.zero_grad()
                lossD.backward()
                self.optimizer.step()
            # nn.utils.clip_grad_norm_(self.parameters(), 0.5)


class Leverage(nn.Module):
    def __init__(self, incode, device):
        super(Leverage, self).__init__()
        self.incode = incode
        self.device = device
        self.tencoder = Transformer(incode, 60, 1, 512, 16, 8, 0.1)
        self.determine = nn.Linear(60, 2)
        self.v = nn.Linear(60, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.03)

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

    def train_model(self, s, s_prime, a, prob_a, reward, gamma=0.98, tau=0.97, weight=0.87,
                    eps_clip=0.1):
        for k in range(10):
            td_target = reward + gamma * self.value(s_prime)
            value = self.value(s)
            delta = (td_target - value).detach()
            pi = self.setleverage(s)
            pi_a = pi.gather(1, torch.tensor([a], dtype=torch.int64).reshape(-1, 1))
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))
            surr1 = ratio * delta
            surr2 = torch.clamp(ratio, 1 + eps_clip, 1 - eps_clip) * delta
            lossL = -torch.min(surr1, surr2).mean().item() + func.smooth_l1_loss(value, td_target) * weight
            if not torch.isfinite(lossL):
                print("loss has not finited")
            else:

                # nn.utils.clip_grad_norm_(self .parameters(), 0.5)
                self.optimizer.zero_grad()
                lossL.backward()
                self.optimizer.step()
