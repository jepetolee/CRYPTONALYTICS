import gc

import torch
import torch.nn as nn
import torch.nn.functional as func
from Model import Transformer, Encoder


class Trader(nn.Module):
    def __init__(self, device):
        super(Trader, self).__init__()
        self.device = device
        self.encoder1h = Encoder()
        self.encoder15min = Encoder()
        self.encoder1min = Encoder()
        self.transformer = Transformer(3, 128, 16, 32, 8, 4, 0.1)

        self.position = nn.Sequential(
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Linear(32, 8),
            nn.ELU(),
            nn.Linear(8, 3))

        self.value = nn.Sequential(
            nn.Linear(128, 32),
            nn.ELU(),
            nn.Linear(32, 8),
            nn.ELU(),
            nn.Linear(8, 2),
            nn.ELU(),
            nn.Linear(2, 1))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.06)

    def transformed(self, hour, fifteen, one, hidden):
        hour, hidden1 = self.encoder1h(hour, (hidden[0][0].to(self.device), hidden[0][1].to(self.device)))
        fifteen, hidden2 = self.encoder15min(fifteen, (hidden[1][0].to(self.device), hidden[1][1].to(self.device)))
        one, hidden3 = self.encoder1min(one, (hidden[2][0].to(self.device), hidden[2][1].to(self.device)))
        tensor = torch.stack([hour, fifteen, one]).reshape(-1, 3, 16).to(self.device)
        src_mask = self.transformer.generate_square_subsequent_mask(tensor.shape[1]).to(self.device)
        return func.relu(self.transformer(tensor, src_mask)), [hidden1, hidden2, hidden3]

    def SetPosition(self, hour, fifteen, one, hidden, softmax_dim=1):
        x, hidden = self.transformed(hour, fifteen, one, hidden)
        return func.softmax(self.position(x.clone()), dim=softmax_dim), hidden

    def Value(self, hour, fifteen, one, hidden):
        x, hidden = self.transformed(hour, fifteen, one, hidden)
        del hidden
        return func.elu(self.value(x))

    def TrainModelP(self, soneH, sfifteenM, soneM,
                    sprimeoneH, sprimefifteenM, sprimeoneM, hidden_in, hidden_out,
                    a, prob_a, reward, gamma=0.98, tau=0.97, weight=0.87,
                    eps_clip=0.1):

        (h1, h2, h3) = hidden_in
        (ho1, ho2, ho3) = hidden_out

        hidden_in = (
            (h1[0].detach(), h1[1].detach()), (h2[0].detach(), h2[1].detach()), (h3[0].detach(), h3[1].detach()))
        hidden_out = (
            (ho1[0].detach(), ho1[1].detach()), (ho2[0].detach(), ho2[1].detach()), (ho3[0].detach(), ho3[1].detach()))

        for k in range(10):
            td_target = reward + gamma * self.Value(sprimeoneH, sprimefifteenM, sprimeoneM, hidden_out)
            value = self.Value(soneH, sfifteenM, soneM, hidden_in)
            delta = (td_target - value).reshape(-1).detach()

            advantage_lst = []
            advantage = 0.0

            for i in range(delta.shape[0]):
                advantage = gamma * tau * advantage + delta[delta.shape[0] - 1 - i]
                advantage_lst.append([advantage])

            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.SetPosition(soneH, sfifteenM, soneM, hidden_in)

            pi_a = pi.gather(1, a.reshape(-1,1))
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a.clone().detach()))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 + eps_clip, 1 - eps_clip) * advantage

            lossL = -torch.min(surr1, surr2).mean().item() + func.smooth_l1_loss(value, td_target.detach()) * weight

            if not torch.isfinite(lossL):
                print("loss has not finited")
            else:
                #    nn.utils.clip_grad_norm_(leverage.parameters(), 0.5)
                torch.cuda.empty_cache()

                self.optimizer.zero_grad()
                lossL.backward()
                self.optimizer.step()
        del hidden_in, hidden_out, surr1, surr2, _, pi, pi_a, lossL, delta, value, td_target
        gc.collect()
        return
