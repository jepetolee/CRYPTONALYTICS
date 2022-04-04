from ValueCrypto import PositionDecisioner, Determiner, Leverage
from TradeAlgorithm import CurrentDataOut, TradeDataSetOut, update_future_1min_csv
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
from torch.distributions import Categorical


def TrainingTrader(symbol, device, saved=False, grad_lock=False):


    leverage, determines, position = Leverage(60, device=device).to(device), Determiner(60, device=device).to(device), \
                                     PositionDecisioner(60, device=device).to(device)
    if saved:
        leverage.load_state_dict('./model/' + symbol + '_leverage.pt')
        determines.load_state_dict('./model/' + symbol + '_determine.pt')
        position.load_state_dict('./model/' + symbol + '_position.pt')
    update_future_1min_csv(symbol)
    dataset = TradeDataSetOut(symbol)

    optimizer_position = torch.optim.Adam(position.parameters(), lr=0.03)
    optimizer_leverage = torch.optim.Adam(leverage.parameters(), lr=0.03)
    optimizer_determines = torch.optim.Adam(determines.parameters(), lr=0.03)

    epoch, loss_epoch, epoch_2 = 1000, 10, 3

    best_score = 0.0
    selecter = True
    gamma, tau, weight, eps_clip = 0.98, 0.97, 0.87, 0.1
    for _i in trange(epoch):
        total_score = 0.0
        s = torch.from_numpy(dataset[0]).float().to(device).reshape(-1,1)
        for t in range(dataset.shape[0] - 1):
            t = t + 1

            s_prime = torch.from_numpy(dataset[t]).float().to(device).reshape(-1,1)

            if selecter:
                leverage_t = leverage.setleverage(s)
                position_t = position.setposition(s)
                positioned = Categorical(position_t).sample()
                leveraged = Categorical(leverage_t).sample()
                leveragen = leveraged + 1
                position_a = positioned
                position_prob = position_t[positioned].item()
                selected_price = s[49]
                position_s_prime = s_prime
                leverage_a = leveraged
                leverage_prob = leverage_t[leveraged].item()
                position_s = s

                if positioned == 0:
                    selecter = False
                    position_v = 'LONG'

                elif positioned == 1:
                    selecter = False
                    position_v = 'SHORT'

                else:
                    position_v = 'NONE'
                print(t,position_v)
            else:
                determined = determines.determine(s)
                determined = Categorical(determined).sample()
                if determined == 0:
                    selecter = True

            if position_v == 'LONG':
                reward = (-100 + 100 * s[49] / selected_price) * leveragen
            elif position_v == 'SHORT':
                reward = (100 - 100 * s[49] / selected_price) * leveragen
            else:
                reward = 0
            total_score += reward
            if selecter:
                print(reward)
                for k in range(epoch_2):
                    td_target = r + gamma * position.value(position_s_prime)
                    value = position.value(position_s)
                    delta = td_target - value
                    pi = position.setposition(position_s)
                    pi_a = pi.gather(1, position_a)
                    ratio = torch.exp(torch.log(pi_a), torch.log(position_prob))
                    surr1 = ratio * delta
                    surr2 = torch.clamp(ratio, 1 + eps_clip, 1 - eps_clip) * delta
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(value, td_target.detach())
                    if not torch.isfinite(loss):
                        print("loss has not finited")
                    else:
                        if grad_lock:
                            nn.utils.clip_grad_norm_(position.parameters(), 0.5)
                        optimizer_position.zero_grad()
                        loss.backward()
                        optimizer_position.step()

                    td_target = r + gamma * leverage.value(position_s_prime)
                    value = leverage.value(position_s)
                    delta = td_target - value
                    pi = leverage.setleverage(position_s)
                    pi_a = pi.gather(1, leverage_a)
                    ratio = torch.exp(torch.log(pi_a), torch.log(leverage_prob))
                    surr1 = ratio * delta
                    surr2 = torch.clamp(ratio, 1 + eps_clip, 1 - eps_clip) * delta
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(value, td_target.detach())
                    if not torch.isfinite(loss):
                        print("loss has not finited")
                    else:
                        if grad_lock:
                            nn.utils.clip_grad_norm_(leverage.parameters(), 0.5)
                        optimizer_leverage.zero_grad()
                        loss.backward()
                        optimizer_leverage.step()
            else:
                for v in range(loss_epoch):
                    td_target = r + gamma * determines.value(s_prime)
                    value = determines.value(s)
                    delta = td_target - value
                    pi = determines.determine(s)
                    pi_a = pi.gather(1, leverage_a)
                    ratio = torch.exp(torch.log(pi_a), torch.log(leverage_prob))
                    surr1 = ratio * delta
                    surr2 = torch.clamp(ratio, 1 + eps_clip, 1 - eps_clip) * delta
                    loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(value, td_target.detach())
                    if not torch.isfinite(loss):
                        print("loss has not finited")
                    else:
                        if grad_lock:
                            nn.utils.clip_grad_norm_(leverage.parameters(), 0.5)
                        optimizer_leverage.zero_grad()
                        loss.backward()
                        optimizer_leverage.step()

            s = s_prime
        print(total_score)
        if best_score < total_score:
            best_score = total_score
            torch.save(leverage.state_dict(), './model/' + symbol + '_leverage.pt')
            torch.save(determine.state_dict(), './model/' + symbol + '_determine.pt')
            torch.save(position.state_dict(), './model/' + symbol + '_position.pt')


TrainingTrader('BTCUSDT', 'cpu')
