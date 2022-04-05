import numpy as np
import sys

#sys.path.append('..')
from ValueCrypto import PositionDecisioner
from TradeAlgorithm import CurrentDataOut, TradeDataSetOut, update_future_1min_csv
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
from torch.distributions import Categorical


def TrainingTrader(symbol, device, saved=False, grad_lock=False):
    position = PositionDecisioner(50, device=device).to(device)

    if saved:
        position.load_state_dict('./model/' + symbol + '_position.pt')

    update_future_1min_csv(symbol)
    dataset = TradeDataSetOut(symbol)

    epoch = 1000

    best_score = 0.0
    selecter = True

    for _i in trange(epoch):
        total_score = 0.0
        s_o = torch.from_numpy(dataset[0]).float().to(device).reshape(1, -1, 1)
        s = torch.from_numpy(np.gradient(dataset[0])).float().to(device).reshape(1, -1, 1)

        position_keeper = 10
        for t in range(dataset.shape[0]):
            t = t

            s_prime = torch.from_numpy(np.gradient(dataset[t])).float().to(device).reshape(1, -1, 1)
            s_prime_o = torch.from_numpy(dataset[t]).float().to(device).reshape(1, -1, 1)
            if selecter:
                leverage_t = position.setleverage(s).reshape(-1)
                position_t = position.setposition(s).reshape(-1)
                position_a = Categorical(position_t).sample().item()
                leverage_a = Categorical(leverage_t).sample().item()
                leveragen = leverage_a + 1

                position_prob = position_t[position_a]
                selected_price = s_o[0][49].item()
                position_s_prime = s_prime
                leverage_prob = leverage_t[leverage_a]
                position_s = s

                if position_a == 0:
                    selecter = False
                    position_v = 'LONG'
                    best_reward = 0

                elif position_a == 1:
                    selecter = False
                    position_v = 'SHORT'
                    best_reward = 0
                else:
                    position_v = 'NONE'
                    best_reward = 0

                print(str(t) + "th," + str(position_v), position_a)

            if position_v == 'LONG':
                reward = (-100 + 100 * s_o[0][49].item() / selected_price) * leveragen
                reward_original = reward

            elif position_v == 'SHORT':
                reward = (100 - 100 * s_o[0][49].item() / selected_price) * leveragen
                reward_original = reward
            else:
                reward = -0.2

            if reward > best_reward:
                best_reward = reward
            else:
                reward_original = reward
                reward -= best_reward

            if reward < -2 or reward_original < -1:
                reward -= 0.4
                print("position changed_intentionally")
                selecter = True

            if position_v is not 'NONE':
                determiner = position.determine(s).reshape(-1)
                determined = Categorical(determiner).sample().item()
                determined_prob = determiner[determined]
                if determined == 0:
                    if position_keeper == 0:
                        total_score += reward_original
                        print("position changed")
                        reward -= 0.4
                        selecter = True
                        position_keeper = 10
                    else:
                        position_keeper -= 1
                        print("position changed but intended to keep it.")

                position.train_model_d(s, s_prime, determined, determined_prob, 10 * reward)

            if selecter:
                position.train_model_l(position_s, position_s_prime, position_a, position_prob, 5 * reward_original)
                position.train_model_p(position_s, position_s_prime, position_a, position_prob, 5 * reward_original)

            s = s_prime
            s_o = s_prime_o

            print(str(position_v) + " " + str(leveragen) + ": reward is " + str(reward_original) + " " + str(t))

        print(total_score)
        if best_score < total_score:
            best_score = total_score
            torch.save(position.state_dict(), './model/' + symbol + '_position.pt')


TrainingTrader('BTCUSDT', 'cpu')
