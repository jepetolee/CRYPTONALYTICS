from ValueCrypto import PositionDecisioner, Determiner, Leverage
from TradeAlgorithm import CurrentDataOut, TradeDataSetOut, update_future_1min_csv
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
from torch.distributions import Categorical


def TrainingTrader(symbol, device, saved=False, grad_lock=False):
    leverage, determines, position = Leverage(50, device=device).to(device), Determiner(50, device=device).to(device), \
                                     PositionDecisioner(50, device=device).to(device)

    if saved:
        leverage.load_state_dict('./model/' + symbol + '_leverage.pt')
        determines.load_state_dict('./model/' + symbol + '_determine.pt')
        position.load_state_dict('./model/' + symbol + '_position.pt')
    update_future_1min_csv(symbol)
    dataset = TradeDataSetOut(symbol)

    epoch = 1000

    best_score = 0.0
    selecter = True

    for _i in trange(epoch):
        total_score = 0.0
        s = torch.from_numpy(dataset[0]).float().to(device).reshape(1, -1, 1)
        for t in range(dataset.shape[0] - 1):
            t = t + 1
            s_prime = torch.from_numpy(dataset[t]).float().to(device).reshape(1, -1, 1)

            if selecter:
                leverage_t = leverage.setleverage(s).reshape(-1)
                position_t = position.setposition(s).reshape(-1)
                position_a = Categorical(position_t).sample().item()
                leverage_a = Categorical(leverage_t).sample().item()
                leveragen = leverage_a + 1

                position_prob = position_t[position_a]
                selected_price = s[0][49].item()
                position_s_prime = s_prime
                leverage_prob = leverage_t[leverage_a]
                position_s = s

                if position_a == 0:
                    selecter = False
                    position_v = 'LONG'

                elif position_a == 1:
                    selecter = False
                    position_v = 'SHORT'

                else:
                    position_v = 'NONE'
                print(str(t) + "th," + str(position_v))

                if position_v == 'LONG':
                    reward = (-100 + 100 * s[0][49].item() / selected_price) * leveragen

                elif position_v == 'SHORT':
                    reward = (100 - 100 * s[0][49].item() / selected_price) * leveragen
                else:
                    reward = 0
                total_score += reward

            if position_v is not 'NONE':
                determiner = determines.determine(s).reshape(-1)
                determined = Categorical(determiner).sample().item()
                determined_prob = determiner[determined]
                if determined == 0:
                    print("position changed")
                    selecter = True
                determines.train_model(s, s_prime, determined, determined_prob, reward)

            if selecter:
                position.train_model(position_s, position_s_prime, position_a, position_prob, reward)
                leverage.train_model(position_s, position_s_prime, leverage_a, leverage_prob, reward)

            s = s_prime
            print("reward is " + str(reward))

        print(total_score)
        if best_score < total_score:
            best_score = total_score
            torch.save(leverage.state_dict(), './model/' + symbol + '_leverage.pt')
            torch.save(determine.state_dict(), './model/' + symbol + '_determine.pt')
            torch.save(position.state_dict(), './model/' + symbol + '_position.pt')


TrainingTrader('BTCUSDT', 'cpu')
