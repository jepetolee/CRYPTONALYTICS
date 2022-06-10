import warnings
import numpy as np
import sys
import PIL

from binance import ThreadedWebsocketManager
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.append('..')
from time import sleep
from ValueCrypto import Trader
from binance.client import Client
from datetime import datetime
from make_pd import *
from torchvision import transforms
from TradeAlgorithm import *
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
import pandas as pd
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_finance import candlestick2_ohlc


def PretrainingTrader(symbol, device, leveragen, saved=False, grad_lock=False):
    trader = Trader(device).to(device)
    if saved:
        trader.load_state_dict(torch.load('./model/' + symbol + '_trader.pt'))

    DatasetFinal(symbol)
    dataset = TradeDataSetOut(symbol)
    epoch = 1000
    total = 98136 - 72000
    selecter = True
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(size=(500, 500)),
                                transforms.Normalize(0.5, 0.5)])

    total_score = 0.0
    for _i in range(epoch):

        hidden = (torch.zeros([1, 1, 16], dtype=torch.float), torch.zeros([1, 1, 16], dtype=torch.float))
        hidden_in = [hidden, hidden, hidden]

        sprime_oneHP = PIL.Image.open(
            'D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/1hour/1201.png').convert("L")
        s_oneHP = PIL.Image.open(
            'D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/1hour/1200.png').convert("L")
        sprime_oneH = trans(sprime_oneHP).float().to(device).reshape(-1, 1, 500, 500)
        s_oneH = trans(s_oneHP).float().to(device).reshape(-1, 1, 500, 500)
        sprime_oneHP.close()
        s_oneHP.close()

        sprime_fifteenMP = PIL.Image.open(
            'D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/15min/5181.png').convert("L")
        s_fifteenMP = PIL.Image.open(
            'D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/15min/5180.png').convert("L")

        sprime_fifteenM = trans(sprime_fifteenMP).float().to(device).reshape(-1, 1, 500, 500)
        s_fifteenM = trans(s_fifteenMP).float().to(device).reshape(-1, 1, 500, 500)
        sprime_fifteenMP.close()
        s_fifteenMP.close()

        temp1, temp2 = 1202, 5182
        benefit = 100
        for t in range(total - 7141):
            t += 7141 + 72000
            data = dataset.copy()[t]

            if t % 60 == 0:
                s_oneH = sprime_oneH
                sprime_oneHP = PIL.Image.open('D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/1hour/' + str(
                    temp1) + '.png').convert("L")

                sprime_oneH = trans(sprime_oneHP).float().to(device).reshape(-1, 1, 500, 500)
                sprime_oneHP.close()
                temp1 += 1

            if t % 15 == 0:
                s_fifteenM = sprime_fifteenM
                sprime_fifteenMP = PIL.Image.open(
                    'D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/15min/' + str(
                        temp2) + '.png').convert("L")
                sprime_fifteenM = trans(sprime_fifteenMP).float().to(device).reshape(-1, 1, 500, 500)
                sprime_fifteenMP.close()
                temp2 += 1

            if selecter:

                s_oneMP = PIL.Image.open(
                    'D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/1min/' + str(t - 1) + '.png').convert("L")
                s_oneM = trans(s_oneMP).float().to(device).reshape(-1, 1, 500, 500)
                s_oneMP.close()

                sprime_oneMP = PIL.Image.open(
                    'D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/1min/' + str(t) + '.png').convert("L")

                sprime_oneM = trans(sprime_oneMP).float().to(device).reshape(-1, 1, 500, 500)
                sprime_oneMP.close()

                if t == 7141 + 72000:
                    h_inP = hidden_in
                else:
                    torch.cuda.empty_cache()
                    h_inP = h_outP

                with torch.no_grad():
                    position_t, h_outP = trader.SetPosition(s_oneH, s_fifteenM, s_oneM, h_inP)

                position_t = position_t.detach().reshape(-1)
                position_a = Categorical(position_t).sample().item()
                position_prob = position_t[position_a]

                selected_price = data[59].item()
                s_prime_select_1h = sprime_oneH
                s_prime_select_15m = sprime_fifteenM
                s_prime_select_1m = sprime_oneM

                s_select_1h = s_oneH
                s_select_15m = s_fifteenM
                s_select_1m = s_oneM

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
                    selecter = False

            reward = (-100 + 100 * data[59].item() / selected_price)

            if position_v == 'SHORT':
                reward *= -1

            if position_v == 'NONE' and reward > 0:
                reward *= -1

            else:
                reward *= leveragen

            reward_original = reward

            if reward > best_reward:
                best_reward = reward
            else:
                reward_original = reward
                reward -= best_reward

            if reward_original < -1:
                reward_original = -1.0198
                selecter = True

            elif reward < -1:

                reward_original = (-100 + 99.96 * data[59].item() / selected_price)
                if position_v == 'SHORT':
                    reward_original *= -1

                if position_v == 'NONE' and reward > 0:
                    reward_original *= -1

                else:
                    reward_original *= leveragen
                selecter = True
            elif position_v is 'NONE' and reward_original < -0.03:
                selecter = True

            if selecter:
                if position_v is not 'NONE':
                    benefit *= 1 + reward_original / 100

                temp = reward_original
                if reward_original < 0:
                    reward_original *= 5

                trader.TrainModelP(s_select_1h, s_select_15m, s_select_1m,
                                   s_prime_select_1h, s_prime_select_15m, s_prime_select_1m, h_inP, h_outP,
                                   position_a, position_prob, reward_original)

                reward_original = temp

            if t % 100 == 0:
                print(str(t) + ": total_benenfit is " + str(
                    round(benefit, 2)) + ",and " + position_v + " reward is " + str(
                    round(reward_original, 2)))

            if total_score < benefit:
                total_score = benefit
                torch.save(trader.state_dict(), './model/' + symbol + '_trader.pt')


def PretrainingTrader2(symbol, device, leveragen, saved=False, grad_lock=False):
    trader = Trader(device).to(device)
    if saved:
        trader.load_state_dict(torch.load('./model/' + symbol + '_FINAL.pt'))
    # <---------------------------------------------------------------------->
    #  DatasetFinal2(symbol)
    dataset = TradeData2SetOut(symbol)

    epoch = 10
    total = 92477 - 600-34944

    # <---------------------------------------------------------------------->
    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(size=(500, 500)),
                                transforms.Normalize(0.5, 0.5)])

    # <---------------------------------------------------------------------->
    for _i in range(epoch):

        hidden = (torch.zeros([1, 1, 16], dtype=torch.float), torch.zeros([1, 1, 16], dtype=torch.float))
        hidden_in = [hidden, hidden, hidden]

        # <---------------------------------------------------------------------->
        s_FourHP = PIL.Image.open('D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/4hour/2183.png').convert("L")
        s_FourH = trans(s_FourHP).float().to(device).reshape(-1, 1, 500, 500)
        s_FourHP.close()

        s_OneHP = PIL.Image.open('D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/1hour/8855.png').convert("L")
        s_OneH = trans(s_OneHP).float().to(device).reshape(-1, 1, 500, 500)
        s_OneHP.close()

        s_FiftMP = PIL.Image.open(
            'D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/15min2/35543.png').convert(
            "L")
        s_FiftM = trans(s_FiftMP).float().to(device).reshape(-1, 1, 500, 500)
        s_FiftMP.close()
        # <---------------------------------------------------------------------->
        temp1, temp2 = 2184, 8856
        benefit = 100
        locker, ring = 5, 1
        position = False
        selecter = True
        count = 0
        win =0
        # <---------------------------------------------------------------------->
        for v in range(total):

            if v % 16 == 0:
                temp1 += 1
            if v % 4 == 0:
                temp2 += 1

            t = v + 600+34944
            data = dataset.copy()[t - 1]  # price
            # <---------------------------------------------------------------------->
            if selecter:
                if v == 0:
                    h_inP = hidden_in
                else:
                    torch.cuda.empty_cache()
                    h_inP = h_outP

                with torch.no_grad():
                    position_t, h_outP = trader.SetPosition(s_FourH, s_OneH, s_FiftM, h_inP)

                position_t = position_t.detach().reshape(-1)
                position_a = Categorical(position_t).sample()
                position_prob = position_t[position_a.item()]

                selected_price = data[119][1].item()

                s_select_4h = s_FourH
                s_select_1h = s_OneH
                s_select_15m = s_FiftM

                if position_a == 0:
                    selecter = False
                    position_v = 'LONG'
                    count += 1
                elif position_a == 1:
                    selecter = False
                    position_v = 'SHORT'
                    count += 1
                else:
                    position_v = 'NONE'
                    selecter = False
            '''
            reward_close = leveragen * (0.9998 * data[119][1].item() - selected_price) / selected_price * 100
            reward_low = leveragen * (0.9998 * data[119][0].item() - selected_price) / selected_price * 100
            reward_high = leveragen * (0.9998 * data[119][2].item() - selected_price) / selected_price * 100

            if position_v == 'SHORT':
                reward_close = leveragen * (0.9998 * selected_price - data[119][1].item()) / selected_price * 100
                reward_low = leveragen * (0.9998 * selected_price - data[119][0].item()) / selected_price * 100
                reward_high = leveragen * (0.9998 * selected_price - data[119][2].item()) / selected_price * 100'''

            reward_close = 0.9998 * data[119][1].item() - selected_price
            reward_low = 0.9998 * data[119][0].item() - selected_price
            reward_high = 0.9998 * data[119][2].item() - selected_price

            if position_v == 'SHORT':
                reward_close = 0.9998 * selected_price - data[119][1].item()
                reward_low = 0.9998 * selected_price - data[119][0].item()
                reward_high = 0.9998 * selected_price - data[119][2].item()

            # <---------------------------------------------------------------------->

            percent = reward_close
            '''
            if percent > reward_close:
                percent = reward_close
            if percent > reward_low:
                percent = reward_low
            if percent > reward_high:
                percent = reward_high
               '''
            # <---------------------------------------------------------------------->

            if percent > 150:

                ring += 1
                position = True
                selecter = True
                reward = 1
                win+=1
                percent = 5
                print(position_v + " reward:" + str(round(100*win/count, 2)))

            if percent < -150:
                percent = -5
                reward = -1
                selecter = True

            if position_v == 'NONE':
                percent = 0
                selecter = True
                reward = 0

            if selecter:
                if position_v is not 'NONE':
                    benefit *= 1 + percent / 100

                print(str(t) + ": total_benenfit is " + str(
                        round(benefit, 2)) + ",and " + position_v + " reward is " + str(
                        round(percent, 2)))

                sprime_FourHP = PIL.Image.open('D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/4hour/' + str(
                    temp1) + '.png').convert("L")
                sprime_FourH = trans(sprime_FourHP).float().to(device).reshape(-1, 1, 500, 500)
                sprime_FourHP.close()

                sprime_OneHP = PIL.Image.open(
                    'D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/1hour/' + str(
                        temp2) + '.png').convert("L")
                sprime_OneH = trans(sprime_OneHP).float().to(device).reshape(-1, 1, 500, 500)
                sprime_OneHP.close()

                sprime_FiftMP = PIL.Image.open(
                    'D:/CRYPTONALYTICS/TradeAlgorithm/dataset/' + symbol + '/15min2/' + str(t) + '.png').convert("L")
                sprime_FiftM = trans(sprime_FiftMP).float().to(device).reshape(-1, 1, 500, 500)
                sprime_FiftMP.close()

                trader.TrainModelP(s_select_4h, s_select_1h, s_select_15m,
                                   sprime_FourH, sprime_OneH, sprime_FiftM, h_inP, h_outP,
                                   position_a, position_prob, reward)

                s_FiftM = sprime_FiftM
                s_OneH = sprime_OneH
                s_FourH = sprime_FourH

                torch.save(trader.state_dict(), './model/' + symbol + '_FINAL.pt')


PretrainingTrader2('BTCUSDT', 'cpu', 5, saved=True, grad_lock=False)
