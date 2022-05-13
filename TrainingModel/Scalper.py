import time
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
from TradeAlgorithm import DatasetFinal, TradeDataSetOut, update_future_15min_csv, update_future_1min_csv, \
    update_future_1hour_csv
from tqdm import tqdm, trange
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
import pandas as pd
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ScalpingEnv as sc
from mpl_finance import candlestick2_ohlc


def build_numpy(data, temper):
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(2, 4)
    axes = list()
    axes.append(plt.subplot(gs[0, :]))
    axes.append(plt.subplot(gs[1, :], sharex=axes[0]))
    candlestick2_ohlc(axes[0], data[0], data[1], data[2], data[3], width=1, colorup='r', colordown='b')
    axes[1].bar(data.index + 1, data[4], color='k', width=0.8, align='center')
    copies = data[3].copy()
    axes[0].plot(data.index + 1, copies.rolling(window=3).mean(), label='Ma3')
    axes[0].plot(data.index + 1, copies.rolling(window=14).mean(), label='Ma14')
    plt.savefig('D:/CRYPTONALYTICS/TrainingModel/' + temper + '.png', dpi=100)
    plt.close('all')
    X = PIL.Image.open('D:/CRYPTONALYTICS/TrainingModel/' + temper + '.png').convert("L")
    x = np.array(X)
    X.close()
    return x


def ScalpRealWorld(symbol, device, leveragen, shaker=0.43, impulse=-1.5, saved=False,
                   grad_lock=False):  # (XRP,BNB,BTC,ETH)
    print("스켈핑")
    trader = Trader(device).to(device)
    client = Client(api_key="", api_secret="")

    if saved:
        trader.load_state_dict(torch.load('./model/' + symbol + '_scalper.pt'))

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(size=(500, 500)),
                                transforms.Normalize(0.5, 0.5)])
    # <---------------------------------------------------------------------->

    current_time = datetime.now().strftime("%H:%M:%S")
    print(current_time + '에 시작')

    # <---------------------------------------------------------------------->

    onehour = client.futures_klines(symbol=symbol, interval='1h', limit=1500)
    onehour = pd.DataFrame(np.array(onehour, dtype=np.float)[-72:].T[1:6].T)
    onehour = build_numpy(onehour, symbol)
    s_oneH = trans(onehour).float().to(device).reshape(-1, 1, 500, 500)
    sleep(0.1)

    fifteen_data = client.futures_klines(symbol=symbol, interval='15m', limit=1500)
    fifteen_data = pd.DataFrame(np.array(fifteen_data, dtype=np.float)[-90:].T[1:6].T)
    fifteen_data = build_numpy(fifteen_data, symbol)
    s_oneF = trans(fifteen_data).float().to(device).reshape(-1, 1, 500, 500)

    oneminute_data = client.futures_klines(symbol=symbol, interval='1m', limit=1500)
    oneminute_data = pd.DataFrame(np.array(oneminute_data, dtype=np.float)[-120:].T[1:6].T)
    oneminute_data = build_numpy(oneminute_data, symbol)
    s_oneM = trans(oneminute_data).float().to(device).reshape(-1, 1, 500, 500)

    # <---------------------------------------------------------------------->
    hidden = (
        torch.zeros([1, 1, 16], dtype=torch.float).to(device), torch.zeros([1, 1, 16], dtype=torch.float).to(device))
    h_in = [hidden, hidden, hidden]
    total_score = 0.0
    benefit = 100
    selecter = True
    position = 0
    t = 0
    changer, urge = shaker, impulse
    while True:
        onehour = client.futures_klines(symbol=symbol, interval='1h', limit=1500)
        onehour = pd.DataFrame(np.array(onehour, dtype=np.float)[-72:].T[1:6].T)
        onehour = build_numpy(onehour, symbol)
        sprime_oneH = trans(onehour).float().to(device).reshape(-1, 1, 500, 500)

        sleep(0.1)
        oneminute_data = client.futures_klines(symbol=symbol, interval='1m', limit=1500)
        oneminute_data = pd.DataFrame(np.array(oneminute_data, dtype=np.float)[-120:].T[1:6].T)
        oneminute_data = build_numpy(oneminute_data, symbol)
        sprime_oneM = trans(oneminute_data).float().to(device).reshape(-1, 1, 500, 500)

        sleep(0.1)
        fifteen_data = client.futures_klines(symbol=symbol, interval='15m', limit=1500)
        fifteen_data = pd.DataFrame(np.array(fifteen_data, dtype=np.float)[-90:].T[1:6].T)
        fifteen_data = build_numpy(fifteen_data, symbol)
        sprime_oneF = trans(fifteen_data).float().to(device).reshape(-1, 1, 500, 500)

        sleep(0.1)
        current_price = float(client.futures_symbol_ticker(symbol=symbol, limit=1500)['price'])

        sleep(0.1)
        # <---------------------------------------------------------------------->

        if selecter:
            with torch.no_grad():
                position_t, h_out = trader.SetPosition(s_oneH, s_oneF, s_oneM, h_in)

            position_t = position_t.detach().reshape(-1)
            position_a = Categorical(position_t).sample()
            position_prob = position_t[position_a.item()]

            s_oneHP = s_oneH
            s_oneMP = s_oneM
            s_oneFP = s_oneF

            h_inP = h_in
            h_outP = h_out
            if position_a == 0:
                position_v = 'LONG'
                best_reward = 0
                selected_price = current_price

                selecter = False
            elif position_a == 1:
                position_v = 'SHORT'
                best_reward = 0
                selected_price = current_price

                selecter = False
            else:
                position_v = 'NONE'
                best_reward = 0
                selected_price = current_price
                selecter = False
            print(position_v + ': ', current_price)

        # <---------------------------------------------------------------------->
        reward = (0.9998 * current_price - selected_price) / selected_price * 100

        if position_v == 'SHORT':
            reward = (0.9998 * selected_price - current_price) / selected_price * 100

        if position_v == 'NONE' and reward > 0:
            reward_original = 0
        else:
            reward *= leveragen

        reward_original = reward

        if reward > best_reward:
            best_reward = reward
        else:
            reward -= best_reward

        if t % 15 == 0:
            print(current_price, reward_original)
        if t % 45 == 0:
            if reward_original < -1.2:
                urge *= 0.95

        if reward_original < urge:
            if reward_original < impulse:
                reward_original = impulse
            selecter = True
            changer = shaker
            urge = impulse

        elif reward_original > changer:  # 큰거 -0.76 작은거:-0.43
            changer = shaker
            selecter = True
            urge = impulse

        elif position_v is 'NONE':
            selecter = True
            changer = shaker
            urge = impulse
            time.sleep(300)
        # <---------------------------------------------------------------------->

        if selecter:
            if position_v is not 'NONE':
                sleep(120)
                benefit *= (1 + reward_original / 100)
            print(str(round(benefit, 2)) + "% " + position_v + " reward is " + str(round(reward_original, 2)),
                  current_price)
            if reward_original < 0:
                reward_original *= 10

            trader.TrainModelP(s_oneHP, s_oneFP, s_oneMP,
                               sprime_oneH, sprime_oneF, sprime_oneM, h_inP, h_outP,
                               position_a, position_prob, reward_original * 5)
            torch.save(trader.state_dict(), './model/' + symbol + '_scalper.pt')

        s_oneM = sprime_oneM
        s_oneF = sprime_oneF
        s_oneH = sprime_oneH

        h_in = h_out
        if t % 75 == 0:
            if reward_original > 0 and abs(reward) < changer:
                changer *= 0.75

        t += 1


ScalpRealWorld('BTCUSDT', 'cpu', 20, saved=True, shaker=5, impulse=-5, grad_lock=False)
