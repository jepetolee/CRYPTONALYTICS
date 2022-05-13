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
    plt.savefig('D:/CRYPTONALYTICS/TrainingModel/' + temper + '_day.png', dpi=100)
    plt.close('all')
    X = PIL.Image.open('D:/CRYPTONALYTICS/TrainingModel/' + temper + '_day.png').convert("L")
    x = np.array(X)
    X.close()
    return x


def DayRealChad(symbol, device, leveragen, impulse=-1, saved=False, grad_lock=False):  # (XRP,BNB,BTC,ETH)
    print("AwesomeChad")
    trader = Trader(device).to(device)
    client = Client(api_key="", api_secret="")

    if saved:
        trader.load_state_dict(torch.load('./model/' + symbol + '_trader.pt'))

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(size=(500, 500)),
                                transforms.Normalize(0.5, 0.5)])
    # <---------------------------------------------------------------------->
    current_time = datetime.now().strftime("%H:%M:%S")
    print(current_time + '에 시작')
    # <---------------------------------------------------------------------->

    onehour = client.futures_klines(symbol=symbol, interval='1d', limit=1500)
    onehour = pd.DataFrame(np.array(onehour, dtype=np.float)[-45:].T[1:6].T)
    onehour = build_numpy(onehour, symbol)
    s_oneH = trans(onehour).float().to(device).reshape(-1, 1, 500, 500)
    sleep(0.1)

    fifteen_data = client.futures_klines(symbol=symbol, interval='15m', limit=1500)
    fifteen_data = pd.DataFrame(np.array(fifteen_data, dtype=np.float)[-120:].T[1:6].T)
    fifteen_data = build_numpy(fifteen_data, symbol)
    s_oneF = trans(fifteen_data).float().to(device).reshape(-1, 1, 500, 500)

    oneminute_data = client.futures_klines(symbol=symbol, interval='4h', limit=1500)
    oneminute_data = pd.DataFrame(np.array(oneminute_data, dtype=np.float)[-60:].T[1:6].T)
    oneminute_data = build_numpy(oneminute_data, symbol)
    s_oneM = trans(oneminute_data).float().to(device).reshape(-1, 1, 500, 500)

    # <---------------------------------------------------------------------->
    hidden = (
        torch.zeros([1, 1, 16], dtype=torch.float).to(device), torch.zeros([1, 1, 16], dtype=torch.float).to(device))
    h_in = [hidden, hidden, hidden]
    benefit = 100
    selecter = True
    position = False
    t = 0
    locker, ring = 20, 1
    while True:
        sleep(1)
        # <---------------------------------------------------------------------->
        current_price = float(client.futures_symbol_ticker(symbol=symbol, limit=1500)['price'])
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
                selected_price = current_price
                selecter = False
            elif position_a == 1:
                position_v = 'SHORT'
                selected_price = current_price
                selecter = False
            else:
                position_v = 'NONE'
                selected_price = current_price
                selecter = False
            print(position_v + ': ', current_price)

        # <---------------------------------------------------------------------->
        difference = (0.9998 * current_price - selected_price)
        if position_v == 'SHORT':
            difference = (0.9998 * selected_price - current_price)
        if position_v == 'NONE' and difference > 0:
            difference *= -1

        # <---------------------------------------------------------------------->

        percent = leveragen * difference / selected_price * 100
        # <---------------------------------------------------------------------->
        if percent < impulse:
            selecter = True
            percent = impulse
            reward = -1
            ring = 1

        if difference > locker * ring:
            ring += 1
            position = True

        if position:
            if difference <= locker * (ring - 1)-10:
                difference = locker * (ring - 1)-10
                reward = ring-1
                selecter = True
                position = False
                ring = 1

        elif position_v is 'NONE':
            percent = 0
            sleep(10800)
            reward = 0
            ring = 1
            selecter = True

        if selecter:
            if position_v is not 'NONE':
                sleep(700)
                benefit *= (1 + percent / 100)

            onehour = client.futures_klines(symbol=symbol, interval='4h', limit=1500)
            onehour = pd.DataFrame(np.array(onehour, dtype=np.float)[-45:].T[1:6].T)
            onehour = build_numpy(onehour, symbol)
            sprime_oneH = trans(onehour).float().to(device).reshape(-1, 1, 500, 500)

            oneminute_data = client.futures_klines(symbol=symbol, interval='1h', limit=1500)
            oneminute_data = pd.DataFrame(np.array(oneminute_data, dtype=np.float)[-60:].T[1:6].T)
            oneminute_data = build_numpy(oneminute_data, symbol)
            sprime_oneM = trans(oneminute_data).float().to(device).reshape(-1, 1, 500, 500)
            fifteen_data = client.futures_klines(symbol=symbol, interval='15m', limit=1500)
            fifteen_data = pd.DataFrame(np.array(fifteen_data, dtype=np.float)[-120:].T[1:6].T)
            fifteen_data = build_numpy(fifteen_data, symbol)
            sprime_oneF = trans(fifteen_data).float().to(device).reshape(-1, 1, 500, 500)

            trader.TrainModelP(s_oneHP, s_oneFP, s_oneMP,
                               sprime_oneH, sprime_oneF, sprime_oneM, h_inP, h_outP,
                               position_a, position_prob, reward)
            torch.save(trader.state_dict(), './model/' + symbol + '_trader.pt')
            print(str(round(benefit, 2)) + "% " + position_v + " reward is " + str(round(percent, 2)),
                  current_price)

            s_oneM = sprime_oneM
            s_oneF = sprime_oneF
            s_oneH = sprime_oneH
            h_in = h_out

        if t % 75 == 0:
            print(current_price, percent)
        t += 1


DayRealChad('BTCUSDT', 'cpu', 125, impulse=-10, saved=True)  # 큰거 20 작은거 5

'''
SOLUSDT
 'TRXUSDT',
 'AVAXUSDT',
 'NEARUSDT',
 'USDCUSDT'               
'''
