import warnings
import numpy as np
import sys
import PIL
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


def pixelart():
    print(' ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
    print('■                                                                                 ■')
    print('■                                CRYPTONALYTICS                                   ■')
    print('■                                                                                 ■')
    print('■                                     THE                                         ■')
    print('■                              MAESTRO OF TRADER                                  ■')
    print('■                                                                                 ■')
    print('■                                                                                 ■')
    print('■                                                     MADE BY JEPETOLEE           ■')
    print('■                                                                                 ■')
    print(' ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')


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


def DayRealWorld(symbol, device, leveragen, saved=False, grad_lock=False):  # (XRP,BNB,BTC,ETH)

    trader = Trader(device).to(device)
    client = sc.FutureAgent(api_key=""
                            , api_secret="")
    client.check_account()

    client.change_leverage(leveragen)
    pixelart()
    if saved:
        trader.load_state_dict(torch.load('./model/' + symbol + '_trader.pt'))

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(size=(500, 500)),
                                transforms.Normalize(0.5, 0.5)])

    onehour = client.agent.futures_klines(symbol=symbol, interval='1h', limit=1500)
    onehour = pd.DataFrame(np.array(onehour, dtype=np.float)[-60:].T[1:6].T)
    onehour = build_numpy(onehour, symbol)
    s_oneH = trans(onehour).float().to(device).reshape(-1, 1, 500, 500)

    sleep(0.1)
    fifteen_data = client.agent.futures_klines(symbol=symbol, interval='15m', limit=1500)
    fifteen_data = pd.DataFrame(np.array(fifteen_data, dtype=np.float)[-120:].T[1:6].T)
    fifteen_data = build_numpy(fifteen_data, symbol)
    s_oneF = trans(fifteen_data).float().to(device).reshape(-1, 1, 500, 500)

    sleep(0.1)
    four_hour = client.agent.futures_klines(symbol=symbol, interval='4h', limit=1500)
    four_hour = pd.DataFrame(np.array(four_hour, dtype=np.float)[-45:].T[1:6].T)
    four_hour = build_numpy(four_hour, symbol)
    s_fourH = trans(four_hour).float().to(device).reshape(-1, 1, 500, 500)

    hidden = (torch.zeros([1, 1, 16], dtype=torch.float).to(device), torch.zeros([1, 1, 16], dtype=torch.float)
              .to(device))
    h_in = [hidden, hidden, hidden]

    account_info = client.agent.futures_account()
    for asset in account_info["assets"]:
        if asset["asset"] == "USDT":
            av_balance = float(asset["initialMargin"])
            account = float(asset["availableBalance"])
            saved_account = account

    benefit = 100
    selecter = True
    while True:

        if selecter:
            current_price = float(client.agent.futures_symbol_ticker(symbol=symbol, limit=1500)['price'])
            with torch.no_grad():
                position_t, h_out = trader.SetPosition(s_fourH, s_oneH, s_oneF, h_in)

            position_t = position_t.detach().reshape(-1)
            position_a = Categorical(position_t).sample()
            position_prob = position_t[position_a.item()]

            s_oneHP = s_oneH
            s_FourHP = s_fourH
            s_oneFP = s_oneF

            h_inP = h_in
            h_outP = h_out
            if position_a == 0:
                position_v = 'LONG'
                selecter = False
                calling_size = float(int(np.floor(1000 * saved_account * leveragen / current_price)) / 1000)
                try:
                    client.agent.futures_create_order(symbol=symbol, type='STOP_MARKET', timeInForce='GTC',
                                                      stopPrice=current_price - 300
                                                      , side='SELL', quantity=calling_size, closePosition='true')
                    client.agent.futures_create_order(symbol=symbol, type='LIMIT', timeInForce='GTC',
                                                      price=current_price, side='BUY', quantity=calling_size)
                    client.agent.futures_create_order(symbol=symbol, type='TAKE_PROFIT_MARKET', timeInForce='GTC',
                                                      stopPrice=current_price + 300
                                                      , side='BUY', quantity=calling_size, closePosition='true')
                except:
                    pass

            elif position_a == 1:
                position_v = 'SHORT'
                calling_size = float(int(np.floor(1000 * saved_account * leveragen / current_price)) / 1000)
                try:
                    client.agent.futures_create_order(symbol=symbol, type='STOP_MARKET', timeInForce='GTC',
                                                      stopPrice=current_price + 300
                                                      , side='BUY', quantity=calling_size, closePosition='true')
                    client.agent.futures_create_order(symbol=symbol, type='LIMIT', timeInForce='GTC',
                                                      price=current_price, side='SELL', quantity=calling_size)
                    client.agent.futures_create_order(symbol=symbol, type='TAKE_PROFIT_MARKET', timeInForce='GTC',
                                                      stopPrice=current_price - 300
                                                      , side='BUY', quantity=calling_size, closePosition='true')
                except:
                    pass

                for i in range(20):
                    account_info = client.agent.futures_account()
                    av_balance = None
                    for asset in account_info["assets"]:
                        if asset["asset"] == "USDT":
                            av_balance = float(asset["initialMargin"])
                    if av_balance > 1:
                        selecter = False
                        break
                    else:
                        sleep(1)
                if selecter:
                    client.agent.futures_cancel_all_open_orders(symbol=symbol)
                    free_pass = True

            else:
                position_v = 'NONE'
                stop_price = current_price
                selected_price = current_price
                selecter = False

            print(position_v + ': ', current_price)

        account_info = client.agent.futures_account()
        av_balance, account = None, None
        for asset in account_info["assets"]:
            if asset["asset"] == "USDT":
                av_balance = float(asset["initialMargin"])
        sleep(1)

        if av_balance == 0:
            account_info = client.agent.futures_account()
            av_balance = None
            for asset in account_info["assets"]:
                if asset["asset"] == "USDT":
                    av_balance = float(asset["initialMargin"])
                    account = float(asset["availableBalance"])
            if av_balance == 0:
                selecter = True
                if position_v != 'NONE':
                    if account >= saved_account:
                        difference = 300
                        reward = 1
                    else:
                        difference = -300
                        reward = -1.5
                    saved_account = account
                else:
                    reward = -0.125
                    sleep(900)
                    difference = 0
                percent = leveragen * difference / selected_price * 100

        if selecter:
            if free_pass:
                free_pass = False
            else:
                if position_v is not 'NONE':
                    benefit *= (1 + percent / 100)
                    print(str(round(benefit, 2)) + "% " + position_v + " reward is " + str(round(percent, 2)))

                sleep(0.1)
                onehour = client.agent.futures_klines(symbol=symbol, interval='1h', limit=1500)
                onehour = pd.DataFrame(np.array(onehour, dtype=np.float)[-60:].T[1:6].T)
                onehour = build_numpy(onehour, symbol)
                sprime_oneH = trans(onehour).float().to(device).reshape(-1, 1, 500, 500)

                sleep(0.1)
                four_hour = client.agent.futures_klines(symbol=symbol, interval='4h', limit=1500)
                four_hour = pd.DataFrame(np.array(four_hour, dtype=np.float)[-45:].T[1:6].T)
                four_hour = build_numpy(four_hour, symbol)

                sleep(0.1)
                sprime_fourH = trans(four_hour).float().to(device).reshape(-1, 1, 500, 500)
                fifteen_data = client.agent.futures_klines(symbol=symbol, interval='15m', limit=1500)
                fifteen_data = pd.DataFrame(np.array(fifteen_data, dtype=np.float)[-120:].T[1:6].T)
                fifteen_data = build_numpy(fifteen_data, symbol)
                sprime_oneF = trans(fifteen_data).float().to(device).reshape(-1, 1, 500, 500)

                trader.TrainModelP(s_FourHP, s_oneHP, s_oneFP,
                                   sprime_fourH, sprime_oneH, sprime_oneF,
                                   h_inP, h_outP, position_a, position_prob, reward)

                torch.save(trader.state_dict(), './model/' + symbol + '_trader.pt')

                s_fourH = sprime_fourH
                s_oneF = sprime_oneF
                s_oneH = sprime_oneH
                h_in = h_out


DayRealWorld('BTCUSDT', 'cpu', 5, saved=True)
