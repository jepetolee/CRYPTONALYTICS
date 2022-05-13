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


def DayRealWorld(symbol, device, leveragen, impulse=-1, saved=False, grad_lock=False):  # (XRP,BNB,BTC,ETH)

    trader = Trader(device).to(device)
    client = sc.FutureAgent(api_key=""
                            , api_secret="")
    client.check_account()
    sleep(0.2)
    free_pass = False
    client.change_leverage(leveragen)
    pixelart()
    if saved:
        trader.load_state_dict(torch.load('./model/' + symbol + '_trader.pt'))

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(size=(500, 500)),
                                transforms.Normalize(0.5, 0.5)])
    # <---------------------------------------------------------------------->

    current_time = datetime.now().strftime("%H:%M:%S")
    print("일일 거래")
    print(current_time + '에 시작')

    # <---------------------------------------------------------------------->

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

    # <---------------------------------------------------------------------->
    hidden = (
        torch.zeros([1, 1, 16], dtype=torch.float).to(device), torch.zeros([1, 1, 16], dtype=torch.float).to(device))
    h_in = [hidden, hidden, hidden]
    benefit = 100
    selecter = True
    t = 0
    best_difference = 100
    ring = 0
    # <---------------------------------------------------------------------->
    while True:
        sleep(0.5)
        # <---------------------------------------------------------------------->
        current_price = float(client.agent.futures_symbol_ticker(symbol=symbol, limit=1500)['price'])
        sleep(1)
        account_info = client.agent.futures_account()
        av_balance, account = None, None
        for asset in account_info["assets"]:
            if asset["asset"] == "USDT":
                av_balance = float(asset["initialMargin"])
                account = float(asset["availableBalance"])

        # <---------------------------------------------------------------------->
        if selecter:


            with torch.no_grad():
                position_t, h_out = trader.SetPosition(s_fourH, s_oneH, s_oneF, h_in)

            position_t = position_t.detach().reshape(-1)
            position_a = Categorical(position_t).sample()
            position_prob = position_t[position_a.item()]

            # <---------------------------------------------------------------------->
            s_oneHP = s_oneH
            s_FourHP = s_fourH
            s_oneFP = s_oneF

            h_inP = h_in
            h_outP = h_out
            if position_a == 0:
                position_v = 'LONG'
                selected_price = current_price
                selecter = False
                calling_size = float(int(np.floor(1000 * account * leveragen / current_price)) / 1000)
                stop_price = int((current_price + impulse) / 10) * 10
                try:
                    client.agent.futures_create_order(symbol=symbol, type='STOP_MARKET', timeInForce='GTC',
                                                      stopPrice=stop_price
                                                      , side='SELL', quantity=calling_size, closePosition='true')
                    client.agent.futures_create_order(symbol=symbol, type='LIMIT', timeInForce='GTC',
                                                      price=current_price, side='BUY', quantity=calling_size)
                except:
                    pass

            elif position_a == 1:
                position_v = 'SHORT'
                selected_price = current_price
                calling_size = float(int(np.floor(1000 * account * leveragen / current_price)) / 1000)
                stop_price = int((current_price - impulse) / 10) * 10
                try:
                    client.agent.futures_create_order(symbol=symbol, type='STOP_MARKET', timeInForce='GTC',
                                                      stopPrice=stop_price
                                                      , side='BUY', quantity=calling_size, closePosition='true')
                    client.agent.futures_create_order(symbol=symbol, type='LIMIT', timeInForce='GTC',
                                                      price=current_price, side='SELL', quantity=calling_size)
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
                    sleep(1)
                if selecter:
                    client.agent.futures_cancel_all_open_orders(symbol=symbol)
                    free_pass = True
            else:
                position_v = 'NONE'
                stop_price = current_price
                selected_price = current_price
                selecter = False
                sleep(100)
            print(position_v + ': ', current_price)

        # <---------------------------------------------------------------------->
        difference = (0.9998 * current_price - selected_price)
        if position_v == 'SHORT':
            difference = (0.9998 * selected_price - current_price)
        if position_v == 'NONE' and difference > 0:
            difference *= -1

        # <---------------------------------------------------------------------->
        if difference > best_difference:
            best_difference = difference + 65
            client.agent.futures_cancel_all_open_orders(symbol=symbol)
            if position_v == 'LONG':
                stop_price_saved = stop_price
                stop_price = int((current_price - 60) / 10) * 10
                try:
                    client.agent.futures_create_order(symbol=symbol, type='STOP_MARKET', timeInForce='GTC',
                                                      stopPrice=stop_price
                                                      , side='SELL', quantity=calling_size, closePosition='true')
                except:
                    client.agent.futures_create_order(symbol=symbol, type='STOP_MARKET', timeInForce='GTC',
                                                      stopPrice=stop_price_saved
                                                      , side='BUY', quantity=calling_size, closePosition='true')

            if position_v == 'SHORT':
                stop_price_saved = stop_price
                stop_price = int((current_price + 60) / 10) * 10
                try:
                    client.agent.futures_create_order(symbol=symbol, type='STOP_MARKET', timeInForce='GTC',
                                                      stopPrice=stop_price
                                                      , side='BUY', quantity=calling_size, closePosition='true')
                except:
                    client.agent.futures_create_order(symbol=symbol, type='STOP_MARKET', timeInForce='GTC',
                                                      stopPrice=stop_price_saved
                                                      , side='BUY', quantity=calling_size, closePosition='true')

            ring += int(difference / 100) + 1
            reward = ring

        percent = leveragen * difference / selected_price * 100
        if av_balance == 0:
            account_info = client.agent.futures_account()
            av_balance = None
            for asset in account_info["assets"]:
                if asset["asset"] == "USDT":
                    av_balance = float(asset["initialMargin"])
            if av_balance == 0:
                if ring == 0:
                    reward = -1
                selecter = True
                ring = 0
                difference = (0.9998 * stop_price - selected_price)
                if position_v == 'SHORT':
                    difference = (0.9998 * selected_price - stop_price)
                if position_v == 'NONE' and difference > 0:
                    difference *= -1

        if position_v is 'NONE':
            reward = -0.25
            ring = 0
            sleep(300)
            selecter = True
        # <---------------------------------------------------------------------->
        if selecter:
            if free_pass:
                free_pass = False
            else:
                if position_v is not 'NONE':
                    benefit *= (1 + percent / 100)
                    print(str(round(benefit, 2)) + "% " + position_v + " reward is " + str(round(percent, 2)),
                          stop_price)

                print("POSITION FINISHED")
                client.check_account()
                difference = 0
                best_difference = 100

                # <---------------------------------------------------------------------->
                sleep(1)
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
                                   sprime_fourH, sprime_oneH, sprime_oneF, h_inP, h_outP,
                                   position_a, position_prob, reward)
                torch.save(trader.state_dict(), './model/' + symbol + '_trader.pt')
                # <---------------------------------------------------------------------->

                selected_price = 0
                s_fourH = sprime_fourH
                s_oneF = sprime_oneF
                s_oneH = sprime_oneH
                h_in = h_out

        # <---------------------------------------------------------------------->
        if t % 200 == 0:
            print(current_price, percent)
        t += 1
        # <---------------------------------------------------------------------->


DayRealWorld('BTCUSDT', 'cpu', 3, impulse=-100, saved=True)

# <---------------------------------------------------------------------->
'''
if difference > locker * counter:
    if not position:
        client.agent.futures_cancel_all_open_orders(symbol=symbol)
        if position_v == 'LONG':
            stop_price = stop_price+locker
            client.agent.futures_create_order(symbol=symbol, type='STOP_MARKET', timeInForce='GTC',
                                              stopPrice=stop_price
                                              , side='SELL', quantity=calling_size, closePosition='true')
        if position_v == 'SHORT':
            stop_price = stop_price-locker
            client.agent.futures_create_order(symbol=symbol, type='STOP_MARKET', timeInForce='GTC',
                                              stopPrice=stop_price
                                              , side='BUY', quantity=calling_size, closePosition='true')
        counter += 1'''

# <---------------------------------------------------------------------->
