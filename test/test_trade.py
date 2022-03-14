import gc
import time
from numba import jit
from tqdm import trange
#from TradeAlgorithm import trace_current_price as tc

import os
api_key = ""
api_secret = ""
from binance.client import Client

client = Client(api_key=api_key, api_secret=api_secret)


while True:

    time.sleep(5)
    os.system('cls')
    ticker1 = client.futures_orderbook_ticker(symbol='ZECUSDT')
    ticker1m = client.get_orderbook_ticker(symbol='ZECUSDT')
    ticker2 = client.futures_orderbook_ticker(symbol='BTCUSDT')
    ticker2m = client.get_orderbook_ticker(symbol='BTCUSDT')

    print("future:" + str(ticker1['symbol']) + ' ' + str(ticker1['askPrice']) + ' ' + str(ticker2['symbol']) + ' ' + str(
            ticker2['askPrice']))

    print("market:" + str(ticker1m['symbol']) + ' ' + str(ticker1m['askPrice']) + ' ' + str(
        ticker2m['symbol']) + ' ' + str(ticker2m['askPrice']))

