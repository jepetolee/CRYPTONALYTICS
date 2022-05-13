import gc
import time
from numba import jit
from tqdm import trange
# from TradeAlgorithm import trace_current_price as tc
from binance.client import Client
import os

api_key = ""
api_secret = ""

client = Client(api_key=api_key, api_secret=api_secret)

#client.futures_cancel_all_open_orders(symbol= 'BTCUSDT')
# -----------------------------------------------------------------------

# client.futures_cancel_order(symbol='BTCUSDT', orderId=52495941955)
'''
price = float(client.futures_symbol_ticker(symbol='BTCUSDT', limit=1500)['price'])
client.futures_create_order(symbol='BTCUSDT', type='TAKE_PROFIT_MARKET', timeInForce='GTC',stopPrice=price-30, side='BUY',
                         quantity='0.001',closePosition='true')'''

'''price = float(client.futures_symbol_ticker(symbol='BTCUSDT', limit=1500)['price'])
client.futures_create_order(symbol='BTCUSDT', type='STOP_MARKET', timeInForce='GTC', stopPrice=32935, side='SELL',
                           quantity='0.010',closePosition='true')'''

# ----------------------------------------------------------
current_order = client.futures_get_all_orders()
account_info = client.futures_account()

av_balance = None
for asset in account_info["assets"]:
    print(asset)
    if asset["asset"] == "USDT":
        av_balance = float(asset["availableBalance"])
print(av_balance)

for order in current_order:
    print(order)

'''while True:
    price = float(client.futures_symbol_ticker(symbol='BTCUSDT', limit=1500)['price'])
    print(price)

    time.sleep(8)'''
