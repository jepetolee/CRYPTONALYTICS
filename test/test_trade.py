import gc
import time
from numba import jit
from tqdm import trange

api_key = ""
api_secret = ""
from binance.client import Client


def make_symbol_list(tickers):
    ticker_list = list()
    for i in range(len(tickers)):
        ticker_list.append(list(tickers[i].values())[0])
    del tickers
    gc.collect()
    return ticker_list


def make_ticker_list(tickers):
    ticker_list = list()
    for i in range(len(tickers)):
        ticker_list.append(list(tickers[i].values()))
    del tickers
    gc.collect()
    return ticker_list


client = Client(api_key=api_key, api_secret=api_secret)
tickers = client.get_all_tickers()
ts = time.time()
symbol = make_symbol_list(tickers)
print(time.time() - ts)

for i in trange(len(symbol)):
    day = client.get_historical_klines(symbol=symbol[i], interval='1d', start_str="2020-01-01", limit=1000)
    day
    print("finished")
