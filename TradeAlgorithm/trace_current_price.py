"""
현재 가격을 계속해서 받아오는 함수
"""
from gc import collect


def make_symbol_list(tickers):
    ticker_list = list()
    for i in range(len(tickers)):

        if list(tickers[i].values())[0][-4:] == "USDT":
            ticker_list.append(list(tickers[i].values())[0])
    del tickers
    # delete datas to prevent cpu's out of memory
    collect()
    return ticker_list


def make_ticker_list(tickers):
    ticker_list = list()
    for i in range(len(tickers)):
        if list(tickers[i].values())[0][-4:] == "USDT":
            ticker_list.append(list(tickers[i].values()))
    # delete datas to prevent cpu's out of memory
    del tickers
    collect()
    return ticker_list
