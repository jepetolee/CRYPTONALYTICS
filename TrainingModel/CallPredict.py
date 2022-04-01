from Prophet import FutureProphetFifteenMinute, FutureProphetOneHour
from TradeAlgorithm import update_future_1hour_csv, update_future_15min_csv


def call_1hour_prophets():
    update_future_1hour_csv()
    FutureProphetOneHour()


def call_15min_prophets():
    update_future_15min_csv()
    FutureProphetFifteenMinute()

