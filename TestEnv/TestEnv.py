from time import sleep
from pandas import DataFrame


class TrainingAgent:
    def __init__(self, api_key, api_secret, test=False):


        self.leverage = 1
        # 현재 잔고 조회
        self.withdrawAvailable = 1
        self.shutdown = False
        self.time_steps = 0
        self.position = 'BUY'

        self.symbol = 'BTCUSDT'

        # 현재 내가 진입한 포지션 가격
        self.position_price = 1
        # 지금 내가 들고 있는 개수
        self.quantity = 7
        # 현재가
        self.current_price = 1
        # 내가 지정한 종료가
        self.stop_price = 38000
        self.profit_price = 37000
        self.isposition = False
        self.account = 1
        self.current_percent = 1
        self.check_account()
        self.seed_money = self.account

    # 포지션 변경(내부값)
    def reverse_position(self):

        if self.position == 'SELL':
            self.position = 'BUY'
        else:
            self.position = 'SELL'



    # 해당 모델의 손해가 지정 %가 넘어버렸을 경우 재빠르게 종료
    def finisher(self, percent=-5):
        if self.percent() <= percent:
            ty=0
        return

    def select_symbol(self, symbol):
        self.symbol = symbol
        return self.symbol

    # 에이전트 스탭
    def step(self, position, retry=False):

        if position[1] == 'HOLD':
            return 0
        else:
            try:
                self.agent.futures_cancel_all_open_orders(symbol=self.symbol)

                self.position = position[1]
                self.reverse_position()

                self.position_price = position[0]
                self.quantity = position[2]
                self.time_steps += 1
                self.check_account()
                self.isposition = True
                return self.percent()
            except BinanceAPIException as e:
                if retry is True:
                    print("Agent: 주문 에러 형성! 강제종료에 진입합니다." + str(e))
                    self.shutdown = True

                    return
                else:
                    print("Agent: 주문이 안됬어요! 다시 시도해볼게요.")
                    self.step(position, retry=True)

    # 현 계좌 체크
    def check_account(self):
        try:
            account = DataFrame.from_dict(account)

            account = account.loc[account['asset'] == 'USDT']

            self.account = float(account['balance'].values)
            self.withdrawAvailable = float(account['withdrawAvailable'].values)
        except BinanceAPIException as e:
            self.shutdown = True
            print("Agent: 계좌와 연동에 실패했습니다! api를 확인해주세요!" + str(e))
        return

    # 포지션 종료가 형성
    def define_TPSL(self, profit_position, stop_position):
        self.profit_price = profit_position
        self.stop_price = stop_position
        return

    # 레버리지 변경
    def change_leverage(self, leverage):
        self.leverage = leverage
        return

    # 순수익 측정
    def interests(self):
        return self.account - self.seed_money

    # 마진거래 채결중 현재 수익/손실 측정
    def percent(self):
        return self.leverage * (100 - 100 * (self.current_price / self.position_price))


