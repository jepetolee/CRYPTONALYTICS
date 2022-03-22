class TrainingAgent:
    def __init__(self):
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
            ty = 0
        return

    def select_symbol(self, symbol):
        self.symbol = symbol
        return self.symbol

    # 에이전트 스탭
    def step(self, position):

        if position[1] == 'HOLD':
            return 0
        else:


                self.position = position[1]
                self.reverse_position()

                self.position_price = position[0]
                self.quantity = position[2]
                self.time_steps += 1
                self.isposition = True
                return self.percent()




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
