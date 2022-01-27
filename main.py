# import crypto packages
import ccxt as crypto_main
import pyupbit as upbit

# import settings
import sys
import time

# import binance from ccxt
binance = crypto_main.binance()

# GUI
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtChart import *

# series data
from pandas import *

#call libs
from analytics import *




# UI design
class CRYPTONALYTICS(QMainWindow):

    def __init__(self):
        super().__init__()
        self.centralwidget = QStackedWidget()
        self.initUI()

        # global variables- keys
        binance_file = open("./api/binance", "r")
        self.binance_api = str(binance_file.read())
        binance_file.close()

        upbit_file = open("./api/upbit", "r")
        self.upbit_api = str(upbit_file.read())
        upbit_file.close()

    # upbit_API_popup
    def upbit_API(self):
        API, ok = QInputDialog.getText(self, '업비트 API 인증', "API를 입력해주세요:")
        if ok:
            self.upbit_api = str(API)
            upbit_file = open("./API/upbit", "w")
            upbit_file.write(str(self.upbit_api))
            upbit_file.close()

    # binance_API_popup
    def binance_API(self):
        API, ok = QInputDialog.getText(self, '바이낸스 API 인증', "API를 입력해주세요:")
        if ok:
            self.binance_api = str(API)
            binance_file = open("./API/binance", "w")
            binance_file.write(str(self.binance_api))
            binance_file.close()

    # change to analytic widget
    def change2analytics(self):
        self.centralwidget.setCurrentIndex(1)

    # change to main widget
    def change2info(self):
        self.centralwidget.setCurrentIndex(0)

    @pyqtSlot(float)
    def get_price(self, cur_price):
        # append current price
        dt = QDateTime.currentDateTime()
        self.statusBar().showMessage(dt.toString())
        self.ticks[dt] = cur_price

        # check whether minute changed
        # if dt.time().minute() != self.minute_cur.time().minute():

        ts = dt.toMSecsSinceEpoch()

        sets = self.series.sets()
        last_set = sets[-1]

        open = last_set.open()
        high = last_set.high()
        low = last_set.low()
        close = last_set.close()
        ts1 = last_set.timestamp()
        self.series.remove(last_set)  # remove last set

        new_set = QCandlestickSet(open, high, low, cur_price, ts1)
        self.series.append(new_set)

    def initUI(self):
        # images,deposit, title
        self.setWindowTitle('CRYPTONALYTICS')
        self.setWindowIcon(QIcon('./images/bitcoin.png'))
        self.resize(1200, 741)

        # backgorund method

        back_action = QAction('돌아가기', self)
        back_action.setShortcut("Ctrl+X")
        back_action.setStatusTip('화면으로 돌아갑니다')
        back_action.triggered.connect(lambda: self.change2info())

        # exiting method
        exitAction = QAction('나가기', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        # keys
        upbit_action = QAction('업비트 API 입력', self)
        upbit_action.setShortcut('Ctrl+U')
        upbit_action.setStatusTip('check the upbit api')
        upbit_action.triggered.connect(self.upbit_API)

        binance_action = QAction('바이낸스 API 입력', self)
        binance_action.setShortcut('Ctrl+B')
        binance_action.setStatusTip('check the binance api')
        binance_action.triggered.connect(self.binance_API)

        # analytic method
        chart_analysis = QAction('기술적 분석', self)
        chart_analysis.setShortcut('Ctrl+A')
        chart_analysis.setStatusTip('차트를 분석합니다')
        chart_analysis.triggered.connect(lambda: self.change2analytics())

        # menubar & widget
        menubar = self.menuBar()

        # menu-settings
        filemenu = menubar.addMenu('설정')
        filemenu.addAction(exitAction)
        filemenu.addAction(back_action)

        # menu- keys
        key = menubar.addMenu('API 입력')
        key.addAction(upbit_action)
        key.addAction(binance_action)

        # menu-chart
        analytics = menubar.addMenu('기술적 분석')
        analytics.addAction(chart_analysis)  # 로그차트만

        # reinforcement trading
        auto_trading = menubar.addMenu('자동 투자')
        # 자동투자 인공지능 스켈핑- 예측 투자후 안되면 빠지기-생존의 스켈핑 즉 주식을 오래 보유 하지 않고 이득을 먹고 탁탁 나오게끔\
        # 바이낸스 이자율 0.02% 99.98%의 자금 손실비
        # 2 자동투자 스윙 투자: 상승장, 박스장, 하락장을 구분할 수 있어야함.
        # 메커니즘: 사용자가 종목 중 확고한 종목을 여러개 선정하여 분산투자가 가능하게 함. 또한 위험률(적용률)을 자기가 선정해 위험에 맞는 투자를 꾀하게 함.
        # 확증 편향 제거

        auto_portfolio = menubar.addMenu('자동투자 포트 폴리오')
        # 수익률 로그 그래프, 잔고

        training = menubar.addMenu('인공지능 성능 업데이트')
        # 신경망 파일 학습

        # info

        main_layout = QVBoxLayout()

        main_info = QLabel(self)
        main_info.setAlignment(Qt.AlignCenter)
        main_image = QPixmap('./Images/main.jpg')
        main_info.setPixmap(main_image)
        main_info.resize(main_image.width(), main_image.height())

        main_text = QLabel("성공적인투자, 안정적인 손실", self)
        main_text.setAlignment(Qt.AlignCenter)
        main_font = main_text.font()
        main_font.setPointSize(30)
        main_text.setFont(main_font)

        main_layout.addWidget(main_info)
        main_layout.addWidget(main_text)

        info_widget = QWidget()
        info_widget.setLayout(main_layout)

        self.centralwidget.addWidget(info_widget)

        self.setCentralWidget(self.centralwidget)

        self.show()


if __name__ == '__main__':
    # start_line
    def my_exception_hook(exctype, value, traceback):
        # Print the error and traceback
        print(exctype, value, traceback)
        # Call the normal Exception hook after
        sys._excepthook(exctype, value, traceback)
        # sys.exit(1)


    # Back up the reference to the exceptionhook
    sys._excepthook = sys.excepthook
    # Set the exception hook to our wrapping function
    sys.excepthook = my_exception_hook
    app = QApplication(sys.argv)
    ex = CRYPTONALYTICS()
    sys.exit(app.exec_())
