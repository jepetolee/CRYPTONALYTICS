# import crypto packages
import ccxt as crypto_main
import pyupbit as upbit

# import sys
import sys

# import binance from ccxt
binance = crypto_main.binance()

# GUI
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, qApp, QWidget, QInputDialog
from PyQt5.QtGui import QIcon


# UI design
class CRYPTONALYTICS(QMainWindow):

    def __init__(self):
        super().__init__()
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
            upbit_file = open("./api/upbit","w")
            upbit_file.write(str(self.upbit_api))
            upbit_file.close()

    # binance_API_popup
    def binance_API(self):
        API, ok = QInputDialog.getText(self, '바이낸스 API 인증', "API를 입력해주세요:")
        if ok:
            self.binance_api = str(API)
            binance_file = open("./api/binance","w")
            binance_file.write(str(self.binance_api))
            binance_file.close()

    def initUI(self):
        # images,deposit, title
        self.setWindowTitle('CRYPTONALYTICS')
        self.setWindowIcon(QIcon('bitcoin.png'))
        self.setGeometry(400, 200, 900, 600)

        # exiting method
        exitAction = QAction('나가기', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        # keys
        upbit_action = QAction('업비트 API 입력', self)
        upbit_action.setShortcut('Ctrl+U')
        upbit_action.setStatusTip('check the upbit key')
        upbit_action.triggered.connect(self.upbit_API)

        binance_action = QAction('바이낸스 API 입력', self)
        binance_action.setShortcut('Ctrl+B')
        binance_action.setStatusTip('check the binance key')
        binance_action.triggered.connect(self.binance_API)

        # menubar & widget
        menubar = self.menuBar()

        # menu-settings
        filemenu = menubar.addMenu('설정')
        filemenu.addAction(exitAction)

        # menu- keys
        key = menubar.addMenu('API 입력')
        key.addAction(upbit_action)
        key.addAction(binance_action)

        self.show()


if __name__ == '__main__':
    # start_line
    app = QApplication(sys.argv)
    ex = CRYPTONALYTICS()
    sys.exit(app.exec_())
