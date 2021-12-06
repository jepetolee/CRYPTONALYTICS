#import crypto packages
import ccxt as crypto_main
import pyupbit as upbit

#global variables- keys
binance_key = " "
upbit_key = " "

#import sys
import sys

#import binance from ccxt
binance = crypto_main.binance()

#GUI
from PyQt5.QtWidgets import QApplication, QMainWindow,QAction,qApp,QWidget
from PyQt5.QtGui import QIcon

#UI design
class CRYPTONALYTICS(QMainWindow):

  def __init__(self):
      super().__init__()
      self.initUI()

  def initUI(self):
      #images,deposit, title
      self.setWindowTitle('CRYPTONALYTICS')
      self.setWindowIcon(QIcon('bitcoin.png'))
      self.setGeometry(400, 200, 900, 600)

      #exiting method
      exitAction = QAction( '나가기', self)
      exitAction.setShortcut('Ctrl+Q')
      exitAction.setStatusTip('Exit application')
      exitAction.triggered.connect(qApp.quit)

      #keys
      upbit_action = QAction('업비트 API 입력',self)
      upbit_action.setShortcut('Ctrl+U')
      upbit_action.setStatusTip('check the upbit key')
      upbit_action.triggered.connect(qApp.quit)

      binance_action = QAction('바이낸스 API 입력', self)
      binance_action.setShortcut('Ctrl+B')
      binance_action.setStatusTip('check the binance key')
      binance_action.triggered.connect(qApp.quit)

      #menubar & widget
      menubar = self.menuBar()


      #menu-settings
      filemenu = menubar.addMenu('설정')
      filemenu.addAction(exitAction)

      #menu- keys
      key = menubar.addMenu('API 입력')
      key.addAction(upbit_action)
      key.addAction(binance_action)

      self.show()



if __name__ == '__main__':
    #start_line
  app = QApplication(sys.argv)
  ex = CRYPTONALYTICS()
  sys.exit(app.exec_())
