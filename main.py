#import crypto packages
import ccxt as crypto_main
import pyupbit as upbit

#import sys
import sys

#import binance from ccxt
binance = crypto_main.binance()

#GUI
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon


#UI design
class CRYPTONALYTICS(QWidget):

  def __init__(self):
      super().__init__()
      self.initUI()

  def initUI(self):
      self.setWindowTitle('CRYPTONALYTICS')
      self.setWindowIcon(QIcon('bitcoin.png'))
      self.setGeometry(400, 200, 900, 600)

      menu = self.

      self.show()



if __name__ == '__main__':
    #start_line
  app = QApplication(sys.argv)
  ex = CRYPTONALYTICS()
  sys.exit(app.exec_())
