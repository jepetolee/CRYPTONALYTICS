#import crypto packages
import ccxt as crypto_main
import pyupbit as upbit

#global variables- keys
binance

#import sys
import sys

#import binance from ccxt
binance = crypto_main.binance()

#GUI
from PyQt5.QtWidgets import QApplication, QMainWindow,QAction,qApp
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

      #menubar & widget
      menubar = self.menuBar()
      menubar.setNativeMenuBar(False)

      #menu-settings
      filemenu = menubar.addMenu('설정')
      filemenu.addAction(exitAction)

      #menu- keys
      key = menubar.addMenu('키값 넣기')

      self.show()



if __name__ == '__main__':
    #start_line
  app = QApplication(sys.argv)
  ex = CRYPTONALYTICS()
  sys.exit(app.exec_())
