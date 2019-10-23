import sys
import os
from PyQt5 import QtGui, QtCore, QtWidgets, uic 
from PyQt5.QtGui import QIcon


class Window(QtWidgets.QMainWindow):

    def __init__(self):
        super(Window, self).__init__()


        uic.loadUi('Main_UI.ui', self)
	self.setWindowIcon(QtGui.QIcon('Icons/Logo.png'))

 	self.btn1.setIcon(QtGui.QIcon('Icons/Scatter.png'))
	self.btn1.setIconSize(QtCore.QSize(50,70))

 	self.btn2.setIcon(QtGui.QIcon('Icons/Histogram.png'))
	self.btn2.setIconSize(QtCore.QSize(60,70))

 	self.btn3.setIcon(QtGui.QIcon('Icons/Bar.png'))
	self.btn3.setIconSize(QtCore.QSize(55,70))

 	self.btn4.setIcon(QtGui.QIcon('Icons/BoxPlot.png'))
	self.btn4.setIconSize(QtCore.QSize(60,70))

 	self.btn5.setIcon(QtGui.QIcon('Icons/Predicate.png'))
	self.btn5.setIconSize(QtCore.QSize(55,70))



 	self.extractAction = QtWidgets.QAction(QIcon('Icons/About.png'),"About", self)
       	self.extractAction.setShortcut("Ctrl+A")
        self.extractAction.setStatusTip('Info on the project') 
	self.extractAction.triggered.connect(self.info)
	self.options.addAction(self.extractAction)

 	
        self.extractAction = QtWidgets.QAction(QIcon('Icons/Quit.png'),"Quit", self)
       	self.extractAction.setShortcut("Ctrl+Q")
        self.extractAction.setStatusTip('Leave the application')    
	self.extractAction.triggered.connect(self.leave)
	self.options.addAction(self.extractAction)
	
	
	self.button1()
	self.button2()
	self.button3()
	self.button4()
	self.button5()
		

    def button1(self):
	self.btn1.clicked.connect(self.button1_code)

    def button2(self):
	self.btn2.clicked.connect(self.button2_code)

    def button3(self):
	self.btn3.clicked.connect(self.button3_code)

    def button4(self):
	self.btn4.clicked.connect(self.button4_code)

    def button5(self):
	self.btn5.clicked.connect(self.button5_code)
	

    def button1_code(self):
	os.system('python Graphs/scatter.py')

    def button2_code(self):
	os.system('python Graphs/histogram.py')

    def button3_code(self):
	os.system('python Graphs/barg.py')

    def button4_code(self):
	os.system('python Graphs/boxplot.py')

    def button5_code(self):
	os.system('python Graphs/prediction.py > output.txt')
	os.system('gedit output.txt')

    def info(self):
	os.system('gedit README.txt')	

    def leave(self):
      	sys.exit()

	



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    GUI = Window()
    GUI.show()
    sys.exit(app.exec_())






		


