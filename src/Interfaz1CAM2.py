#!/usr/bin/env python3
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import cv2
import PyQt5
from PyQt5 import QtGui
from PyQt5.QtGui import * #QPainter, QFont, QColor, QPixmap, QPen, QBrush, QImage, QIcon
from PyQt5.QtCore import * #Qt, QRect, QPoint
from PyQt5.QtWidgets import * #QMainWindow, QAction, qApp, QApplication, QWidget, QApplication, QCheckBox, QLineEdit, QLabel, QSlider, QLCDNumber, QCalendarWidget, \
   # QGroupBox, QRadioButton, QVBoxLayout, QPushButton, QFontDialog, QHBoxLayout,


class MyWindow(QMainWindow):
	def __init__(self):
		super(MyWindow,self).__init__()
		self.initUI()
		self.thread= camarathread()

	def button_clicked(self):
	    print("clicked")

	def initUI(self):
	    self.setGeometry(200, 200, 300, 300)
	    self.setWindowTitle("ventana")

	    self.label = QtWidgets.QLabel(self)
	    self.label.setText("my first label!")
	    self.label.move(50,50)

	    self.b1 = QtWidgets.QPushButton(self)
	    self.b1.setText("click me!")
	    self.b1.clicked.connect(self.button_clicked)
	    
	
class camarathread(QThread):

    output = pyqtSignal(QRect,QImage)
    def __init__(self, parent=None):
    	QThread.__init__(self,parent)
    	self.exiting=False
    	self.size=QSize(0,0)
    	self.stars=0
    	
    		
    	self.capture3 = cv2.VideoCapture(0)
    	self.capture3.set(3,820)
    	self.capture3.set(4,600)
    
    	#self.text = QLabel('No camera detected', self)
    	#self.text.setFont(QFont('arial',12))
    	#self.text.setStyleSheet("color:rgb(255,255,255)")
    	#self.text.move(395,330)
    
    	self.image3 = QLabel(' ',self)
    	self.image3.setGeometry(40,5,800, 700)
    
    
    	self.timer5 = QTimer(self)
    	self.timer5.setInterval(int(1000/30))
    	self.timer5.timeout.connect(self.get_frame5)
    	self.timer5.start()
    
    	self.setFixedSize(900,680)

def get_frame5(self):
    global lecturaparametros
    global greenLower
    check, frame = self.capture3.read()
    if not check:
    	print("Camara1 Desconectada")
    	self.capture3.release()
    	apagada3 = False
    	self.timer5.stop()
    	self.timer6.start()
    else:
    	image = QImage(frame, *frame.shape[1::-1], QImage.Format_RGB888).rgbSwapped()
    	pixmap = QPixmap.fromImage(image)

    	self.image3.setPixmap(pixmap)
        # print("termino get frame")
	    
	    #self.name=PicButton(QPixmap('sombra1.png'),self)
        #self.name.clicked.connect(self.selectSWTLSW)
        #self.name.move(20,400)
        #self.name.setFixedSize(150,195)



        #self.name=PicButton(QPixmap('but1.png'),self)
        #self.name.clicked.connect(self.selectSWTLSW)
        #self.name.move(30,450)
        #self.name.setFixedSize(50,30)

        #self.name=PicButton(QPixmap('but2.png'),self)
        #self.name.clicked.connect(self.selectSWTLSW)
        #self.name.move(95,450)
        #self.name.setFixedSize(50,30)

        #self.name=PicButton(QPixmap('but3.png'),self)
        #self.name.clicked.connect(self.selectSWTLSW)
        #self.name.move(65,410)
        #self.name.setFixedSize(50,30)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())






