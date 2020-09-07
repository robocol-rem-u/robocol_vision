#!/usr/bin/env python3
import sys
import cv2
#from PIL import ImageGrab

import PyQt5
from PyQt5 import QtGui
from PyQt5.QtGui import * #QPainter, QFont, QColor, QPixmap, QPen, QBrush, QImage, QIcon
from PyQt5.QtCore import * #Qt, QRect, QPoint
from PyQt5.QtWidgets import * #QMainWindow, QAction, qApp, QApplication, QWidget, QApplication, QCheckBox, QLineEdit, QLabel, QSlider, QLCDNumber, QCalendarWidget, \
   # QGroupBox, QRadioButton, QVBoxLayout, QPushButton, QFontDialog, QHBoxLayout, QFileDialog, QComboBox, QListWidgetItem, QTableWidget, QTableWidgetItem, QListWidget


from collections import deque
from imutils.video import VideoStream
#from scipy.interpolate import interp1d
import numpy as np
import argparse
import cv2
import imutils
import time
#import pyautogui, sys


import cv2
import os

import numpy as np






class PicButton(QAbstractButton):

    #global ALLDATA
    #global DUMPINFO

    def __init__(self, pixmap, parent=None):
        super(PicButton, self).__init__(parent)
        self.pixmap = pixmap

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(event.rect(), self.pixmap)


    def sizeHint(self):
        return self.pixmap.size()





print("esta funcionando xd")



global apagada1
global apagada2
global apagada3


apagada1=True
apagada2=True
apagada3=True

greenLower = (58,49, 125)
greenUpper = (87, 157, 255)


hmn=0
hmx=0
smn=0
smx=0
vmn=0
vmx=0

listicay=[0,0,0,0,0]
listicax=[0,0,0,0,0]
listota=[0]
contador=0
diam1y=0
diam2y=0
diam1x=0
diam2x=0
posix=0
posiy=0	
   
clicXX=0
clicYY=0
    
haypelota1=False
haypelota2=False
listo=False
primeraV=0
activo=0
supderx=0
supdery=0

global lecturaparametros
lecturaparametros=True

class Example(QWidget):

    def __init__(self):
        super().__init__()


 
        
#Camara 1



        self.capture3 = cv2.VideoCapture(0)
        self.capture3.set(3,820)
        self.capture3.set(4,600)



        self.text = QLabel('No camera detected', self)
        self.text.setFont(QFont('arial',12))
        self.text.setStyleSheet("color:rgb(255,255,255)")
        self.text.move(395,330)



        self.image3 = QLabel(' ',self)
        #self.image3.setStyleSheet('border: gray; border-style:solid; border-width: 1px;')
        self.image3.setGeometry(40,5,800, 700)



        xxx=-650
        yyy=20




        self.timer5 = QTimer(self)
        self.timer5.setInterval(int(1000/30))
        self.timer5.timeout.connect(self.get_frame5)
        self.timer5.start()

        #self.timer6 = QTimer(self)
        #self.timer6.setInterval(int(2000))
        #self.timer6.timeout.connect(self.get_frame6)
        #self.timer6.start()






        self.name=PicButton(QPixmap('sombra1.png'),self)
        #self.name.clicked.connect(self.selectSWTLSW)
        self.name.move(20,400)
        self.name.setFixedSize(150,195)



        self.name=PicButton(QPixmap('but1.png'),self)
        #self.name.clicked.connect(self.selectSWTLSW)
        self.name.move(30,450)
        self.name.setFixedSize(50,30)

        self.name=PicButton(QPixmap('but2.png'),self)
        #self.name.clicked.connect(self.selectSWTLSW)
        self.name.move(95,450)
        self.name.setFixedSize(50,30)

        self.name=PicButton(QPixmap('but3.png'),self)
        #self.name.clicked.connect(self.selectSWTLSW)
        self.name.move(65,410)
        self.name.setFixedSize(50,30)




        self.setFixedSize(900,680)




    def get_frame5(self):
        global lecturaparametros
        global greenLower
        global greenUpper
        global apagada3

        # self.capture = cv2.VideoCapture(1)
        #if apagada3 == True:
        check, frame = self.capture3.read()
        #print("entro al primer if")

        if not check:
            print("Camara1 Desconectada")
            self.capture3.release()
            apagada3 = False
            self.timer5.stop()
            self.timer6.start()

            # self.timer3.stop()

            # self.capture2 = cv2.VideoCapture(0)
            # self.capture2.set(3,520)
            # self.capture2.set(4,440)
            # self.timer3.start()

        else:
            image = QImage(frame, *frame.shape[1::-1], QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(image)

            self.image3.setPixmap(pixmap)
        # print("termino get frame")



#    def get_frame6(self):
#        global apagada3
#        self.capture3 = cv2.VideoCapture(0)
#        self.capture3.set(3,320)
#        self.capture3.set(4,240)
#        print("entro a get 2 frame")
##        if apagada3==False:
#       print("entro a GET FRAME 2")
#        check, frame = self.capture3.read()
 #      if check:
#        print("Camara conectada")
#        #self.capture.release()
#        apagada=1
#        self.timer5.start()
#        self.timer6.stop()
#        print("salio de get frame 2")




    def paintEvent(self, event):

        qp = QPainter()
        qp.begin(self)
        self.drawImage(event, qp, 'fondo2.png')
        qp.end()
        ####################
        self.update()
        #self.close()

    def drawImage(self, event, qp, image):
        pixmap = QPixmap(image)
        qp.drawPixmap(event.rect(), pixmap)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    win = Example()
    win.show()
    sys.exit(app.exec_())






