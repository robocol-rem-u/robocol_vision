#!/usr/bin/env python3
import cv2
import numpy as np
import glob
import random
import os
import time

def get_maxima(coor,img):
    count = np.shape(coor)[0]
    sizes = []
    for i in range(count):
        if any(t < 0 for t in coor[i]):
            continue
        else:
            hight = (coor[i][3]-coor[i][1])
            width = (coor[i][2]-coor[i][0])
            area = hight*width
            sizes.append(area)
    if coor == []:
        return None
    else:
        maxima = np.array(sizes).argmax(axis=0)
        coor_max = coor[maxima][:]
        cv2.imshow('Station',img[coor_max[1]:coor_max[3],abs(coor_max[0]):coor_max[2],:])
        return coor_max


def turn(coor_max,img):
    shape = np.shape(img)
    margin = 150 #pixels margin
    center = np.array([shape[0]/2,shape[1]/2])
    center_up = center+margin
    center_down = center-margin
    center = [center_up,center_down]
    #Recognized mid point
    avg_y = (coor_max[3]+coor_max[1])/2
    avg_x = (coor_max[2]+coor_max[0])/2
    recog_mid = np.array([avg_y,avg_x])
    direction = 0
    if recog_mid[1] >= center[0][1]:
        direction = 2
        #print('Girar a la derecha')
    elif recog_mid[1] <= center[1][1]:
        direction = 1
        #print('Girar a la izquierda')
    elif recog_mid[1] >= center[1][1] and recog_mid[1] <= center[0][1]:
        direction = 3
        #print('Derecho')
    return direction



#

