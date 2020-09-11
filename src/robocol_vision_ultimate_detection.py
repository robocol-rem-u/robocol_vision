#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from std_msgs.msg import Float32

import imutils
import threading
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import PIL
from cv2 import aruco
import glob
import os
#from PIL import Image



global aruco_dict
global parameters,folder_path
global detectamos,rango,direccion,profundidad,CoorMsg, x1,x2,y1,y2, constante
global CoorMsg, imgMsg,imgMsgMask


aruco_dict= aruco.custom_dictionary(20,7)
aruco_dict.bytesList = np.empty(shape = (20, 7, 4), dtype = np.uint8)


#0
mybits1 = np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,1,1,1,1,1,0],[0,1,1,1,1,1,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[0] = aruco.Dictionary_getByteListFromBits(mybits1)
#1
mybits2 = np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,0,0,1,1,0,0],[0,1,1,1,0,1,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[1] = aruco.Dictionary_getByteListFromBits(mybits2)
#2
mybits3 = np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,1,0,1,1,0,0],[0,1,0,1,1,0,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[2] = aruco.Dictionary_getByteListFromBits(mybits3)
#3
mybits4 = np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,0,1,1,1,1,0],[0,1,0,1,0,0,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[3] = aruco.Dictionary_getByteListFromBits(mybits4)
#4
mybits5 = np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,0,1,1,1,0,0],[0,0,1,1,1,0,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[4] = aruco.Dictionary_getByteListFromBits(mybits5)
#5
mybits6 = np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,1,0,1,1,1,0],[0,0,1,1,0,0,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[5] = aruco.Dictionary_getByteListFromBits(mybits6)
#6
mybits7 = np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,0,0,1,1,1,0],[0,0,0,1,1,1,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[6] = aruco.Dictionary_getByteListFromBits(mybits7)
#7
mybits8 = np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,1,1,1,1,0,0],[0,0,0,1,0,1,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[7] = aruco.Dictionary_getByteListFromBits(mybits8)
#8
mybits9 =np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,0,0,1,0,1,0],[0,1,1,1,1,0,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[8] = aruco.Dictionary_getByteListFromBits(mybits9)
#9
mybits10 = np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,1,1,1,0,0,0],[0,1,1,1,0,0,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[9] = aruco.Dictionary_getByteListFromBits(mybits10)
#10
mybits11 = np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,0,1,1,0,0,0],[0,1,0,1,1,1,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[10] = aruco.Dictionary_getByteListFromBits(mybits11)
#11
mybits12 = np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,1,0,1,0,1,0],[0,1,0,1,0,1,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[11] = aruco.Dictionary_getByteListFromBits(mybits12)
#12
mybits13 = np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,1,0,1,0,0,0],[0,0,1,1,1,1,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[12] = aruco.Dictionary_getByteListFromBits(mybits13)
#13
mybits14 = np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,0,1,1,0,1,0],[0,0,1,1,0,1,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[13] = aruco.Dictionary_getByteListFromBits(mybits14)
#14
mybits15= np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,1,1,1,0,1,0],[0,0,0,1,1,0,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[14] = aruco.Dictionary_getByteListFromBits(mybits15)
#15
mybits16= np.array([[0,0,0,0,0,0,0],[0,1,1,0,1,1,0],[0,1,1,0,1,1,0],[0,1,0,1,0,1,0],[0,0,0,1,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[15] = aruco.Dictionary_getByteListFromBits(mybits16)
#16
mybits17= np.array([[0,0,0,0,0,0,0],[0,0,0,0,1,0,0],[0,1,1,0,0,1,0],[0,1,0,1,0,1,0],[0,1,1,1,1,1,0],[0,1,1,1,1,1,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[16] = aruco.Dictionary_getByteListFromBits(mybits17)
#17
mybits18= np.array([[0,0,0,0,0,0,0],[0,0,0,0,1,0,0],[0,1,1,0,0,1,0],[0,1,0,1,0,1,0],[0,0,0,1,1,0,0],[0,1,1,1,0,1,0],[0,0,0,0,0,0,0]], dtype = np.uint8)
aruco_dict.bytesList[17] = aruco.Dictionary_getByteListFromBits(mybits18)







#print(aruco_dict)
#for i in range(len(aruco_dict.bytesList)):
#    cv2.imwrite("custom_aruco_" + str(i) + ".png", aruco.drawMarker(aruco_dict, i, 128))
parameters = aruco.DetectorParameters_create()  # Marker detection parameters
#aruco_dict = aruco.custom_dictionary(0, 4, 1)

CoorMsg=""
imgMsg=np.ones((480,620),np.uint8)
imgMsgMask=np.ones((270,270),np.uint8)



detectamos=0
rango=0
direccion=0
profundidad=0
constante=0
CoorMsg = str([0.0,0.0,0.0,0.0,0.0])


folder_path = glob.glob(os.path.join('markers','*.png'))
#print(np.shape(folder_path))
#print((folder_path))

images = []
dim = (270,270)
for i in range(len(folder_path)):
    images.append(cv2.imread(folder_path[i],0))
    images[i] = cv2.resize(images[i],dim,interpolation=cv2.INTER_CUBIC)



# known pump geometry
#  - units are pixels (of half-size image)
PUMP_DIAMETER = 360
PISTON_DIAMETER = 90
PISTON_COUNT = 7

def start_node():
    rospy.init_node('camera_subscriber_hd8')
    rospy.loginfo('camera_subscriber_hd8_started')
    rospy.Subscriber('robocol_vision_object_distance', String , process_YOLO)
    rospy.Subscriber('/zed2/rgb/image_rect_color', Image , process_image)

    t = threading.Thread(target=impresion)
    t.start()


    rospy.spin()
    
    
def impresion():
    global CoorMsg, imgMsg,imgMsgMask
    pub=rospy.Publisher("/robocol_vision_object_FINAL", String, queue_size=10)
    pub2 = rospy.Publisher("/robocol_vision_Segmentation", Image, queue_size=10)
    pub3 = rospy.Publisher("/robocol_vision_HSVMASK", Image, queue_size=10)

    while not rospy.is_shutdown():
        #Variables globales
        #bridge = CvBridge()
        pub.publish(CoorMsg)
        pub2.publish(CvBridge().cv2_to_imgmsg(imgMsg))
        pub3.publish(CvBridge().cv2_to_imgmsg(imgMsgMask))
        #print(type(imgMsg))
        rospy.Rate(100).sleep()  # 1 Hz



def process_YOLO(msg):
    
    global detectamos, rango, direccion, profundidad,  constante
    
    msn_YOLO=msg.msg[1:-1]
    msn_YOLO=msn_YOLO.split(',')
    detectamos=int(msn_YOLO[0])
    rango=int(msn_YOLO[1])
    direccion=int(msn_YOLO[2])
    profundidad=int(msn_YOLO[3])




def process_image(msg):
    
    global aruco_dict
    global parameters,folder_path,detectamos,rango,direccion, profundidad,CoorMsg, constante
    
    
    try:
        
        
        folder_path = glob.glob(os.path.join('markers','*.png'))

        images = []
        dim = (270,270)
        for i in range(len(folder_path)):
        	images.append(cv2.imread(folder_path[i],0))
        	images[i] = cv2.resize(images[i],dim,interpolation=cv2.INTER_CUBIC)

 
        frame= CvBridge().imgmsg_to_cv2(msg)
        cv2.imshow("imagen_recibida", frame)
        
        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict,parameters=parameters)
        print(corners)
        print(ids)
        print(rejectedImgPoints)
        
        
        if ids is not None:
                try:

                    corners = corners[0][0]
                    start, stop = drawRectangle(corners, frame, ids)
                    frame_cut = frame[int(start[1]):int(stop[1]), int(start[0]):int(stop[0]), :]
                    
                    

                    dim = (270, 270)
                    frame_cut = cv2.resize(frame_cut, dim, interpolation=cv2.INTER_CUBIC)
                    
                    img1 = cv2.cvtColor(frame_cut, cv2.COLOR_BGR2GRAY)

                    ret, img1 = cv2.threshold(img1, 100, 255, cv2.THRESH_BINARY)


                    for i in range(len(folder_path)):
                        images[i] = cv2.resize(images[i], dim, interpolation=cv2.INTER_CUBIC)
                        ret2, img2 = cv2.threshold(images[i], 100, 255, cv2.THRESH_BINARY)


                        err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
                        err /= float(img1.shape[0] * img2.shape[1])
                        print("Error MSE")
                        if err < 5000:
                        #print("Reconocido")
                            constante = str(i)
                        #print("constaaaaaaaaaaaaaaannnnnnnnteeeeeeeeeeee")
                            print(constante)

                            CoorMsgAA = [detectamos,rango,direccion,profundidad,constante]
                            CoorMsg = str(CoorMsgAA)
                            
                            
                            break
                            
                        else:
                            detectamos=0
                        print(err)
                    
                    
                    sift = cv2.xfeatures2d.SIFT_create()

                    kp1, des1 = sift.detectAndCompute(img1, None)
                    kp2, des2 = sift.detectAndCompute(img2, None)

                # feature matching
                    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

                    matches = bf.match(des1, des2)
                    matches2 = sorted(matches, key=lambda x: x.distance)

                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(des1, des2, k=2)

                    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches2[:50], img2, flags=2)

                    cv2.imshow("imagenpro",img3)
                    imgMsgMask=img3
        
        
        
                    x1=corners[0][0]
                    x2=corners[0][1]
                    y1=corners[2][0]
                    y2=corners[2][1]


                    cv2.rectangle(frame,(corners[0][0],corners[0][1]),(corners[2][0],corners[2][1]),(255, 255, 255), 2)
                    
                    imgMsg=frame
                    
                    CoorMsgAA = [detectamos,rango,direccion,profundidad,constante]
                    CoorMsg = str(CoorMsgAA)
                            


                #MENSAJEEEEEEEEEEEEEEEEEEEEEEEEEeee
                    print("MENSAJEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
                    print(CoorMsg)
                    
                except Exception as err:
                    print(err)



        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()


    except Exception as err:
        print(err)
        
 
        
def showImage(img):
    cv2.imshow('image', img)
    cv2.waitKey(10)
    
    
    
    
if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass
