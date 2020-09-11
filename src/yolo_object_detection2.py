#!/usr/bin/env python
import cv2
import numpy as np
import glob
import random
import os
import time
from yolo_utils import get_maxima, turn
import threading
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
import time

global coor_max, OUTLIST, img, depth
coor_max = [0,0,0,0]
OUTLIST = []
img = []
depth = 0

def yolo():
    global coor_max, OUTLIST, img, depth
    time.sleep(10) #Esperar a que se asigne primero la variable imagenTopico del callback
    net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
    classes = ["Station"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    #cap = cv2.VideoCapture('videoROSbag2.mp4')
    k = 0
    while True:
        #ret, img = cap.read()
        #img = cv2.resize(img, None, fx=0.4, fy=0.4)
        imgnp=np.array(img)
        height, width, channels = imgnp.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        if np.mod(k,180) == 0: 
            outs = net.forward(output_layers)
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            coor = []
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    x1 = x
                    x2 = x1+w
                    y1 = y
                    y2 = y1+h
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    coor.append([x1,y1,x2,y2])
            if coor == []:
                pass
                detection = 0
                distance  = 0
                direction = 0
                depth = 100
            else:
                coor_max = get_maxima(coor,img)
                direction = turn(coor_max,img)
                detection = 1
                if depth <= 5:
                    distance = 1
                else:
                    distance = 0
                    depth = 100
            OUTLIST = str([detection,distance,direction,depth,0])
            #OUTLIST = str(OUTLIST)
            print(OUTLIST)
        cv2.imshow("Image", img)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        k = k+1
    cv2.destroyAllWindows()



def call_back_depth(data):
    global coor_max, depth
    x1 = coor_max[0]
    x2 = coor_max[2]
    y1 = coor_max[1]
    y2 = coor_max[3]

    image = CvBridge().imgmsg_to_cv2(data)
    ##################################
    depth_bb = image[y1:y2,x1:x2].astype(float)
    depth,_,_,_ = cv2.mean(depth_bb)

def call_back_img(data):
    global img
    img = CvBridge().imgmsg_to_cv2(data)


def listener():
    global OUTLIST
    #Subscriber
    rospy.init_node('robocol_vision_yolo_distance',anonymous=False) #robocol_vision_distance
    rospy.Subscriber('/zed2/depth_registered',Image,call_back_depth)
    #rospy.Subscriber('/zed2/rgb/image_rect_color',Image,call_back_depth)
    rospy.Subscriber('/zed2/rgb/image_rect_color',Image,call_back_img)
    
    threading.Thread(target=yolo).start()
    #Publisher
    dist_publisher = rospy.Publisher('/robocol_vision_object_FINAL', String, queue_size=10)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        dist_publisher.publish(OUTLIST)
        rate.sleep()
    rospy.spin()

if __name__=='__main__':
    listener()
