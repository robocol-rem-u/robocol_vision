#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import imutils
import threading

# known pump geometry
#  - units are pixels (of half-size image)



estado=2
# Variables
hmn = 0
hmx = 1
smn = 0
smx = 1
vmn = 0
vmx = 1
supizqx = 0
supizqy = 0
supderx = 0
supdery = 0
infizqx = 0
infizqy = 0
infderx = 0
infdery = 0
radius = 0
height, width = 200, 400
im_dst = np.zeros((height,width,3),dtype=np.uint8)
pts_dst = np.empty((0,2))
pts_dst = np.append(pts_dst, [(0,0)], axis=0)
pts_dst = np.append(pts_dst, [(width-1,0)], axis=0)
pts_dst = np.append(pts_dst, [(width-1,height-1)], axis=0)
pts_dst = np.append(pts_dst, [(0,height-1)],  axis=0)

greenLower = (46, 42, 97)
greenUpper = (96, 218, 255)
blueLower = (100, 127, 26)
blueUpper = (141, 237, 172)



global CoorMsg, imgMsg,imgMsgMask
CoorMsg=""
imgMsg=np.ones((300,500),np.uint8)
imgMsgMask=np.ones((300,500),np.uint8)



def read_Binary(frame):
    # Read rectangle (scale to specific size)

    resized_frame=frame
    #resized_frame = cv2.resize(frame,(100,30),interpolation=cv2.INTER_CUBIC)
    # Separate in mini-batches (returns 3 images)
    box1 = resized_frame[:,0:75]
    box2 = resized_frame[:,76:150]
    box3 = resized_frame[:,151:225]
    box4 = resized_frame[:,226:300]
    boxes = [box1,box2,box3,box4]

    #cv2.imshow("box1",box1)
    #cv2.imshow("box2",box2)
    #cv2.imshow("box3",box3)
    #cv2.imshow("box4",box4)

    # Read minibatch info with blue-mask
    bin_list = []
    hist_value2=[]
    for i in range(4):
        hist_value = np.histogram(boxes[i],bins=2)[0].argmax(axis=0) # Get Histogram
        #hist_value2.append(np.histogram(boxes[i],bins=2)[0])
        if hist_value==0:
            hist_value=1
        else:
            hist_value=0

        bin_list.append(hist_value) # Assign value, 0 or 1
    # Concat output and convert to string # int(binary,2) #str1 = ''.join(str(e) for e in list1)
    binary = ''.join(str(e) for e in bin_list)
    decimal = int(binary,2)
    
    
    return binary
    



def start_node():

    global CoorMsg, imgMsg,imgMsgMask

    rospy.init_node('camera_subscriber_hd1')
    rospy.loginfo('camera_subscriber_hd1 started')
    rospy.Subscriber("rgb/image_rect_color", Image , process_image)
    


    t=threading.Thread(target=impresion)
    t.start()
    
    
    rospy.spin()


def impresion():
    global CoorMsg, imgMsg,imgMsgMask
    pub=rospy.Publisher("/robocol_vision_object_coordinates", String,queue_size=10)
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


def process_image(msg):

    global imgMsg
    global CoorMsg
    global imgMsgMask


    #Image
    try:
        
        kernel=np.ones((5,5),np.uint8)
        frame=CvBridge().imgmsg_to_cv2(msg)
        #print(frame)
        original2 = frame
        (h, w) = frame.shape[:2]
        zeros = np.zeros((h, w), dtype="uint8")

        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        dilation = cv2.dilate(frame, kernel, iterations=1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        frame2 = cv2.GaussianBlur(closing, (5, 5), 0)
        hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, greenLower, greenUpper)


        dilation1 = cv2.dilate(original2, kernel, iterations=1)
        closing1 = cv2.morphologyEx(dilation1, cv2.MORPH_CLOSE, kernel)
        original2 = cv2.GaussianBlur(closing1, (5, 5), 0)

        hsv1 = cv2.cvtColor(original2, cv2.COLOR_BGR2HSV)
        original2 = cv2.inRange(hsv1, blueLower, blueUpper)


        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        haypelota1 = False
        contours_poly = [None] * len(cnts)
        boundRect = [None] * len(cnts)
        hull=0
        decimalAlvin=0

        if len(cnts) > 0:


            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid

            c = max(cnts, key=cv2.contourArea)

            for i,c in enumerate(cnts):
                hull=cv2.convexHull(c)
                contours_poly[i] = cv2.approxPolyDP(c, 3, True)
                boundRect[i] = cv2.boundingRect(contours_poly[i])
            #print(((boundRect)))
            #dst = cv2.cornerHarris(mask, 2, 3, 0.04)
            maxdimx=0
            maxdimy=0
            PosTarget=0
            for i in range (len(cnts)):
                if(boundRect[i][2]>maxdimx):
                    if(boundRect[i][3]>maxdimy):
                        maxdimx=boundRect[i][2]
                        maxdimy=boundRect[i][3]
                        posTarget=i

            supizX=100000
            supizY=100000
            infizX=100000
            infizY=0
            infdeX=0
            infdeY=0
            supdeX=0
            supdeY=10000
            coordenadas = [[supizX, supizY], [supdeX, supdeY], [infdeX, infdeY], [infizX, infizY]]

            for i in hull:

                if(i[0][0]-10<supizX and i[0][1]-10<supizY):
                    supizX=i[0][0]
                    supizY=i[0][1]

                if(i[0][0]-10<infizX and i[0][1]+10>infizY):
                    infizX=i[0][0]
                    infizY=i[0][1]

                if (i[0][0]+10 > supdeX and i[0][1]-10 < supdeY):
                    supdeX = i[0][0]
                    supdeY = i[0][1]

                if (i[0][0]+10 > infdeX and i[0][1]+10 > infdeY):
                    infdeX = i[0][0]
                    infdeY = i[0][1]

                coordenadas=[[supizX,supizY],[supdeX,supdeY],[infdeX,infdeY],[infizX,infizY]]

            pts_src = np.empty((0, 2))
            pts_src = np.append(pts_src, [(supizX-10, supizY-10)], axis=0)
            pts_src = np.append(pts_src, [(supdeX+10, supdeY-10)], axis=0)
            pts_src = np.append(pts_src, [(infdeX+10, infdeY+10)], axis=0)
            pts_src = np.append(pts_src, [(infizX-10, infizY+10)], axis=0)
            original=frame


            for i in range (len(cnts)):
                cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])),
                              (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])),
                              (255, 255, 255), 2)


                cv2.drawContours(frame, contours_poly, i, (255,255,255),-1)
            cv2.drawContours(frame,hull,-1,(0,255,0),5)

            cv2.circle(frame, ((supizX,supizY)), 5, (0, 255, 255), 2)
            cv2.circle(frame, ((supdeX,supdeY)), 5, (0, 255, 255), 2)
            cv2.circle(frame, ((infizX,infizY)), 5, (0, 255, 255), 2)
            cv2.circle(frame, ((infdeX,infdeY)), 5, (0, 255, 255), 2)


            tform, status = cv2.findHomography(pts_src, pts_dst)
            im_dst = cv2.warpPerspective(original, tform, (width, height))

            original2 = original2[int(boundRect[len(cnts)-1][1]):int(boundRect[len(cnts)-1][1])+int(boundRect[len(cnts)-1][3]),
                        int(boundRect[len(cnts)-1][0]):int(boundRect[len(cnts)-1][0])+int(boundRect[len(cnts)-1][2])]

            original2 = cv2.resize(original2, (300, 80), interpolation=cv2.INTER_CUBIC)
            decimalAlvin=read_Binary(original2)


        cv2.putText(frame,str(decimalAlvin),(10,40),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,255,3)
        cv2.putText(frame,str(int(decimalAlvin,2)),(10,70),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,255,3)

        #cv2.imshow("IMAGEN RAW",IMAGENRAW)
        #cv2.imshow("Mask2", original2)
        #cv2.imshow("Image", im_dst)

        CoorMsgAA= [int(boundRect[len(cnts)-1][0]), int(boundRect[len(cnts)-1][1]),int(boundRect[len(cnts)-1][0] + boundRect[len(cnts)-1][2]), int(boundRect[len(cnts)-1][1] + boundRect[len(cnts)-1][3])]
        CoorMsg=str(CoorMsgAA)
        
        imgMsgMask= original2
        
        imgMsg=frame
        #cv2.imshow("a",imgMsg)

        #print("asasasasas")



        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()




    except Exception as err:
        print("fallaaaaaaa")
        print(err)
        


if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass
