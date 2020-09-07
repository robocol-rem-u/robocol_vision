#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

# known pump geometry
#  - units are pixels (of half-size image)

def start_node():
    rospy.init_node('camera_subscriber_hd1')
    rospy.loginfo('camera_subscriber_hd1 started')
    rospy.Subscriber("/camera_publisher_hd1/image_raw", Image , process_image)
    rospy.spin()

def process_image(msg):
    try:
 	
	#Image

        cv2.imshow("imagen_recibida", CvBridge().imgmsg_to_cv2(msg))

	#CompressedImage
	#np_arr = np.fromstring(msg.data, np.uint8)
        #image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #cv2.imshow('cv_img', image_np)

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
