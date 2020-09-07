#!/usr/bin/env python3
import rospy
import sys
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def start_node():
    rospy.init_node('image_pub')
    rospy.loginfo('image_pub node started')
    pub = rospy.Publisher('image', Image, queue_size=10)



    cap= cv2.VideoCapture(0)
    
    while(1):
        _, frame= cap.read()
   
        #cv2.imwrite("1.png",frame)
        #img = cv2.imread("1.png")
        #cv2.imshow("image",frame)
        bridge = CvBridge()
        imgMsg = bridge.cv2_to_imgmsg(frame, "bgr8")
        #print(imgMsg)


    #while not rospy.is_shutdown():
        pub.publish(imgMsg)
        rospy.Rate(100).sleep()  # 1 Hz



        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    try:
    	start_node( )
    except rospy.ROSInterruptException:
        pass
