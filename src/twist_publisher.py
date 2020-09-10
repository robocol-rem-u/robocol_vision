#!/usr/bin/env python3
import rospy
import sys
import cv2

from cv_bridge import CvBridge
from geometry_msgs.msg import Twist

def start_node():
	rospy.init_node('nodo_twist_publisher')
	rospy.loginfo('nodo_twist_pubisher started')
	pub = rospy.Publisher('/robocol/pose', Twist, queue_size=10)
	msgTwist = Twist()
    
	while not rospy.is_shutdown():
		msgTwist.linear.x = 1
		msgTwist.linear.y = 1
		msgTwist.angular.z = 1
    


    #while not rospy.is_shutdown():
		pub.publish(msgTwist)
		rospy.Rate(10).sleep()  # 1 Hz
	


        

if __name__ == '__main__':
	try:
		start_node( )
	except rospy.ROSInterruptException:
		pass
