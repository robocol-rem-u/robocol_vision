#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge
import cv2
from std_msgs.msg import String
from std_msgs.msg import Float32
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


global x1,x2,y1,y2, dist, bandera

# Coordinates
x1 = 0
x2 = 0
y1 = 0
y2 = 0
# Depth
dist = 0
bandera = 0


def call_back_coordinates(data):
	global x1,x2,y1,y2
	#print("datooooooss")
	#print(data)
	data = data.data[1:-1]
	data = data.split(',')
	x1 = int(data[0])
	y1 = int(data[1])
	x2 = int(data[2])
	y2 = int(data[3])

def call_back_depth(data):
	global x1,x2,y1,y2, dist, bandera
    
	image = CvBridge().imgmsg_to_cv2(data)
	depth = image[x1:x2,y1:y2].astype(float)
	dist,_,_,_ = cv2.mean(depth)
	if dist<25:
		badera=1
	print(dist)

def listener():
	global dist, bandera
	#Subscriber
	rospy.init_node('robocol_vision_distance',anonymous=False) #robocol_vision_distance
	rospy.Subscriber('/robocol_vision_object_coordinates',String,call_back_coordinates)
	rospy.Subscriber('depth/depth_registered',Image,call_back_depth)
	#rospy.Subscriber('rgb/image_rect_color',Image,call_back_depth)
	#Publisher
	dist_publisher = rospy.Publisher('/robocol_vision_object_distance', String, queue_size=10)
	distance = str([bandera,distance])
	rate = rospy.Rate(10)
    
    
	while not rospy.is_shutdown():
		distance = dist
		dist_publisher.publish(distance)
		rate.sleep()
	rospy.spin()



if __name__=='__main__':
	listener()


#def publisher():
#	global dist
#	rospy.init_node('robocol_vision_publish_distance', anonymous=False)
#	dist_publisher = rospy.Publisher('/robocol_vision_object_distance', Float32, queue_size=10) #
#	rate = rospy.Rate(10)
#	distance = Float32()
#	while not rospy.is_shutdown():
#		distance = dist
#		dist_publisher.publish(distance)
#		rate.sleep()
