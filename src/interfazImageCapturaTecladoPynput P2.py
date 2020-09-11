#!/usr/bin/env python
import cv2
import os
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import time
import rospy
import threading
from pynput.keyboard import Key, Listener

global tomarFoto
global img_counter
global posicionTwist
posicionTwist=Twist()
posicionTwist.linear.x=0
posicionTwist.linear.y=0
posicionTwist.linear.z=0
img_counter=0
tomarFoto = False
#cam = cv2.VideoCapture(0)

def iniciarCamara():
	while True:
		ret, frame = cam.read()
		if not ret:
			print("failed to grab frame")
			break
			cv2.imshow("test", frame)
		k = cv2.waitKey(1)
		if k%256 == 27:
		# ESC pressed
			print("Escape hit, closing...")
			break
		elif k%256 == 32:
			# SPACE pressed
			img_name = "frame_{}.png".format(img_counter)
			print(os.getcwd())
			cv2.imwrite(img_name, frame)
			print("{} written!".format(img_name))
			img_counter += 1

	cam.release()


	cv2.destroyAllWindows()

def hiloPynput():
	# Collect events until released
	with Listener(on_press=on_press,on_release=on_release) as listener:
	    listener.join()



def on_press(key):
	global tomarFoto
	if key == Key.space:
		tomarFoto=True
       

def on_release(key):
	global tomarFoto
	if key == Key.space:
		tomarFoto=False


def algMaximizacion(lista):
	maxi=0
	for i in range(len(lista)):
		temp = int(lista[i].split('_')[1].split('.')[0])
		if maxi < temp:
			maxi=temp
	return maxi
		
		

def callbackProcessImage(msg):
	global imagenTopico, img_counter, tomarFoto, posicionTwist

	font                   = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (5,450)
	fontScale              = 1
	fontColor              = (252, 3, 248)
	lineType               = 2


	imagenTopico=CvBridge().imgmsg_to_cv2(msg) #es un np array


	#cv2.imshow("test", imagenTopico)

	if tomarFoto == True:
	# SPACE pressed
		img_name = "frame_{}.png".format(img_counter)

		ruta=os.getcwd()
		
		#print("ruta ",ruta)
		rutaCarpetaImagenes=ruta+"/imagenes/"
		rutaImagen=ruta+"/imagenes/"+img_name
		
		if os.path.isfile(rutaImagen):
			lista=os.listdir("imagenes/")
			#print(lista)
			ultimoNumero=algMaximizacion(lista) #Encontrar el frame de numero mas alto. Para no sobreescribir si la consola se cierra.
			ultimoNumero+=1
			img_name="frame_{}.png".format(ultimoNumero)
			imageCounter=ultimoNumero
		
	
		posicionText = "["+str(round(posicionTwist.linear.x,2))+","+str(round(posicionTwist.linear.y,2))+","+str(round(posicionTwist.linear.y,2))+"]"
	#cv2.imwrite('imagenes/'+img_name, frame)
		cv2.putText(imagenTopico, posicionText, bottomLeftCornerOfText,font,fontScale,fontColor,lineType)
		cv2.imwrite('imagenes/'+img_name, imagenTopico)

		print("{} written! Guardado en {}".format(img_name,rutaImagen))


	
def callbackPosicion(msg):
	global posicionTwist
	posicionTwist = msg
	
	







def startNode(topico_escogido):
	rospy.init_node('robocol_vision_camara_image',anonymous=True)
	rospy.loginfo('camera_subscriber_hd1 started')
	
	
	sub =rospy.Subscriber(topico_escogido, Image , callbackProcessImage)
	
	#nombreVentana="Topico: ",topico_escogido
	#cv2.namedWindow(nombreVentana)
	#rospy.Subscriber("/camera_publisher_hd1/image_raw", Image , callbackProcessImage)
	
	sub1 =rospy.Subscriber("/robocol/pose", Twist , callbackPosicion)
	threading.Thread(target=hiloPynput).start()
	rospy.spin()
	#cam.release()
	cv2.destroyAllWindows()



if __name__ == "__main__":
	os.chdir("..") #Cambiar a directorio robocol_ws/vision_pkg/src
	print('Topicos disponibles:')
	topicos_publicados = rospy.get_published_topics()
	texto_interfaz=""
	for i in range(len(topicos_publicados)):
		texto_interfaz+=str(topicos_publicados[i][0])+"\n"
	print(texto_interfaz)
	
	
	#threading.Thread(target=waitKey).start()
	#threading.Thread(target=mostrarInterfaz).start()
	topico_escogido = input("Cual topico quieres ver?")
	
	startNode(topico_escogido)
		

