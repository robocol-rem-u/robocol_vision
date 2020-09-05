#!/usr/bin/env python3
import cv2
import os
#import tkinter as tk
import threading
from tkinter import *
from PIL import ImageTk
from PIL import Image as Image2 #Hay que llamarlo diferente o se confunde con el Image de ROS
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import numpy as np
import time
global textExample
global lmain, cap, frame, imageCounter, lbltexto, imagenTopico, bandera, topico_escogido
imageCounter=0
imagenTopico=0
topico_escogido=''
bandera = False


def IniciarCamara():
	cam = cv2.VideoCapture(0)

	cv2.namedWindow("test")

	img_counter = 0
	ruta=os.getcwd()

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
	    	cv2.imwrite(ruta+'/src/vision_pkg/imagenes/'+img_name, frame)
	    	print("{} written!".format(img_name))
	    	img_counter += 1

	cam.release()

	cv2.destroyAllWindows()
	
#def getTextInput():
#	global textExample, sub, bandera, result
#	result=textExample.get("1.0","end")
#	bandera=True

		
	#except:
	#	print('error')
def mostrarInterfaz():
	global lmain, cap, lbltexto, textExample, imagenTopico, topico_escogido
	time.sleep(10) #Esperar a que se asigne primero la variable imagenTopico del callback
	root = Tk()
	root.title('Topico: '+topico_escogido)
	# Create a frame
	app = Frame(root, bg="white")
	app.pack()
	# Create a label in the frame
	lmain = Label(app)
	lmain.pack()
	
	bottomframe = Frame(root)
	bottomframe.pack( side = BOTTOM )
	
	topicosframe = Frame(root)
	topicosframe.pack( side = LEFT )
	
	redbutton = Button(bottomframe, text="Tomar Captura", fg="red", command = tomarCaptura)
	redbutton.pack( side = LEFT)
	
	#bluebutton = Button(bottomframe, text="Blue", fg="blue")
	#bluebutton.pack( side = LEFT )
	
	strTopicos = StringVar()
	lbltexto = Label (topicosframe, text="Topicos")
	lbltexto.pack(side = RIGHT)
	
	textExample=Text(root, height=1)
	textExample.pack(side = RIGHT)
	
	#btnRead=Button(bottomframe, height=1, width=10, text="Cambiar Topico", command=getTextInput)
	#btnRead.pack()
	
	#cap = cv2.VideoCapture(0)
	mostrarVideo()
	mostrarTopicos()
	
	root.mainloop()
	sys.exit()
	#cap.release()
	
def mostrarTopicos():
	global lbltexto
	topicos_publicados=rospy.get_published_topics()
	print(topicos_publicados)
	texto_interfaz=''
	for i in range(len(topicos_publicados)):
		texto_interfaz+=str(topicos_publicados[i][0])+"\t"
		
	lbltexto.configure(text=texto_interfaz)


def algMaximizacion(lista):
	maxi=0
	for i in range(len(lista)):
		temp = int(lista[i].split('_')[1].split('.')[0])
		if maxi < temp:
			maxi=temp
	return maxi
		
		
	

def tomarCaptura():
	global frame, imageCounter, imagenTopico
	img_name = "frame_{}.png".format(imageCounter)

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
		
	
		
	#cv2.imwrite('imagenes/'+img_name, frame)
	cv2.imwrite('imagenes/'+img_name, imagenTopico)

	print("{} written! Guardado en {}".format(img_name,rutaImagen))

	#imageCounter += 1
	
	

def mostrarVideo(): 
	global lmain, frame, imagenTopico #frame se usa cuando se esta usando la camara del computador no usando un nodo

#Usando la camara del compu videoCapture##########################	
	#global cap
	#_, frame = cap.read()
	#cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	#img = Image2.fromarray(cv2image)
##################################################################	

	#cv2image = cv2.cvtColor(imagenTopico, cv2.COLOR_BGR2RGBA)
	#a = np.asarray(imagenTopico)
	#print(imagenTopico)
	imageRGB = cv2.cvtColor(imagenTopico, cv2.COLOR_BGR2RGB)
	img = Image2.fromarray(imageRGB)
	imgtk = ImageTk.PhotoImage(image=img)
	lmain.imgtk = imgtk
	lmain.configure(image=imgtk)
	lmain.after(1, mostrarVideo)

def startNode(topico_escogido):
	global sub
	rospy.init_node('robocol_vision_camara_image',anonymous=True)
	rospy.loginfo('camera_subscriber_hd1 started')
	#rospy.Subscriber("/camera_publisher_hd1/image_raw", Image , callbackProcessImage)
	sub =rospy.Subscriber(topico_escogido, Image , callbackProcessImage)
	rospy.spin()

def callbackProcessImage(msg):
	global imagenTopico, bandera, sub, result

	imagenTopico=CvBridge().imgmsg_to_cv2(msg) #es un np array
	

if __name__ == "__main__":
	os.chdir("..") #Cambiar a directorio robocol_ws/vision_pkg/src
	print('Topicos disponibles:')
	topicos_publicados = rospy.get_published_topics()
	texto_interfaz=""
	for i in range(len(topicos_publicados)):
		texto_interfaz+=str(topicos_publicados[i][0])+"\n"
	print(texto_interfaz)
	
	threading.Thread(target=mostrarInterfaz).start()
	topico_escogido = input("Cual topico quieres ver?")
	startNode(topico_escogido)
		

