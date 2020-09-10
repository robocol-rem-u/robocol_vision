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
import PIL
from cv2 import aruco
import glob
import os
from PIL import Image


global detectamos,rango,direccion,profundidad,CoorMsg, x1,x2,y1,y2
detectamos=0
rango=0
direccion=0
profundidad=0
constante=0


folder_path = glob.glob(os.path.join('markers','*.png'))
print(np.shape(folder_path))
print((folder_path))

images = []
dim = (270,270)
for i in range(len(folder_path)):
    images.append(cv2.imread(folder_path[i],0))
    images[i] = cv2.resize(images[i],dim,interpolation=cv2.INTER_CUBIC)




aruco_dict = aruco.custom_dictionary(20,7, 2)
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




#for i in range(len(aruco_dict.bytesList)):
#    cv2.imwrite("custom_aruco_" + str(i) + ".png", aruco.drawMarker(aruco_dict, i, 128))

parameters = aruco.DetectorParameters_create()  # Marker detection parameters


#markerImage = np.zeros((200, 200), dtype=np.uint8)
#markerImage = cv2.aruco.drawMarker(aruco_dict, 8, 200, markerImage, 1);
#cv2.imwrite("marker4.png", markerImage)
#cap = cv2.VideoCapture(1)















def start_node():
    global detectamos, rango, direccion, profundidad, CoorMsg

    rospy.init_node('camera_subscriber_hd2')
    rospy.loginfo('camera_subscriber_hd2 started')
    rospy.Subscriber("/zed2/left_raw/image_raw_color", Image, process_image)
    rospy.Subscriber('zed2/depth_registered', Image, call_back_depth)

    t = threading.Thread(target=impresion)
    t.start()

    rospy.spin()


def call_back_depth(data):
    global x1, x2, y1, y2, rango, profundidad

    image = CvBridge().imgmsg_to_cv2(data)
    depth = image[x1:x2, y1:y2].astype(float)
    dist, _, _, _ = cv2.mean(depth)
    if dist < 5:
        rango = 1
        profundidad=dist

    else:
        rango=0
        profundidad=dist




def impresion():
    global detectamos, rango, direccion, profundidad, CoorMsg


    pub=rospy.Publisher("/robocol_vision_object_FINAL", String,queue_size=10)


    while not rospy.is_shutdown():
        #Variables globales
        #bridge = CvBridge()
        pub.publish(CoorMsg)

        #print(type(imgMsg))
        rospy.Rate(100).sleep()  # 1 Hz












def drawRectangle(coor,frame,ids):
    x_max = coor[0][0]
    x_min = coor[2][0]
    y_max = coor[0][1]
    y_min = coor[2][1]
    start = (x_max,y_max)
    stop = (x_min,y_min)
    return start,stop

def gray2bin(frame,val):
    frame[frame>val] = 255
    frame[frame<val] = 0
    return frame


def get_value(frame_bb):
    dim = (270,270)
    frame_resized = cv2.resize(frame_bb,dim,interpolation=cv2.INTER_CUBIC)
    frame = cv2.cvtColor(frame_resized,cv2.COLOR_RGB2GRAY)
    val = filters.threshold_otsu(frame)
    frame = gray2bin(frame,val)
    decod = []
    vector = np.linspace(0,240,9).astype(int)
    for i in vector:
        for j in vector:
            temp_img = frame[j:j+29,i:i+29]
            hist_value = np.histogram(temp_img,bins=2)[0].argmax(axis=0) # Get Histogram
            decod.append(hist_value) # Assign value, 0 or 1
    return decod




def process_image(msg):
    global detectamos, rango, direccion, profundidad, CoorMsg,x1,x2,y1,y2



    while(True):

    # Capture frame-by-frame
        ret, frame = cap.read()
    #frame=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        if ret == True:
        #lists of ids and the corners beloning to each id
            corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict,parameters=parameters)
        #print(np.shape(corners))
        #print(corners)
            if ids is not None:
                try:

                    corners = corners[0][0]
                    start, stop = drawRectangle(corners, frame, ids)
                    frame_cut = frame[int(start[1]):int(stop[1]), int(start[0]):int(stop[0]), :]

                    dim = (270, 270)
                    frame_cut = cv2.resize(frame_cut, dim, interpolation=cv2.INTER_CUBIC)
                    cv2.imshow("aa",frame_cut)

                #COMP=cv2.imread("Comparacion.png")
                    dim = (270, 270)
                #COMP = cv2.resize(COMP, dim, interpolation=cv2.INTER_CUBIC)

                    img1 = cv2.cvtColor(frame_cut, cv2.COLOR_BGR2GRAY)
                #img2 = cv2.cvtColor(COMP, cv2.COLOR_BGR2GRAY)

                ##################################
                #Binarización!!!!!!!!!!!!!!!!111
                #################################

                    ret, img1 = cv2.threshold(img1, 100, 255, cv2.THRESH_BINARY)
                #ret2, img2 = cv2.threshold(img2, 100, 255, cv2.THRESH_BINARY)
                #cv2.imshow("aa2",img1)
                #cv2.imshow("aa22",img2)

                    array = np.zeros([7, 7], dtype=np.uint8)



                # the 'Mean Squared Error' between the two images is the
                # sum of the squared difference between the two images;
                # NOTE: the two images must have the same dimension

                    for i in range(len(folder_path)):
                        images[i] = cv2.resize(images[i], dim, interpolation=cv2.INTER_CUBIC)
                        ret2, img2 = cv2.threshold(images[i], 100, 255, cv2.THRESH_BINARY)

#                    cv2.imshow("imagen de la carpeta", images[i])
                    #cv2.waitKey()

                        err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
                        err /= float(img1.shape[0] * img2.shape[1])
                        print("Error MSE")
                        if err < 5000:
                        #print("Reconocido")
                            constante = str(i)
                        #print("constaaaaaaaaaaaaaaannnnnnnnteeeeeeeeeeee")
                            print(constante)
                            detectamos=1
                            break
                        else:
                            detectamos=0
                        print(err)

                #print(constante)


                #for x in range(7):
                #    for y in range(7):
                #        cuadrito=img1Binary[x*30:(x+1)*30, y*30:(y+1)*30]

                #        cv2.imshow("cuadro",cuadrito)

                #        histogram = np.histogram(cuadrito, bins=2)[0].argmax(axis=0)
                #        if histogram == 0:
                #            array[x, y] = 255  # 0 is black
                            #print("2222222222222222222222222222222222222222222")
                #        if histogram == 1:
                #            array[x, y] = 0  # 0 is white
                            #print("1111111111111111111111111111111111111111111111")



                #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

                #array = np.zeros([7, 7], dtype=np.uint8)
                #array[2,2]=255

                #img4 = Image.fromarray(array)
                #pil_image = img4.convert('RGB')
                #open_cv_image = np.array(pil_image)
                #dim = (270, 270)
                #open_cv_image = cv2.resize(open_cv_image, dim, interpolation=cv2.INTER_CUBIC)
                #cv2.imshow("asas", open_cv_image)

                # Set grey value to black or white depending on x position








#############################################################
############################################################
#                print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")

#                im_pil1 = Image.fromarray(img1)
#                hash0 = imagehash.average_hash(im_pil1)

#                im_pil2 = Image.fromarray(img2)
#                hash1 = imagehash.average_hash(im_pil2)

#                print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB")
#                cutoff = 5

#                if hash0 - hash1 < cutoff:
#                    print('images are similar')
#                else:
#                    print('images are not similar')
##############################################################
###############################################################

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







                    x1=corners[0][0]
                    x2=corners[0][1]
                    y1=corners[2][0]
                    y2=corners[2][1]


                    cv2.rectangle(frame,(corners[0][0],corners[0][1]),(corners[2][0],corners[2][1]),(255, 255, 255), 2)
                    if(corners[0][0]+corners[2][0]>240+50):
                        direccion=2
                    elif(corners[0][0]+corners[2][0]<240-50):
                        direccion=1
                    else:
                        direccion=3





                    CoorMsgAA = [detectamos,rango,direccion,profundidad,constante ]
                    CoorMsg = str(CoorMsgAA)

                #MENSAJEEEEEEEEEEEEEEEEEEEEEEEEEeee
                    print("MENSAJEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE")
                    print(CoorMsg)




                ####################################################################33

                #kp1, des1 = surf.detectAndCompute(imagenbase, None)
                #kp2, des2 = surf.detectAndCompute(imagencapturada, None)

                # BFMatcher with default params
                #bf = cv2.BFMatcher()
                #matches = bf.knnMatch(des1, des2, k=2)

                # Apply ratio test
                    good = []
                    percent = 0
                # SURF

                #print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaaa")
                    for m, n in matches:
                    #print("asasasaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                        if m.distance < 0.75 * n.distance:
                            good.append([m])
                            a = len(good)
                            percent = (a * 100) / len(kp2)
                            if percent >= 40.00:
                                print('Match Found')
                                detectamos=1

                            percent = round(percent, 2)

                except:
                    pass


            cv2.imshow('frame',frame)
















        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



# When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()






if __name__ == '__main__':
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass