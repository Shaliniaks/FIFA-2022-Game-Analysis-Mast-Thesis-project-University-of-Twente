# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 21:50:50 2022

@author: shali

"""



import cv2
import numpy as np
from itertools import combinations
import math
import pandas as pd
from openpyxl import load_workbook
from Create_pitch_multiple_square import createPitch
import matplotlib.pyplot as plt

# Kalmans Filter
class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)


    def predict(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y
        return x, y
Zone1=0
Zone2=0
Zone3=0
Zone4=0
Zone5=0
Zone6=0     
kernel = None
#Color bound limits for white color(ball)
l_w=np.array([20, 109, 80], np.uint8)# lower hsv bound for white [20, 109, 80],[33, 213, 245]]
u_w=np.array([33, 213, 245], np.uint8)# upper hsv bound to red   
#Initialise Kalmans filter 
kf = KalmanFilter()
  
#Upload the YOLO weight nd cfg files)
net = cv2.dnn.readNet("yolov3_training_last_17_april.weights", "yolov3_testing111.cfg")

classes = []
with open("classes_151.txt", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture('C:\\Users\\shali\\Documents\\Shalini Masters\\esports thesis\\yolo\\Bryan video\\Fifa_video1_ten_min.mp4')
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
df = pd.DataFrame()
df11 = pd.DataFrame([])
df1 = pd.DataFrame()
df22 = pd.DataFrame() #mid-field line
df33 = pd.DataFrame() #penalty line
createPitch()
while True:
    _, img = cap.read()
    height, width, _ = img.shape
    #img =cv2.resize(img,None,fx=0.65,fy=0.65)

    #blob = cv2.dnn.blobFromImage(img, 1/255, (608, 608), (0,0,0), swapRB=True, crop=False)
    blob = cv2.dnn.blobFromImage(img, 1/255, (224, 224), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    centroid_dict = dict()
    new_dict=dict()
    empty = {0: '', 1:  '', 2:  '',3:'',4:'',5:''}
    #empty = {0:, 1:, 2:}
    
    df = pd.DataFrame()
    df11 = pd.DataFrame([])
    df1 = pd.DataFrame()
    df22=pd.DataFrame()
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #df = pd.DataFrame()
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)
                cX = int((x + w) / 2.0)
                cY = int((y + h) / 2.0)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
                #centroid_dict[class_id] = (cX, cY, x, y, w, h)
                centroid_dict[class_id] = (cX, cY, x, y, w, h)
                new_dict[class_id]=(cX, cY)
                #print("OOOOOOOOOOOOOOOOOOOO",new_dict)
                result = {k: centroid_dict.get(k, (0.1,0.1,0.1,0.1,0.1,0.1)) for k in empty}
                #xx = area_lines(img)
                
                #print(type(xx[2]))
                df11['id'] = result .keys()
                df11['idValue'] = result.values()

                df11.to_csv("ball_player_location_thesis_test.csv", mode='a', index=False, header=False)
    
            for i in range(5):
            
                if all(key in new_dict for key in (0,i+1)):
                
             
                    a=np.array(new_dict.get(0))#1,2,3,4,5
                    #b=np.array(new_dict.get(i+1))
                    b=new_dict.get(i+1)
                    dist = np.linalg.norm(a-b)
                  
                    df1 = pd.DataFrame({'col_1': [i+1],'col_2': [b],'col_3':[dist]})
                    
                    #print("df1_plyer",df1.col_1)
                    df=pd.concat([df,df1])
           
            
            if df.empty:
                
                pass
            else:
                min_dist=df[df.col_3==df.col_3.min()]
                min_dist.to_csv("ball_possession.csv", mode='a', index=False, header=False)
                #print("min_value",min_dist)   
                break
            
            
            
            
            ######################################################
            ####################################################
            #from mini map
            
            #Select the minimap
        

        # Extract Region of interest
        ############################################################################
        #roi = frame[786: 1184,787: 1185] #x= frame[720: 1000,800: 1110] #[y(top to bottom increse),y(bottom )]

    height, width, _ = img.shape
    roi=img[795: 1145,840: 1107]
 
     # Apply the background object on the frame to get the segmented mask.
    fgmask = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
     # Apply some morphological operations to make sure you have a good mask
    fgmask_3 = cv2.GaussianBlur(fgmask, (21, 21), 0)
     
    fgmask = cv2.erode(fgmask_3, kernel, iterations = 2)
    fgmask = cv2.dilate(fgmask, kernel, iterations = 2)
     #cv2.imshow("afterdilation",fgmask)
    # real_part = cv2.bitwise_and(frame,frame,mask=fgmask)
    fgmask_2 = cv2.inRange(fgmask, l_w, u_w)
     # Detect contours in the frame.
    contours, _ = cv2.findContours(fgmask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
     # Create a copy of the frame to draw bounding boxes around the detected cars.
    frameCopy = img.copy()
     
     # loop over each contour found in the frame.
    for cnt in contours:
         
         if (cv2.contourArea(cnt) > 60) & (cv2.contourArea(cnt) <180 ) :#correct measure for the ball(0-500)
             
             # Retrieve the bounding box coordinates from the contour.
             x, y, width, height = cv2.boundingRect(cnt)
             
             #cx = int((x + width) / 2)
             #cy = int((y + height) / 2)
             ccx = (x + x + width) // 2
             ccy = (y + y + height) // 2
             
             # Draw a bounding box around the car.
             cv2.rectangle(roi, (x , y), (x + width, y + height),(0, 0, 255), 2)
             
             # Write Car Detected near the bounding box drawn.
             cv2.putText(roi, 'Soccer ball', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
             predicted = kf.predict(ccx, ccy)
    
             cv2.circle(roi, (predicted[0], predicted[1]), 10, (255, 0, 0), 2)

         if df.empty:
        #print("checking",new_dict.keys())
            pass
        
         elif df1.col_1[0]==5:    
             plt.scatter(ccx, ccy, color="green")
             #print("player_id",df1.col_1[0])
             if ccx < 130 and  ccy <=100:
                
                Zone4=Zone4+1
                plt.scatter(ccx, ccy, color="green")
                print("(Zone4)",Zone4)
                print("cx",ccx,ccy)
        
             elif ccx < 130 and (200 > ccy >100):
                Zone5=Zone5+1
                plt.scatter(ccx, ccy, color="green")
                print("(Zone5)",Zone5)
                print("cx",ccx,ccy)
        
             elif ccx < 130 and  ccy >=200:
                Zone6 = Zone6+1
                plt.scatter(ccx, ccy, color="green")  

             else:
                pass
         else:
            pass
        
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    

    
    if len(indexes)>0:
         for i in indexes.flatten():
             x, y, w, h = boxes[i]
             label = str(classes[class_ids[i]])
             confidence = str(round(confidences[i],2))
             color = colors[i]
             cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
             cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
#             


                        
            
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
