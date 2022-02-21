# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 15:34:48 2022

@author: shali
"""

import cv2

cap = cv2.VideoCapture('C:\\Users\\shali\\Documents\\Shalini Masters\\esports thesis\\sample video.mp4')
# For streams:
#   cap = cv2.VideoCapture('rtsp://url.to.stream/media.amqp')
# Or e.g. most common ID for webcams:
#   cap = cv2.VideoCapture(0)
count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imwrite('frame{:d}.jpg'.format(count), frame)
        count += 30 # i.e. at 30 fps, this advances one second
        cap.set(cv2.CAP_PROP_POS_FRAMES, count)
    else:
        cap.release()
        break