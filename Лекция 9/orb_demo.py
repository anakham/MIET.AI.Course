# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:48:22 2015

@author: anatoly
"""

#import numpy as np
import cv2

mousePos=[0,0]

def onMouse( event, x, y, flag, param ):
    param[0]=x
    param[1]=y


img = cv2.imread('orb_demo.jpg',0)

# Initiate STAR detector
orb = cv2.SIFT()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
cv2.namedWindow('orb')
cv2.setMouseCallback('orb', onMouse, mousePos)
keycode=0
while (keycode not in [27,ord('q'),ord('Q')]):
    img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=4)
    mouse_text='mouse x={}, mouse y={}'.format(mousePos[0],mousePos[1])
    cv2.putText(img2, mouse_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255),2)
    cv2.imshow('orb',img2)
    keycode=cv2.waitKey(20)
    
cv2.destroyAllWindows()
