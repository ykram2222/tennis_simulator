import cv2
import numpy as np

def detect_orange_ball(image):
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,np.array([5,100,80]),np.array([28,255,255]))
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours,key=cv2.contourArea)
    if cv2.contourArea(c)<3: return None
    m = cv2.moments(c)
    if abs(m["m00"])<1e-9: return None
    return np.array([m["m10"]/m["m00"],m["m01"]/m["m00"]],float)
