import cv2
import numpy as np

def triangulate(left_camera,right_camera,uv_left,uv_right):
    a = np.asarray(uv_left,float).reshape(2,1)
    b = np.asarray(uv_right,float).reshape(2,1)
    xh = cv2.triangulatePoints(left_camera.P,right_camera.P,a,b)
    return (xh[:3]/xh[3]).reshape(3)
