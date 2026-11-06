import cv2
import numpy as np
from environment.court import court_segments

def render_frame(camera, ball_position, cfg, blur_px=0, noise_std=0.0, rng=None):
    img = np.zeros((cfg.image_height,cfg.image_width,3),np.uint8)
    img[:] = (45,105,55)
    for a,b in court_segments(cfg.court_width,cfg.court_length):
        uv,z = camera.project([a,b])
        if np.all(z>0):
            cv2.line(img,tuple(np.round(uv[0]).astype(int)),tuple(np.round(uv[1]).astype(int)),(235,235,235),3,cv2.LINE_AA)
    uv,z = camera.project([ball_position])
    if z[0]>0:
        radius = max(2,int(round(camera.K[0,0]*cfg.ball_radius/z[0])))
        center = tuple(np.round(uv[0]).astype(int))
        cv2.circle(img,center,radius,(0,140,255),-1,cv2.LINE_AA)
        cv2.circle(img,center,radius,(20,20,20),1,cv2.LINE_AA)
    if blur_px and blur_px>1:
        k = blur_px if blur_px%2 else blur_px+1
        img = cv2.GaussianBlur(img,(k,k),0)
    if noise_std>0:
        rng = rng or np.random.default_rng()
        img = np.clip(img.astype(float)+rng.normal(0,noise_std,img.shape),0,255).astype(np.uint8)
    return img
