import numpy as np
import cv2

class PinholeCamera:
    def __init__(self, position, target, width, height, hfov_deg):
        self.position = np.asarray(position, float)
        self.width, self.height = int(width), int(height)
        f = (self.width/2) / np.tan(np.deg2rad(hfov_deg)/2)
        self.K = np.array([[f,0,self.width/2],[0,f,self.height/2],[0,0,1]], float)
        forward = np.asarray(target,float) - self.position
        forward /= np.linalg.norm(forward)
        up = np.array([0.,0.,1.])
        right = np.cross(forward, up); right /= np.linalg.norm(right)
        down = np.cross(forward, right); down /= np.linalg.norm(down)
        self.R = np.vstack([right, down, forward])
        self.t = -self.R @ self.position
        self.P = self.K @ np.column_stack([self.R,self.t])

    def project(self, points):
        pts = np.asarray(points,float).reshape(-1,3)
        cam = (self.R @ pts.T + self.t[:,None]).T
        uvw = (self.K @ cam.T).T
        uv = uvw[:,:2] / np.maximum(uvw[:,2:3],1e-9)
        return uv, cam[:,2]

def make_stereo_cameras(cfg):
    half = cfg.stereo_baseline/2
    target = [0., cfg.court_length*0.53, 0.]
    # Camera distance from the baseline
    y = -7.5
    return (
        PinholeCamera([-half,y,cfg.camera_height],target,cfg.image_width,cfg.image_height,cfg.horizontal_fov_deg),
        PinholeCamera([ half,y,cfg.camera_height],target,cfg.image_width,cfg.image_height,cfg.horizontal_fov_deg))
