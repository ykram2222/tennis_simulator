from dataclasses import dataclass

@dataclass
class SimConfig:
    court_width: float = 10.97
    court_length: float = 23.77
    camera_height: float = 8.0
    stereo_baseline: float = 2.0
    image_width: int = 1920
    image_height: int = 1080
    horizontal_fov_deg: float = 90.0
    fps: float = 120.0
    duration: float = 1.8
    ball_radius: float = 0.0335
    ball_speed_mph: float = 100.0
    impact_x: float = 0.15
    impact_y: float = 11.885
    camera_noise_std: float = 1.5
    motion_blur_px: int = 3
    image_noise_std: float = 1.5
    random_seed: int = 7
