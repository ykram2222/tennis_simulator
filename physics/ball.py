import numpy as np

MPH_TO_MPS = 0.44704

def simulate_ball(cfg):
    """Integrate a simple ball model with gravity and quadratic aerodynamic drag."""
    dt = 1.0 / cfg.fps
    times = np.arange(int(cfg.duration * cfg.fps) + 1) * dt
    start = np.array([0.0, 0.8, 2.2], dtype=float)
    target = np.array([cfg.impact_x, cfg.impact_y, cfg.ball_radius], dtype=float)
    speed = cfg.ball_speed_mph * MPH_TO_MPS
    delta = target[:2] - start[:2]
    horizontal_dist = np.linalg.norm(delta)
    direction = delta / max(horizontal_dist, 1e-9)

    # Solve the no-drag ballistic launch angle for the requested horizontal range
    # while keeping the requested initial speed. Drag is then integrated below.
    angles = np.deg2rad(np.linspace(-75.0, 35.0, 2001))
    def height_error(theta):
        c = max(np.cos(theta), 1e-6)
        flight = horizontal_dist / (speed * c)
        return start[2] + speed*np.sin(theta)*flight - 0.5*9.81*flight**2 - target[2]
    vals = np.array([height_error(a) for a in angles])
    crossings = np.where(vals[:-1] * vals[1:] <= 0)[0]
    if len(crossings):
        j = crossings[0]
        lo, hi = angles[j], angles[j+1]
        for _ in range(40):
            mid = (lo+hi)/2
            if height_error(lo)*height_error(mid) <= 0: hi = mid
            else: lo = mid
        theta = (lo+hi)/2
    else:
        theta = angles[np.argmin(np.abs(vals))]
    v = np.array([direction[0]*speed*np.cos(theta),
                  direction[1]*speed*np.cos(theta), speed*np.sin(theta)])
    p = start.copy()
    rho, cd, mass = 1.20, 0.55, 0.057
    area = np.pi * cfg.ball_radius**2
    gravity = np.array([0.0, 0.0, -9.81])
    positions, velocities, contact = [], [], None
    for i, t in enumerate(times):
        positions.append(p.copy())
        velocities.append(v.copy())
        if i > 0 and p[2] <= cfg.ball_radius and v[2] < 0:
            contact = i
            positions[-1][2] = cfg.ball_radius
            break
        s = np.linalg.norm(v)
        drag = -0.5 * rho * cd * area * s * v / mass if s > 1e-9 else np.zeros(3)
        v += (gravity + drag) * dt
        p += v * dt
    positions, velocities = np.asarray(positions), np.asarray(velocities)
    times = times[:len(positions)]
    if contact is None:
        contact = int(np.argmin(np.abs(positions[:,2] - cfg.ball_radius)))
    return times, positions, velocities, min(contact, len(positions)-1)
