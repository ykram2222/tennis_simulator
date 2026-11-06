import argparse, json, os, csv
import numpy as np
import matplotlib.pyplot as plt
from config import SimConfig
from physics.ball import simulate_ball
from cameras.model import make_stereo_cameras
from cameras.render import render_frame
from vision.detect import detect_orange_ball
from vision.stereo import triangulate
from evaluation.metrics import summarize_errors,line_call

def run(cfg):
    os.makedirs("outputs",exist_ok=True)
    rng=np.random.default_rng(cfg.random_seed)
    times,gt,vel,contact=simulate_ball(cfg)
    left,right=make_stereo_cameras(cfg)
    estimates=[]; est_times=[]; saved=None
    for i,(t,p) in enumerate(zip(times,gt)):
        li=render_frame(left,p,cfg,cfg.motion_blur_px,cfg.image_noise_std,rng)
        ri=render_frame(right,p,cfg,cfg.motion_blur_px,cfg.image_noise_std,rng)
        if i==max(0,contact-1): saved=(li.copy(),ri.copy())
        ul,ur=detect_orange_ball(li),detect_orange_ball(ri)
        if ul is not None and ur is not None:
            ul+=rng.normal(0,cfg.camera_noise_std,2)
            ur+=rng.normal(0,cfg.camera_noise_std,2)
            xyz=triangulate(left,right,ul,ur)
            if np.all(np.isfinite(xyz)):
                estimates.append(xyz); est_times.append(t)
    if saved is None:
        saved=(render_frame(left,gt[0],cfg),render_frame(right,gt[0],cfg))
    plt.imsave("outputs/camera_left.png",saved[0][:,:,::-1])
    plt.imsave("outputs/camera_right.png",saved[1][:,:,::-1])
    estimates=np.asarray(estimates); est_times=np.asarray(est_times)
    gt_samples=np.array([gt[np.argmin(abs(times-t))] for t in est_times]) if len(estimates) else np.empty((0,3))
    errors=np.linalg.norm(estimates-gt_samples,axis=1) if len(estimates) else []
    metrics={"camera_count":2,"fps":cfg.fps,"resolution":[cfg.image_width,cfg.image_height],
        "hfov_deg":cfg.horizontal_fov_deg,"baseline_m":cfg.stereo_baseline,
        "trajectory_reconstruction":summarize_errors(errors),
        "ground_truth_impact_xyz_m":gt[contact].tolist(),
        "ground_truth_line_call":line_call(gt[contact,:2],cfg.court_width,cfg.court_length),
        "detected_frames":int(len(estimates)),"simulated_frames":int(len(times))}
    impact_est=None
    if len(estimates):
        impact_est=estimates[np.argmin(abs(estimates[:,2]-cfg.ball_radius))]
        metrics["estimated_impact_xyz_m"]=impact_est.tolist()
        metrics["impact_position_error_mm"]=float(np.linalg.norm(impact_est-gt[contact])*1000)
        metrics["estimated_line_call"]=line_call(impact_est[:2],cfg.court_width,cfg.court_length)
    else:
        metrics.update({"estimated_impact_xyz_m":None,"impact_position_error_mm":None,"estimated_line_call":"NO_DETECTION"})
    with open("outputs/metrics.json","w") as f: json.dump(metrics,f,indent=2)
    with open("outputs/trajectory.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["time_s","gt_x_m","gt_y_m","gt_z_m","est_x_m","est_y_m","est_z_m"])
        for j,t in enumerate(times):
            k=np.argmin(abs(est_times-t)) if len(est_times) else None
            row=[t,*gt[j]]
            row+=list(estimates[k]) if k is not None and abs(est_times[k]-t)<1e-6 else ["","",""]
            w.writerow(row)
    fig=plt.figure(figsize=(11,8)); ax=fig.add_subplot(111,projection="3d")
    ax.plot(gt[:,0],gt[:,1],gt[:,2],label="Ground truth")
    if len(estimates): ax.scatter(estimates[:,0],estimates[:,1],estimates[:,2],s=8,label="Stereo estimates")
    ax.scatter(*gt[contact],marker="x",s=100,label="True impact")
    if impact_est is not None: ax.scatter(*impact_est,marker="o",s=80,label="Estimated impact")
    ax.set(xlabel="X (m)",ylabel="Y (m)",zlabel="Z (m)",title="Tennis stereo simulation")
    ax.legend(); plt.tight_layout(); plt.savefig("outputs/simulation.png",dpi=150); plt.close(fig)
    print(json.dumps(metrics,indent=2)); print("Saved results to ./outputs/")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--fps",type=float,default=120)
    p.add_argument("--duration",type=float,default=1.8)
    p.add_argument("--speed-mph",type=float,default=100)
    p.add_argument("--impact-x",type=float,default=.15)
    p.add_argument("--impact-y",type=float,default=11.885)
    a=p.parse_args()
    run(SimConfig(fps=a.fps,duration=a.duration,ball_speed_mph=a.speed_mph,impact_x=a.impact_x,impact_y=a.impact_y))
