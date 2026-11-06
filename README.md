# Tennis Stereo / Line-Call Simulator

Python MVP combining ball physics, stereo camera geometry, synthetic images, color-based detection,
triangulation, and evaluation.

## Install and run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```
Outputs: `outputs/camera_left.png`, `camera_right.png`, `simulation.png`, `metrics.json`, `trajectory.csv`.

Optional:
```bash
python main.py --fps 120 --speed-mph 100 --impact-x 0.15 --impact-y 11.885
```

## Important limitations
This is a research prototype, not a validated officiating system. Physics currently includes gravity
and quadratic drag, but not Magnus force/spin. Images are synthetic projections with a simple ball
render and court markings. Detection uses orange-color segmentation. Cameras are assumed synchronized
and triangulation uses known calibration. The first impact estimate selects the reconstructed point
closest to ball height; it does not yet infer contact time from vision alone.

The project deliberately separates ground-truth physics from the vision pipeline so real camera frames
can eventually replace synthetic observations.
