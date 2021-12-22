# Reconstructing Robot Motion from Video

Website: https://sites.google.com/ucsd.edu/robotmotionreconstruction/home

## Usage

### Dependencies
Recommend set up the environment using Anaconda

- Python(3.8)
- PyTorch(1.10.0)
- Deeplabcut(2.0+)
- Pyrep

Require CoppeliaSim(4.1) for running the simulation reconstruction.

### Keypoint Estimation

Extracting keypoints from Baxter wobbler dataset:

```
cd keypoint_detection
python extract_keypoints_baxter.py --dataset baxter_wobbler
```
Extracting keypoints from Baxter poses dataset:

```
cd keypoint_detection
python extract_keypoints_baxter.py --dataset baxter_poses
```

### State Estimation

For Baxter wobbler dataset:

```
cd state_estimation
python state_estimation_baxter_wobbler.py [--n_l] [--n_q] [--n_c]
```
For Baxter poses dataset:

```
cd state_estimation
python state_estimation_baxter_poses.py [--n_l] [--n_q] [--n_c]
```

### Reconstruction in Simulation

See ipynotebooks for reconstructing robot motion in simulation.


## Notes

The code for Boston Dynamic Spot will be released soon!
