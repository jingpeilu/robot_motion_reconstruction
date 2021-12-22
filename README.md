# Reconstructing Robot Motion from Video

Website: https://sites.google.com/ucsd.edu/robotmotionreconstruction/home

## Usage

### Dependencies
Recommend set up the environment using Anaconda.

- Python(3.8)
- PyTorch(1.10.0)
- [Deeplabcut(2.0+)](https://github.com/DeepLabCut/DeepLabCut)
- [PyRep](https://github.com/stepjam/PyRep)

Require CoppeliaSim(4.1) for running the robot simulation.

Codes are tested on Ubuntu 20.04.

### Dataset

Download the following dataset and extract in dataset folder.

1. [Baxter wobbler dataset](https://drive.google.com/file/d/1UlbUgTQFce4Bqci0im8ieCTvXa_Ylr0j/view)
2. [Baxter poses dataset](https://drive.google.com/file/d/19_PdlJw-uOlUGS5Vp6oK5tcKXiMD-QUQ/view)


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

```simulation/Baxter_wobbler.ipynb ```: demonstration of motion reconstruction for Baxter wobbler dataset.

```simulation/Baxter_poses.ipynb ```: demonstration of motion reconstruction for Baxter poses dataset.

```simulation/kuka.ipynb ```: demonstration of motion reconstruction for Baxter datasets on Kuka robot arm.


## Notes

TODO: The code for Boston Dynamic Spot will be released soon!
