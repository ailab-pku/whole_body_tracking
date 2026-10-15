# BeyondMimic Motion Tracking for X2 Ultra

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

## Overview

BeyondMimic is a versatile humanoid control framework that provides highly dynamic motion tracking with the
state-of-the-art motion quality on real-world deployment and steerable test-time control with guided diffusion-based
controllers.

This repo covers the motion tracking training in BeyondMimic. **You should be able to
train any sim-to-real-ready motion in the LAFAN1 dataset, without tuning any parameters**.

For sim-to-sim and sim-to-real deployment, please refer to
the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

> **Fork note:** This fork adapts BeyondMimic for AgiBot **X2 Ultra**. The checkout/repository directory is named
> `whole_body_tracking_X2_ultra`, while the robot configuration key is `x2_ultra`. The inner Python package remains
> named `whole_body_tracking`; do not rename `source/whole_body_tracking` or its package directories when renaming the
> outer checkout directory.

### Alternative Implementations

- There is an alternative reproduction of BeyondMimic in [mjlab](https://github.com/mujocolab/mjlab), a new Isaac Lab-style manager API powered by MuJoCo-Warp for RL and robotics research. See the implementation [here](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/tracking_env_cfg.py).

## Installation

- Install Isaac Lab v2.1.0 by following
  the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend
  using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: SSH
git clone git@github.com:MrH-creater/whole_body_tracking_X2_ultra.git

# Option 2: HTTPS
git clone https://github.com/MrH-creater/whole_body_tracking_X2_ultra.git
```

- Enter this checkout:

```bash
cd whole_body_tracking_X2_ultra
```

- For the original G1 tasks, pull the Unitree robot description files from GCS:

```bash
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

- For the X2 Ultra tasks, place the robot description files at:

```text
source/whole_body_tracking/whole_body_tracking/assets/robots/x2_ultra_description
```

The X2 configuration uses `x2_ultra_simple_collision.urdf`. Robot assets are excluded by this repository's
`.gitignore`, so they must be copied separately after cloning.

- Using a Python interpreter that has Isaac Lab installed, install the library

```bash
python -m pip install -e source/whole_body_tracking
```

## Motion Tracking

### Motion Preprocessing

Reference motions are stored as local NPZ files. The reference motion should be retargeted and use generalized
coordinates only.

- Gather the reference motion datasets (please follow the original licenses), we use the same convention as .csv of
  Unitree's dataset

    - Unitree-retargeted LAFAN1 Dataset is available
      on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    - Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
    - Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP).
    - Balance motions are from [HuB](https://hub-robot.github.io/)

- Convert retargeted motions to include the maximum coordinates information (body pose, body velocity, and body
  acceleration) via forward kinematics,

```bash
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 \
--output_file data/{motion_name}.npz --headless
```

The processed motion is written directly to the path passed through `--output_file`.

For named-column X2 Ultra CSV files containing root position, root quaternion, and all 31 joints, use the dedicated
MuJoCo converter:

```bash
python scripts/x2_csv_to_npz_mujoco.py \
  --input-csv motion_data/x2_motion/taijixiuzheng2_x2_fast2x.csv \
  --model source/whole_body_tracking/whole_body_tracking/assets/robots/x2_ultra_description/scene.xml \
  --output-npz motion_data/x2_motion/taijixiuzheng2_x2_fast2x_31dof.npz
```

Add `--output-video motion_data/x2_motion/taijixiuzheng2_x2_fast2x_mujoco.mp4` to render a MuJoCo preview. This
converter requires `mujoco`; video output also requires `ffmpeg`.

- Replay a converted motion in Isaac Sim:

```bash
python scripts/replay_npz.py --robot=x2_ultra --motion=data/formal/prepare.npz
```

### Policy Training

- Train an X2 Ultra policy with a local motion and TensorBoard logging:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-x2_ultra-Wo-State-Estimation-v0 \
--motion data/formal/prepare.npz --headless --logger tensorboard --run_name prepare
```

TensorBoard is the default logger, so `--logger tensorboard` may be omitted. View the training logs with:

```bash
tensorboard --logdir logs/rsl_rl/x2_ultra_beyondmimic --port 6006
```

### Policy Evaluation

- Play a local checkpoint with a local motion:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-x2_ultra-Wo-State-Estimation-v0 \
--num_envs=2 --load_run={run-directory} --checkpoint={checkpoint-file} \
--motion=data/formal/badamu.npz
```

## Code Structure

Below is an overview of the code structure for this repository:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  This directory contains the atomic functions to define the MDP for BeyondMimic. Below is a breakdown of the functions:

    - **`commands.py`**
      Command library to compute relevant variables from the reference motion, current robot state, and error
      computations. This includes pose and velocity error calculation, initial state randomization, and adaptive
      sampling.

    - **`rewards.py`**
      Implements the DeepMimic reward functions and smoothing terms.

    - **`events.py`**
      Implements domain randomization terms.

    - **`observations.py`**
      Implements observation terms for motion tracking and data collection.

    - **`terminations.py`**
      Implements early terminations and timeouts.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  Contains the environment (MDP) hyperparameters configuration for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  Contains the PPO hyperparameters for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  Contains robot-specific settings, including armature parameters, joint stiffness/damping calculation, and action scale
  calculation.

- **`scripts`**
  Includes utility scripts for preprocessing motion data, training policies, and evaluating trained policies.

This structure is designed to ensure modularity and ease of navigation for developers expanding the project.

## X2 Ultra Adaptation Details

### Robot model and articulation

- Robot configuration: `source/whole_body_tracking/whole_body_tracking/assets/robots/x2_ultra.py`
- Asset root index: `ROBOT_X2_ULTRA_ASSET_DIR` in
  `source/whole_body_tracking/whole_body_tracking/assets/__init__.py`
- Training URDF: `x2_ultra_simple_collision.urdf`
- Floating base with contact sensors and self-collision enabled
- 29 policy-controlled body joints plus two head joints held at their default pose by a fixed PD controller
- Joint order follows the 31 non-fixed joints in the training URDF exactly
- Initial floating-base height: `0.95 m`
- Initial leg pose: hip pitch `-0.312 rad`, knee `0.669 rad`, and ankle pitch `-0.363 rad`
- Soft joint-position limit factor: `0.9`
- Reflected joint armature: `0.01`

Effort limits come from the v1.3 URDF. The robot configuration follows the official AimDK example: legs and feet use
explicit delayed-PD actuators with a randomized command delay of 1--6 physics steps, while the waist and arms use
implicit actuators. The RL PD gains are:

| Group | Stiffness | Damping |
| --- | --- | --- |
| Hip pitch | 120 | 5 |
| Hip roll/yaw | 100 | 4 |
| Knee | 150 | 5 |
| Ankle pitch/roll | 40 | 2 |
| Waist yaw | 40.18 | 2.56 |
| Waist pitch/roll | 200 | 2 |
| Shoulders/elbows | 50 | 3 |
| Wrists | 20 | 2 |
| Head hold controller | 20 | 2 |

Action scales are computed per joint as `0.25 * effort_limit / stiffness`. Actions, motion commands, and simulated
joint states all use the 31-joint URDF order. The two head action dimensions have zero scale, while a fixed PD
controller holds the head at its default zero pose.

### Registered X2 Ultra tasks

- `Tracking-Flat-x2_ultra-v0`
- `Tracking-Flat-x2_ultra-Wo-State-Estimation-v0`
- `Tracking-Flat-x2_ultra-Low-Freq-v0`

The `Wo-State-Estimation` actor removes motion-anchor position and base linear velocity. Its policy observation contains
the 31-DoF reference position/velocity command, relative torso orientation, base angular velocity, 31 relative joint
positions, 31 joint velocities, and the previous 31 actions (164 values total). Deployment still requires joint
encoders and an IMU/attitude source for the torso-relative orientation.

The critic retains privileged body pose and base velocity observations during training; these are not actor inputs at
deployment.

### Motion mapping and rewards

The standard X2 motion NPZ format contains:

```text
joint_pos          [num_frames, 31]
joint_vel          [num_frames, 31]
body_pos_w         [num_frames, 32, 3]
body_quat_w        [num_frames, 32, 4]
body_lin_vel_w     [num_frames, 32, 3]
body_ang_vel_w     [num_frames, 32, 3]
joint_names        [31]
body_names         [32]
fps
```

`MotionLoader` checks that the NPZ joint count and `joint_names` agree with the X2 configuration. `MotionCommand`
resolves the simulated joint indices with `preserve_order=True`, so reference joint positions/velocities and simulated
joint positions/velocities are compared column-for-column.

The imitation reward remains the original body-space BeyondMimic objective: torso anchor pose plus the position,
orientation, linear velocity, and angular velocity of 14 selected bodies. Joint position and velocity errors are
computed as metrics, but are not currently direct reward terms. Additional penalties cover action rate, joint limits,
and undesired contacts.

For X2, the tracked end bodies are the ankle-roll and wrist-roll links. These names replace the G1 wrist-yaw endpoints
in X2-specific termination and contact settings because the X2 wrist chain ends at the roll link.

### Included motion example

The Tai Chi conversion workspace is under `motion_data/x2_motion/`. The final training file is:

```text
motion_data/x2_motion/taijixiuzheng2_x2_fast2x_31dof.npz
```

It contains 5,868 frames at 50 FPS, with all 31 standard joint position and velocity columns in URDF order. Both head
joint trajectories are zero for the entire motion. CSV and NPZ files are ignored by Git, so this motion must be copied
separately or stored in a W&B artifact; it will not be included automatically in a normal GitHub clone.

### Training notes

- The default environment runs at 50 Hz (`sim.dt=0.005`, decimation `4`), matching the example NPZ's 50 FPS.
- Episode reset positions and velocities come from the selected motion frame; `init_state.pos` is not the per-episode
  reset height.
- Reset pose randomization can add up to `+/-0.01 m` vertically.
- The reference motion stays inside the URDF hard position and velocity limits. Some frames exceed the configured 0.9
  soft position limits, so review the soft-limit factor or reference margins before a long production training run.
- Checkpoints trained with the previous 29-action layout are not compatible with the current 31-action X2 policy.
