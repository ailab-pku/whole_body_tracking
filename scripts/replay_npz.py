"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_npz.py --motion_file data/formal/prepare.npz --robot x2_ultra
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import pathlib

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--motion", "--motion_file", dest="motion", type=str, required=True, help="Motion NPZ file.")
parser.add_argument(
    "--robot",
    choices=("g1", "x2_ultra"),
    default="g1",
    help="Robot model used to replay the motion.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG, G1_TRACKING_JOINT_NAMES
from whole_body_tracking.assets.robots.x2_ultra import X2_ULTRA_CYLINDER_CFG, X2_ULTRA_TRACKING_JOINT_NAMES
from whole_body_tracking.tasks.beyond_mimic.mdp import MotionLoader

ROBOT_CFG = G1_CYLINDER_CFG if args_cli.robot == "g1" else X2_ULTRA_CYLINDER_CFG
ROBOT_JOINT_NAMES = G1_TRACKING_JOINT_NAMES if args_cli.robot == "g1" else X2_ULTRA_TRACKING_JOINT_NAMES


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    robot_joint_indexes = robot.find_joints(ROBOT_JOINT_NAMES, preserve_order=True)[0]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    motion_path = pathlib.Path(args_cli.motion).expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file does not exist: {motion_path}")
    if motion_path.suffix.lower() != ".npz":
        raise ValueError(f"Motion file must be an NPZ file: {motion_path}")

    motion = MotionLoader(
        str(motion_path),
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
        joint_names=ROBOT_JOINT_NAMES,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(
            motion.joint_pos[time_steps],
            motion.joint_vel[time_steps],
            joint_ids=robot_joint_indexes,
        )
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
