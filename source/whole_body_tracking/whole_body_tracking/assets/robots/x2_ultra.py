import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ROBOT_X2_ULTRA_ASSET_DIR


# Values resolved from the official x2_config/joint_gains.yaml used by the
# AimDK example. Keeping the maps here makes this extension self-contained.
X2_ULTRA_RL_STIFFNESS = {
    "left_hip_pitch_joint": 120.0,
    "left_hip_roll_joint": 100.0,
    "left_hip_yaw_joint": 100.0,
    "left_knee_joint": 150.0,
    "left_ankle_pitch_joint": 40.0,
    "left_ankle_roll_joint": 40.0,
    "right_hip_pitch_joint": 120.0,
    "right_hip_roll_joint": 100.0,
    "right_hip_yaw_joint": 100.0,
    "right_knee_joint": 150.0,
    "right_ankle_pitch_joint": 40.0,
    "right_ankle_roll_joint": 40.0,
    "waist_yaw_joint": 40.18,
    "waist_pitch_joint": 200.0,
    "waist_roll_joint": 200.0,
    "left_shoulder_pitch_joint": 50.0,
    "left_shoulder_roll_joint": 50.0,
    "left_shoulder_yaw_joint": 50.0,
    "left_elbow_joint": 50.0,
    "left_wrist_yaw_joint": 20.0,
    "left_wrist_pitch_joint": 20.0,
    "left_wrist_roll_joint": 20.0,
    "right_shoulder_pitch_joint": 50.0,
    "right_shoulder_roll_joint": 50.0,
    "right_shoulder_yaw_joint": 50.0,
    "right_elbow_joint": 50.0,
    "right_wrist_yaw_joint": 20.0,
    "right_wrist_pitch_joint": 20.0,
    "right_wrist_roll_joint": 20.0,
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
}

X2_ULTRA_RL_DAMPING = {
    "left_hip_pitch_joint": 5.0,
    "left_hip_roll_joint": 4.0,
    "left_hip_yaw_joint": 4.0,
    "left_knee_joint": 5.0,
    "left_ankle_pitch_joint": 2.0,
    "left_ankle_roll_joint": 2.0,
    "right_hip_pitch_joint": 5.0,
    "right_hip_roll_joint": 4.0,
    "right_hip_yaw_joint": 4.0,
    "right_knee_joint": 5.0,
    "right_ankle_pitch_joint": 2.0,
    "right_ankle_roll_joint": 2.0,
    "waist_yaw_joint": 2.56,
    "waist_pitch_joint": 2.0,
    "waist_roll_joint": 2.0,
    "left_shoulder_pitch_joint": 3.0,
    "left_shoulder_roll_joint": 3.0,
    "left_shoulder_yaw_joint": 3.0,
    "left_elbow_joint": 3.0,
    "left_wrist_yaw_joint": 2.0,
    "left_wrist_pitch_joint": 2.0,
    "left_wrist_roll_joint": 2.0,
    "right_shoulder_pitch_joint": 3.0,
    "right_shoulder_roll_joint": 3.0,
    "right_shoulder_yaw_joint": 3.0,
    "right_elbow_joint": 3.0,
    "right_wrist_yaw_joint": 2.0,
    "right_wrist_pitch_joint": 2.0,
    "right_wrist_roll_joint": 2.0,
    "head_yaw_joint": 2.0,
    "head_pitch_joint": 2.0,
}

_LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
]
_FOOT_JOINT_NAMES = [
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]
_WAIST_JOINT_NAMES = ["waist_roll_joint", "waist_pitch_joint"]
_WAIST_YAW_JOINT_NAMES = ["waist_yaw_joint"]
_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]


X2_ULTRA_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=os.path.join(ROBOT_X2_ULTRA_ASSET_DIR, "x2_ultra_simple_collision.urdf"),
        usd_dir=os.path.join(os.path.expanduser("~"), ".cache", "whole_body_tracking", "x2_ultra"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            ".*_elbow_joint": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit={
                ".*_hip_yaw_joint": 120.0,
                ".*_hip_roll_joint": 120.0,
                ".*_hip_pitch_joint": 120.0,
                ".*_knee_joint": 120.0,
            },
            velocity_limit_sim=12.0,
            stiffness={joint: X2_ULTRA_RL_STIFFNESS[joint] for joint in _LEG_JOINT_NAMES},
            damping={joint: X2_ULTRA_RL_DAMPING[joint] for joint in _LEG_JOINT_NAMES},
            armature=0.01,
            min_delay=1,
            max_delay=6,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit={
                ".*_ankle_pitch_joint": 36.0,
                ".*_ankle_roll_joint": 24.0,
            },
            velocity_limit_sim=15.0,
            stiffness={joint: X2_ULTRA_RL_STIFFNESS[joint] for joint in _FOOT_JOINT_NAMES},
            damping={joint: X2_ULTRA_RL_DAMPING[joint] for joint in _FOOT_JOINT_NAMES},
            armature=0.01,
            min_delay=1,
            max_delay=6,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=48.0,
            velocity_limit_sim=13.0,
            stiffness={joint: X2_ULTRA_RL_STIFFNESS[joint] for joint in _WAIST_JOINT_NAMES},
            damping={joint: X2_ULTRA_RL_DAMPING[joint] for joint in _WAIST_JOINT_NAMES},
            armature=0.01,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=120.0,
            velocity_limit_sim=12.0,
            stiffness={joint: X2_ULTRA_RL_STIFFNESS[joint] for joint in _WAIST_YAW_JOINT_NAMES},
            damping={joint: X2_ULTRA_RL_DAMPING[joint] for joint in _WAIST_YAW_JOINT_NAMES},
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 36.0,
                ".*_shoulder_roll_joint": 36.0,
                ".*_shoulder_yaw_joint": 24.0,
                ".*_elbow_joint": 24.0,
                ".*_wrist_roll_joint": 4.8,
                ".*_wrist_pitch_joint": 4.8,
                ".*_wrist_yaw_joint": 24.0,
            },
            velocity_limit_sim=15.0,
            stiffness={joint: X2_ULTRA_RL_STIFFNESS[joint] for joint in _ARM_JOINT_NAMES},
            damping={joint: X2_ULTRA_RL_DAMPING[joint] for joint in _ARM_JOINT_NAMES},
            armature=0.01,
        ),
        # The head is held at its default zero pose and is intentionally not
        # exposed to policy actions.
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
            effort_limit_sim={"head_yaw_joint": 2.6, "head_pitch_joint": 0.6},
            velocity_limit_sim={"head_yaw_joint": 6.019, "head_pitch_joint": 6.28},
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)

# BeyondMimic uses a per-joint position-action scale. Derive it from the same
# official torque limits and gains; head actions have zero scale.
_EFFORT_LIMITS = {
    **{joint: 120.0 for joint in _LEG_JOINT_NAMES},
    "left_ankle_pitch_joint": 36.0,
    "right_ankle_pitch_joint": 36.0,
    "left_ankle_roll_joint": 24.0,
    "right_ankle_roll_joint": 24.0,
    "waist_yaw_joint": 120.0,
    "waist_pitch_joint": 48.0,
    "waist_roll_joint": 48.0,
    "left_shoulder_pitch_joint": 36.0,
    "right_shoulder_pitch_joint": 36.0,
    "left_shoulder_roll_joint": 36.0,
    "right_shoulder_roll_joint": 36.0,
    "left_shoulder_yaw_joint": 24.0,
    "right_shoulder_yaw_joint": 24.0,
    "left_elbow_joint": 24.0,
    "right_elbow_joint": 24.0,
    "left_wrist_yaw_joint": 24.0,
    "right_wrist_yaw_joint": 24.0,
    "left_wrist_pitch_joint": 4.8,
    "right_wrist_pitch_joint": 4.8,
    "left_wrist_roll_joint": 4.8,
    "right_wrist_roll_joint": 4.8,
}
X2_ULTRA_ACTION_SCALE = {
    joint: 0.25 * _EFFORT_LIMITS[joint] / X2_ULTRA_RL_STIFFNESS[joint]
    if joint in _EFFORT_LIMITS and X2_ULTRA_RL_STIFFNESS[joint] > 0.0
    else 0.0
    for joint in X2_ULTRA_RL_STIFFNESS
}
