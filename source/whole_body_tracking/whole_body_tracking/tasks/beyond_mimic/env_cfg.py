from isaaclab.utils import configclass

from whole_body_tracking.assets.robots.x2_ultra import (
    X2_ULTRA_ACTION_SCALE,
    X2_ULTRA_CYLINDER_CFG,
    X2_ULTRA_TRACKING_JOINT_NAMES,
)
from whole_body_tracking.tasks.beyond_mimic.tracking_env_cfg import TrackingEnvCfg


@configclass
class X2_ULTRA_FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = X2_ULTRA_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.joint_names = X2_ULTRA_TRACKING_JOINT_NAMES
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = X2_ULTRA_ACTION_SCALE
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.joint_names = X2_ULTRA_TRACKING_JOINT_NAMES
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_roll_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_roll_link",
        ]

@configclass
class X2_ULTRA_FlatWoStateEstimationEnvCfg(X2_ULTRA_FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

@configclass
class Play_X2_ULTRA_FlatWoStateEstimationEnvCfg(X2_ULTRA_FlatWoStateEstimationEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.play = True
        self.events.push_robot = None
        self.episode_length_s = 120
        self.observations.policy.enable_corruption = False

        # # test
        # self.observations.policy.enable_corruption = False

        # self.events.physics_material = None
        # self.events.add_joint_default_pos = None
        # self.events.base_com = None