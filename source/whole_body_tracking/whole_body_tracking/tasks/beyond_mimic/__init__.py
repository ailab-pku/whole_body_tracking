import gymnasium as gym

from . import env_cfg
from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-x2_ultra-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.X2_ULTRA_FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2_ULTRA_FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-x2_ultra-v0-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.Play_X2_ULTRA_FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:X2_ULTRA_FlatPPORunnerCfg",
    },
)
