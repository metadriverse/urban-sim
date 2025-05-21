# Copyright (c) 2025-2030, UrbanSim Project Developers from Zhou Lab @ UCLA.
# All rights reserved.
# Author: Honglin He
# SPDX-License-Identifier: BSD-3-Clause
# Acknowledgment:
# The robot is from IsaacLab: https://github.com/isaac-sim/IsaacLab
# We thank the IsaacLab team for their contributions.

from __future__ import annotations
from dataclasses import MISSING
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.navigation.mdp as nmdp
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
UNITREE_GO2_CFG.init_state.pos = (0.0, 0.0, 0.4)
from urbansim.primitives.locomotion.general import ObservationsCfg as loc_ObservationsCfg

# ============================
# Locomotion
# ============================
@configclass
class GO2LocActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)

# Flat without height scanner
def GO2FlatModifyEnv(env):
    # event
    env.events.push_robot = None
    env.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
    env.events.add_base_mass.params["asset_cfg"].body_names = "base"
    env.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
    env.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env.events.reset_base.params = {
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }

    # rewards
    env.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
    env.rewards.feet_air_time.weight = 0.01
    env.rewards.undesired_contacts = None
    env.rewards.dof_torques_l2.weight = -0.0002
    env.rewards.track_lin_vel_xy_exp.weight = 1.5
    env.rewards.track_ang_vel_z_exp.weight = 0.75
    env.rewards.dof_acc_l2.weight = -2.5e-7

    # terminations
    env.terminations.base_contact.params["sensor_cfg"].body_names = "base"
    
    # override rewards
    env.rewards.flat_orientation_l2.weight = -2.5
    env.rewards.feet_air_time.weight = 0.25
    
# Terrain with height scanner
def GO2RoughModifyEnv(env):
    import numpy as np
    import torch
    import random
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # event
    env.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
    # scale down the terrains because the robot is small
    env.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
    env.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
    env.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

    # reduce action scale
    env.actions.joint_pos.scale = 0.25

    # event
    env.events.push_robot = None
    env.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
    env.events.add_base_mass.params["asset_cfg"].body_names = "base"
    env.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
    env.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env.events.reset_base.params = {
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }

    # rewards
    env.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
    env.rewards.feet_air_time.weight = 0.01
    env.rewards.undesired_contacts = None
    env.rewards.dof_torques_l2.weight = -0.0002
    env.rewards.track_lin_vel_xy_exp.weight = 1.5
    env.rewards.track_ang_vel_z_exp.weight = 0.75
    env.rewards.dof_acc_l2.weight = -2.5e-7

    # terminations
    env.terminations.base_contact.params["sensor_cfg"].body_names = "base"

# ============================
# Navigation
# ============================
@configclass
class GO2NavActionsCfg:
    """Action specifications for the MDP."""
    pre_trained_policy_action: nmdp.PreTrainedPolicyActionCfg = nmdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"assets/ckpts/locomotion/unitree_go2/general.pt",
        low_level_decimation=4,
        low_level_actions=mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True),
        low_level_observations=loc_ObservationsCfg.PolicyCfg(),
        debug_vis=False,
        align_heading_with_velocity=False,
    )

# ============================
# Trainig Config
# ============================
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "go2_locomotion"
    empirical_normalization = False
    clip_actions = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )