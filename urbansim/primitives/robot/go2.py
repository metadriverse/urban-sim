# Copyright (c) 2025-2030, UrbanSim Project Developers from Zhou Lab @ UCLA.
# All rights reserved.
# Author: Honglin He
# SPDX-License-Identifier: BSD-3-Clause
# Acknowledgment:
# The template is from IsaacLab: https://github.com/isaac-sim/IsaacLab
# We thank the IsaacLab team for their contributions.

from __future__ import annotations
from dataclasses import MISSING
import torch
import math
import copy
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.navigation.mdp as nmdp
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file
from isaaclab.envs.mdp.actions import JointPositionAction, JointVelocityAction, JointPositionActionCfg, JointVelocityActionCfg, JointEffortActionCfg, JointEffortAction
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
UNITREE_GO2_CFG.init_state.pos = (0.0, 0.0, 0.3)

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
    env.events.push_robot = None
    env.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
    env.events.add_base_mass.params["asset_cfg"].body_names = "base"
    env.events.base_external_force_torque.params["asset_cfg"].body_names = "base"
    env.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env.events.reset_base.params = {
        "pose_range": {"x": (16.0, 16.0), "y": (16.0, 16.0), "yaw": (-3.14 / 4, 3.14 / 4)},
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
        low_level_observations=flat_ObservationsCfg.PolicyCfg(),
        debug_vis=False,
        align_heading_with_velocity=True,
    )