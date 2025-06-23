# Copyright (c) 2025-2030, UrbanSim Project Developers from Zhou Lab @ UCLA.
# All rights reserved.
# Author: Honglin He
# SPDX-License-Identifier: BSD-3-Clause
# Acknowledgment:
# The robot is from IsaacLab: https://github.com/isaac-sim/IsaacLab
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
from urbansim.primitives.locomotion.general import ObservationsCfg as loc_ObservationsCfg

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG
ANYMAL_C_CFG.init_state.pos = (0.0, 0.0, 0.5)

# ============================
# Locomotion
# ============================
@configclass
class AnymalCLocActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
    
# Flat without height scanner
def AnymalCFlatModifyEnv(env):
    pass
    
# Terrain with height scanner
def AnymalCRoughModifyEnv(env):
    return env

# ============================
# Navigation
# ============================
@configclass
class AnymalCNavActionsCfg:
    """Action specifications for the MDP."""
    pre_trained_policy_action: nmdp.PreTrainedPolicyActionCfg = nmdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"assets/ckpts/locomotion/anymal_C.pt",
        low_level_decimation=4,
        low_level_actions=mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True),
        low_level_observations=loc_ObservationsCfg.PolicyCfg(),
        debug_vis=False,
        align_heading_with_velocity=True,
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
    experiment_name = "anymalC_locomotion"
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