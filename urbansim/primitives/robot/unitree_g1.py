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

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG
G1_MINIMAL_CFG.init_state.pos = (0.0, 0.0, 0.74)

# ============================
# Locomotion
# ============================
@configclass
class G1LocActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

# Flat without height scanner
def G1FlatModifyEnv(env):
    pass


from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg
@configclass
class G1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_pitch_joint",
                    ".*_elbow_roll_joint",
                ],
            )
        },
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_five_joint",
                    ".*_three_joint",
                    ".*_six_joint",
                    ".*_four_joint",
                    ".*_zero_joint",
                    ".*_one_joint",
                    ".*_two_joint",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )

# Terrain with height scanner
def G1RoughModifyEnv(env):
    env.rewards = G1Rewards()
    
    env.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"

    # Randomization
    env.events.push_robot = None
    env.events.add_base_mass = None
    env.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
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

    # Rewards
    env.rewards.lin_vel_z_l2.weight = 0.0
    env.rewards.undesired_contacts = None
    env.rewards.flat_orientation_l2.weight = -1.0
    env.rewards.action_rate_l2.weight = -0.005
    env.rewards.dof_acc_l2.weight = -1.25e-7
    env.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
    )
    env.rewards.dof_torques_l2.weight = -1.5e-7
    env.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
    )

    # Commands
    env.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
    env.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
    env.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

    # terminations
    env.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"

# ============================
# Navigation
# ============================
@configclass
class G1NavActionsCfg:
    """Action specifications for the MDP."""
    pre_trained_policy_action: nmdp.PreTrainedPolicyActionCfg = nmdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"assets/ckpts/locomotion/unitree_g1.pt",
        low_level_decimation=4,
        low_level_actions=mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True),
        low_level_observations=loc_ObservationsCfg.PolicyCfg(),
        debug_vis=False,
        align_heading_with_velocity=True,
    )

def G1NavModifyEnv(env):
    from isaaclab.managers import EventTermCfg as EventTerm
    env.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/head_link"
    env.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/head_link/front_cam"
    # env.scene.camera.offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 1.3), rot=(0.5, -0.5, 0.5, -0.5), convention="ros")
    # terminations
    if hasattr(env.terminations, 'base_contact'):
        env.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"
        env.terminations.base_contact.params['threshold'] = 1.0
    if hasattr(env.rewards, 'contact_penalty'):
        env.rewards.contact_penalty.params['sensor_cfg'].body_names = ["torso_link"]
    env.events.reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    env.events.physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    env.events.base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    
    return env

# ============================
# Trainig Config
# ============================
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_locomotion"
    empirical_normalization = False
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
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )