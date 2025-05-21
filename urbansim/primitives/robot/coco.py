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

# Robot instance
COCO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"assets/robots/coco_one/coco_one.usd",
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0., 0., 0.3),
        joint_pos={
            ".*wheel_joint*": 0.0,"base_to_front_axle_joint":0.0,
        },
        joint_vel={
            ".*wheel_joint":0.0,"base_to_front_axle_joint":0.0,#TODO: WHY NOT WORKING
        },
    ),
    actuators={"wheels": DelayedPDActuatorCfg(
            joint_names_expr=[".*wheel_joint"],
            velocity_limit=100.0,
            min_delay=0,  
            max_delay=4, 
            stiffness={
                ".*_wheel_joint": 0.0,
            },
            damping={".*_wheel_joint": 0.3},
            friction={
                ".*_wheel_joint": 0.0,
            },
            armature={
                ".*_wheel_joint": 0.0,
            },

        ), "axle": 
            DCMotorCfg(
            joint_names_expr=["base_to_front_axle_joint"],
            saturation_effort=64.0,
            effort_limit=64.0,
            velocity_limit=20.,
            stiffness=25.0,
            damping=0.5,
            friction=0.
        ),
        
    "shock": ImplicitActuatorCfg(
            joint_names_expr=[".*shock_joint"],
            stiffness=0.0,
            damping=0.0,
        )
        
        },
    soft_joint_pos_limit_factor=1.0,# TODO: 0.95
)

DT = 0.1
MAX_W = 2.0
MAX_V = 2.0
ACTION_INTERVAL = 4

# Support two types of action space
# 2. velocity and angular velocity
# 1. waypoint

# ===========================
# Action space: Velocity and angular velocity
# ===========================
class ClassicalCarAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: ClassicalCarActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ClassicalCarActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._counter = 0
        self.last_wheel_angle = torch.zeros(self.num_envs, 1, device=self.device)

        self.axle_names = ["base_to_front_axle_joint"]
        self.wheel_names = ["front_left_wheel_joint","front_right_wheel_joint", "rear_left_wheel_joint", "rear_right_wheel_joint"]
        self.shock_names = [".*shock_joint"]
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)

        # prepare low level actions
        self.acceleration_action: JointVelocityAction = JointVelocityAction(JointVelocityActionCfg(asset_name="robot", joint_names=[".*_wheel_joint"], scale=10.0, use_default_offset=False), env)
        self.steering_action: JointPositionAction = JointPositionAction(JointPositionActionCfg(asset_name="robot", joint_names=self.axle_names, scale=1., use_default_offset=True), env)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions
    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions


    def apply_actions(self):
        if self._counter % ACTION_INTERVAL == 0:
            max_wheel_v = 4.
            wheel_base = 1.5
            radius_rear = 0.3
            max_ang = 40 * torch.pi / 180
            velocity = self.raw_actions[..., :1].clamp(-max_wheel_v, max_wheel_v) / radius_rear 
            angular = self.raw_actions[..., 1:2].clamp(-max_ang, max_ang)
            angular[angular.abs() < 0.05] = torch.zeros_like(angular[angular.abs() < 0.05])
            R = wheel_base / torch.tan(angular)
            left_wheel_angle = torch.arctan(wheel_base / (R - 0.5 * 1.8))
            right_wheel_angle = torch.arctan(wheel_base / (R + 0.5 * 1.8))

        
            self.steering_action.process_actions(((right_wheel_angle + left_wheel_angle) / 2.))
            self.acceleration_action.process_actions(torch.cat([velocity, velocity, velocity, velocity], dim=1))
        
        self.steering_action.apply_actions()
        self.acceleration_action.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass
    

@configclass
class ClassicalCarActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = ClassicalCarAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    """Whether to visualize debug information. Defaults to False."""
@configclass
class COCOVelocityActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = ClassicalCarActionCfg(asset_name="robot")
    
# ===========================
# Action space: Waypoint
# ===========================
def clip_angle(angle):
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi
def pd_controller(waypoint):
    dx = waypoint[:, 0]
    dy = waypoint[:, 1]
    v = dx / DT
    w = torch.arctan2(dy, dx) / DT
    v = torch.clip(v, 0, MAX_V)
    w = torch.clip(w, -MAX_W, MAX_W)
    return v, w

class ClassicalCarWaypointAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: ClassicalCarWaypointActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: ClassicalCarWaypointActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._counter = 0
        self.last_wheel_angle = torch.zeros(self.num_envs, 1, device=self.device)

        ### TODO: find it out later
        self.axle_names = ["base_to_front_axle_joint"]
        self.wheel_names = ["front_left_wheel_joint","front_right_wheel_joint", "rear_left_wheel_joint", "rear_right_wheel_joint"]
        self.shock_names = [".*shock_joint"]
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)

        # prepare low level actions
        self.acceleration_action: JointVelocityAction = JointVelocityAction(JointVelocityActionCfg(asset_name="robot", joint_names=[".*_wheel_joint"], scale=10.0, use_default_offset=False), env)
        self.steering_action: JointPositionAction = JointPositionAction(JointPositionActionCfg(asset_name="robot", joint_names=self.axle_names, scale=1., use_default_offset=True), env)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.raw_actions
    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        actions = actions * 2 / 10
        v, w = pd_controller(actions)
        eps = 1e-6  # Small value to avoid division by zero
        R = torch.where(
            torch.abs(w) > eps,
            v / (w + eps),
            torch.full_like(v, 1e6)
        )
        steering_angle = torch.atan(1.5 / R)
        self._raw_actions[:, 0] = v
        self._raw_actions[:, 1] = steering_angle


    def apply_actions(self):
        if self._counter % ACTION_INTERVAL == 0:
            max_wheel_v = 4.
            wheel_base = 1.5
            radius_rear = 0.3
            max_ang = 40 * torch.pi / 180
            velocity = self.raw_actions[..., :1].clamp(-max_wheel_v, max_wheel_v) / radius_rear 
            angular = self.raw_actions[..., 1:2].clamp(-max_ang, max_ang)
            angular[angular.abs() < 0.05] = torch.zeros_like(angular[angular.abs() < 0.05])
            R = wheel_base / torch.tan(angular)
            left_wheel_angle = torch.arctan(wheel_base / (R - 0.5 * 1.8))
            right_wheel_angle = torch.arctan(wheel_base / (R + 0.5 * 1.8))
        
            self.steering_action.process_actions(((right_wheel_angle + left_wheel_angle) / 2.))
            self.acceleration_action.process_actions(torch.cat([velocity, velocity, velocity, velocity], dim=1))
        
        self.steering_action.apply_actions()
        self.acceleration_action.apply_actions()
        self._counter += 1

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass
    

@configclass
class ClassicalCarWaypointActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = ClassicalCarWaypointAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    """Whether to visualize debug information. Defaults to False."""
@configclass
class COCOWaypointActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = ClassicalCarWaypointActionCfg(asset_name="robot")
    
def COCONavModifyEnv(env):
    # change to base link
    # sensors in the scene
    env.scene.height_scanner.prim_path = '{ENV_REGEX_NS}/Robot/base_link'
    env.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/base_link/front_cam"
    # terminations
    env.terminations.base_contact.params["sensor_cfg"].body_names = "body_link"
    env.terminations.base_contact.params['threshold'] = 1.0
    
    return env