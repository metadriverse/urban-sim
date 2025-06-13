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
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv

WheelChairCfg = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"assets/robots/wheel_chair.usd",
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
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0., 0., 0.3),
    joint_pos={
            "base_to_front_axle_joint": 0.0,
        },
        joint_vel={
            "base_to_front_axle_joint":0.0,
        },
    ),
    actuators={"axle": 
            DCMotorCfg(
            joint_names_expr=["base_to_front_axle_joint"],
            saturation_effort=64.0,
            effort_limit=64.0,
            velocity_limit=20.,
            stiffness=25.0,
            damping=0.5,
            friction=0.
        ),
        
        },
    soft_joint_pos_limit_factor=1.0,# TODO: 0.95
)

class MovingAction(ActionTerm):
    r"""Pre-trained policy action term.

    This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
    The raw actions correspond to the commands for the pre-trained policy.

    """

    cfg: MovingActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: MovingActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self._counter = 0

        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)

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
        if not hasattr(self, "vx"):
            self.vx = torch.zeros(self.num_envs, device=self.device)
            self.vy = torch.zeros(self.num_envs, device=self.device)
            self.dh = torch.zeros(self.num_envs, device=self.device)
        if self._counter % ACTION_INTERVAL == 0:
            vx = self._raw_actions[:, 0].clamp(-0.5, 1.0)
            vy = self._raw_actions[:, 1].clamp(-1., 1.)
            dh = torch.arctan2(vy, vx).clamp(-math.pi / 5, math.pi / 5)
            self.vx = vx
            self.vy = vy
            self.dh = dh
        # raw_x = self.robot.data
        
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
class MovingActionCfg(ActionTermCfg):
    """Configuration for pre-trained policy action term.

    See :class:`PreTrainedPolicyAction` for more details.
    """

    class_type: type[ActionTerm] = MovingAction
    """ Class of the action term."""
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    """Whether to visualize debug information. Defaults to False."""
@configclass
class WheelChairMovingActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = MovingActionCfg(asset_name="robot")

def WheelChairNavModifyEnv(env):
    # change to base link
    # sensors in the scene
    env.scene.height_scanner.prim_path = '{ENV_REGEX_NS}/Robot'
    env.scene.contact_forces = None
    env.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/front_cam"
    # terminations
    env.terminations.collision = None
    
    return env
