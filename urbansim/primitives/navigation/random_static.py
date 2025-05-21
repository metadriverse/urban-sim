from __future__ import annotations
from dataclasses import MISSING
import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loc_mdp
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp

from urbansim.scene.urban_scene import UrbanSceneCfg

#------------------------------------------------
# Scene Config
@configclass
class SceneCfg(UrbanSceneCfg):
    """Configuration"""
    pass

#------------------------------------------------
# Specific settings for training
# Note:
# 1. define action cfg for each type of robots
# 2. if you want to change the reward, command or any other settings, change it in custom_env
#------------------------------------------------
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    pose_command = nav_mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(30.0, 30.0),
        debug_vis=True,
        ranges=nav_mdp.UniformPose2dCommandCfg.Ranges(pos_x=(7.0, 10.0), pos_y=(7.0, 10.0), heading=(-math.pi, math.pi)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        pose_command = ObsTerm(func=loc_mdp.generated_commands, params={"command_name": "pose_command"})
        
    @configclass
    class SensorCfg(ObsGroup):
        rgb = ObsTerm(func=nav_mdp.image_processed, params={"sensor_cfg": SceneEntityCfg("camera")})
    
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    sensor: SensorCfg = SensorCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=loc_mdp.is_terminated, weight=-100.0)
    position_tracking = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=1.0,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    moving_towards_goal = RewTerm(
        func=nav_mdp.moving_towards_goal_reward, 
        weight=1.0, 
        params={"command_name": "pose_command"})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=loc_mdp.time_out, time_out=False)
    collision = DoneTerm(
        func=nav_mdp.illegal_contact, time_out=False,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body_link"), # change path to your robot
            "threshold": 1.0
            },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass
