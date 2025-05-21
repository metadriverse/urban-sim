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

    pass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pass


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    pass


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    pass


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    pass


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    pass
