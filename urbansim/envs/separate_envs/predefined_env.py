# Copyright (c) 2022-2025, The UrbanSim Project Developers.
# Author: Honglin He
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import time
import gymnasium as gym
import numpy as np
from collections import defaultdict
from typing import Union, Dict, AnyStr, Optional, Tuple, Callable, Any
import copy

# launch omniverse
import argparse
from omni.isaac.kit import SimulationApp
import builtins
from collections.abc import Sequence
from dataclasses import MISSING
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to URBAN-SIM Environments!")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--color_scheme",
    type=str,
    default="none",
    choices=["height", "random", "none"],
    help="Color scheme to use for the terrain generation.",
)
parser.add_argument(
    "--use_curriculum",
    action="store_true",
    default=False,
    help="Whether to use the curriculum for the terrain generation.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument(
    "--show_flat_patches",
    action="store_true",
    default=False,
    help="Whether to show the flat patches computed during the terrain generation.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
AppLauncher.add_app_launcher_args(parser)
from urbansim import cli_args
cli_args.add_rsl_rl_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless, 'enable_cameras':args_cli.enable_cameras}
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.utils.configclass import configclass
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
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
from isaaclab.sensors import CameraCfg, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.assets import RigidObject, RigidObjectCfg

from urbansim.scene.urban_scene_cfg import UrbanSceneCfg
from urbansim.envs.abstract_env import AbstractEnv
from urbansim.primitives.navigation.random_static import ObservationsCfg, RewardsCfg, TerminationsCfg, CommandsCfg, EventCfg, CurriculumCfg
from urbansim.primitives.robot.coco import COCO_CFG, COCOVelocityActionsCfg, COCONavModifyEnv

"""
Define the Urban Scene
"""
@configclass
class BaseUrbanSceneCfg(UrbanSceneCfg):
    # scenario type
    scenario_generation_method: str = "predefined"
    
    # place objects
    temp_object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.1,
            height=0.2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    
    # robot
    robot: ArticulationCfg = MISSING
    
    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    
    # sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # cameras
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=['rgb', 'distance_to_image_plane'],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=30.0, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )
    # height scanner
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=[f"/World/ground"],
    )
    
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            #intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
            intensity=750.0,
            #color=(0.75, 0.75, 0.75),
            # visible_in_primary_ray=False,
        ),
    )


"""
Define the Environment
"""
@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    scene = BaseUrbanSceneCfg(num_envs=4, env_spacing=2.0)
    
    observations = ObservationsCfg()
    actions = GO2MapActionsCfg()
    
    rewards = RewardsCfg()
    
    terminations = TerminationsCfg()
    
    commands = CommandsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()
    
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 20
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = 4
        self.sim.disable_contact_processing = True
        
        if hasattr(self.scene, 'height_scanner'):
            if self.scene.height_scanner is not None:
                self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if hasattr(self.scene, 'contact_forces'):
            if self.scene.contact_forces is not None:
                self.scene.contact_forces.update_period = self.sim.dt
                
        self.scene.robot = GO2_MAP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if hasattr(self.scene, 'height_scanner'):
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # modify env
        GO2MapModifyEnv(self)
    
    
"""
register env
"""
import gymnasium as gym
gym.register(
    id=f"UrbanSim-Base",
    entry_point="urbansim.envs:AbstractEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BaseEnvCfg,
        "rsl_rl_cfg_entry_point": None,
    },
)


def main():
    env = gym.make("UrbanSim-Base", cfg=BaseEnvCfg(), render_mode='rgb_array')
    env.reset()
    for t in range(1000000):
        action = env.action_space.sample()
        action = torch.from_numpy(action).float().to(env.device)
        obs, reward, terminated, truncated, info = env.step(action)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
    