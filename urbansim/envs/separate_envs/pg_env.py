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
parser.add_argument("--use_async", action="store_true", default=False, help="Run simulation in async mode.")
parser.add_argument("--use_terrain", action="store_true", default=False, help="Generate terrain in the environment.")
parser.add_argument("--save_images", action="store_true", default=False, help="Whether to save images during the simulation.")
parser.add_argument("--scenario_type", type=str, default="static", choices=['clean', 'static', 'dynamic'])
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
from urbansim.primitives.navigation.pg_env_cfg import SceneCfg, ObservationsCfg, RewardsCfg, TerminationsCfg, CommandsCfg, EventCfg, CurriculumCfg
from urbansim.primitives.robot.coco import COCO_CFG, COCOVelocityActionsCfg, COCONavModifyEnv

COCO_CFG.init_state.pos = (1.0, 1.0, 0.5)
pg_config = dict(
    type=args_cli.scenario_type, # [clean, static, dynamic]
    with_terrain=args_cli.use_terrain,
    with_boundary=True,
    map_region=20,
    buffer_width=1,
    num_object=13,
    num_pedestrian=4,
    walkable_seed=0,
    non_walkable_seed=1,
    seed=423,
    unique_env_num=20,
    ped_forward_inteval=10,
    moving_max_t=80,
)

"""
Define the Environment
"""
@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""
    eye: tuple[float, float, float] = (-20, -20, 7.2)

    lookat: tuple[float, float, float] = (20, 20, -5)

    cam_prim_path: str = "/OmniverseKit_Persp"

    resolution: tuple[int, int] = (1920, 1080)

    origin_type: str = "world"

    env_index: int = 0

    asset_name: str | None = None
    
@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    if args_cli.use_async:
        print("Using async mode for the environment.")
        scene = SceneCfg(scenario_generation_method = "async procedural generation", 
                        #  pg_config=pg_config,
                         num_envs=args_cli.num_envs, env_spacing=21.0)
    else:
        print("Using sync mode for the environment.")
        scene = SceneCfg(scenario_generation_method = "sync procedural generation", 
                        #  pg_config=pg_config,
                         num_envs=args_cli.num_envs, env_spacing=21.0)
    
    viewer = ViewerCfg()
    
    observations = ObservationsCfg()
    actions = COCOVelocityActionsCfg()
    
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
                
        self.scene.robot = COCO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if hasattr(self.scene, 'height_scanner'):
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # modify env
        COCONavModifyEnv(self)
    
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
    if args_cli.save_images:
        root_dir = './log_images'
        import os
        from isaacsim.sensors.camera import Camera
        os.makedirs(root_dir, exist_ok=True)
        cam_prim_path = "/OmniverseKit_Persp"
        camera = Camera(prim_path=cam_prim_path, resolution=(1920, 1080))
        camera.initialize()
        
    for t in range(1000000):
        action = env.action_space.sample()
        action = torch.from_numpy(action).float().to(env.device)
        obs, reward, terminated, truncated, info = env.step(action)
        if args_cli.save_images:
            import cv2
            rgb_image = camera.get_rgba()
            cv2.imwrite(f"{root_dir}/rgb_{t:04d}.png", rgb_image[:, :, :3][..., ::-1])
            if t == 200:
                print(f"Saved images to {root_dir}")
                env.close()
                break
    
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
    