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
from datetime import datetime
import os, sys
import math
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
parser.add_argument("--env", type=str, default=None, help="Configuration file for the environment.")
parser.add_argument("--framework", type=str, default='rlgames', choices=['rlgames', 'rsl'], help="Learning framework. Recommended to use [rsl] for locomotion and [rlgames] for navigation.")
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
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
# parser.add_argument("--enable_cameras", action="store_true", default=False, help="cameras")
AppLauncher.add_app_launcher_args(parser)
from urbansim import cli_args
cli_args.add_rsl_rl_args(parser)
args_cli = parser.parse_args()
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
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

# rl
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
import isaaclab_tasks  # noqa: F401
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.common.algo_observer import IsaacAlgoObserver
from isaaclab.utils import update_class_from_dict, update_dict
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
# parser config
import yaml
with open(args_cli.env, 'r') as f:
    env_config = yaml.safe_load(f)
print('|===================================|')
print('[Global Config]')
print(env_config)
print('|===================================|')

# specific env
task_name = env_config['Task'].lower()
assert task_name in ['navigation', 'locomotion'], '[Task] Only support [navigation, locomotion] currently'
setting_name = env_config['Env']['name']
if task_name == 'navigation':
    assert setting_name in ['emppty', 'terrain', 'static', 'dynamic'], '[Env] Only support [emppty, terrain, static, dynamic] currently in navigation currently'
elif task_name == 'locomotion':
    assert setting_name in ['flat', 'slope', 'stair', 'rough', 'general'], '[Env] Only support [flat, slope, stair, rough, general] in locomotion currently'
robot_name = env_config['Robot']['type']
assert robot_name in ['unitree_go2', 'coco', 'anymal_c', 'unitree_b2w', 'unitree_g1'], '[Robot] Only support [unitree_go2, coco, anymal_c, unitree_b2w, unitree_g1] currently'

print('[INFO] Task: {}'.format(task_name))
print('[INFO] Setting: {}'.format(setting_name))
print('[INFO] Robot: {}'.format(robot_name))

# Basic settings
observation_cfg = None
command_cfg = None
# MDP settings
reward_cfg = None
termination_cfg = None
event_cfg = None
curriculum_cfg = None
# Scene settings
scene_cfg = None
# Robot settings
robot_cfg = None
action_cfg = None
def dummy_fn(env): return env
modify_env_fn = dummy_fn

if setting_name == 'general':
    from urbansim.primitives.locomotion.general import ObservationsCfg
    from urbansim.primitives.locomotion.general import CommandsCfg
    from urbansim.primitives.locomotion.general import RewardsCfg
    from urbansim.primitives.locomotion.general import TerminationsCfg
    from urbansim.primitives.locomotion.general import EventCfg
    from urbansim.primitives.locomotion.general import CurriculumCfg
    
    observation_cfg = ObservationsCfg
    command_cfg = CommandsCfg
    reward_cfg = RewardsCfg
    termination_cfg = TerminationsCfg
    event_cfg = EventCfg
    curriculum_cfg = CurriculumCfg
    
    from urbansim.primitives.locomotion.general import SceneCfg
    scene_cfg = SceneCfg
    
if task_name.lower() == 'locomotion':
    assert not robot_name.lower() == 'coco', '[Robot] Coco is not supported in locomotion currently'
    if robot_name.lower() == 'unitree_go2':
        from urbansim.primitives.robot.unitree_go2 import UNITREE_GO2_CFG
        from urbansim.primitives.robot.unitree_go2 import GO2LocActionsCfg
        if setting_name != 'flat':
            from urbansim.primitives.robot.unitree_go2 import GO2RoughModifyEnv
            robot_cfg = UNITREE_GO2_CFG
            action_cfg = GO2LocActionsCfg
            modify_env_fn = GO2RoughModifyEnv
        else:
            from urbansim.primitives.robot.unitree_go2 import GO2FlatModifyEnv
            robot_cfg = UNITREE_GO2_CFG
            action_cfg = GO2LocActionsCfg
            modify_env_fn = GO2FlatModifyEnv
    elif robot_name.lower() == 'anymal_c':
        from urbansim.primitives.robot.anymal_c import ANYMAL_C_CFG
        from urbansim.primitives.robot.anymal_c import AnymalCLocActionsCfg
        if setting_name != 'flat':
            from urbansim.primitives.robot.anymal_c import AnymalCRoughModifyEnv
            robot_cfg = ANYMAL_C_CFG
            action_cfg = AnymalCLocActionsCfg
            modify_env_fn = AnymalCRoughModifyEnv
        else:
            from urbansim.primitives.robot.anymal_c import AnymalCFlatModifyEnv
            robot_cfg = ANYMAL_C_CFG
            action_cfg = AnymalCLocActionsCfg
            modify_env_fn = AnymalCFlatModifyEnv
    elif robot_name.lower() == 'unitree_g1':
        from urbansim.primitives.robot.unitree_g1 import G1_MINIMAL_CFG
        from urbansim.primitives.robot.unitree_g1 import G1LocActionsCfg
        if setting_name != 'flat':
            from urbansim.primitives.robot.unitree_g1 import G1RoughModifyEnv
            robot_cfg = G1_MINIMAL_CFG
            action_cfg = G1LocActionsCfg
            modify_env_fn = G1RoughModifyEnv
        else:
            from urbansim.primitives.robot.unitree_g1 import G1FlatModifyEnv
            robot_cfg = G1_MINIMAL_CFG
            action_cfg = G1LocActionsCfg
            modify_env_fn = G1FlatModifyEnv
            
# generate env cfg
@configclass
class EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene = scene_cfg(num_envs=env_config['Training']['num_envs'], env_spacing=env_config['Omniverse']['env_spacing'])
    # Basic settings
    observations = observation_cfg()
    actions = action_cfg()
    commands = command_cfg()
    # MDP settings
    rewards = reward_cfg()
    terminations = termination_cfg()
    events = event_cfg()
    curriculum = curriculum_cfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = env_config['Omniverse']['decimation']
        self.episode_length_s = env_config['Omniverse']['episode_length_s']
        # simulation settings
        self.sim.dt = env_config['Omniverse']['simulation_dt']
        self.sim.render_interval = env_config['Omniverse']['rendering_interval']
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        
        if hasattr(self.scene, 'height_scanner'):
            if self.scene.height_scanner is not None:
                self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if hasattr(self.scene, 'contact_forces'):
            if self.scene.contact_forces is not None:
                self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
                
        self.scene.robot = robot_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if hasattr(self.scene, 'height_scanner'):
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # modify env
        modify_env_fn(self)
  
# register env
import gymnasium as gym
gym.register(
    id=f"IsaacUrban-{task_name}-{robot_name}-{setting_name}",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": EnvCfg,
        "rsl_rl_cfg_entry_point":  f"urbansim.primitives.robot.{robot_name}:PPORunnerCfg",
        "rl_games_cfg_entry_point": f"configs/rl_configs/{task_name}/{robot_name}/{setting_name}_train.yaml",
    },
)
args_cli.task = f"IsaacUrban-{task_name}-{robot_name}-{setting_name}"

# RL framework
rl_framework = env_config['Training']['framework']
use_ddp = env_config['Training']['use_ddp']
if use_ddp:
    rl_framework = 'rl_games'
    print('Only rl_games supports DDP training, change framework to rl_games')
assert rl_framework in ['rsl', 'rl_games'], '[Framework] Only support [rsl, rl_games] currently'

print('[INFO] All settings are done.')

def parse_env_cfg():
    use_gpu = not args_cli.cpu
    num_envs=args_cli.num_envs
    use_fabric=not args_cli.disable_fabric
    args_cfg = {"sim": {"physx": dict()}, "scene": dict()}
    # resolve pipeline to use (based on input)
    if use_gpu is not None:
        if not use_gpu:
            args_cfg["sim"]["device"] = "cpu"
        else:
            args_cfg["sim"]["device"] = "cuda:0"

    # disable fabric to read/write through USD
    if use_fabric is not None:
        args_cfg["sim"]["use_fabric"] = use_fabric

    # number of environments
    if num_envs is not None:
        args_cfg["scene"]["num_envs"] = num_envs
    
    cfg = EnvCfg()
    # update the main configuration
    if isinstance(cfg, dict):
        cfg = update_dict(cfg, args_cfg)
    else:
        update_class_from_dict(cfg, args_cfg)
        
    args_cfg["sim"]['enable_scene_query_support'] = True
    return cfg


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def train_with_rsl(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"[INFO] Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == '__main__':
    train_with_rsl()
    simulation_app.close()