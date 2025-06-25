from __future__ import annotations
from dataclasses import MISSING
import torch
import numpy as np
import math
import isaaclab.sim as sim_utils
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as loc_mdp
import isaaclab_tasks.manager_based.navigation.mdp as nav_mdp
from isaaclab.sensors import CameraCfg, TiledCameraCfg

from urbansim.scene.urban_scene import UrbanSceneCfg

# Random path
material_path = './assets/materials'

walkable_material_path_list = [
    f'{material_path}/Ground/Small_Cobblestone.mdl',
    f'{material_path}/Ground/Large_Granite_Paving.mdl',
    f'{material_path}/Ground/Rough_Gravel.mdl',
    f'{material_path}/Ground/Paving_Stones.mdl',
    # f'{material_path}/Ground/Mulch.mdl',
    
]
non_walkable_material_path_list = [    
    f'{material_path}/Ground/Mulch.mdl',
    f'{material_path}/Ground/Gravel_Track_Ballast.mdl',
    f'{material_path}/Ground/Cobblestone_Big_and_Loose.mdl',
    f'{material_path}/Ground/Asphalt_Fine.mdl',
]
terrain_gen_cfg = ROUGH_TERRAINS_CFG.replace(curriculum=False, color_scheme='none')

light = np.random.choice(
    [
        # 'Night/kloppenheim_02_4k.hdr',
        # 'Night/moonlit_golf_4k.hdr',
        # 'Cloudy/abandoned_parking_4k.hdr',
        'Cloudy/champagne_castle_1_4k.hdr',
        # 'Cloudy/evening_road_01_4k.hdr',
        # 'Cloudy/kloofendal_48d_partly_cloudy_4k.hdr',
        # 'Cloudy/lakeside_4k.hdr',
        # 'Cloudy/sunflowers_4k.hdr',
        # 'Cloudy/table_mountain_1_4k.hdr',
        # 'Evening/evening_road_01_4k.hdr',
        # 'Storm/approaching_storm_4k.hdr',
    ]
)
light_intensity = np.random.uniform(880.0, 900.0)

#------------------------------------------------
# Scene Config
@configclass
class SceneCfg(UrbanSceneCfg):
    """Configuration"""
    # scenario type
    scenario_generation_method: str = "limited sync procedural generation"
    # [sync or async]
    
    # procedural generation config
    pg_config: dict = dict(
        type='dynamic', # [clean, static, dynamic]
        with_terrain=True,
        with_boundary=True,
        map_region=20,
        buffer_width=1,
        num_object=10,
        num_pedestrian=9,
        walkable_seed=0,
        non_walkable_seed=1,
        seed=0,
        unique_env_num=20,
        ped_forward_inteval=10,
        moving_max_t=80,
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
    
    terrain_gen_cfg = ROUGH_TERRAINS_CFG.replace(curriculum=False, color_scheme='none')
    
    terrain_importer = TerrainImporterCfg(
        prim_path=f"/World/Obstacle_terrain",
        max_init_terrain_level=None,
        terrain_type="plane",
        terrain_generator=terrain_gen_cfg,
        debug_vis=False,
        visual_material=None
    )
    
    terrain_importer_walkable_list = [
        TerrainImporterCfg(
            prim_path=f"/World/Walkable_{i:03d}",
            max_init_terrain_level=None,
            terrain_type="plane",
            terrain_generator=terrain_gen_cfg,
            debug_vis=False,
            visual_material=None
        ) for i in range(4)
    ]
    terrain_non_walkable_list = [
        TerrainImporterCfg(
            prim_path=f"/World/NonWalkable_{i:03d}",
            max_init_terrain_level=None,
            terrain_type="plane",
            terrain_generator=terrain_gen_cfg,
            debug_vis=False,
            visual_material=None
        ) for i in range(4)
    ]
    terrain_walkable_material_list = [
        sim_utils.MdlFileCfg(mdl_path=walkable_material_path_list[i], project_uvw=True, texture_scale=1000) for i in range(len(terrain_importer_walkable_list))
    ]
    terrain_non_walkable_material_list = [
        sim_utils.MdlFileCfg(mdl_path=non_walkable_material_path_list[i], project_uvw=True, texture_scale=1000) for i in range(len(terrain_non_walkable_list))
    ]
    
    # sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # cameras
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=1080 // 8,
        width=1920 // 8,
        data_types=['rgb', 'distance_to_camera'],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix = [531., 0., 960., 0., 531., 540., 0., 0., 1.],
            width=1920,
            height=1080,
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.51, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
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
            texture_file=f"{NVIDIA_NUCLEUS_DIR}/Assets/Skies/{light}",
            intensity=light_intensity,
        ),
    )


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
        debug_vis=False,
        ranges=nav_mdp.UniformPose2dCommandCfg.Ranges(pos_x=(15.0, 30.0), pos_y=(15.0, 30.0), heading=(-math.pi, math.pi)),
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
        pose_command = ObsTerm(func=loc_mdp.advanced_generated_commands, params={"command_name": "pose_command", 
                                                                                 "max_dim": 2,
                                                                                 "normalize": True})
        
    @configclass
    class SensorCfg(ObsGroup):
        rgb = ObsTerm(func=nav_mdp.rgbd_processed, params={"sensor_cfg": SceneEntityCfg("camera")})
    
    # observation groups
    policy: PolicyCfg = PolicyCfg()
    sensor: SensorCfg = SensorCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    reset_base = EventTerm(
        func=loc_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.3, 0.3), "y": (0.3, 0.3), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    arrived_reward = RewTerm(
        loc_mdp.is_terminated_term, weight=2000.0, params={"term_keys": "arrive"}
    )
    # out_of_region_penalty = RewTerm(func=loc_mdp.is_terminated_term, weight=-1.0, params={"term_keys": "out_of_region"})
    collision_penalty = RewTerm(func=loc_mdp.is_terminated_term, weight=-200.0, params={"term_keys": "collision"})
    position_tracking = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=10.0,
        params={"std": 5.0, "command_name": "pose_command"},
    )
    position_tracking_fine = RewTerm(
        func=nav_mdp.position_command_error_tanh,
        weight=50.0,
        params={"std": 1.0, "command_name": "pose_command"},
    )
    moving_towards_goal = RewTerm(
        func=nav_mdp.moving_towards_goal_reward, 
        weight=20.0, 
        params={"command_name": "pose_command"})
    target_vel_rew = RewTerm(
        func=nav_mdp.target_vel_reward, 
        weight=10.0, 
        params={"command_name": "pose_command"})


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=loc_mdp.time_out, time_out=True)
    collision = DoneTerm(
        func=nav_mdp.illegal_contact, time_out=False,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body_link"), # change path to your robot
            "threshold": 1.0
            },
    )
    arrive = DoneTerm(
        func=nav_mdp.arrive, time_out=False,
        params={"threshold": 1.0, "command_name": "pose_command"},
    )
    # out_of_region = DoneTerm(
    #     func=nav_mdp.out_of_region, time_out=False,
    #     params={"threshold_l": -1.0, "threshold_h": 31.0},
    # )

from isaaclab.envs import ManagerBasedRLEnv
from collections.abc import Sequence
map_region = 30.0
def increase_moving_distance(
        env:ManagerBasedRLEnv,
        env_ids:Sequence[int],    # pylint: disable=unused-argument
        command_name: str = 'pose_command',
        x_moving_min=(11.0, map_region - 2.),
        y_moving_min=(5.0, map_region - 2.),
        total_iterations=10,
        num_steps_per_iteration=102400):
    cur_iteration = env.common_step_counter // num_steps_per_iteration
    start_x, end_x = x_moving_min
    start_y, end_y = y_moving_min
    x = start_x + (end_x - start_x) * (cur_iteration / total_iterations)

    x_left = start_x + (end_x - start_x) * (cur_iteration / total_iterations)
    x_left = min(x_left, end_x)
    env.command_manager.get_term(command_name).cfg.ranges.pos_x = (10., x_left)
    env.command_manager.get_term(command_name).cfg.ranges.pos_y = (0., map_region)

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    increased_moving_distance = CurrTerm(func=increase_moving_distance,
                                         params={})
