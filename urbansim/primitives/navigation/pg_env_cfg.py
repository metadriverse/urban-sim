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
        'Night/kloppenheim_02_4k.hdr',
        'Night/moonlit_golf_4k.hdr',
        'Cloudy/abandoned_parking_4k.hdr',
        'Cloudy/champagne_castle_1_4k.hdr',
        'Cloudy/evening_road_01_4k.hdr',
        'Cloudy/kloofendal_48d_partly_cloudy_4k.hdr',
        'Cloudy/lakeside_4k.hdr',
        'Cloudy/sunflowers_4k.hdr',
        'Cloudy/table_mountain_1_4k.hdr',
        'Evening/evening_road_01_4k.hdr',
        'Storm/approaching_storm_4k.hdr',
    ]
)
light_intensity = np.random.uniform(700.0, 800.0)

#------------------------------------------------
# Scene Config
@configclass
class SceneCfg(UrbanSceneCfg):
    """Configuration"""
    # scenario type
    scenario_generation_method: str = "sync procedural generation"
    # [sync or async]
    
    # procedural generation config
    pg_config: dict = dict(
        type='clean', # [clean, static, dynamic]
        map_config='XCX', # [X for intersection, S for straight road, C for curved road] [user can use any combination of these letters to generate a map, such as XSX]
        lane_num = 2,
        lane_width = 3.5,
        exit_length = 50.,
        sidewalk_type='Medium Commercial',
        # ['Narrow Sidewalk', 'Narrow Sidewalk with Trees', 'Ribbon Sidewalk', 'Neighborhood 1', 'Neighborhood 2',
        #     'Medium Commercial', 'Wide Commercial'
        # ]
        object_density=0.7,
        seed=0,
        use_orca_for_agent_trajectory_generation=False,
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
    
    # sensor
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # cameras
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=['rgb', 'distance_to_image_plane'],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0, focus_distance=400.0, horizontal_aperture=30.0, clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
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
