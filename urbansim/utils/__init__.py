from urbansim.utils.config import Config, merge_config_with_unknown_keys, merge_config
from urbansim.utils.coordinates_shift import panda_heading, panda_vector, metadrive_heading, metadrive_vector
from urbansim.utils.math import safe_clip, clip, norm, distance_greater, safe_clip_for_small_array, Vector
from urbansim.utils.random_utils import get_np_random, random_string
from urbansim.utils.utils import is_mac, import_pygame, recursive_equal, setup_logger, merge_dicts, \
    concat_step_infos, is_win, time_me
from urbansim.utils.map_manager import PGMapManager

from metaurban.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metaurban.component.map.base_map import BaseMap
from metaurban.component.map.pg_map import parse_map_config, MapGenerateMethod
from metaurban.envs.base_env import BaseEnv
from metaurban.utils import clip, Config
import gymnasium as gym
import numpy as np
from metaurban.constants import DEFAULT_AGENT
from metaurban.constants import RENDER_MODE_NONE
BASE_DEFAULT_CONFIG = dict(
     # ===== agent =====
    # Whether randomize the car model for the agent, randomly choosing from 4 types of cars
    random_agent_model=False,
    # The ego config is: env_config["vehicle_config"].update(env_config"[agent_configs"]["default_agent"])
    agent_configs={DEFAULT_AGENT: dict(use_special_color=True, spawn_lane_index=None)},

    # ===== multi-agent =====
    # This should be >1 in MARL envs, or set to -1 for spawning as many vehicles as possible.
    num_agents=1,
    # Turn on this to notify the simulator that it is MARL env
    is_multi_agent=False,
    # The number of agent will be fixed adn determined at the start of the episode, if set to False
    allow_respawn=False,
    # How many substeps for the agent to stay static at the death place after done. (Default for MARL: 25)
    delay_done=0,
    
    # ===== Action/Control =====
    # Please see Documentation: Action and Policy for more details
    # What policy to use for controlling agents
    # TODO: agent policy
    agent_policy=None,
    # If set to True, agent_policy will be overriden and change to ManualControlPolicy
    manual_control=False,
    # What interfaces to use for manual control, options: "steering_wheel" or "keyboard" or "xbos"
    controller="keyboard",
    # Used with EnvInputPolicy. If set to True, the env.action_space will be discrete
    discrete_action=False,
    # If True, use MultiDiscrete action space. Otherwise, use Discrete.
    use_multi_discrete=False,
    # How many discrete actions are used for steering dim
    discrete_steering_dim=5,
    # How many discrete actions are used for throttle/brake dim
    discrete_throttle_dim=5,
    # Check if the action is contained in gym.space. Usually turned off to speed up simulation
    action_check=False,

    # ===== Observation =====
    # Please see Documentation: Observation for more details
    # Whether to normalize the pixel value from 0-255 to 0-1
    norm_pixel=True,
    # The number of timesteps for stacking image observation
    stack_size=3,
    # Whether to use image observation or lidar. It takes effect in get_single_observation
    image_observation=False,
    # Like agent_policy, users can use customized observation class through this field
    agent_observation=None,

    # ===== Termination =====
    # The maximum length of each agent episode. Set to None to remove this constraint
    horizon=None,
    # If set to True, the terminated will be True as well when the length of agent episode exceeds horizon
    truncate_as_terminate=False,
    
    # ===== Main Camera =====
    # A True value makes the camera follow the reference line instead of the vehicle, making its movement smooth
    use_chase_camera_follow_lane=False,
    # Height of the main camera
    camera_height=2.2,
    # Distance between the camera and the vehicle. It is the distance projecting to the x-y plane.
    camera_dist=7.5,
    # Pitch of main camera. If None, this will be automatically calculated
    camera_pitch=None,  # degree
    # Smooth the camera movement
    camera_smooth=True,
    # How many frames used to smooth the camera
    camera_smooth_buffer_size=20,
    # FOV of main camera
    camera_fov=65,
    # Only available in MARL setting, choosing which agent to track. Values should be "agent0", "agent1" or so on
    prefer_track_agent=None,
    # Setting the camera position for the Top-down Camera for 3D viewer (pressing key "B" to activate it)
    top_down_camera_initial_x=0,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=200,
    
    # ===== PG Map Config =====
    map=3,  # int or string: an easy way to fill map_config
    block_dist_config=PGBlockDistConfig,
    random_lane_width=False,
    random_lane_num=False,
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        BaseMap.GENERATE_CONFIG: None,  # it can be a file path / block num / block ID sequence
        BaseMap.LANE_WIDTH: 2.5,
        BaseMap.LANE_NUM: 3,
        "exit_length": 50,
        "start_position": [0, 0],
    },
    store_map=True,
    
    # ===== Terrain =====
    # The size of the square map region, which is centered at [0, 0]. The map objects outside it are culled.
    map_region_size=2048,
    # Whether to remove lanes outside the map region. If True, lane localization only applies to map region
    cull_lanes_outside_map=False,
    # Road will have a flat marin whose width is determined by this value, unit: [m]
    drivable_area_extension=7,
    # Height scale for mountains, unit: [m]. 0 height makes the terrain flat
    height_scale=50,
    # If using mesh collision, mountains will have physics body and thus interact with vehicles.
    use_mesh_terrain=False,
    # If set to False, only the center region of the terrain has the physics body
    full_size_mesh=True,
    # Whether to show crosswalk
    show_crosswalk=True,
    # Whether to show sidewalk
    show_sidewalk=True,
    sensors={},

    # ===== Vehicle =====
    vehicle_config=dict(
        # Vehicle model. Candidates: "s", "m", "l", "xl", "default". random_agent_model makes this config invalid
        vehicle_model="default",
        # If set to True, the vehicle can go backwards with throttle/brake < -1
        enable_reverse=False,
        # Whether to show the box as navigation points
        show_navi_mark=True,
        # Whether to show a box mark at the destination
        show_dest_mark=False,
        # Whether to draw a line from current vehicle position to the designation point
        show_line_to_dest=False,
        # Whether to draw a line from current vehicle position to the next navigation point
        show_line_to_navi_mark=False,
        # If set to True, the vehicle will be in color green in top-down renderer or MARL setting
        use_special_color=False,
        # Clear wheel friction, so it can not move by setting steering and throttle/brake. Used for ReplayPolicy
        no_wheel_friction=False,

        # ===== image capturing =====
        # Which camera to use for image observation. It should be a sensor registered in sensor config.
        image_source="rgb_camera",

        # ===== vehicle spawn and navigation =====
        # A BaseNavigation instance. It should match the road network type.
        navigation_module=None,
        # A lane id specifies which lane to spawn this vehicle
        spawn_lane_index=None,
        # destination lane id. Required only when navigation module is not None.
        destination=None,
        # the longitudinal and lateral position on the spawn lane
        spawn_longitude=5.0,
        spawn_lateral=0.0,

        # If the following items is assigned, the vehicle will be spawn at the specified position with certain speed
        spawn_position_heading=None,
        spawn_velocity=None,  # m/s
        spawn_velocity_car_frame=False,

        # ==== others ====
        # How many cars the vehicle has overtaken. It is deprecated due to bug.
        overtake_stat=False,
        # If set to True, the default texture for the vehicle will be replaced with a pure color one.
        random_color=False,
        # The shape of vehicle are predefined by its class. But in special scenario (WaymoVehicle) we might want to
        # set to arbitrary shape.
        width=None,
        length=None,
        height=None,
        mass=None,

        # Set the vehicle size only for pygame top-down renderer. It doesn't affect the physical size!
        top_down_width=None,
        top_down_length=None,

        # ===== vehicle module config =====
        lidar=dict(
            num_lasers=240, distance=50, num_others=0, gaussian_noise=0.0, dropout_prob=0.0, add_others_navi=False
        ),
        side_detector=dict(num_lasers=0, distance=50, gaussian_noise=0.0, dropout_prob=0.0),
        lane_line_detector=dict(num_lasers=0, distance=20, gaussian_noise=0.0, dropout_prob=0.0),
        show_lidar=False,
        show_side_detector=False,
        show_lane_line_detector=False,
        # Whether to turn on vehicle light, only available when enabling render-pipeline
        light=False,
    ),
    # ===== Debug =====
    # Please see Documentation: Debug for more details
    pstats=False,  # turn on to profile the efficiency
    debug=False,  # debug, output more messages
    debug_panda3d=False,  # debug panda3d
    debug_physics_world=False,  # only render physics world without model, a special debug option
    debug_static_world=False,  # debug static world
    log_level='CRITICAL',  # log level. logging.DEBUG/logging.CRITICAL or so on
    show_coordinates=False,  # show coordinates for maps and objects for debug
    
    # ===== Record/Replay Metadata =====
    # Please see Documentation: Record and Replay for more details
    # When replay_episode is True, the episode metadata will be recorded
    record_episode=False,
    # The value should be None or the log data. If it is the later one, the simulator will replay logged scenario
    replay_episode=None,
    # When set to True, the replay system will only reconstruct the first frame from the logged scenario metadata
    only_reset_when_replay=False,
    # If True, when creating and replaying object trajectories, use the same ID as in dataset
    force_reuse_object_name=False,

    # ===== randomization =====
    num_scenarios=100, # the number of scenarios in this environment
    start_seed=1213, # 12 153XCT 1212121XSX
    
        # ===== GUI =====
    # Please see Documentation: GUI for more details
    # Whether to show these elements in the 3D scene
    show_fps=True,
    show_logo=False,
    show_mouse=True,
    show_skybox=True,
    show_terrain=True,
    show_interface=True,
    # Show marks for policies for debugging multi-policy setting
    show_policy_mark=False,
    # Show an arrow marks for providing navigation information
    show_interface_navi_mark=True,
    # A list showing sensor output on window. Its elements are chosen from sensors.keys() + "dashboard"
    interface_panel=[],
    
     # ===== Logical Engine Core config =====
    # If true pop a window to render
    use_render=False,
    # (width, height), if set to None, it will be automatically determined
    window_size=(1200, 900),
    # Physics world step is 0.02s and will be repeated for decision_repeat times per env.step()
    physics_world_step_size=2e-2,
    decision_repeat=5,
    # This is an advanced feature for accessing image without moving them to ram!
    image_on_cuda=False,
    # Don't set this config. We will determine the render mode automatically, it runs at physics-only mode by default.
    _render_mode=RENDER_MODE_NONE,
    # If set to None: the program will run as fast as possible. Otherwise, the fps will be limited under this value
    force_render_fps=None,
    # We will maintain a set of buffers in the engine to store the used objects and can reuse them when possible
    # enhancing the efficiency. If set to True, all objects will be force destroyed when call clear()
    force_destroy=False,
    # Number of buffering objects for each class.
    num_buffering_objects=200,
    # Turn on it to use render pipeline, which provides advanced rendering effects (Beta)
    render_pipeline=False,
    # daytime is only available when using render-pipeline
    daytime="19:00",  # use string like "13:40", We usually set this by editor in toolkit
    # Shadow range, unit: [m]
    shadow_range=50,
    # Whether to use multi-thread rendering
    multi_thread_render=True,
    multi_thread_render_mode="Cull",  # or "Cull/Draw"
    # Model loading optimization. Preload pedestrian for avoiding lagging when creating it for the first time
    preload_models=True,
    # model compression increasing the launch time
    disable_model_compression=True,
)