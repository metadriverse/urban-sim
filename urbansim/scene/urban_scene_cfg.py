# Copyright (c) 2022-2025, The UrbanSim Project Developers.
# Author: Honglin He
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

@configclass
class UrbanSceneCfg:
    """Configuration for the urban scene.
    """

    num_envs: int = MISSING
    """Number of environment instances handled by the scene."""

    env_spacing: float = MISSING
    """Spacing between environments.

    This is the default distance between environment origins in the scene. Used only when the
    number of environments is greater than one.
    """

    lazy_sensor_update: bool = True
    """Whether to update sensors only when they are accessed. Default is True.

    If true, the sensor data is only updated when their attribute ``data`` is accessed. Otherwise, the sensor
    data is updated every time sensors are updated.
    """

    replicate_physics: bool = True
    """Enable/disable replication of physics schemas when using the Cloner APIs. Default is True.

    If True, the simulation will have the same asset instances (USD prims) in all the cloned environments.
    Internally, this ensures optimization in setting up the scene and parsing it via the physics stage parser.

    If False, the simulation allows having separate asset instances (USD prims) in each environment.
    This flexibility comes at a cost of slowdowns in setting up and parsing the scene.

    .. note::
        Optimized parsing of certain prim types (such as deformable objects) is not currently supported
        by the physics engine. In these cases, this flag needs to be set to False.
    """

    filter_collisions: bool = True
    """Enable/disable collision filtering between cloned environments. Default is True.

    If True, collisions will not occur between cloned environments.

    If False, the simulation will generate collisions between environments.

    .. note::
        Collisions can only be filtered automatically in direct workflows when physics replication is enabled.
        If ``replicated_physics=False`` and collision filtering is desired, make sure to call ``scene.filter_collisions()``.
    """
    
    scenario_generation_method: str = None
    """
    The method used to generate the scenario. Default is None.
    Choose from:
    'async procedural generation'
    'sync procedural generation'
    'predefined'
    '3dgs'
    """
    
    maximum_static_object_number: int = 1e10
    """
    maximum number of static objects in the scene. Default is 1e10.
    """
    
    maximum_dynamic_object_number: int = 1e10
    """
    maximum number of dynamic objects in the scene. Default is 1e10.
    """
    
    scenario_type: str = None
    """
    scenario type. Default is None.
    Choose from:
    'map only'
    'static'
    'dynamic'
    """
    
    task_type: str = None
    """
    task type. Default is None.
    Choose from:
    'navigation'
    'locomotion'
    """
    
    static_assets_path: str = None
    """
    path to the static assets. Default is None.
    """
    
    dynamic_assets_path: str = None
    """
    path to the dynamic assets. Default is None.
    """
    
    area_size: float = None
    """
    area size. Default is None.
    """
