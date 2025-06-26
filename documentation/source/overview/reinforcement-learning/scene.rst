Scene Configuration
=====================

URBAN-SIM supports multiple pipelines for generating simulation scenes with varying complexity, spatial structure, and randomness.  
Each environment binds a scene configuration (`SceneCfg`) that determines what types of scenes are loaded, how they are arranged, and how they are reused across multiple environments.

Scene Binding
---------------------

Scene configuration is specified in the environment config via:

.. code-block:: python

   @configclass
   class EnvCfg(ManagerBasedRLEnvCfg):
       ...
       scene = SceneCfg(...)

The scene class determines the number of environments, their spatial arrangement, and the scenario generation method used.

Scenario Generation Pipelines
------------------------------

URBAN-SIM includes two main pipelines for generating scenes:

1. **Random Object Placement**  
   Fast and lightweight. Places obstacles, buildings, and pedestrians randomly within a defined area.  
   - Used via: ``random_env.py``
   - Suitable for large-scale training and pretraining

2. **Procedural Generation (PG)**  
   Structured scene generation based on layout rules and scenario graphs.  
   - Used via: ``pg_env.py``
   - Suitable for curriculum learning and benchmarking

Each pipeline supports:

- Scene-level asset instantiation
- Environment duplication (`--num_envs`)
- Asynchronous or synchronous layout control

Example YAML Config
---------------------

In a YAML config, the scene section may look like:

.. code-block:: yaml

   Env:
    name: static
    map_region: 30
    num_objects: 25
   Omniverse:
    headless: True
    simulation_dt: 0.005
    rendering_interval: 4
    decimation: 20
    episode_length_s: 30.0
    env_spacing: 31.0
    scenario_generation_method: limited async procedural generation

This will spawn environments with randomized object placement and layout variation.

Scene Customization
---------------------

Scene behavior can be further customized via:

- Changing asset spawn positions or scale
- Overriding `pg_config` in the environment config
- Specifying deterministic vs stochastic layouts

You can also inject dynamic agents (e.g., pedestrians, moving vehicles) through scenario setup like

.. code-block:: python

   pg_config: dict = dict(
        type='dynamic', # [clean, static, dynamic]
        with_terrain=False,
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

Spatial Arrangement
--------------------

URBAN-SIM supports both:

- **Synchronous scenes**: All environments share the same map and layout (ideal for evaluation)
- **Asynchronous scenes**: Each environment is different (ideal for generalization training)

Each environment is placed on a grid, spaced by `env_spacing`, and receives its own camera, sensor, and physics handle.

More details on scene generation and configuration can be found in the file ``urbansim/scene/urban_scene.py``.
