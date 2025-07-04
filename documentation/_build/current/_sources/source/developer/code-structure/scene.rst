UrbanScene: Scenario Construction
==================================

The `UrbanScene` class in URBAN-SIM is a central abstraction responsible for creating and managing dynamic urban environments for training and evaluation.

It extends `InteractiveScene` from Isaac Lab, and supports both predefined maps and procedurally generated scenes with varying complexity.

.. note::
   Users do not directly call `UrbanScene` in most cases; instead, they configure it via the ``SceneCfg`` object in their environment YAML file.

Core Responsibilities
-----------------------

The `UrbanScene` class handles:

- **Asset instantiation** from configuration (robots, objects, terrain, sensors)
- **Scene reset logic**, including pedestrian and object dynamics
- **Scenario generation**, from static maps to procedural pipelines

Key Methods
------------

.. code-block:: python

   class UrbanScene(InteractiveScene):
       def __init__(self, cfg: UrbanSceneCfg)
       def _add_entities_from_cfg(self, procedural_generation=False)
       def reset(self, env_ids: Sequence[int] | None = None)
       def update(self, dt)

- ``__init__``: Binds the configuration and initializes internal variables.
- ``_add_entities_from_cfg``: Adds all assets (robot, terrain, lights, pedestrians) to the scene. If `procedural_generation=True`, the map is built on-the-fly.
- ``reset``: Re-generates or resets entities for the given environments.
- ``update``: Performs scene update on each timestep, including moving pedestrians or updating lights.

Scenario Generation Modes
---------------------------

Scenario generation is determined by the ``scenario_generation_method`` field in the ``UrbanSceneCfg`` configuration.

.. code-block:: python

   if self.cfg.scenario_generation_method == "predefined":
       self.generate_predefined_scene()
   elif self.cfg.scenario_generation_method == "async procedural generation":
       self.generate_async_procedural_scene()
   elif self.cfg.scenario_generation_method == "sync procedural generation":
       self.generate_sync_procedural_scene()
   elif self.cfg.scenario_generation_method == "limited async procedural generation":
       self.generate_limited_async_procedural_scene()
   elif self.cfg.scenario_generation_method == "limited sync procedural generation":
       self.generate_limited_sync_procedural_scene()

Explanation of Modes:

- **predefined**: Load from static assets (e.g., fixed USD map).
- **async procedural generation**: Asynchronously create a new map per environment (for full randomization).
- **sync procedural generation**: All environments share the same map (synchronized PG).
- **limited async procedural generation**: Randomize within a fixed region and number of assets per env.
- **limited sync procedural generation**: Share limited PG map across environments.

Use Cases
-----------

- For static evaluation, use `predefined`.
- For generalization training, prefer `async procedural generation` or `limited async procedural generation`.
- For curriculum or debugging, `sync` versions are faster and easier to visualize.

Related Files
--------------

While `urban_scene.py` defines the implementation logic of the scene itself, two supporting files are worth noting:

- **urban_scene_cfg.py**: Contains the `UrbanSceneCfg` dataclass, which defines scenario parameters (e.g., number of objects, lighting conditions, region size) passed into `UrbanScene`.

- **utils.py**: Includes helper functions for asset manipulation, pedestrian trajectory generation, and other utilities used by the scene construction pipeline.

Takeaway
---------

`UrbanScene` is where the **world gets built**. Whether you're running a thousand parallel simulations or a single debug rollout, this class ensures every environment is initialized with coherent terrain, lighting, and asset placement — in line with your scenario generation strategy.
