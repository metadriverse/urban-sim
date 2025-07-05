Environment Configurations
====================================

The `EnvCfg` class defines the configuration schema for URBAN-SIM training environments.  
It inherits from `ManagerBasedRLEnvCfg` in Isaac Lab and encapsulates settings for the simulation scene, observation/action space, rewards, curriculum, and termination logic.

.. code-block:: python

   @configclass
   class EnvCfg(ManagerBasedRLEnvCfg):
       """
       Environment configuration schema for URBAN-SIM.
       Inherits from ManagerBasedRLEnvCfg.
       """

       # Scene configuration
       scene: SceneCfg = SceneCfg()

       # Simulation parameters
       viewer: ViewerCfg = ViewerCfg()
       observations: ObservationCfg = ObservationCfg()
       actions: ActionCfg = ActionCfg()
       commands: CommandCfg = CommandCfg()

       # MDP components
       rewards: RewardCfg = RewardCfg()
       terminations: TerminationCfg = TerminationCfg()
       events: EventCfg = EventCfg()
       curriculum: CurriculumCfg = CurriculumCfg()

       # Robot configuration
       robot_name: str = "COCO"  # Default robot type

Scene Configuration
-------------------

The `scene` field instantiates a `SceneCfg`-compatible class, determining how the environment is constructed.

- If `pg_config` is **not provided**, the system defaults to random object placement:

  .. code-block:: python

     scene = scene_cfg(num_envs=..., env_spacing=...)

- If `pg_config` **is provided**, a procedural scenario is built:

  .. code-block:: python

     scene = scene_cfg(
         num_envs=...,
         env_spacing=...,
         pg_config=pg_config,
         scenario_generation_method=...,
     )

Key options:

- ``num_envs``: Number of parallel environments
- ``env_spacing``: Spacing between environments in simulation world
- ``pg_config``: Optional procedural generation config
- ``scenario_generation_method``: Overrides default random placement

Simulation Configuration
------------------------

- ``viewer``: Defines viewer resolution, camera mode, etc. via `ViewerCfg`
- ``observations``: Sensor and observation space definitions via `observation_cfg`
- ``actions``: Action space definition (e.g., continuous, discrete) via `action_cfg`
- ``commands``: External command signal structure via `command_cfg`

MDP Configuration
-----------------

- ``rewards``: Reward shaping logic via `reward_cfg`
- ``terminations``: Termination condition logic via `termination_cfg`
- ``events``: Optional triggerable in-sim events
- ``curriculum``: Curriculum learning parameters via `curriculum_cfg`

All these components are dynamically loaded based on the environment configuration, allowing for flexible and extensible environment setups.

Robot Configuration
====================

URBAN-SIM supports multiple robot embodiments, each with its own physical parameters, control interface, and integration strategy.

The robot selection is determined via the ``robot_name`` field in the environment configuration, and dynamically loads the corresponding config modules.

Supported Robots
----------------

1. **COCO** (wheeled base)

   - **Config**: ``COCO_CFG`` from ``urbansim.primitives.robot.coco``
   - **Action space**: ``COCOVelocityActionsCfg``
   - **Environment modifier**: ``COCONavModifyEnv``
   - **Default height**: `z = 0.4`

2. **Unitree Go2** (quadruped robot)

   - **Config**: ``UNITREE_GO2_CFG`` from ``urbansim.primitives.robot.unitree_go2``
   - **Action space**: ``GO2NavActionsCfg``
   - **Environment modifier**: ``GO2NavModifyEnv``
   - **Default height**: `z = 0.3`

3. **Unitree G1** (humanoid / bipedal)

   - **Config**: ``G1_MINIMAL_CFG`` from ``urbansim.primitives.robot.unitree_g1``
   - **Action space**: ``G1NavActionsCfg``
   - **Environment modifier**: ``G1NavModifyEnv``
   - **Default height**: `z = 0.74`

Dynamic Initialization
----------------------

The configuration system selects robot-specific components based on name:

.. code-block:: python

   if robot_name.lower() == "unitree_go2":
       from urbansim.primitives.robot.unitree_go2 import UNITREE_GO2_CFG, GO2NavActionsCfg, GO2NavModifyEnv
       robot_cfg = UNITREE_GO2_CFG
       action_cfg = GO2NavActionsCfg
       modify_env_fn = GO2NavModifyEnv

   # Set robot spawn position
   robot_cfg.init_state.pos = env_config["Robot"].get("init_position", default_xyz)

Action Configuration
---------------------

Each robot defines its own ``action_cfg`` class, determining:

- Control mode (e.g., velocity commands, joint torques)
- Action dimension and limits
- Mapping to simulation API

These configurations are injected into the full environment config (e.g., ``EnvCfg``) to ensure proper wiring during instantiation.

