Environments
=======================

This section describes the different categories of environments available in URBAN-SIM.

Environments for RL Training
----------------------------

These environments are designed for training agents using reinforcement learning. They follow the standard Gymnasium interface and support batched simulation.

- Base class: ``AbstractRLEnv`` in ``urbansim/envs/abstract_rl_env.py``
- **Inheritance**: Subclass of Isaac Lab's ``ManagerBasedRLEnv``
- **Configuration**: Bound to a subclass of ``UrbanSceneCfg`` which defines and spawns assets, maps, agents, and sensors, etc.

To spawn such environment for training, we need to define the class as provided in ``urbansim/learning/RL/train.py``:

.. code:: 

    @configclass
    class EnvCfg(ManagerBasedRLEnvCfg):
        """Configuration for the locomotion velocity-tracking environment."""

        # Scene settings
        if pg_config is None:
            scene = scene_cfg(num_envs=env_config['Training']['num_envs'], 
                              env_spacing=env_config['Omniverse']['env_spacing'],)
        else:
            scene = scene_cfg(num_envs=env_config['Training']['num_envs'], 
                            env_spacing=env_config['Omniverse']['env_spacing'],
                            pg_config=pg_config,
                            scenario_generation_method=env_config['Omniverse'].get('scenario_generation_method', None),)
        # Basic settings
        viewer = ViewerCfg()
        observations = observation_cfg()
        actions = action_cfg()
        commands = command_cfg()
        # MDP settings
        rewards = reward_cfg()
        terminations = termination_cfg()
        events = event_cfg()
        curriculum = curriculum_cfg()

        def __post_init__(self):
            """Post-initialization to set up the environment."""
            super().__post_init__()
            # Additional setup can be done here if needed
            ...

and register it in gymnasium:

.. code::

    import gymnasium as gym
    gym.register(
        id=f"URBANSIM-{task_name}-{robot_name}-{setting_name}",
        entry_point="urbansim.envs:AbstractRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": EnvCfg,
            "rsl_rl_cfg_entry_point":  f"urbansim.primitives.robot.{robot_name}:PPORunnerCfg",
            "rl_games_cfg_entry_point": f"configs/rl_configs/{task_name}/{robot_name}/{setting_name}_train.yaml",
        },
    )

Environment Generation Pipeline
--------------------------------------------------

URBAN-SIM supports two distinct pipelines for scenario generation, each suited to different use cases in training and evaluation.

Environments with Random Object Placement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This lightweight pipeline places static props, buildings, and pedestrians at randomly sampled positions within a predefined region. It is ideal for large-scale training and introduces high scene diversity. Key characteristics include:
- Fast initialization
- Parameterized randomness (e.g., density, seed)
- Low spatial structure, suitable for robust policy training

Run with:

.. code-block:: bash

   python urbansim/envs/separate_envs/random_env.py --enable_cameras --num_envs 16 --use_async

.. image:: ../../assets/random_detail.png
   :align: center
   :width: 95%

Environments with Procedural Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The PG pipeline creates structured scenes using rule-based or programmatic layouts. It supports curriculum learning and consistent benchmarking, thanks to repeatable spatial patterns. It is especially useful for:
- Generalization tasks
- Scene logic control
- Progressive difficulty setups

Run with:

.. code-block:: bash

   python urbansim/envs/separate_envs/pg_env.py --enable_cameras --num_envs 16 --use_async

.. image:: ../../assets/pg_detail.png
   :align: center
   :width: 95%

