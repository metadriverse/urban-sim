.. _urbansim-quickstart:

Quickstart Guide
=======================


This guide includes running RL, finding environments, creating new projects, and some basic usage of URBAN-SIM.

To get started, we will first install URBAN-SIM following the previsou section.
After that, we can launch a training script to train a reinforcement learning agent in a URBAN-SIM environment.

Launch Training
-------------------

The various envs of URBAN-SIM are accessed through their corresponding ``train.py`` scripts located in the ``urbansim/learning/RL`` directory.
Invoking these scripts will require a **Task Configuration** to the gymnasium API. For example,

.. code-block:: bash

    python urbansim/learning/RL/train.py --env configs/env_configs/navigation/coco.yaml --enable_cameras --num_envs 32

This will train the coco delivery robot to navigate.  Note specifically the ``--num_envs`` option and the ``--headless`` flag,
both of which can be useful when trying to develop and debug a new environment. 

In the script, the environment is registered with the gymnasium API, and the training script is launched with the
``task`` argument in Isaac Sim.  The task is a string that identifies the environment and the robot used in the training. 

.. code-block:: python

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
    args_cli.task = f"URBANSIM-{task_name}-{robot_name}-{setting_name}"

This is the function that actually registers an environment for future use.  Notice that the ``entry_point`` is literally
just the python module path to the environment definition, you can change this to point to any environment class you want to use.

Also, the learning framework is specified by the ``rsl_rl_cfg_entry_point`` and the ``rl_games_cfg_entry_point``.  Notably, if you want to use
rl games instead of the RSL RL framework, you can change the ``rsl_rl_cfg_entry_point`` to point to the rl games runner configuration file that you created, such as 
``configs/rl_configs/navigation/coco/static_train.yaml`` used in  the example above. 
This is the configuration file that defines the training parameters, such as the number of steps, the learning rate, etc.

Configurations
---------------

Regardless of what you are going to be doing with URBAN-SIM, you will need to deal with **Configurations**. Configurations
can all be identified by the inclusion of the ``@configclass`` decorator above their class definition and the lack of an ``__init__`` function. For example,

.. code-block:: python

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

Configurations provide a direct path to any variable in the configuration hierarchy, making it easy
to modify anything "configured" by the environment at launch time.
