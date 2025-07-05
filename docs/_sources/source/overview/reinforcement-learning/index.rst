Reinforcement Learning in URBAN-SIM
===================================

URBAN-SIM provides a modular and extensible framework for reinforcement learning (RL) in complex urban environments.  
It is built on top of Isaac Lab's `ManagerBasedRLEnv` architecture and supports multi-environment parallel simulation, curriculum learning, and rich observations/actions.

This section introduces the major configurable components of the RL environment pipeline.

.. toctree::
   :maxdepth: 1
   :caption: RL Environment Components
   
   scene
   action
   observation
   reward
   reset
   curriculum
   event

Component Overview
-------------------

- **Scene Binding**  
  RL environments are bound to a ``SceneCfg`` that defines the world layout, agent configuration, and asset loading.  
  See: :doc:`scene`

- **Actions**  
  URBAN-SIM supports multiple action interfaces, including velocity commands for wheeled and legged robots.  
  See: :doc:`action`

- **Observations**  
  Rich multimodal observations are available (e.g., RGB, depth, lidar, robot state).  
  See: :doc:`observation`

- **Rewards**  
  Task-specific rewards are defined via modular reward terms for navigation, collision avoidance, etc.  
  See: :doc:`reward`

- **Reset Conditions**  
  Environments reset based on terminal conditions like collisions, goal reached, or episode timeout.  
  See: :doc:`reset`

- **Curriculum**  
  Training difficulty can be gradually increased using a curriculum manager.  
  See: :doc:`curriculum`

- **Events**  
  Optional simulation events (e.g., trigger zone entered, pedestrian spawn) that can influence rewards or resets.  
  See: :doc:`event`

Usage Tip
----------

Each component can be configured independently via the central ``EnvCfg`` class.  
You can also subclass individual configs to customize robot interfaces, rewards, observations, or scene logic.

Example and Target Result
-----------------------------

We have provided  several example environments that demonstrate the use of these components in practice.
You can use the command:

.. code-block:: bash

   python urbansim/learning/RL/train.py --env configs/env_configs/navigation/coco.yaml --enable_cameras --num_envs 256 --headless --video

You will get the target training curves saved in TensorBoard-compatible format, like

.. image:: ../../../assets/reward.png
   :alt: RL Training Curve
   :width: 100%
   :align: center

with additional details of each reward term, terminations, etc.

.. image:: ../../../assets/detail.png
   :alt: RL Training Curve Detail
   :width: 100%
   :align: center

as well as videos during training saved in the ``logged_videos`` directory.
