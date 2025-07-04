.. _code-structure-learning:

Learning Module
===============

The `urbansim/learning` directory defines the **training and inference logic** for reinforcement learning (RL) in URBAN-SIM. It builds on top of the modular environment and policy definitions and supports distributed training, curriculum learning, and video logging.

Training Entrypoint
--------------------

The primary script for training is:

- **train.py**
  This script configures the RL environment, loads robot/scenario/action/reward definitions, and launches training.

  Supported features:

  - Compatible with both **rsl-rl** and **rl-games** training frameworks
  - Launches Isaac Sim and sets up GPU configuration, headless mode, camera views
  - Loads training hyperparameters from YAML configuration files
  - Supports multi-GPU training (DDP) and curriculum learning
  - Records videos, reward curves, and logs

  Example command:

  .. code-block:: bash

     python urbansim/learning/RL/train.py \
         --env configs/env_configs/navigation/coco.yaml \
         --enable_cameras \
         --headless --video

  Key Components:

  - **AppLauncher**: Used to boot Omniverse with optional camera and rendering control
  - **EnvCfg**: Assembles all config classes for scene, robot, reward, observations, etc.
  - **modify_env_fn**: Customizes the environment for each robot-task-setting
  - **gym.register**: Registers the environment using Gymnasium API

  Logging:

  - TensorBoard-compatible logs
  - Videos 
  - YAML/Pickle configuration dump for reproducibility

Inference and Evaluation
--------------------------

In addition to training, URBAN-SIM also supports **policy inference** and **demo playback** via the `play.py` script.

- **play.py**
  A standalone script to run trained policies in simulation environments.  
  It supports rendering, video recording, and evaluating performance metrics such as success rate or reward.

  Usage:

  .. code-block:: bash

     python urbansim/learning/RL/play.py \
         --env configs/env_configs/navigation/coco.yaml \
         --checkpoint ./assets/ckpts/navigation/coco_static.pth \
         --enable_cameras \
         --num_envs 1

  Features:

  - Loads a pretrained policy checkpoint and runs it in the selected environment.
  - Supports both visualizing and **headless** mode.
  - Compatible with curriculum and custom scene generation logic.
  - Records videos.
