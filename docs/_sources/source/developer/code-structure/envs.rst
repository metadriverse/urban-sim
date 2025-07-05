Environment: Spawning and simulation for learning
=============

The `envs` module contains the foundational environment classes for simulation and reinforcement learning within URBAN-SIM.  
These classes wrap around the simulation backend and manager-based modular design from IsaacLab, enabling scalable, customizable RL environments.

Overview
--------


Core Files
~~~~~~~~~~

- `abstract_env.py`  
  Defines the `AbstractEnv` class, a base class that wraps around `ManagerBasedEnv`. It handles:
  - Simulation setup and stepping
  - Scene generation (`UrbanScene`)
  - Manager initialization
  - Environment reset and rendering
  This class is task-agnostic and can be used for both learning-based and classical control tasks.

- `abstract_rl_env.py`  
  Defines the `AbstractRLEnv` class, which inherits from `ManagerBasedRLEnv`.  
  It provides an extended structure tailored for reinforcement learning, including:
  - Full RL-compatible `step()` method: action → sim → reward → reset → obs
  - Support for batch environments (vectorized execution)
  - Integration with Gymnasium APIs
  - Curriculum-aware resets
  - Automatic event triggering and metric logging

  Key components handled include:
  - `CommandManager` — generates goals, targets, or behavior signals
  - `RewardManager` — computes modular reward terms
  - `TerminationManager` — evaluates terminal conditions
  - `CurriculumManager` — adapts task difficulty over time
  - `ObservationManager`, `ActionManager`, `EventManager`, `RecorderManager`

- `separate_envs/`  
  Contains task-specific environment subclasses:
  
  - `predefined_env.py` — Load fixed environment configurations (e.g., pedestrian positions, objects).
  - `random_env.py` — Stochastic environment sampling for reinforcement learning.
  - `pg_env.py` — Procedural generation of scenes, typically used for inference.

The hierarchy is:

- `ManagerBasedRLEnv` (from IsaacLab)  
  - `AbstractRLEnv` (URBAN-SIM RL base class)  
    - `PGEnv`, `PredefinedEnv`, etc. (concrete tasks)

Workflow Summary
-----------------

Each RL environment implements:

1. `__init__()`  
   Sets up the simulation, scene, viewer, and all managers.

2. `step(action)`  
   Runs decimated simulation steps, applies actions, computes rewards, resets if needed, and returns batched observations and stats.

3. `render()`  
   Renders a frame (if enabled) and supports both GUI-based and headless usage.

4. `reset()`  
   Invoked automatically for terminated environments, handles curriculum, randomness, and metric tracking.
