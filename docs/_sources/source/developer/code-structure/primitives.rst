Primitives Module
=================

The ``urbansim.primitives`` package contains the foundational configuration scripts used to build training environments for both **navigation** and **locomotion** tasks. It is divided into three main submodules:

- ``navigation``: configuration and MDP logic for point-to-point navigation environments.
- ``locomotion``: terrain-based locomotion tasks for legged robots.
- ``robot``: robot-specific configuration files (URDF paths, controllers, action spaces, etc.).

While ``locomotion`` follows a similar configuration pattern, we focus on ``navigation`` in this section to illustrate how procedural generation and task logic are composed.

Navigation: random_env_cfg.py
-----------------------------

We take ``random_env_cfg.py`` under ``urbansim.primitives.navigation`` as the primary example to explain the primitives module. This file defines the full environment configuration for random, procedurally generated navigation scenarios.

Scene Configuration (SceneCfg)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``SceneCfg`` class extends ``UrbanSceneCfg`` and defines the simulation environment, including:

- **Procedural Generation (pg_config)**: This dictionary controls scenario diversity — including the number of obstacles, pedestrians, environment size, and boundary setup.
- **Terrain Importing**: Includes default ground, obstacle terrains, walkable and non-walkable tiles. Each region is assigned random visual materials to simulate diverse textures.
- **Lighting**: A random HDR dome light is chosen at runtime from a curated list.
- **Sensors**:
  - *RGB-D Camera*: Positioned on the robot base, provides `rgb` and `distance_to_camera` for learning or logging.
  - *Height Scanner*: Ray-based terrain scanner mounted above the robot, useful for terrain-aware policies.
  - *Contact Sensors*: Detects collisions for reward and termination purposes.

MDP Components
~~~~~~~~~~~~~~

The random navigation environment is designed around modular MDP configuration classes:

- **CommandsCfg**: Defines goal pose sampling in a bounded region using `UniformPose2dCommandCfg`.
- **ObservationsCfg**:
  - `policy`: Receives normalized pose commands.
  - `sensor`: Receives processed RGB image from the front camera.
- **ActionsCfg**: Empty placeholder, to be overridden by robot-specific action spaces.
- **RewardsCfg**:
  - `arrived_reward`: High reward for reaching the goal.
  - `position_tracking`: Encourages trajectory following.
  - `collision_penalty`: Strong penalty for contact.
  - `moving_towards_goal`, `target_vel_rew`: Additional shaping terms.
- **TerminationsCfg**:
  - `collision`: Checks illegal contact via contact sensor.
  - `arrive`: Stops when the robot is close enough to the goal.
  - `time_out`: Ends episodes after a max step count.
- **EventCfg**: Defines how to reset the robot’s base pose and velocity on episode start.
- **CurriculumCfg**: Implements curriculum learning, such as `increase_moving_distance`, which gradually expands the goal sampling region to increase difficulty.

Integration
~~~~~~~~~~~

The `random_env_cfg.py` is used in `train.py` for training setup, and is registered into Gym via:

.. code-block:: python

   gym.register(
       id="URBANSIM-navigation-coco-dynamic",
       entry_point="urbansim.envs:AbstractRLEnv",
       kwargs={
           "env_cfg_entry_point": EnvCfg,
           ...
       },
   )

This enables training in both RSL-RL and RL-Games pipelines, with full compatibility for curriculum, procedural variation, and video recording.

Takeaway
~~~~~~~~

The primitives module enables decoupling between robot specifications, task design, and simulation setup. By combining different configurations (e.g., new robot + random terrain + new reward), users can flexibly prototype new scenarios with minimal code duplication.

Robot
-----------

We take the ``coco.py`` file under ``urbansim.primitives.robot`` as an example to demonstrate how a robot is defined and integrated into the simulation.

The COCO robot is a four-wheel steerable vehicle with a front axle joint and delayed PD control. It supports **two types of action spaces**:

1. **Velocity-based commands** (linear velocity + angular steering)
2. **Waypoint-based commands** (local Cartesian goal points)

Robot Configuration
~~~~~~~~~~~~~~~~~~~

The robot is instantiated with an ``ArticulationCfg`` class, specifying:

- **URDF (USD) Path**: Under ``assets/robots/coco_one/coco_one.usd``
- **Initial State**: Position + joint position/velocity initialization
- **Actuators**:
  - *Wheels*: PD-velocity control with optional actuation delay
  - *Front axle*: DC motor controlling the steering angle
  - *Shocks*: Passive (zero stiffness/damping) implicit actuator

.. code-block:: python

   COCO_CFG = ArticulationCfg(
       spawn=sim_utils.UsdFileCfg(usd_path="assets/robots/coco_one/coco_one.usd", ...),
       init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., 0.3), ...),
       actuators={
           "wheels": DelayedPDActuatorCfg(...),
           "axle": DCMotorCfg(...),
           "shock": ImplicitActuatorCfg(...),
       },
   )

Action Space 1: Velocity + Angular Commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `ClassicalCarAction` class defines how a `(v, ω)` command is transformed into wheel velocity and steering angle.

- Steering is calculated via Ackermann geometry (wheelbase, turning radius).
- Acceleration is applied to all 4 wheels.
- Actions are updated every ``ACTION_INTERVAL`` steps to emulate delayed control.

.. code-block:: python

   class ClassicalCarAction(ActionTerm):
       def apply_actions(self):
           velocity = ...
           angular = ...
           self.steering_action.process_actions(...)
           self.acceleration_action.process_actions(...)

- Config class: ``ClassicalCarActionCfg``
- MDP entry point: ``COCOVelocityActionsCfg``

Action Space 2: Waypoint-based Commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `ClassicalCarWaypointAction` transforms a local 2D waypoint into velocity and steering commands using a PD controller.

- `(dx, dy)` is interpreted as a short-range goal in the robot's frame.
- PD controller infers velocity and desired yaw rate, then translates that into steering angle.

.. code-block:: python

   class ClassicalCarWaypointAction(ActionTerm):
       def process_actions(self, actions):
           v, w = pd_controller(actions)
           ...
           self.steering_action.process_actions(...)
           self.acceleration_action.process_actions(...)

- Config class: ``ClassicalCarWaypointActionCfg``
- MDP entry point: ``COCOWaypointActionsCfg``

Integration Hook
~~~~~~~~~~~~~~~~

The `COCONavModifyEnv` function is provided to modify the scene and termination logic to match the COCO robot configuration.

- Adjusts sensor locations
- Overrides termination collision parameters

.. code-block:: python

   def COCONavModifyEnv(env):
       env.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/base_link/front_cam"
       env.terminations.collision.params["sensor_cfg"].body_names = "body_link"
       return env


While the `COCO` robot is the default example used in our tasks, we understand that many users may work with other platforms.  
To facilitate this, we also provide two additional robot configurations out of the box:

- `go2.py`: Unitree Go2 robot configuration (quadruped locomotion)
- `g1.py`: Unitree G1 robot configuration (humanoid locomotion)
- `anymal_c.py`: ANYmal C robot configuration (quadruped locomotion)

These files follow the same structure and conventions as `coco.py`, making it easy to reuse task logic across different platforms.

If you wish to add support for a new robot:

1. **Create a new config file** under `urbansim/primitives/robot/`, e.g., `myrobot.py`.
2. **Define its articulation config**, including URDF path, actuators, and initial state.
3. **Implement or reuse** an action space class that suits your control interface (e.g., velocity-based, waypoint-based, joint command).
4. **(Optional)** Provide a `ModifyEnv` function to adjust sensor placements, terminations, or robot-specific resets.
5. **Register** your robot into the environment configuration (e.g., `random_env_cfg.py`) or pass it dynamically at runtime.


Takeaway
~~~~~~~~

The robot module is **robot-centric**, separating hardware specs (URDF, actuators, joint names) from policy logic.  
This decoupling allows different robots to plug into the same training scene and share task logic (e.g., rewards, observations).
