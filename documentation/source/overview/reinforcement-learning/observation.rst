Observation Space
=================

URBAN-SIM supports configurable multimodal observation spaces tailored to different robot embodiments and tasks.  
Each environment binds an observation configuration class that defines the set of sensors, modalities, and preprocessing methods available to the policy.

The observation space is defined in the scenecfg file, for example, ``urbansim/primitives/navigation/random_env_cfg.py``.

Sensor-based Observations
----------------------------

URBAN-SIM supports a modular sensor system, where observations from various physical or virtual sensors  
(e.g., cameras, raycasters, contact sensors) can be processed and passed to RL policies.  
These sensors are defined under the robot's scene configuration and referenced by observation terms.

Taking **COCO** as an example:

Sensor Configuration
----------------------

The following sensors are typically spawned in the robot’s scene config (e.g., `SceneCfg.sensors`):

.. code-block:: python

   # Contact sensor for foot or chassis contact
   contact_forces = ContactSensorCfg(
       prim_path="{ENV_REGEX_NS}/Robot/.*",
       history_length=3,
       track_air_time=True
   )

   # RGB + depth camera
   camera = TiledCameraCfg(
       prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
       update_period=0.1,
       height=135,
       width=240,
       data_types=["rgb", "distance_to_camera"],
       spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
           intrinsic_matrix=[531., 0., 960., 0., 531., 540., 0., 0., 1.],
           width=1920,
           height=1080,
       ),
       offset=CameraCfg.OffsetCfg(
           pos=(0.51, 0.0, 0.015),
           rot=(0.5, -0.5, 0.5, -0.5),
           convention="ros"
       ),
   )

   # Height scanner using raycasting grid
   height_scanner = RayCasterCfg(
       prim_path="{ENV_REGEX_NS}/Robot/base",
       offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
       attach_yaw_only=True,
       pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
       debug_vis=True,
       mesh_prim_paths=["/World/ground"]
   )

Observation Binding
--------------------

Once sensors are defined, their data is passed to the policy via an `ObservationsCfg` class:

.. code-block:: python

   @configclass
   class ObservationsCfg:
       @configclass
       class PolicyCfg(ObsGroup):
           pose_command = ObsTerm(
               func=loc_mdp.advanced_generated_commands,
               params={
                   "command_name": "pose_command",
                   "max_dim": 2,
                   "normalize": True
               }
           )

       @configclass
       class SensorCfg(ObsGroup):
           rgb = ObsTerm(
               func=nav_mdp.rgbd_processed,
               params={"sensor_cfg": SceneEntityCfg("camera")}
           )

       policy: PolicyCfg = PolicyCfg()
       sensor: SensorCfg = SensorCfg()

Explanation:

- `ObsGroup`: A logical grouping of observations (e.g., policy inputs vs. auxiliary sensors).
- `ObsTerm`: Binds a specific function that processes sensor data.
- `SceneEntityCfg("camera")`: Indicates this term pulls data from the named sensor defined in the scene.

Processing Chain:

.. code-block::

   Scene Sensor → Simulator → ObsTerm.func (e.g., rgbd_processed) → Policy Input Tensor

This design decouples **sensor definition** from **observation usage**, allowing different policies to reuse or remap sensor outputs flexibly.

Extending Observation Space
----------------------------

To add new sensor-based observations:

1. Define the sensor in `SceneCfg.sensors` with a unique name.
2. Use `SceneEntityCfg(<name>)` to reference it in your `ObsTerm`.
3. Provide a custom processing function in your MDP module.
