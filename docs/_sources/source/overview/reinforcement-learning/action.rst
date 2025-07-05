Action Space
=================

URBAN-SIM supports multiple types of action spaces depending on the robot embodiment.  
Each robot defines its own control interface by providing a corresponding action configuration class.  
This configuration determines the **action dimension**, **control type**, and **scaling behavior** used during RL training.

Taking **COCO** (a wheeled robot) as an example:

The action space is defined in the ``urbansim/primitives/robot/coco.py``.

COCO Action Binding
--------------------

When `robot_name = "coco"`, the following modules are loaded:

.. code-block:: python

   from urbansim.primitives.robot.coco import COCOVelocityActionsCfg
   action_cfg = COCOVelocityActionsCfg

The class `COCOVelocityActionsCfg` defines a **continuous 2D velocity command** space, consisting of:

- **linear_velocity** (forward/backward)
- **angular_velocity** (rotation rate)

It is used in the full environment config:

.. code-block:: python

   @configclass
   class EnvCfg(ManagerBasedRLEnvCfg):
       ...
       actions = COCOVelocityActionsCfg()

More specifically, the action space is defined as:

.. code-block:: python

   class ClassicalCarAction(ActionTerm):
      r"""Pre-trained policy action term.

      This action term infers a pre-trained policy and applies the corresponding low-level actions to the robot.
      The raw actions correspond to the commands for the pre-trained policy.

      """

      cfg: ClassicalCarActionCfg
      """The configuration of the action term."""

      def __init__(self, cfg: ClassicalCarActionCfg, env: ManagerBasedRLEnv) -> None:
          # initialize the action term
          super().__init__(cfg, env)

          self.robot: Articulation = env.scene[cfg.asset_name]
          self._counter = 0
          self.last_wheel_angle = torch.zeros(self.num_envs, 1, device=self.device)

          self.axle_names = ["base_to_front_axle_joint"]
          self.wheel_names = ["front_left_wheel_joint","front_right_wheel_joint", "rear_left_wheel_joint", "rear_right_wheel_joint"]
          self.shock_names = [".*shock_joint"]
          self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)

          # prepare low level actions
          self.acceleration_action: JointVelocityAction = JointVelocityAction(JointVelocityActionCfg(asset_name="robot", joint_names=[".*_wheel_joint"], scale=10.0, use_default_offset=False), env)
          self.steering_action: JointPositionAction = JointPositionAction(JointPositionActionCfg(asset_name="robot", joint_names=self.axle_names, scale=1., use_default_offset=True), env)

      """
      Properties.
      """

      @property
      def action_dim(self) -> int:
          return 2

      @property
      def raw_actions(self) -> torch.Tensor:
          return self._raw_actions

      @property
      def processed_actions(self) -> torch.Tensor:
          return self.raw_actions
      """
      Operations.
      """

      def process_actions(self, actions: torch.Tensor):
          self._raw_actions[:] = actions


      def apply_actions(self):
          if self._counter % ACTION_INTERVAL == 0:
              max_wheel_v = 4.
              wheel_base = 1.5
              radius_rear = 0.3
              max_ang = 40 * torch.pi / 180
              velocity = self.raw_actions[..., :1].clamp(0.0, max_wheel_v) / radius_rear 
              angular = self.raw_actions[..., 1:2].clamp(-max_ang, max_ang)
              angular[angular.abs() < 0.05] = torch.zeros_like(angular[angular.abs() < 0.05])
              R = wheel_base / torch.tan(angular)
              left_wheel_angle = torch.arctan(wheel_base / (R - 0.5 * 1.8))
              right_wheel_angle = torch.arctan(wheel_base / (R + 0.5 * 1.8))

          
              self.steering_action.process_actions(((right_wheel_angle + left_wheel_angle) / 2.))
              self.acceleration_action.process_actions(torch.cat([velocity, velocity, velocity, velocity], dim=1))
          
          self.steering_action.apply_actions()
          self.acceleration_action.apply_actions()
          self._counter += 1

If you want to build your own action term, you can subclass `ActionTerm` and implement the required methods.

Action Application
------------------

In each simulation step, the RL policy outputs a 2D vector ``[v_lin, v_ang]``, which is interpreted as:

- ``v_lin``: forward speed (e.g., mapped to ``base_command.v_target``)
- ``v_ang``: yaw rate (e.g., mapped to ``base_command.w_target``)

This command is then used to drive the robot within the simulation loop.

Other Robot Types
------------------

Other robot embodiments use different action configs:

- **Unitree Go2**: `GO2NavActionsCfg` → joint velocity commands or high-level velocity
- **Unitree G1**: `G1NavActionsCfg` → biped locomotion controls

These robots take pretrained neural networks as actions, which are applied to the robot's joints or high-level velocity commands.
More details can be found in the respective robot configuration files under `urbansim/primitives/robot/` and we will provide pretrained model weights.
