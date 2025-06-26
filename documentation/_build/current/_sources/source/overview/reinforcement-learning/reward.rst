Reward
=================

The reward function in URBAN-SIM is modular and composed of multiple weighted terms.  
Each term corresponds to a specific behavior or outcome in the MDP (Markov Decision Process),  
and is defined using a `RewTerm` with an associated function, weight, and optional parameters.

Taking **COCO** (a wheeled robot) as an example:

COCO Reward Binding
--------------------

When ``robot_name = "coco"``, the following class is loaded:

.. code-block:: python

   from urbansim.primitives.robot.coco import COCORewardCfg
   rewards = COCORewardCfg()

The class `COCORewardCfg` defines the following reward terms:

.. code-block:: python

   @configclass
   class RewardsCfg:
       """Reward terms for the MDP."""

       arrived_reward = RewTerm(
           func=loc_mdp.is_terminated_term,
           weight=2000.0,
           params={"term_keys": "arrive"}
       )

       collision_penalty = RewTerm(
           func=loc_mdp.is_terminated_term,
           weight=-200.0,
           params={"term_keys": "collision"}
       )

       position_tracking = RewTerm(
           func=nav_mdp.position_command_error_tanh,
           weight=10.0,
           params={"std": 5.0, "command_name": "pose_command"}
       )

       position_tracking_fine = RewTerm(
           func=nav_mdp.position_command_error_tanh,
           weight=50.0,
           params={"std": 1.0, "command_name": "pose_command"}
       )

       moving_towards_goal = RewTerm(
           func=nav_mdp.moving_towards_goal_reward,
           weight=20.0,
           params={"command_name": "pose_command"}
       )

       target_vel_rew = RewTerm(
           func=nav_mdp.target_vel_reward,
           weight=10.0,
           params={"command_name": "pose_command"}
       )

Reward Term Descriptions
--------------------------

- **arrived_reward**  
  Provides a large positive reward when the robot successfully reaches the goal (termination condition).

- **collision_penalty**  
  Penalizes termination due to collision with objects or pedestrians.

- **position_tracking**  
  Penalizes positional error using a smooth tanh function with coarse tolerance (std=5.0).

- **position_tracking_fine**  
  Applies a finer penalty when closer to the target, encouraging precise final alignment.

- **moving_towards_goal**  
  Encourages movement in the direction of the goal based on heading alignment.

- **target_vel_rew**  
  Provides a dense reward for matching commanded target velocity.

Each `RewTerm` is evaluated during environment stepping and combined (via weighted sum) to produce the total reward for each timestep.

Customizing Rewards
---------------------

To customize rewards:

1. Subclass `RewardCfg` or override terms in YAML.
2. Adjust the weights or add new `RewTerm` entries.
3. Define custom reward functions in `nav_mdp` or `loc_mdp`.

You can toggle or ablate specific terms (e.g., disable `collision_penalty`) by setting their weight to 0 or removing them from the config.