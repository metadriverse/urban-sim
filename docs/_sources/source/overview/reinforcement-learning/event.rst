Events
=================

In URBAN-SIM, **events** are simulation-time triggers that modify the environment state or robot behavior  
without terminating the episode. Events can occur on reset, periodically, or under specific conditions.  
They are defined using the `EventTerm` class and grouped under an `EventCfg`.

Taking **COCO** (a wheeled robot) as an example:

COCO Event Binding
--------------------

When ``robot_name = "coco"``, the following event config is registered:

.. code-block:: python

   @configclass
   class EventCfg:
       reset_base = EventTerm(
           func=loc_mdp.reset_root_state_uniform,
           mode="reset",
           params={
               "pose_range": {
                   "x": (0.3, 0.3),
                   "y": (0.3, 0.3),
                   "yaw": (0.0, 0.0)
               },
               "velocity_range": {
                   "x": (0.0, 0.0),
                   "y": (0.0, 0.0),
                   "z": (0.0, 0.0),
                   "roll": (0.0, 0.0),
                   "pitch": (0.0, 0.0)
               },
           }
       )

This event is triggered in `reset` mode, meaning it will be applied at the beginning of each episode.

Event Term Parameters
-----------------------

- **pose_range**  
  Specifies the randomization range for the robot’s base position and yaw (heading).  
  In this case, it's a fixed spawn at ``(x=0.3, y=0.3, yaw=0.0)``.

- **velocity_range**  
  Specifies the initial linear and angular velocity ranges.  
  Setting all values to zero initializes the robot at rest.

Event Modes
------------

Each `EventTerm` has a `mode`, which determines when the event is triggered:

- `"reset"`: Applied at the start of an episode
- `"step"`: Applied during environment stepping (e.g., at intervals or conditions)
- `"manual"`: Triggered via API calls or scripted logic

Defining Custom Events
------------------------

To define your own event:

1. Create a Python function with signature:  
   ``def my_event(env: ManagerBasedRLEnv, env_ids: Sequence[int], ...)``  
2. Wrap it in an `EventTerm(func=..., mode=..., params=...)`
3. Register it in your robot-specific `EventCfg` class.

Use cases include:

- Randomizing initial state
- Injecting wind or external forces
- Resetting memory or trajectory buffers
