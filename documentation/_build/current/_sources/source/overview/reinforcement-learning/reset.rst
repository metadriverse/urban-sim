Termination Conditions
==============================

URBAN-SIM uses a set of modular termination conditions to determine when an episode should end.  
Each condition is implemented as a `DoneTerm`, and bound via a `TerminationsCfg` config class.

Taking **COCO** (a wheeled robot) as an example:

Termination Binding
--------------------

When ``robot_name = "coco"``, the following termination config is used:

.. code-block:: python

   @configclass
   class TerminationsCfg:
       time_out = DoneTerm(func=loc_mdp.time_out, time_out=True)

       collision = DoneTerm(
           func=nav_mdp.illegal_contact,
           time_out=False,
           params={
               "sensor_cfg": SceneEntityCfg("contact_forces", body_names="body_link"),
               "threshold": 1.0
           },
       )

       arrive = DoneTerm(
           func=nav_mdp.arrive,
           time_out=False,
           params={
               "threshold": 1.0,
               "command_name": "pose_command"
           },
       )

Termination Term Descriptions
-------------------------------

- **time_out**  
  Ends episode after a predefined number of steps.

- **collision**  
  Terminates if contact force (e.g., from walls or pedestrians) exceeds a threshold.

- **arrive**  
  Terminates when the robot reaches its goal (within distance threshold).

- *(Optional)* **out_of_region**  
  Can be enabled to terminate the episode when the robot leaves a bounded region.

Each `DoneTerm` can be used to trigger reward terms and is evaluated per timestep.  
Termination decisions also propagate into curriculum updates and training logs.
