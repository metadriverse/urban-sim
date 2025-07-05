Curriculum Learning
====================

URBAN-SIM supports curriculum learning by gradually adjusting task difficulty during training.  
Each curriculum behavior is defined as a `CurrTerm` function and bound via `CurriculumCfg`.

A common use case is to expand the movement range or goal distance as training progresses.

Example: Increasing Goal Distance
----------------------------------

The following curriculum function is defined for COCO:

.. code-block:: python

   def increase_moving_distance(env, env_ids, command_name='pose_command', ...):
       cur_iteration = env.common_step_counter // num_steps_per_iteration
       x_left = start_x + (end_x - start_x) * (cur_iteration / total_iterations)
       env.command_manager.get_term(command_name).cfg.ranges.pos_x = (10., min(x_left, end_x))
       env.command_manager.get_term(command_name).cfg.ranges.pos_y = (0., map_region)

This function progressively expands the allowed x-axis goal region as training proceeds.

Curriculum Binding
--------------------

It is registered via:

.. code-block:: python

   @configclass
   class CurriculumCfg:
       increased_moving_distance = CurrTerm(
           func=increase_moving_distance,
           params={}
       )

This term will be called automatically during each environment step  
to update internal task parameters (e.g., goal position sampling bounds).

Design Notes
-------------

- Curriculum functions operate *outside* the RL policy loop and are called by the environment manager.
- You can define any callable that modifies task-related config values (goal ranges, difficulty level, distractor count, etc.).
- These changes take effect in the next episode reset.

