Simulation
=======================

URBAN-SIM supports both **synchronous** and **asynchronous** multi-environment simulations. These two modes differ in how environments are spawned.

⏱️ Synchronous Simulation
-------------------------

In synchronous mode, all environments are advanced **together in lock-step**. They share the **same scene layout**, including map structure, object placement, and pedestrian routes. This setting is ideal for:

- Controlled debugging
- Frame-aligned evaluation
- Consistent benchmarking across environments

.. code-block:: bash

   python urbansim/envs/separate_envs/random_env.py --num_envs 16 --scenario_type {clean,static,dynamic}

**Example Scenario Types:**

- ``clean``: Empty environment without obstacles
- ``static``: With static obstacles
- ``dynamic``: With both static and dynamic (e.g., pedestrians) elements

⚡ Asynchronous Simulation
--------------------------

In asynchronous mode, each environment is generated **independently**, allowing for diverse scenario layouts, object instances, appearances, etc. This is especially useful for improving robustness, or simulating real-world noise.

.. code-block:: bash

   python urbansim/envs/separate_envs/random_env.py --num_envs 16 --scenario_type {clean,static,dynamic} --use_async

**Advantages:**

- Enables greater variation in observation-action loops
- Closer to real-world multi-agent or multi-robot setups
- Can increase training speed under high environment counts

🔄 Semantic Differences: Synchronous vs. Asynchronous Scenarios
---------------------------------------------------------------

In **synchronous simulation**, all environments share the **same scene layout**, obstacle placement, and map structure. This is ideal for evaluation and debugging where controlled consistency is needed.

In **asynchronous simulation**, each environment instance is procedurally varied. This allows:

- Diverse layouts and building placements
- Varying pedestrian paths and densities
- Scene-level randomness in lighting or asset combinations

This diversity helps improve the generalization and robustness of navigation policies.

.. note::

   ✅ Asynchronous simulation is ideal for large-scale training across diverse environments.

🖼️ Scenario Visualizations
--------------------------

Different scenario types can be combined with both simulation modes. For example:

- ``clean`` (empty world)
- ``static`` (with immovable structures)
- ``dynamic`` (with moving pedestrians)

.. list-table::
   :widths: 33 33 33
   :header-rows: 0

   * - .. image:: ../../assets/sync_clean.gif
         :width: 100%
     - .. image:: ../../assets/sync_static.gif
         :width: 100%
     - .. image:: ../../assets/sync_dynamic.gif
         :width: 100%

   * - Sync - Clean
     - Sync - Static
     - Sync - Dynamic

.. list-table::
   :widths: 33 33 33
   :header-rows: 0

   * - .. image:: ../../assets/async_clean.gif
         :width: 100%
     - .. image:: ../../assets/async_static.gif
         :width: 100%
     - .. image:: ../../assets/async_dynamic.gif
         :width: 100%

   * - Async - Clean
     - Async - Static
     - Async - Dynamic