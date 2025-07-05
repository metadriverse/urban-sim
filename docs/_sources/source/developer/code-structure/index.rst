Code Structure in URBAN-SIM
===========================

URBAN-SIM is organized to support extensibility, debugging, and algorithmic innovation.  
This section introduces core components of the codebase and their intended roles in the simulation and learning pipeline.

.. toctree::
   :maxdepth: 1
   :caption: Code Modules

   envs
   learning
   primitives
   scene

Module Overview
----------------

- **envs**  
  Defines environment logic, including abstract base classes and wrappers for RL training.  
  See: :doc:`envs`

- **learning**  
  Contains RL algorithm implementations and training entry points.  
  See: :doc:`learning`

- **primitives**  
  Encapsulates reusable low-level capabilities such as locomotion, navigation heuristics, and robot configurations.  
  See: :doc:`primitives`

- **scene**  
  Manages scene layout, asset placement, and scenario configuration for environments.  
  See: :doc:`scene`

- **utils**  
  General utility functions and wrappers used across modules, such as logging, parsing, and random seeds.  

Developer Notes
----------------

The project follows a component-based architecture, where environments, robot actions, assets, and reward logic are independently modularized.  
Developers can subclass and register custom components for rapid experimentation.

New modules can be added under `urbansim/` and integrated via entry-point registries.
