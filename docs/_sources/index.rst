Welcome to URBAN-SIM!
=====================

.. figure:: assets/URBANSIM.png
   :width: 100%
   :alt: Async Simulation in URBAN-SIM

**URBAN-SIM** is a unified framework for robot learning in urban scenarios, 
including point navigation, obstacle avoidance, pedestrian interaction, etc. 
It is designed to support a wide range of learning paradigms such as reinforcement learning, imitation learning. 
Built on top of `NVIDIA Isaac Sim`_ and `NVIDIA Isaac Lab`_, URBAN-SIM enables high-fidelity photo-realistic rendering,
efficient, asynchronous simulation in large-scale, dynamic environments.

URBAN-SIM provides a variety of environments featuring diverse robots and scenario generation pipelines. We are 
actively expanding the set of supported scenarios to accommodate broader use cases and research needs.

Based on Omniverse, PhysX, OpenUSD provided by NVIDIA Isaac ecosystem, 
the platform is also designed so that you can add your own robots!

.. figure:: assets/teaser.gif
   :width: 100%
   :alt: Example robots


License
=======

The URBAN-SIM framework is open-sourced under the Apache 2.0 license.
Please refer to :ref:`license` for more details.

Acknowledgement
===============
If you find URBAN-SIM helpful for your research, please cite the following BibTeX entry:

.. code:: bibtex

   @inproceedings{wu2025towards,
      title={Towards autonomous micromobility through scalable urban simulation},
      author={Wu, Wayne and He, Honglin and Zhang, Chaoyuan and He, Jack and Zhao, Seth Z and Gong, Ran and Li, Quanyi and Zhou, Bolei},
      booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
      pages={27553--27563},
      year={2025}
   }

Table of Contents
=================

.. toctree::
   :maxdepth: 2
   :caption: Getting Started with URBAN-SIM

   source/setup/installation/index
   source/setup/quickstart

.. toctree::
   :maxdepth: 3
   :caption: RL Training with URBAN-SIM
   :titlesonly:
   
   source/overview/simulation
   source/overview/environments
   source/overview/configuration
   source/overview/reinforcement-learning/index

.. toctree::
   :maxdepth: 3
   :caption: Developer

   source/developer/code-structure/index

.. toctree::
   :maxdepth: 1
   :caption: References

   source/refs/assets
   source/refs/issues
   source/refs/license
   source/refs/3license

.. _NVIDIA Isaac Sim: https://docs.isaacsim.omniverse.nvidia.com/latest/index.html
.. _NVIDIA Isaac Lab: https://isaac-sim.github.io/IsaacLab/main/index.html
