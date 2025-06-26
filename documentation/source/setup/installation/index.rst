.. _urbansim-installation-root:

Local Installation
==================

.. image:: https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg
   :target: https://docs.isaacsim.omniverse.nvidia.com/latest/index.html
   :alt: IsaacSim 4.5.0

.. image:: https://img.shields.io/badge/IsaacLab-2.0.1-silver.svg
   :target: https://isaac-sim.github.io/IsaacLab/main/index.html
   :alt: IsaacSim 4.5.0

.. image:: https://img.shields.io/badge/python-3.10-blue.svg
   :target: https://www.python.org/downloads/release/python-31013/
   :alt: Python 3.10

.. image:: https://img.shields.io/badge/platform-linux--64-orange.svg
   :target: https://releases.ubuntu.com/20.04/
   :alt: Ubuntu 20.04

.. note::

    We recommend system requirements with at least 32GB RAM and 16GB VRAM for URBAN-SIM.
    For workflows with rendering enabled, additional VRAM may be required.
    For the full list of system requirements for Isaac Sim, please refer to the
    `Isaac Sim system requirements <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html#system-requirements>`_.

URBAN-SIM is built on top of the Isaac ecosystem. Therefore, it is required to first install Isaac Sim
before using URBAN-SIM. We highly recommend installing Isaac Sim and our project 
through binary download/source file to use the latest features and improvements.

For users getting started with URBAN-SIM, we recommend installing URBAN-SIM by cloning the repo.


.. toctree::
    :maxdepth: 2

    Binary installation (recommended) <binaries_installation>
    Asset caching <asset_caching>
    Verifying the installation <verifying_installation>
