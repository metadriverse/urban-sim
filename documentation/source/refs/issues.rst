Known Issues and Limitations
=============================

This section outlines known issues and current limitations in URBAN-SIM, including hardware requirements, missing dependencies, upcoming asset releases, etc.

Module Not Found Error
----------------------

If you meet  the error ``ModuleNotFoundError: No module named 'xxx'``, ensure that the conda environment is activated and use
    ``source _isaac_sim/setup_conda_env.sh`` to set up the conda environment.

CUDA & GLIBCXX Incompatibility
-------------------------------

URBAN-SIM relies on many libraries and systems like NVIDIA Isaac Sim, which requires supports of `libstdc++` for successful runtime.  
On some systems, especially with custom Anaconda installations, the following error may occur:

.. code-block:: bash

   ImportError: ... libstdc++.so.6: version `GLIBCXX_3.4.32' not found

**Solution**: Ensure your environment uses a system-level `libstdc++.so.6` with `GLIBCXX_3.4.32+`.  
You can verify this with:

.. code-block:: bash

   strings $(g++ -print-file-name=libstdc++.so.6) | grep GLIBCXX

If you can find `GLIBCXX_3.4.32` or higher in the output, your system is compatible. At this time, if you are using a custom Anaconda installation, 
you may need to install the system-level `libstdc++` package or replace the `link` of  `libstdc++.so.6` in Anaconda with the system version.

A100 GPU Compatibility
------------------------

Isaac Sim currently does **not support running on NVIDIA A100 GPUs** due to limitations in the OptiX and PhysX backends.  
Attempting to launch on an A100 may result in failure to create Vulkan/OpenGL contexts or invalid CUDA device errors.

We recommend using RTX 30xx / 40xx / 50xx consumer cards or supported workstation GPUs.

Asset Availability
-------------------

Some assets in URBAN-SIM scenarios are under construction or being curated for release.  
More specifically:

- Static objects: Static objects are currently partial; full set with more than 1.5W well-annotated assets will be released progressively.
- Pedestrians: We use pedestrian assets from `Synbody <https://synbody.github.io/>`_ project.  
  To reduce space and streamline runtime loading, only a **subset of Synbody** models is currently included in URBAN-SIM.  
  We will release the code to process and load the Synbody asset and convert them to `.usd` format in the future.
  Currently, you can convert Synbody assets to `.usd` format using the Isaacsim GUI.

Please stay tuned via our repo or project page for updates.

