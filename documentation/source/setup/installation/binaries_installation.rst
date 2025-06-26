.. _urbansim-binaries-installation:

Installation using Binaries
=====================================

URBAN-SIM requires Isaac Sim. This tutorial installs Isaac Sim first from binaries, then URBAN-SIM from source code.

Installing Isaac Sim
--------------------

Downloading pre-built binaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Please follow the Isaac Sim
`documentation <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_workstation.html>`__
to install the latest Isaac Sim release.

From Isaac Sim 4.5 release, Isaac Sim binaries can be `downloaded <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html#download-isaac-sim-short>`_ directly as a zip file.

To check the minimum system requirements, refer to the documentation
`here <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html>`__.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. note::

         We have tested Isaac Lab with Isaac Sim 4.5 release on Ubuntu
         22.04 LTS with NVIDIA driver 535.230 on NVIDIA 4080 Super, 4090 and L40S.

         From Isaac Sim 4.5 release, Isaac Sim binaries can be downloaded directly as a zip file.
         The below steps assume the Isaac Sim folder was unzipped to the ``${HOME}/isaacsim`` directory.

      On Linux systems, Isaac Sim directory will be named ``${HOME}/isaacsim``.

Verifying the Isaac Sim installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To avoid the overhead of finding and locating the Isaac Sim installation
directory every time, we recommend exporting the following environment
variables to your terminal for the remaining of the installation instructions:

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         # Isaac Sim root directory
         export ISAACSIM_PATH="${HOME}/isaacsim"
         # Isaac Sim python executable
         export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"


For more information on common paths, please check the Isaac Sim
`documentation <https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_faq.html#common-path-locations>`__.


-  Check that the simulator runs as expected:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            # note: you can pass the argument "--help" to see all arguments possible.
            ${ISAACSIM_PATH}/isaac-sim.sh

-  Check that the simulator runs from a standalone python script:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: bash

            # checks that python path is set correctly
            ${ISAACSIM_PYTHON_EXE} -c "print('Isaac Sim configuration is now complete.')"
            # checks that Isaac Sim can be launched from python
            ${ISAACSIM_PYTHON_EXE} ${ISAACSIM_PATH}/standalone_examples/api/isaacsim.core.api/add_cubes.py

If the simulator does not run or crashes while following the above
instructions, it means that something is incorrectly configured. To
debug and troubleshoot, please check Isaac Sim
`documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html>`__
and the
`forums <https://docs.isaacsim.omniverse.nvidia.com/latest/isaac_sim_forums.html>`__.

.. note:: 
    If you meet  the error ``ModuleNotFoundError: No module named 'xxx'``, ensure that the conda environment is activated and use
    ``source _isaac_sim/setup_conda_env.sh`` to set up the conda environment.

Installing URBAN-SIM
--------------------

Cloning URBAN-SIM
~~~~~~~~~~~~~~~~~

Clone the URBAN-SIM repository into your workspace:

.. tab-set::

   .. tab-item:: HTTPS

      .. code:: bash

         git clone -b main --depth 1 https://github.com/metadriverse/urban-sim.git

.. note::
   We provide a helper executable `urbansim.sh` that provides
   utilities to manage extensions:

   .. tab-set::
      :sync-group: os

      .. tab-item:: :icon:`fa-brands fa-linux` Linux
         :sync: linux

         .. code:: text

            ./urbansim.sh --help

            usage: urbansim.sh [-h] [-i] [-v] [-c] [-a] -- Utility to manage URBAN-SIM.

            optional arguments:
                -h, --help           Display the help content.
                -i, --install        Install the extensions inside URBAN-SIM and learning frameworks as extra dependencies.
                -v, --vscode         Generate the VSCode settings file from template.
                -c, --conda [NAME]   Create the conda environment for URBAN-SIM. Default name is 'urbansim'.
                -a, --advanced       Run the advanced command.

Creating the Isaac Sim Symbolic Link
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set up a symbolic link between the installed Isaac Sim root folder
and ``_isaac_sim`` in the Isaac Lab directory. This makes it convenient
to index the python modules and look for extensions shipped with Isaac Sim.

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         # enter the cloned repository
         cd urban-sim
         # create a symbolic link
         ln -s ${HOME}/isaacsim ./_isaac_sim
         # You can also use the absolute path instead of ${HOME}/isaacsim

Setting up the conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         bash urbansim.sh -c [env_name]  # The default name is "urbansim"

Once created, be sure to activate the environment before proceeding!

.. code:: bash

   conda activate urbansim  # or "conda activate my_env"

Once you are in the virtual environment, you can use the default python executable in your environment
by running ``python`` or ``python3``.


Installation
~~~~~~~~~~~~

.. tab-set::
   :sync-group: os

   .. tab-item:: :icon:`fa-brands fa-linux` Linux
      :sync: linux

      .. code:: bash

         ./urbansim.sh --install # or "./urbansim.sh -i"
         ./urbansim.sh --advanced # or "./urbansim.sh -a"

.. note::

   By default, the above will install all the learning frameworks. More specifically, `--install` will install the basic pipeline for random scenario generation,
   `--advanced` will install the full pipeline for scenario generation, including the learning frameworks and additional dependencies such as ORCA for pedestrian moving.
