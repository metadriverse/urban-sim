Asset Caching
=============

Assets used in URBAN-SIM should be downloaded from HuggingFace.
We provide utility scripts to help automate this process.

.. code:: bash

   python scripts/tools/collectors/collect_asset.py

Note that for the first run, assets ending in `.gltf` or `.glb` need to be converted to `.usd `format for use in Isaac Sim.

We provide a script to automate this conversion:

.. code:: bash
   
   python scripts/tools/converters/convert_asset.py

