Assets in URBAN-SIM
=====================

URBAN-SIM relies on a set of standardized assets to build scenes, configure robots, and define environment dynamics.  
These assets are typically downloaded from HuggingFace on first run or via our provided download scripts.

Asset Structure
-----------------

The asset directory consists of the following subfolders:

.. code-block::

   assets/
   ├── adj_parameter_folder/
   ├── asset_config.yaml
   ├── ckpts/
   ├── materials/
   ├── objects/
   ├── ped_actions/
   ├── pedestrians/
   ├── peds/
   ├── robots/
   └── usds/

Descriptions and Sources
-------------------------

- **adj_parameter_folder/**  
  Stores parameters like scla, bounding box of objects.  

- **asset_config.yaml**  
  Central configuration file listing available assets and their metadata.  

- **ckpts/**  
  Contains pretrained model checkpoints used for robot navigation or locomotion.  
  → *From official training runs.*

- **materials/**  
  Defines surface materials.  
  → *From Isaac Lab extensions.*

- **objects/**  
  Static environment objects like walls, crates, traffic cones, and trees.  

- **ped_actions/**  
  Low-level motions for pedestrians.  

- **pedestrians/**  
  Meshes for pedestrians in .gltf format.  
  → *Sourced from Synbody.*

- **peds/**  
  Meshes for pedestrians in .usd format.  
  → *Sourced from Synbody.*

- **robots/**  
  URDF or USD descriptions of robots (e.g., COCO).  

- **usds/**  
  All environment assets in `.usd`, `.usda`, or `.usdc` format.  
  → *Includes buildings, roads, maps, and backgrounds... Converted from glTF or Blender exports.*

Asset Download
-----------------

You can use the following command to download all required assets:

.. code-block:: bash

   python scripts/download_assets.py

Assets will be downloaded and extracted to the appropriate directory (`./assets/` by default).

Asset Conversion
------------------

On first run, glTF/GLB assets will be converted to USD via Isaac Sim tools.  
This step is required before simulation if only `.gltf` or `.glb` files are present.

