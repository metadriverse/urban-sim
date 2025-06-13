# Towards Autonomous Micromobility through Scalable Urban Simulation

[![Static Badge](https://img.shields.io/badge/URBANSIM-arxiv-blue)](https://arxiv.org/pdf/2505.00690.pdf)
[![GitHub license](https://img.shields.io/github/license/metadriverse/urban-sim)](https://github.com/metadriverse/urban-sim/blob/main/LICENSE.txt)
[![GitHub contributors](https://img.shields.io/github/contributors/metadriverse/urban-sim)](https://github.com/metadriverse/urban-sim/graphs/contributors)

<p align="center">
  <img src="documentation/assets/teaser.gif" alt="Urban-Sim Teaser"  width="640">
</p>

<div style="text-align: center; width:100%; margin: 0 auto; display: inline-block">
<strong>
[
<a href="https://metadriverse.github.io/urban-sim/">Website</a>
|
<a href="https://arxiv.org/pdf/2505.00690.pdf">Paper</a>
|
<a href="https://metadriverse.github.io/">Relevant Projects</a>
]
</strong>
</div>

## Latest Updates
- [Jun/13/2025] **v0.0.1**: The first official release of URBAN-SIM.

## Table of Contents
TODO

## 🛠 Getting Started

### Hardware Recommendations

To ensure the best experience with **URBAN-SIM**, please review the following hardware guidelines:

- **Recommended Hardware**:  
  - **OS**: Ubuntu 22.04; 24.04.
  - **GPU**: Nvidia GPU with at least **16GB RAM** and **12GB VRAM**.
    - Tested GPUs: **Nvidia RTX-4080, RTX-4090, L40S**.   
  - **Storage**: Minimum of **20GB free space**.  

### Installation
```bash
# Clone the repository
git clone -b main --depth 1 https://github.com/metadriverse/urban-sim.git
cd urban-sim

# Install Isaacsim 4.5 by following: https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/install_workstation.html
# Download official file from: 
# https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone%404.5.0-rc.36%2Brelease.19112.f59b3005.gl.linux-x86_64.release.zip
mkdir ${HOME}/isaacsim
cd ~/Downloads
unzip "isaac-sim-standalone@4.5.0-rc.36+release.19112.f59b3005.gl.linux-x86_64.release.zip" -d ${HOME}/isaacsim
cd ~/urban-sim
ln -s ${HOME}/isaacsim ./_isaac_sim
# Isaac Sim root directory
export ISAACSIM_PATH="${HOME}/isaacsim"
# Isaac Sim python executable
export ISAACSIM_PYTHON_EXE="${ISAACSIM_PATH}/python.sh"
# note: you can pass the argument "--help" to see all arguments possible.
${ISAACSIM_PATH}/isaac-sim.sh

# Create a new conda environment and install required libraries
bash urbansim.sh -c [env_name]  # The default name is "urbansim"
conda activate urbansim         # Or use your specified env_name
bash urbansim.sh -i             # Install dependencies and initialize

# Download the required assets
python scripts/tools/collectors/collect_asset.py

# Convert .glb files to .usd if loading issues occur
python scripts/tools/converters/convert_asset.py
```

It should be noted that you should install several dependencies including ```isaacsim, cmake, make, gcc``` on your system before installing urbansim.

## 🏃‍♂️ Simulation Environment

We provide examples to demonstrate features and basic usages of URBAN-SIM after the local installation.
### Synchronous Navigation Environment
In synchronous navigation environments, spawned 

### Asynchronous Navigation Environment

In a point navigation environment, there will be only static objects besides the ego agent in the scenario.

## 🚀 Model Training

## 📖 Questions and Support

## 📌 TODOs


## 💘 Acknowledgement
The simulator can not be built without the help from Panda3D community and the following open-sourced projects:
- Omniverse: https://www.nvidia.com/en-us/omniverse/
- IsaacSim: https://developer.nvidia.com/isaac/sim
- IsaacLab: https://github.com/isaac-sim/IsaacLab
- Objaverse: https://github.com/allenai/objaverse-xl
- OmniObject3D: https://github.com/omniobject3d/OmniObject3D
- Synbody: https://github.com/SynBody/SynBody
- BEDLAM: https://github.com/pixelite1201/BEDLAM
- ORCA: https://gamma.cs.unc.edu/ORCA/
- panda3d-simplepbr: https://github.com/Moguri/panda3d-simplepbr
- panda3d-gltf: https://github.com/Moguri/panda3d-gltf
- RenderPipeline (RP): https://github.com/tobspr/RenderPipeline
- Water effect for RP: https://github.com/kergalym/RenderPipeline 
- procedural_panda3d_model_primitives: https://github.com/Epihaius/procedural_panda3d_model_primitives
- DiamondSquare for terrain generation: https://github.com/buckinha/DiamondSquare
- KITSUNETSUKI-Asset-Tools: https://github.com/kitsune-ONE-team/KITSUNETSUKI-Asset-Tools

## 📎 Citation

If you find URBAN-SIM helpful for your research, please cite the following BibTeX entry.

```latex
@ inproceedings{wu2025urbansim,
title={Towards Autonomous Micromobility through Scalable Urban Simulation},
author={Wu, Wayne and He, Honglin and Zhang, Chaoyuan and He, Jack and Zhao, Seth Z. and Gong, Ran and Li, Quanyi and Zhou, Bolei},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2025}
}
```
