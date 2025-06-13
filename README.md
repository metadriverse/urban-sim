# Towards Autonomous Micromobility through Scalable Urban Simulation

[![Static Badge](https://img.shields.io/badge/URBANSIM-arxiv-blue)](https://arxiv.org/pdf/2505.00690.pdf)
[![Documentation](https://readthedocs.org/projects/metaurban-simulator/badge/?version=latest)](TODO)
[![GitHub license](https://img.shields.io/github/license/metadriverse/urban-sim)](TODO)
[![GitHub contributors](https://img.shields.io/github/contributors/metadriverse/urban-sim)](TODO)

<p align="center">
  <img src="docs/assets/output.gif" alt="Urban-Sim Teaser" width="100%">
</p>

<div style="text-align: center; width:100%; margin: 0 auto; display: inline-block">
<strong>
[
<a href="TODO">Documentation</a>
|
<a href="TODO">Website</a>
|
<a href="https://arxiv.org/pdf/2505.00690.pdf">Paper</a>
|
<a href="https://metadriverse.github.io/">Relevant Projects</a>
]
</strong>
</div>

## Latest Updates
- [TODO] **v0.1.0**: The first official release of URBAN-SIM :wrench: [[release notes]]()

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
git clone -b main --depth 1 TODO
cd TODO

# Install Isaacsim 4.5
missref

# Create a new conda environment and install required libraries
bash urbansim.sh -c [env_name]  # The default name is "urbansim"
conda activate urbansim         # Or use your specified env_name
bash urbansim.sh -i             # Install dependencies and initialize
```

It should be noted that you should install several dependencies including ```isaacsim,cmake,make,gcc``` on your system before installation, more details can be found in [FAQs](TODO).


## 🏃‍♂️ Simulation Environment

## 🤖 Run with a Pre-Trained PPO Policy

## 🚀 Model Training and Evaluation

## 📖 Questions and Support

## 💘 Acknowledgement
The simulator can not be built without the help from Panda3D community and the following open-sourced projects:
- Omniverse: TODO
- IsaacSim: TODO
- IsaacLab: TODO
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

If you find MetaUrban helpful for your research, please cite the following BibTeX entry.

```latex
@ inproceedings{wu2025urbansim,
title={Towards Autonomous Micromobility through Scalable Urban Simulation},
author={Wu, Wayne and He, Honglin and Zhang, Chaoyuan and He, Jack and Zhao, Seth Z. and Gong, Ran and Li, Quanyi and Zhou, Bolei},
booktitle ={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
year={2025}
}
```
