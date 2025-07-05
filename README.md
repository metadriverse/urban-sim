# Towards Autonomous Micromobility through Scalable Urban Simulation

<p align="center">
  <img src="documentation/assets/teaser.gif" alt="Urban-Sim Teaser"  width="640">
</p>

[![Static Badge](https://img.shields.io/badge/URBANSIM-arxiv-blue)](https://arxiv.org/pdf/2505.00690.pdf)
[![GitHub license](https://img.shields.io/github/license/metadriverse/urban-sim)](https://github.com/metadriverse/urban-sim/blob/main/LICENSE.txt)
[![GitHub contributors](https://img.shields.io/github/contributors/metadriverse/urban-sim)](https://github.com/metadriverse/urban-sim/graphs/contributors)

<div style="text-align: center; margin: 1em auto; max-width: 90%; word-wrap: break-word;">
  <strong>
    [ <a href="https://metadriverse.github.io/urbansim/">Website</a> |
    <a href="https://metadriverse.github.io/urban-sim/docs">Documentation</a> |
    <a href="https://arxiv.org/pdf/2505.00690.pdf">Paper</a> |
    <a href="https://metadriverse.github.io/">Relevant Projects</a> ]
  </strong>
</div>


## Latest Updates
- [Jul/05/2025] **v0.1.0**: The first official release of URBAN-SIM.

## Table of Contents
- [URBANSIM](#-towards-autonomous-micromobility-through-scalable-urban-simulation)
  - [🛠 Getting Started](#-getting-started)
    - [Hardware Recommendations](#hardware-recommendations)
    - [Installation](#installation)
  - [🏃‍♂️ Simulation Environment](#️-simulation-environment)
    - [Asynchronous Navigation Environment](#asynchronous-navigation-environment)
  - [🚀 Reinforcement Learning](#-reinforcement-learning)
  - [📖 Questions and Support](#-questions-and-support)
  - [📎 Citation](#-citation)

## 🛠 Getting Started

### Hardware Recommendations

To ensure the best experience with **URBAN-SIM**, please review the following hardware guidelines:

- **Recommended Hardware**:  
  - **OS**: Ubuntu 22.04; 24.04.
  - **GPU**: Nvidia GPU with at least **16GB RAM** and **12GB VRAM**.
    - Tested GPUs: **Nvidia RTX-4080, RTX-4090, L40S**.   
  - **Storage**: Minimum of **50GB free space**.  

### Installation
#### 1. Install IsaacSim 4.5
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
```

#### 2. Install URBAN-SIM

```bash
# Create a new conda environment and install required libraries
bash urbansim.sh -c [env_name]  # The default name is "urbansim"
conda activate urbansim         # Or use your specified env_name
bash urbansim.sh -i             # Install dependencies and initialize
bash urbansim.sh -a             # Advanced installation including procedural generation pipeline and rl training frameworks

# Download the required assets
python scripts/tools/collectors/collect_asset.py

# Convert .glb files to .usd files
python scripts/tools/converters/convert_asset.py
```

It should be noted that you should install several dependencies including ```isaacsim, cmake, make, gcc``` on your system before installing urbansim.

## 🏃‍♂️ Simulation Environment

We provide examples to demonstrate features and basic usages of URBAN-SIM after the local installation.

### Asynchronous Navigation Environment
We provide source code and scripts to extend and customize the asynchronous navigation environment, enabling users to develop navigation policies and integrate with various robot embodiments.
```bash
python urbansim/envs/separate_envs/random_env.py --enable_cameras --num_envs ${NUM_ENV} --scenario_type ${TYPE} --use_async
```
- `--enable_cameras`: Enables vision-based observation space.
- `NUM_ENV`: Number of parallel environments to simulate (e.g., 256).
- `TYPE`: `{clean, static, dynamic}`  
  - `clean`: No obstacles or pedestrians  
  - `static`: Includes static obstacles  
  - `dynamic`: Includes static obstacles and moving pedestrians
- `--use_async`: Launches environments in asynchronous stepping mode, enabling diverse simulation timings across parallel agents.

In addition to random object placement, we provid procedural generation (PG) pipeline enables scalable creation of large-scale, structured environments—such as MetaUrban layouts—for reinforcement learning at scale.
```bash
python urbansim/envs/separate_envs/pg_env.py --enable_cameras --num_envs ${NUM_ENV} --use_async
```

More details, comparisons, and target results can be found in:
[Scenarios](documentation/scenario_generation.md) and [AsyncSimulation](documentation/async_simulation.md)

## 🚀 Reinforcement Learning
We train policies by specifying configuration files, which define environment settings, algorithm parameters, and training options. For example,

### Training
#### Training with Large-Memory GPUs (> 20 GB VRAM)

```bash
python urbansim/learning/RL/train.py --env configs/env_configs/navigation/coco.yaml --enable_cameras
```

This script is designed for `large-scale` experiments using powerful GPUs. The provided configuration and default parameters are designed to ensure stable and reproducible training results across diverse scenarios.

For PPO training, you can change the parameters in the configuration files to adjust learning rate, number of steps, entropy regularization, and other hyperparameters.

We adopt different training backends tailored to specific tasks:

- `Locomotion` is trained with the RSL-RL framework, which provides fast and stable low-level control.

- `Navigation` is trained with the rl-games framework, which supports training with flexible network architectures.

You don’t need to install these frameworks separately — all dependencies are installed via ```urbansim.sh```.

#### Training with limited environments on a local machine

```bash
python urbansim/learning/RL/train.py --env configs/env_configs/navigation/coco_mini.yaml --enable_cameras --mini
```

This script is provided for debugging on a local machine. To obtain a robust policy, we highly recommend training with larger-scale environments as described above.

### Play with pretrained policy
You can play with the pretrained policy in environments, for example,
```bash
python urbansim/learning/RL/play.py --env configs/env_configs/navigation/coco.yaml --enable_cameras --num_envs 1 --checkpoint ./assets/ckpts/navigation/coco_static.pth
```

This will launch a single environment with camera views enabled, using a pretrained navigation policy in a static urban scene.

We plan to support more pretrained checkpoints and diverse environments in upcoming releases. Please stay tuned !

## 📖 Questions and Support

Can't run URBAN-SIM correctly? Please refer to: [FAQs](documentation/faq.md). 

Can't find the answer to your question? Try posting issues or asking the developers and community in our Discussions forum.

## 📌 TODOs
- [ ] Curated asset and scenarios for better rendering.
- [ ] Release checkpoints for more robots (Booster T1, Unitree G1, ...)
- [ ] Release reactive environmental agents via ORCA -> IDM-JAX.
- [ ] Release useful scripts for downstream tasks such as data generation for imitation learning, instance/appearance randomization for reinforcement learning.
- [ ] Lightweight pipeline for scenario & environment generation.

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
