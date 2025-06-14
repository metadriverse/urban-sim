# MetaUrban: An Embodied AI Simulation Platform for Urban Micromobility

[![Static Badge](https://img.shields.io/badge/MetaUrban-arxiv-blue)](https://arxiv.org/pdf/2407.08725.pdf)
[![Documentation](https://readthedocs.org/projects/metaurban-simulator/badge/?version=latest)](https://metaurban-simulator.readthedocs.io)
[![GitHub license](https://img.shields.io/github/license/metadriverse/metaurban)](https://github.com/metadriverse/metaurban/blob/main/LICENSE.txt)
[![GitHub contributors](https://img.shields.io/github/contributors/metadriverse/metaurban)](https://github.com/metadriverse/metaurban/graphs/contributors)

<div style="text-align: center; width:100%; margin: 0 auto; display: inline-block">
<strong>
[
<a href="https://metaurban-simulator.readthedocs.io">Documentation</a>
|
<a href="https://colab.research.google.com/github/metadriverse/metaurban/blob/main/metaurban/examples/basic_colab_usage.ipynb">Colab Examples</a>
|
<a href="https://metadriverse.github.io/metaurban/">Website</a>
|
<a href="https://arxiv.org/pdf/2407.08725.pdf">Paper</a>
|
<a href="https://metadriverse.github.io/">Relevant Projects</a>
]
</strong>
</div>


**`MetaUrban`** is a cutting-edge simulation platform designed for Embodied AI research in urban spaces. It offers:

- 🌆 **Infinite Urban Scene Generation**: Create diverse, interactive city environments.  
- 🏗️ **High-Quality Urban Objects**: Includes real-world infrastructure and clutter.   
- 🧍 **Rigged Human Models**: SMPL-compatible models with diverse motions.
- 🤖 **Urban Agents**: Including delivery robots, cyclists, skateboarders, and more.  
- 🕹️ **Flexible User Interfaces**: Compatible with mouse, keyboard, joystick, and racing wheel.  
- 🎥 **Configurable Sensors**: Supports RGB, depth, semantic map, and LiDAR.  
- ⚙️ **Rigid-Body Physics**: Realistic mechanics for agents and environments.  
- 🌍 **OpenAI Gym Interface**: Seamless integration for AI and reinforcement learning tasks.
- 🔗 **Framework Compatibility**: Seamlessly integrates with Ray, Stable Baselines, Torch, Imitation, etc.

📖 Check out [**`MetaUrban` Documentation**](https://metaurban-simulator.readthedocs.io) to learn more!

## 📎 Citation

If you find MetaUrban helpful for your research, please cite the following BibTeX entry.

```latex
@article{wu2025metaurban,
  title={MetaUrban: An Embodied AI Simulation Platform for Urban Micromobility},
  author={Wu, Wayne and He, Honglin and He, Jack and Wang, Yiran and Duan, Chenda and Liu, Zhizheng and Li, Quanyi and Zhou, Bolei},
  journal={ICLR},
  year={2025}
}
```

## 💘 Acknowledgement
The simulator can not be built without the help from Panda3D community and the following open-sourced projects:
- panda3d-simplepbr: https://github.com/Moguri/panda3d-simplepbr
- panda3d-gltf: https://github.com/Moguri/panda3d-gltf
- RenderPipeline (RP): https://github.com/tobspr/RenderPipeline
- Water effect for RP: https://github.com/kergalym/RenderPipeline 
- procedural_panda3d_model_primitives: https://github.com/Epihaius/procedural_panda3d_model_primitives
- DiamondSquare for terrain generation: https://github.com/buckinha/DiamondSquare
- KITSUNETSUKI-Asset-Tools: https://github.com/kitsune-ONE-team/KITSUNETSUKI-Asset-Tools
- Objaverse: https://github.com/allenai/objaverse-xl
- OmniObject3D: https://github.com/omniobject3d/OmniObject3D
- Synbody: https://github.com/SynBody/SynBody
- BEDLAM: https://github.com/pixelite1201/BEDLAM
- ORCA: https://gamma.cs.unc.edu/ORCA/