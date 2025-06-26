# Known Issues and Limitations

This section outlines known issues and current limitations in URBAN-SIM, including hardware requirements, missing dependencies, upcoming asset releases, etc.

## Module Not Found Error

If you meet the error `ModuleNotFoundError: No module named 'xxx'`, ensure that the conda environment is activated and use:

```bash
source _isaac_sim/setup_conda_env.sh
```

## CUDA & GLIBCXX Incompatibility

URBAN-SIM relies on many libraries and systems like NVIDIA Isaac Sim, which requires support of `libstdc++` for successful runtime.  
On some systems, especially with custom Anaconda installations, the following error may occur:

```bash
ImportError: ... libstdc++.so.6: version `GLIBCXX_3.4.32' not found
```

**Solution**: Ensure your environment uses a system-level `libstdc++.so.6` with `GLIBCXX_3.4.32+`.  
You can verify this with:

```bash
strings $(g++ -print-file-name=libstdc++.so.6) | grep GLIBCXX
```

If you can find `GLIBCXX_3.4.32` or higher in the output, your system is compatible.  
If you are using a custom Anaconda installation, you may need to install the system-level `libstdc++` package or replace the `libstdc++.so.6` link in Anaconda with the system version.

## A100 GPU Compatibility

Isaac Sim currently does **not support running on NVIDIA A100 GPUs** due to limitations in the OptiX and PhysX backends.  
Attempting to launch on an A100 may result in failure to create Vulkan/OpenGL contexts or invalid CUDA device errors.

We recommend using RTX 30xx / 40xx / 50xx consumer cards or supported workstation GPUs.

## Asset Availability

Some assets in URBAN-SIM scenarios are under construction or being curated for release.  
More specifically:

- **Static objects**: Static objects are currently partial; a full set with more than 15,000 well-annotated assets will be released progressively.
- **Pedestrians**: We use pedestrian assets from the [Synbody](https://synbody.github.io/) project.  
  To reduce space and streamline runtime loading, only a **subset of Synbody** models is currently included in URBAN-SIM.  
  We will release code to process and convert Synbody assets into `.usd` format in the future.  
  For now, you can manually convert Synbody assets using the Isaac Sim GUI.

Please stay tuned via our repo or project page for updates.