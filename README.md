# Constrained Reinforcement Learning for Unstable Point-Feet Bipedal Locomotion Applied to the Bolt Robot
![](./assets/main.png)
## About this repository
This repository contains the official implementation of the paper *[Constrained Reinforcement Learning for Unstable Point-Feet Bipedal Locomotion Applied to the Bolt Robot](https://gepetto.github.io/BoltLocomotion/)* by Constant Roux, Elliot Chane-Sane, Ludovic de Matteïs, Thomas Flayols, Jérôme Manhes, Olivier Stasse and Philippe Souères.

This paper has been accepted for the 2025 IEEE-RAS 24rd International Conference on Humanoid Robots (Humanoids).

## Installation
- Install Isaac Lab 2.x.x by following the [installation guide](https://isaac-sim.github.io/IsaacLab/v2.1.0/index.html).
- Clone the repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory).
- Using a Python interpreter that has Isaac Lab installed, install the library:

```bash
python -m pip install -e exts/cat_envs
```

## Running Bolt locomotion training

Navigate to the `Bolt-CaT-RL` directory and launch a training:

```bash
python scripts/clean_rl/train.py --task=Isaac-Velocity-CaT-Bolt-v0 --headless
```

If everything goes well, you will see monitoring in the terminal as the training progresses. At the end, you can check the result with:

```bash
python scripts/clean_rl/play.py --task=Isaac-Velocity-CaT-Bolt-Play-v0 --headless --video --video_length 200
```

## Citation
If you find this project useful for your work please cite:
```
@inproceedings{roux2025bolt,
      title={Constrained Reinforcement Learning for Unstable Point-Feet Bipedal Locomotion Applied to the Bolt Robot},
      author={Constant Roux and Elliot Chane-Sane and Ludovic de Matteïs and Thomas Flayols and Jérôme Manhes and Olivier Stasse and Philippe Souères and Nicolas Mansard},
      booktitle={2025 IEEE-RAS 24rd International Conference on Humanoid Robots (Humanoids)}, 
      year={2025}
}
```