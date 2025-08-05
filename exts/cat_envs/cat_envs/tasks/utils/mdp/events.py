from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.envs.mdp.events import _randomize_prop_by_op

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_body_coms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    max_displacement: float,
    asset_cfg: SceneEntityCfg,):
    """Randomize the CoM of the bodies by adding a random value sampled from the given range.

    .. tip::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()[:, body_ids, :3]
    
    # Randomize the com in range -max displacement to max displacement
    coms += torch.rand_like(coms) * 2 * max_displacement - max_displacement

    # Set the new coms
    new_coms = asset.root_physx_view.get_coms().clone()
    new_coms[:, asset_cfg.body_ids, 0:3] = coms
    asset.root_physx_view.set_coms(new_coms, env_ids)
    
def random_scale_inertia_tensors(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_min: float,
    scale_max: float,
    asset_cfg: SceneEntityCfg,
):
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve environment IDs
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Get original inertia tensors
    I_URDF = asset.root_physx_view.get_inertias().clone()[env_ids, :, :]
    N_envs, N_bodies, _ = I_URDF.shape
    device = I_URDF.device

    # Generate random scaling factors in the given range
    scale_factors = torch.rand(N_envs, N_bodies, 1, device=device) * (scale_max - scale_min) + scale_min
    
    # Apply scaling to the inertia tensors
    I_URDF_scaled = I_URDF * scale_factors
    
    # Flatten the last two dimensions to match the input shape (N_envs, N_bodies, 9)
    I_URDF_scaled_flattened = I_URDF_scaled.reshape(N_envs, N_bodies, 9)

    # Update the inertia tensors in the simulation
    asset.root_physx_view.set_inertias(I_URDF_scaled_flattened, env_ids)