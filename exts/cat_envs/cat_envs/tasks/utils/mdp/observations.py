# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def joint_pos(
    env: ManagerBasedEnv,
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset.find_joints(names, preserve_order=True)[0]]


def joint_vel(
    env: ManagerBasedEnv,
    names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset.find_joints(names, preserve_order=True)[0]]

def foot_contact_bool(
    env: ManagerBasedEnv,
    threshold: float = 1.0,
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf", "RL_calf", "RR_calf"]),
) -> torch.Tensor:
    """Binary contact indicator for each foot: 1 if contact force > threshold, else 0.
    Returns (N, num_feet) tensor."""
    contact_sensor = env.scene[contact_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]
    norms = torch.norm(forces, dim=-1)  # (N, num_feet)
    return (norms > threshold).float()


def foot_contact_fz(
    env: ManagerBasedEnv,
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
) -> torch.Tensor:
    """Vertical (z) contact force for specified feet.
    Returns (N, num_feet) tensor."""
    contact_sensor = env.scene[contact_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]
    fz = forces[:, :, 2].clamp(min=0.0)  # (N, num_feet)
    return fz


def com_cop_offset(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["RL_calf", "RR_calf"]),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
) -> torch.Tensor:
    """Horizontal offset from CoP to CoM: (com_x - cop_x, com_y - cop_y).
    Returns (N, 2) tensor."""
    from cat_envs.tasks.utils.mdp.rewards import _get_foot_pos_from_calf

    asset = env.scene[asset_cfg.name]
    body_com = asset.data.body_com_pos_w
    body_mass = asset.data.default_mass.to(body_com.device)
    total_mass = body_mass.sum(dim=1, keepdim=True)
    com_xy = ((body_com * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass)[:, :2]

    contact_sensor = env.scene[contact_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]
    fz = forces[..., 2:3].clamp(min=0.0)

    foot_pos = _get_foot_pos_from_calf(asset, foot_cfg)[..., :2]
    cop_xy = (foot_pos * fz).sum(dim=1) / fz.sum(dim=1).clamp(min=1e-6)

    return com_xy - cop_xy  # (N, 2)
