from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject  # 新增：导入 RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def base_height_above(
    env: ManagerBasedRLEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for keeping base height above threshold.
    
    Returns +1.0 if height >= min_height, else 0.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    return (current_height >= min_height).float()

def base_height_in_range(
    env: ManagerBasedRLEnv,
    target_height: float,
    tolerance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for keeping base height within a range.
    
    Returns +1.0 if height is within [target_height - tolerance, target_height + tolerance], else 0.
    
    Args:
        env: The environment.
        target_height: Target height in meters (e.g., 0.7).
        tolerance: Acceptable deviation in meters (e.g., 0.05).
        asset_cfg: The asset configuration.
    
    Returns:
        +1.0 if in range, 0.0 otherwise.
    """
    asset = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    height_diff = torch.abs(current_height - target_height)
    return (height_diff <= tolerance).float()

def base_height_progress(
    env: ManagerBasedRLEnv,
    h0: float,
    h1: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Continuous reward for height progress from h0 to h1.
    
    Returns a value in [0, 1] representing progress:
    - h <= h0: returns 0.0
    - h >= h1: returns 1.0
    - h0 < h < h1: returns (h - h0) / (h1 - h0)
    
    This provides continuous gradient for every centimeter of height gain,
    avoiding the "plateau problem" of pure threshold rewards.
    
    Args:
        env: The environment.
        h0: Starting height in meters (e.g., 0.35 for quadruped stance).
        h1: Target height in meters (e.g., 0.65 for biped stance).
        asset_cfg: The asset configuration.
    
    Returns:
        Progress value in [0, 1].
    
    Example:
        h0=0.35, h1=0.65:
        - At 0.35m: returns 0.0
        - At 0.50m: returns 0.5 (halfway)
        - At 0.65m: returns 1.0 (goal reached)
    """
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    progress = (h - h0) / (h1 - h0)
    return torch.clamp(progress, 0.0, 1.0)

def front_feet_contact_penalty_smooth(
    env: ManagerBasedRLEnv,
    contact_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    min_height: float = 0.55,
    threshold: float = 10.0,
) -> torch.Tensor:
    """
    Smooth version: penalize excess force above threshold.
    
    Instead of binary contact detection, this returns the normalized
    excess force, providing smoother gradients for learning.
    """
    contact_sensor = env.scene[contact_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]  # (N, B, 3)
    norms = torch.norm(forces, dim=-1)  # (N, B)
    
    # Compute excess force above threshold (smoother than binary)
    excess = torch.relu(norms - threshold)  # (N, B)
    
    # Normalize and sum (typical foot contact is ~50-100N)
    # Divide by 50 to get reasonable scale: 60N excess → 1.2 penalty units
    penalty = (excess / 50.0).sum(dim=1)  # (N,)
    
    # Height gate
    robot = env.scene[robot_cfg.name]
    h = robot.data.root_pos_w[:, 2]
    gate = torch.sigmoid((h - min_height - 0.025) / 0.01)
    
    return penalty * gate