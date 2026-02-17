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

def standing_time_bonus_exponential(
    env: ManagerBasedRLEnv,
    min_height: float,
    max_height: float,
    max_front_foot_contact: float,
    alpha: float = 2.0,  # 最大额外奖励
    tau: float = 2.0,    # 时间常数（秒）
    delay: float = 0.0,  # ✅ 新增：延迟时间（秒）
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
) -> torch.Tensor:
    """
    指数增长的站立时间奖励: f(t) = 1 + α * (1 - exp(-t/τ))
    
    连续站立越久，奖励越大，但有上限 (1 + α)。
    
    Args:
        env: 环境
        min_height: 最小高度阈值
        max_height: 最大高度阈值
        max_front_foot_contact: 前脚最大接触力阈值
        alpha: 最大额外奖励（建议 1.0-5.0）
        tau: 时间常数，控制增长速度（建议 1.0-3.0秒）
        asset_cfg: 机器人配置
        contact_cfg: 接触传感器配置
    
    Returns:
        奖励张量，范围 [0, 1 + α]
        - 不满足条件: 0
        - 刚满足条件: 1
        - 长时间站立: 逐渐增长到 1 + α
    
    Example:
        alpha=2.0, tau=2.0:
        - t=0s:  reward=1.0
        - t=1s:  reward=1.79
        - t=2s:  reward=2.26
        - t=5s:  reward=2.84
        - t→∞:   reward=3.0
    """
    
    # 1. 检查高度条件
    asset = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    height_ok = (current_height >= min_height) & (current_height <= max_height)
    
    # 2. 检查前脚离地条件
    contact_sensor = env.scene[contact_cfg.name]
    front_foot_ids = contact_cfg.body_ids
    forces = contact_sensor.data.net_forces_w_history[:, -1, front_foot_ids]  # (N, num_feet, 3)
    norms = torch.norm(forces, dim=-1)  # (N, num_feet)
    max_contact_force = norms.max(dim=1)[0]  # (N,) - 取最大接触力
    front_feet_ok = max_contact_force < max_front_foot_contact
    
    # 3. 综合条件
    standing_condition = height_ok & front_feet_ok
    
    # 4. 更新计时器
    # 需要在环境中维护 standing_timer，如果不存在则初始化
    if not hasattr(env, 'standing_timer'):
        env.standing_timer = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    
    # 满足条件：累加时间；不满足：清零
    dt = env.physics_dt * env.cfg.decimation  # 每步的实际时间
    env.standing_timer = torch.where(
        standing_condition,
        env.standing_timer + dt,
        torch.zeros_like(env.standing_timer)
    )
    
    # 5. 计算指数奖励 f(t) = 1 + α * (1 - exp(-t/τ))
    t = env.standing_timer
    
    effective_time = torch.clamp(t - delay, min=0.0)  # 减去延迟时间
    reward = 1.0 + alpha * (1.0 - torch.exp(-effective_time / tau))
    
    # 6. 只有站立时间超过延迟才给奖励
    reward = torch.where(
        standing_condition & (t >= delay),  # ✅ 必须满足条件且超过延迟
        reward,
        torch.zeros_like(reward)
    )
    
    return reward