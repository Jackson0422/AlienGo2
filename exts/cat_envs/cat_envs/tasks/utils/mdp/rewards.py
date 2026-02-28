from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject  # 新增：导入 RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def base_height_progress(
    env: ManagerBasedRLEnv,
    h0: float,
    h1: float,
    max_front_contact: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    front_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
    rear_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
) -> torch.Tensor:
    """Continuous reward for height progress from h0 to h1.
    Only active when front feet are off ground and at least one rear foot is on ground.
    """
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    progress = (h - h0) / (h1 - h0)
    progress = torch.clamp(progress, 0.0, 1.0)
    reward = progress ** 2

    contact_sensor = env.scene[front_contact_cfg.name]

    # 前脚必须离地
    front_forces = contact_sensor.data.net_forces_w_history[:, -1, front_contact_cfg.body_ids]
    front_norms = torch.norm(front_forces, dim=-1)
    front_max = front_norms.max(dim=1)[0]
    front_gate = (front_max < max_front_contact).float()

    # 至少一只后脚着地
    rear_forces = contact_sensor.data.net_forces_w_history[:, -1, rear_contact_cfg.body_ids]
    rear_norms = torch.norm(rear_forces, dim=-1)
    rear_gate = (rear_norms > 1.0).any(dim=1).float()

    return reward * front_gate * rear_gate

def base_height_above(
    env: ManagerBasedRLEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for keeping base height above threshold.
    Returns +1.0 if height >= min_height, else 0.
    """
    from isaaclab.assets import RigidObject
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
    alpha: float = 2.0,
    tau: float = 2.0,
    delay: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
    rear_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
) -> torch.Tensor:
    """
    指数增长的站立时间奖励: f(t) = 1 + α * (1 - exp(-t/τ))

    连续站立越久，奖励越大，但有上限 (1 + α)。

    条件：高度在范围内 + 前脚离地 + 至少一只后脚着地。
    """

    # 1. 检查高度条件
    asset = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    height_ok = (current_height >= min_height) & (current_height <= max_height)

    # 2. 检查前脚离地条件
    contact_sensor = env.scene[contact_cfg.name]
    front_foot_ids = contact_cfg.body_ids
    forces = contact_sensor.data.net_forces_w_history[:, -1, front_foot_ids]
    norms = torch.norm(forces, dim=-1)
    max_contact_force = norms.max(dim=1)[0]
    front_feet_ok = max_contact_force < max_front_foot_contact

    # 3. 检查后脚着地条件（至少一只）
    rear_forces = contact_sensor.data.net_forces_w_history[:, -1, rear_contact_cfg.body_ids]
    rear_norms = torch.norm(rear_forces, dim=-1)
    rear_any_contact = (rear_norms > 1.0).any(dim=1)

    # 4. 综合条件
    standing_condition = height_ok & front_feet_ok & rear_any_contact

    # 5. 更新计时器
    if not hasattr(env, 'standing_timer'):
        env.standing_timer = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    dt = env.physics_dt * env.cfg.decimation
    env.standing_timer = torch.where(
        standing_condition,
        env.standing_timer + dt,
        torch.zeros_like(env.standing_timer)
    )

    # 6. 计算指数奖励 f(t) = 1 + α * (1 - exp(-t/τ))
    t = env.standing_timer
    effective_time = torch.clamp(t - delay, min=0.0)
    reward = 1.0 + alpha * (1.0 - torch.exp(-effective_time / tau))

    # 7. 只有站立时间超过延迟才给奖励
    reward = torch.where(
        standing_condition & (t >= delay),
        reward,
        torch.zeros_like(reward)
    )

    return reward

def pitch_in_range_duration(
    env: ManagerBasedRLEnv,
    min_pitch_deg: float,
    max_pitch_deg: float,
    min_height: float,
    max_front_foot_contact: float,
    alpha: float = 2.0,
    tau: float = 2.0,
    delay: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
) -> torch.Tensor:
    """奖励机身pitch角维持在目标范围内，且高度足够、前腿离地时才生效。
    维持越久奖励越高。
    """
    asset = env.scene[asset_cfg.name]

    # 条件1：pitch角在范围内
    quat = asset.data.root_quat_w  # (N, 4) -> (w, x, y, z)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = torch.clamp(sin_pitch, -1.0, 1.0)
    pitch_rad = torch.asin(sin_pitch)
    pitch_deg = pitch_rad * (180.0 / torch.pi)
    pitch_ok = (pitch_deg >= min_pitch_deg) & (pitch_deg <= max_pitch_deg)

    # 条件2：机身高度 >= min_height
    current_height = asset.data.root_pos_w[:, 2]
    height_ok = current_height >= min_height

    # 条件3：前腿不接触地面
    contact_sensor = env.scene[contact_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]
    norms = torch.norm(forces, dim=-1)
    max_contact_force = norms.max(dim=1)[0]
    front_feet_ok = max_contact_force < max_front_foot_contact

    # 三个条件同时满足
    all_ok = pitch_ok & height_ok & front_feet_ok

    # 维护计时器
    if not hasattr(env, 'pitch_timer'):
        env.pitch_timer = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    dt = env.physics_dt * env.cfg.decimation
    env.pitch_timer = torch.where(
        all_ok,
        env.pitch_timer + dt,
        torch.zeros_like(env.pitch_timer)
    )

    t = env.pitch_timer
    effective_time = torch.clamp(t - delay, min=0.0)
    reward = 1.0 + alpha * (1.0 - torch.exp(-effective_time / tau))

    reward = torch.where(
        all_ok & (t >= delay),
        reward,
        torch.zeros_like(reward)
    )

    return reward

def upright_gravity_alignment(
    env: ManagerBasedRLEnv,
    k_o: float = 5.0,
    target_pitch_deg: float = 75.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    g_B = data.projected_gravity_b  # (N, 3)

    theta = target_pitch_deg * torch.pi / 180.0
    g_target = torch.tensor(
        [torch.sin(torch.tensor(theta)), 0.0, -torch.cos(torch.tensor(theta))],
        device=g_B.device,
    )

    error_sq = torch.sum((g_B - g_target) ** 2, dim=1)
    return torch.exp(-k_o * error_sq)

def roll_stability(
    env: ManagerBasedRLEnv,
    k_r: float = 10.0,
    min_height: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """r = exp(-k_r * roll^2)

    从四元数提取 roll 角，惩罚侧翻。
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w  # (w, x, y, z)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    reward = torch.exp(-k_r * roll ** 2)

    h = asset.data.root_pos_w[:, 2]
    gate = torch.sigmoid((h - min_height) / 0.02)
    return reward * gate

def com_cop_progress(
    env: ManagerBasedRLEnv,
    d_max: float = 0.20,
    min_height: float = 0.0,
    max_front_contact: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["RL_calf", "RR_calf"]),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
    front_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
) -> torch.Tensor:
    """渐进式 CoM-CoP 对齐奖励，凸函数形状，带高度/前脚/后脚门控。"""
    asset = env.scene[asset_cfg.name]
    com_pos = asset.data.root_com_pos_w[:, :2]

    contact_sensor = env.scene[contact_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]
    force_mag = torch.norm(forces, dim=-1, keepdim=True).clamp(min=1e-6)

    foot_pos = asset.data.body_pos_w[:, foot_cfg.body_ids, :2]

    total_force = force_mag.sum(dim=1)
    cop_pos = (foot_pos * force_mag).sum(dim=1) / total_force.clamp(min=1e-6)

    d = torch.norm(com_pos - cop_pos, dim=1)
    progress = 1.0 - d / d_max
    progress = torch.clamp(progress, 0.0, 1.0)
    reward = progress ** 2

    # 门控1：高度
    h = asset.data.root_pos_w[:, 2]
    height_gate = torch.sigmoid((h - min_height) / 0.02)

    # 门控2：前脚必须离地
    front_forces = contact_sensor.data.net_forces_w_history[:, -1, front_contact_cfg.body_ids]
    front_norms = torch.norm(front_forces, dim=-1)
    front_max = front_norms.max(dim=1)[0]
    front_gate = (front_max < max_front_contact).float()

    # 门控3：至少一只后脚着地
    rear_norms = torch.norm(forces, dim=-1)
    rear_any_contact = (rear_norms > 1.0).any(dim=1).float()

    return reward * height_gate * front_gate * rear_any_contact

def cop_midpoint(
    env: ManagerBasedRLEnv,
    k_cop: float = 50.0,
    min_height: float = 0.0,
    max_front_contact: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["RL_calf", "RR_calf"]),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
    front_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
) -> torch.Tensor:
    """r = exp(-k_cop * ||p_CoP - p_mid||^2) * gate(height) * gate(front) * gate(rear)"""
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene[contact_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]
    force_mag = torch.norm(forces, dim=-1, keepdim=True).clamp(min=1e-6)

    foot_pos = asset.data.body_pos_w[:, foot_cfg.body_ids, :2]

    total_force = force_mag.sum(dim=1)
    cop_pos = (foot_pos * force_mag).sum(dim=1) / total_force.clamp(min=1e-6)

    mid_pos = foot_pos.mean(dim=1)

    error_sq = torch.sum((cop_pos - mid_pos) ** 2, dim=1)
    reward = torch.exp(-k_cop * error_sq)

    # 门控1：高度
    h = asset.data.root_pos_w[:, 2]
    height_gate = torch.sigmoid((h - min_height) / 0.02)

    # 门控2：前脚必须离地
    front_forces = contact_sensor.data.net_forces_w_history[:, -1, front_contact_cfg.body_ids]
    front_norms = torch.norm(front_forces, dim=-1)
    front_max = front_norms.max(dim=1)[0]
    front_gate = (front_max < max_front_contact).float()

    # 门控3：至少一只后脚着地
    rear_norms = torch.norm(forces, dim=-1)  # (N, num_feet)
    rear_any_contact = (rear_norms > 1.0).any(dim=1).float()

    return reward * height_gate * front_gate * rear_any_contact