from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObject  # Added: import RigidObject / 新增：导入 RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def _quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector v by quaternion q (w,x,y,z convention)."""
    w, x, y, z = q.unbind(-1)
    vx, vy, vz = v.unbind(-1)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return torch.stack((
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ), dim=-1)


_FOOT_OFFSET_LOCAL = None

def _get_foot_pos_from_calf(asset, foot_cfg) -> torch.Tensor:
    """From calf link pose, compute foot bottom position in world frame."""
    global _FOOT_OFFSET_LOCAL
    calf_pos_w = asset.data.body_pos_w[:, foot_cfg.body_ids, :]    # (N, 2, 3)
    calf_quat_w = asset.data.body_quat_w[:, foot_cfg.body_ids, :]  # (N, 2, 4) wxyz
    if _FOOT_OFFSET_LOCAL is None or _FOOT_OFFSET_LOCAL.device != calf_pos_w.device:
        _FOOT_OFFSET_LOCAL = torch.tensor([0.0, 0.0, -0.25], device=calf_pos_w.device)
    offset = _FOOT_OFFSET_LOCAL.view(1, 1, 3).expand_as(calf_pos_w)
    foot_pos_w = calf_pos_w + _quat_apply_wxyz(calf_quat_w, offset)
    return foot_pos_w

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

    # Front feet must be off the ground / 前脚必须离地
    front_forces = contact_sensor.data.net_forces_w_history[:, -1, front_contact_cfg.body_ids]
    front_norms = torch.norm(front_forces, dim=-1)
    front_max = front_norms.max(dim=1)[0]
    front_gate = (front_max < max_front_contact).float()

    # At least one rear foot on the ground / 至少一只后脚着地
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
    min_pitch_deg: float = -80.0,   
    max_pitch_deg: float = -45.0,   
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
    rear_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
) -> torch.Tensor:
    """Exponential standing time reward: f(t) = 1 + α * (1 - exp(-t/τ))
    指数增长的站立时间奖励: f(t) = 1 + α * (1 - exp(-t/τ))

    The longer the robot stands, the higher the reward, capped at (1 + α).
    连续站立越久，奖励越大，但有上限 (1 + α)。

    Conditions: height in range + front feet off ground + at least one rear foot on ground.
    条件：高度在范围内 + 前脚离地 + 至少一只后脚着地。
    """

    # 1. Check height condition / 检查高度条件
    asset = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    height_ok = (current_height >= min_height) & (current_height <= max_height)

    # 2. Check front feet off ground / 检查前脚离地条件
    contact_sensor = env.scene[contact_cfg.name]
    front_foot_ids = contact_cfg.body_ids
    forces = contact_sensor.data.net_forces_w_history[:, -1, front_foot_ids]
    norms = torch.norm(forces, dim=-1)
    max_contact_force = norms.max(dim=1)[0]
    front_feet_ok = max_contact_force < max_front_foot_contact

    # 2.5 Check pitch in range / 检查 pitch 角度范围
    quat = asset.data.root_quat_w  # (N, 4) wxyz
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    pitch = torch.atan2(2.0 * (w * y - z * x), 1.0 - 2.0 * (x * x + y * y))
    pitch_deg = pitch * (180.0 / math.pi)
    pitch_ok = (pitch_deg >= min_pitch_deg) & (pitch_deg <= max_pitch_deg)

    # 3. Check rear foot contact (at least one) / 检查后脚着地条件（至少一只）
    rear_forces = contact_sensor.data.net_forces_w_history[:, -1, rear_contact_cfg.body_ids]
    rear_norms = torch.norm(rear_forces, dim=-1)
    rear_any_contact = (rear_norms > 1.0).any(dim=1)

    # 4. Combined condition / 综合条件
    standing_condition = height_ok & front_feet_ok & rear_any_contact & pitch_ok

    # 5. Update standing timer / 更新计时器
    if not hasattr(env, 'standing_timer'):
        env.standing_timer = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    dt = env.physics_dt * env.cfg.decimation
    env.standing_timer = torch.where(
        standing_condition,
        env.standing_timer + dt,
        torch.zeros_like(env.standing_timer)
    )

    # 6. Compute exponential reward f(t) = 1 + α * (1 - exp(-t/τ)) / 计算指数奖励
    t = env.standing_timer
    effective_time = torch.clamp(t - delay, min=0.0)
    reward =  alpha * (1.0 - torch.exp(-effective_time / tau))

    # 7. Only reward when standing time exceeds delay / 只有站立时间超过延迟才给奖励
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
    rear_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),  
) -> torch.Tensor:
    """Reward for maintaining pitch angle within target range, active only when height is sufficient and front legs are off ground.
    The longer the pitch is maintained, the higher the reward.
    奖励机身pitch角维持在目标范围内，且高度足够、前腿离地时才生效。维持越久奖励越高。
    """
    asset = env.scene[asset_cfg.name]

    # Condition 1: pitch angle within range / 条件1：pitch角在范围内
    quat = asset.data.root_quat_w  # (N, 4) -> (w, x, y, z)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sin_pitch = 2.0 * (w * y - z * x)
    sin_pitch = torch.clamp(sin_pitch, -1.0, 1.0)
    pitch_rad = torch.asin(sin_pitch)
    pitch_deg = pitch_rad * (180.0 / torch.pi)
    pitch_ok = (pitch_deg >= min_pitch_deg) & (pitch_deg <= max_pitch_deg)

    # Condition 2: base height >= min_height / 条件2：机身高度 >= min_height
    current_height = asset.data.root_pos_w[:, 2]
    height_ok = current_height >= min_height

    # Condition 3: front legs not touching ground / 条件3：前腿不接触地面
    contact_sensor = env.scene[contact_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]
    norms = torch.norm(forces, dim=-1)
    max_contact_force = norms.max(dim=1)[0]
    front_feet_ok = max_contact_force < max_front_foot_contact

    # Condition 4: at least one rear foot on ground / 条件4：至少一只后脚着地
    rear_forces = contact_sensor.data.net_forces_w_history[:, -1, rear_contact_cfg.body_ids]
    rear_norms = torch.norm(rear_forces, dim=-1)
    rear_any_contact = (rear_norms > 1.0).any(dim=1)

    # All three conditions must be met / 三个条件同时满足
    all_ok = pitch_ok & height_ok & front_feet_ok

    # Maintain pitch timer / 维护计时器
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
    min_height: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    g_B = data.projected_gravity_b

    theta = target_pitch_deg * torch.pi / 180.0
    g_target = torch.tensor(
        [torch.sin(torch.tensor(theta)), 0.0, -torch.cos(torch.tensor(theta))],
        device=g_B.device,
    )

    error_sq = torch.sum((g_B - g_target) ** 2, dim=1)
    reward = torch.exp(-k_o * error_sq)

    h = data.root_pos_w[:, 2]
    height_gate = torch.sigmoid((h - min_height) / 0.02)

    return reward * height_gate

def roll_stability(
    env: ManagerBasedRLEnv,
    k_r: float = 10.0,
    min_height: float = 0.0,
    max_front_contact: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    front_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    reward = torch.exp(-k_r * roll ** 2)

    h = asset.data.root_pos_w[:, 2]
    height_gate = torch.sigmoid((h - min_height) / 0.02)

    # Front feet off ground gate / 前脚离地门控
    contact_sensor = env.scene[front_contact_cfg.name]
    front_forces = contact_sensor.data.net_forces_w_history[:, -1, front_contact_cfg.body_ids]
    front_norms = torch.norm(front_forces, dim=-1)
    front_max = front_norms.max(dim=1)[0]
    front_gate = (front_max < max_front_contact).float()

    return reward * height_gate * front_gate

def com_cop_correction(
    env: ManagerBasedRLEnv,
    d_max: float = 0.20,
    k: float = 4.0,
    correction_scale: float = 1.0,
    min_height: float = 0.0,
    max_front_contact: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["RL_calf", "RR_calf"]),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
    front_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    body_com = asset.data.body_com_pos_w
    body_mass = asset.data.default_mass.to(body_com.device)
    total_mass = body_mass.sum(dim=1, keepdim=True)
    whole_com = (body_com * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass
    com_pos = whole_com[:, :2]

    contact_sensor = env.scene[contact_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]
    fz = forces[..., 2:3].clamp(min=0.0)

    foot_pos = _get_foot_pos_from_calf(asset, foot_cfg)[..., :2]
    total_force = fz.sum(dim=1)
    cop_pos = (foot_pos * fz).sum(dim=1) / total_force.clamp(min=1e-6)

    d = torch.norm(com_pos - cop_pos, dim=1)

    # --- Term 1: Alignment reward (peak at d=0) ---
    alignment = torch.exp(-k * (d / d_max) ** 2)

    # --- Term 2: Correction reward (positive when d is decreasing) ---
    # --- Term 2: Correction reward (CoP moving towards CoM) ---
    if not hasattr(env, '_prev_cop_pos'):
        env._prev_cop_pos = torch.zeros(env.num_envs, 2, device=env.device)
    reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
        env._prev_cop_pos[reset_ids] = cop_pos[reset_ids]

    # Handle envs reset at the END of the previous step (first reward call of
    # the new episode has episode_length_buf == 1). Without this, delta_cop
    # carries the CoP jump between fail-state and fresh-state, producing a
    # spurious correction reward on every new rollout.
    just_reset_mask = env.episode_length_buf == 1
    if just_reset_mask.any():
        env._prev_cop_pos[just_reset_mask] = cop_pos[just_reset_mask]

    delta_cop = cop_pos - env._prev_cop_pos  # (N, 2)
    env._prev_cop_pos = cop_pos.clone()
    direction = com_pos - cop_pos  # (N, 2) vector from CoP to CoM
    unit_dir = direction / d.clamp(min=1e-6).unsqueeze(-1)  # (N, 2) unit vector
    cop_towards_com = (delta_cop * unit_dir).sum(dim=1)  # projection onto CoP→CoM direction
    correction = (cop_towards_com / d_max).clamp(min=0.0, max=1.0)

    # --- Combined reward ---
    reward = alignment + correction_scale * correction

    # Gate 1: height
    h = asset.data.root_pos_w[:, 2]
    height_gate = torch.sigmoid((h - min_height) / 0.02)

    # Gate 2: front feet off ground
    front_forces = contact_sensor.data.net_forces_w_history[:, -1, front_contact_cfg.body_ids]
    front_norms = torch.norm(front_forces, dim=-1)
    front_max = front_norms.max(dim=1)[0]
    front_gate = (front_max < max_front_contact).float()

    # Gate 3: at least one rear foot on ground
    rear_any_contact = (fz.squeeze(-1) > 1.0).any(dim=1).float()

    return reward * height_gate * front_gate * rear_any_contact

def com_cop_correction_2(
    env: ManagerBasedRLEnv,
    k: float = 4.0,
    correction_scale: float = 1.0,
    foot_radius: float = 0.04,        # 新参数：脚底前后向有效半径
    min_height: float = 0.0,
    max_front_contact: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["RL_calf", "RR_calf"]),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
    front_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    # --- CoM (whole body, XY) ---
    body_com = asset.data.body_com_pos_w
    body_mass = asset.data.default_mass.to(body_com.device)
    total_mass = body_mass.sum(dim=1, keepdim=True)
    whole_com = (body_com * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass
    com_xy = whole_com[:, :2]

    # --- CoP (rear feet, XY) ---
    contact_sensor = env.scene[contact_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]
    fz = forces[..., 2:3].clamp(min=0.0)
    foot_pos = _get_foot_pos_from_calf(asset, foot_cfg)          # (N, 2, 3)
    foot_xy = foot_pos[..., :2]                                   # (N, 2, 2)
    total_force = fz.sum(dim=1)
    cop_xy = (foot_xy * fz).sum(dim=1) / total_force.clamp(min=1e-6)

    # === 新增：各向异性 d_norm 计算 ===
    foot_L = foot_xy[:, 0, :]   # (N, 2)  RL foot
    foot_R = foot_xy[:, 1, :]   # (N, 2)  RR foot
    support_vec = foot_R - foot_L                                  # (N, 2)
    support_len = torch.norm(support_vec, dim=1)                   # (N,)
    support_dir = support_vec / support_len.clamp(min=1e-6).unsqueeze(-1)  # (N, 2) 单位向量

    offset = com_xy - cop_xy                                       # (N, 2)
    lateral  = (offset * support_dir).sum(dim=1)                   # 沿脚间连线投影
    sagittal_vec = offset - lateral.unsqueeze(-1) * support_dir    # 垂直分量
    sagittal = torch.norm(sagittal_vec, dim=1)                     # (N,)

    half_width_lateral  = support_len / 2.0                        # 侧向半宽
    half_width_sagittal = foot_radius                              # 前后向半宽

    d_norm = torch.sqrt(
        (lateral / half_width_lateral.clamp(min=0.02)) ** 2 +
        (sagittal / half_width_sagittal) ** 2
    )

    # --- Term 1: Alignment reward (peak at d_norm=0) ---
    alignment = torch.exp(-k * d_norm ** 2)

    # --- Term 2: Correction = d_norm 在减小 ---
    if not hasattr(env, '_prev_d_norm_reward'):
        env._prev_d_norm_reward = torch.zeros(env.num_envs, device=env.device)
    reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
        env._prev_d_norm_reward[reset_ids] = d_norm[reset_ids]

    delta_d_norm = env._prev_d_norm_reward - d_norm   # 正值 = d_norm 在减小 = 更稳定
    env._prev_d_norm_reward = d_norm.clone()
    correction = delta_d_norm.clamp(min=0.0, max=1.0)

    # --- Combined reward ---
    reward = alignment + correction_scale * correction

    # --- Gates (不变) ---
    h = asset.data.root_pos_w[:, 2]
    height_gate = torch.sigmoid((h - min_height) / 0.02)

    front_forces = contact_sensor.data.net_forces_w_history[:, -1, front_contact_cfg.body_ids]
    front_norms = torch.norm(front_forces, dim=-1)
    front_max = front_norms.max(dim=1)[0]
    front_gate = (front_max < max_front_contact).float()

    rear_any_contact = (fz.squeeze(-1) > 1.0).any(dim=1).float()

    return reward * height_gate * front_gate * rear_any_contact

def cp_cop_correction(
    env: ManagerBasedRLEnv,
    k: float = 4.0,
    correction_scale: float = 2.0,
    cop_tracking_scale: float = 0.005,
    foot_radius: float = 0.04,
    velocity_alpha: float = 1.0,
    min_height: float = 0.0,
    max_front_contact: float = 1.0,
    pitch_min_deg: float = 55.0,
    pitch_sharpness: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["RL_calf", "RR_calf"]),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
    front_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]

    # --- CoM (whole body, XY) ---
    body_com = asset.data.body_com_pos_w
    body_mass = asset.data.default_mass.to(body_com.device)
    total_mass = body_mass.sum(dim=1, keepdim=True)
    whole_com = (body_com * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass
    com_xy = whole_com[:, :2]

    # --- Capture Point: com + v / omega ---
    v_com = asset.data.root_lin_vel_w[:, :2]
    h_com = whole_com[:, 2].clamp(min=0.3)
    omega = torch.sqrt(9.81 / h_com)
    capture_xy = com_xy + velocity_alpha * v_com / omega.unsqueeze(-1)

    # --- CoP (rear feet, XY) ---
    contact_sensor = env.scene[contact_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]
    fz = forces[..., 2:3].clamp(min=0.0)
    foot_pos = _get_foot_pos_from_calf(asset, foot_cfg)
    foot_xy = foot_pos[..., :2]
    total_force = fz.sum(dim=1)
    cop_xy = (foot_xy * fz).sum(dim=1) / total_force.clamp(min=1e-6)

    # --- Anisotropic d_norm (CP vs CoP) ---
    foot_L = foot_xy[:, 0, :]
    foot_R = foot_xy[:, 1, :]
    support_vec = foot_R - foot_L
    support_len = torch.norm(support_vec, dim=1)
    support_dir = support_vec / support_len.clamp(min=1e-6).unsqueeze(-1)

    offset = capture_xy - cop_xy
    lateral = (offset * support_dir).sum(dim=1)
    sagittal_vec = offset - lateral.unsqueeze(-1) * support_dir
    sagittal = torch.norm(sagittal_vec, dim=1)

    half_width_lateral = support_len / 2.0
    half_width_sagittal = foot_radius

    d_norm = torch.sqrt(
        (lateral / half_width_lateral.clamp(min=0.02)) ** 2 +
        (sagittal / half_width_sagittal) ** 2
    )

    # --- Term 1: Alignment (CP ≈ CoP) ---
    alignment = torch.exp(-k * d_norm ** 2)

    # --- Term 2: CoP 主动追踪 CP ---
    if not hasattr(env, '_prev_cop_xy'):
        env._prev_cop_xy = cop_xy.clone()
    reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
        env._prev_cop_xy[reset_ids] = cop_xy[reset_ids]

    cop_delta = cop_xy - env._prev_cop_xy
    env._prev_cop_xy = cop_xy.clone()

    gap = capture_xy - cop_xy
    gap_norm = torch.norm(gap, dim=1, keepdim=True).clamp(min=1e-6)
    gap_dir = gap / gap_norm

    cop_toward_cp = (cop_delta * gap_dir).sum(dim=1)
    correction = (cop_toward_cp / cop_tracking_scale).clamp(min=0.0, max=1.0)

    # --- Combined reward ---
    reward = alignment + correction_scale * correction

    # --- Gates ---
    h = asset.data.root_pos_w[:, 2]
    height_gate = torch.sigmoid((h - min_height) / 0.02)

    front_forces = contact_sensor.data.net_forces_w_history[:, -1, front_contact_cfg.body_ids]
    front_norms = torch.norm(front_forces, dim=-1)
    front_max = front_norms.max(dim=1)[0]
    front_gate = (front_max < max_front_contact).float()

    rear_any_contact = (fz.squeeze(-1) > 1.0).any(dim=1).float()

    g = asset.data.projected_gravity_b
    pitch_deg = torch.atan2(-g[:, 0], -g[:, 2]) * (180.0 / 3.14159)
    pitch_gate = torch.sigmoid((pitch_deg - pitch_min_deg) / pitch_sharpness)

    return reward * height_gate * front_gate * rear_any_contact * pitch_gate

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
    fz = forces[..., 2:3].clamp(min=0.0)  # (N, 2, 1) vertical support force only / 只用竖直支撑力

    foot_pos = _get_foot_pos_from_calf(asset, foot_cfg)[..., :2]

    total_force = fz.sum(dim=1)
    cop_pos = (foot_pos * fz).sum(dim=1) / total_force.clamp(min=1e-6)

    mid_pos = foot_pos.mean(dim=1)

    error_sq = torch.sum((cop_pos - mid_pos) ** 2, dim=1)
    reward = torch.exp(-k_cop * error_sq)

    # Gate 1: height / 门控1：高度
    h = asset.data.root_pos_w[:, 2]
    height_gate = torch.sigmoid((h - min_height) / 0.02)

    # Gate 2: front feet must be off ground / 门控2：前脚必须离地
    front_forces = contact_sensor.data.net_forces_w_history[:, -1, front_contact_cfg.body_ids]
    front_norms = torch.norm(front_forces, dim=-1)
    front_max = front_norms.max(dim=1)[0]
    front_gate = (front_max < max_front_contact).float()

    # Gate 3: at least one rear foot on ground / 门控3：至少一只后脚着地
    rear_any_contact = (fz.squeeze(-1) > 1.0).any(dim=1).float()

    return reward * height_gate * front_gate * rear_any_contact

def height_maintenance(
    env: ManagerBasedRLEnv,
    target_height: float,
    sigma: float = 0.05,
    target_pitch_deg: float = -90.0,
    pitch_sigma_deg: float = 20.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    height_reward = torch.exp(-((h - target_height) / sigma) ** 2)

    g = asset.data.projected_gravity_b
    pitch = torch.atan2(-g[:, 0], -g[:, 2])
    target_pitch_rad = target_pitch_deg * 3.14159 / 180.0
    pitch_sigma_rad = pitch_sigma_deg * 3.14159 / 180.0
    pitch_reward = torch.exp(-((pitch - target_pitch_rad) / pitch_sigma_rad) ** 2)

    pitch_grav_deg = pitch * 180.0 / 3.14159

    return height_reward * pitch_reward

def posture_progress_to_target(
    env: ManagerBasedRLEnv,
    target_height: float,
    sigma: float = 0.05,
    target_pitch_deg: float = 80.0,
    pitch_sigma_deg: float = 20.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Shaping reward: reward any positive progress of the posture score
    toward the (height, pitch) target.

    Score definition (MUST be kept identical to `height_maintenance`):
        s_t = exp(-((h - h*)/sigma)^2) * exp(-((pitch - theta*)/sigma_theta)^2)

    Reward:
        r_t = max(0, s_t - s_{t-1})

    Properties:
      - During initial rear-up (score monotonically increasing): every step
        gives positive reward. Does NOT penalize the angular velocity required
        to stand up.
      - Once at target (s_t ~ 1): reward goes to 0. Long-term maintenance is
        handled by `height_maintenance`.
      - After disturbance drops score: reward = 0 (no double-penalty).
      - During recovery (score rising again toward target): positive reward.

    Important:
        target_height / sigma / target_pitch_deg / pitch_sigma_deg MUST be
        kept in sync with `height_maintenance` (including via curriculum),
        so that the "target" in both rewards refers to the same point.

    塑形奖励：朝 (高度, pitch) 目标靠近时给正反馈。公式与 height_maintenance
    完全一致，确保两者共享同一个"目标"定义。
    """
    asset = env.scene[asset_cfg.name]

    # --- Identical to height_maintenance: height term ---
    h = asset.data.root_pos_w[:, 2]
    height_reward = torch.exp(-((h - target_height) / sigma) ** 2)

    # --- Identical to height_maintenance: pitch term ---
    g = asset.data.projected_gravity_b
    pitch = torch.atan2(-g[:, 0], -g[:, 2])
    target_pitch_rad = target_pitch_deg * 3.14159 / 180.0
    pitch_sigma_rad = pitch_sigma_deg * 3.14159 / 180.0
    pitch_reward = torch.exp(-((pitch - target_pitch_rad) / pitch_sigma_rad) ** 2)

    # --- Current score (product, matches height_maintenance output) ---
    score = height_reward * pitch_reward

    # --- Persistent previous score, reset-safe ---
    if not hasattr(env, "_pp_prev_score"):
        env._pp_prev_score = score.clone()
    reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
        env._pp_prev_score[reset_ids] = score[reset_ids]

    # Handle envs reset at the END of the previous step.
    # episode_length_buf == 1 marks the first reward call of a new episode;
    # without this, _pp_prev_score still carries the fail-state score and a
    # spurious positive delta is emitted on episode-1 of every new rollout.
    just_reset_mask = env.episode_length_buf == 1
    if just_reset_mask.any():
        env._pp_prev_score[just_reset_mask] = score[just_reset_mask]

    # --- Only reward positive progress ---
    delta = score - env._pp_prev_score
    reward = delta.clamp(min=0.0)

    # --- Update state for next step ---
    env._pp_prev_score = score.detach().clone()

    return reward

def ang_vel_xy_damping(
    env: ManagerBasedRLEnv,
    sigma_roll: float = 2.0,
    sigma_pitch: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    w = asset.data.root_ang_vel_b
    wx = w[:, 0]
    wy = w[:, 1]
    return torch.exp(-(sigma_roll * wx * wx + sigma_pitch * wy * wy))

def reactive_balance(
    env: ManagerBasedRLEnv,
    offset_threshold: float = 0.05,
    min_height: float = 0.50,
    max_front_contact: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rear_joint_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        joint_names=[
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ],
    ),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["RL_calf", "RR_calf"]),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
    front_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
) -> torch.Tensor:
    """Reward rear-leg corrective action when CoM-CoP offset is large.
    Gated by height, front feet off ground."""
    asset = env.scene[asset_cfg.name]

    # CoM (whole body)
    body_com = asset.data.body_com_pos_w
    body_mass = asset.data.default_mass.to(body_com.device)
    total_mass = body_mass.sum(dim=1, keepdim=True)
    com_xy = ((body_com * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass)[:, :2]

    # CoP (rear feet only)
    contact_sensor = env.scene[contact_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]
    fz = forces[..., 2:3].clamp(min=0.0)
    foot_pos = _get_foot_pos_from_calf(asset, foot_cfg)[..., :2]
    cop_xy = (foot_pos * fz).sum(dim=1) / fz.sum(dim=1).clamp(min=1e-6)

    # Offset
    d = torch.norm(com_xy - cop_xy, dim=1)
    need_correction = torch.clamp(d - offset_threshold, min=0.0)
    correction_gate = torch.tanh(need_correction * 10.0)

    # Term 1: rear-leg activity when off-balance
    rear_joint_vel = asset.data.joint_vel[:, rear_joint_cfg.joint_ids]
    rear_vel_norm = torch.norm(rear_joint_vel, dim=1)
    activity = correction_gate * torch.tanh(rear_vel_norm * 0.1)

    # Term 2: offset is shrinking
    if not hasattr(env, "_prev_reactive_d"):
        env._prev_reactive_d = d.clone()
    reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
        env._prev_reactive_d[reset_ids] = d[reset_ids]
    delta_d = env._prev_reactive_d - d
    env._prev_reactive_d = d.clone()
    improving = (delta_d.clamp(min=0.0) / (offset_threshold + 1e-6)).clamp(max=1.0)

    reward = activity + 0.5 * improving

    # Gate: height
    h = asset.data.root_pos_w[:, 2]
    height_gate = torch.sigmoid((h - min_height) / 0.02)

    # Gate: front feet off ground
    front_forces = contact_sensor.data.net_forces_w_history[:, -1, front_contact_cfg.body_ids]
    front_norms = torch.norm(front_forces, dim=-1)
    front_max = front_norms.max(dim=1)[0]
    front_gate = (front_max < max_front_contact).float()

    return reward * height_gate * front_gate

def rear_stand_alive(
    env: ManagerBasedRLEnv,
    max_front_contact: float = 1.0,
    min_rear_contact: float = 1.0,
    min_height: float = 0.45,
    target_pitch_deg: float = 80.0,     # 新增
    pitch_sigma_deg: float = 10.0,      # 新增
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
    front_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
) -> torch.Tensor:
    """Per-step alive reward: 1.0 when rear feet on ground, front feet off, height above threshold.
    The policy maximizes cumulative reward by staying in this state as long as possible,
    naturally learning to move rear legs for balance without explicit movement rewards."""
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene[contact_cfg.name]

    # Rear feet: at least one must be on ground
    rear_forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]
    rear_norms = torch.norm(rear_forces, dim=-1)
    rear_on = (rear_norms > min_rear_contact).any(dim=1).float()

    # Front feet: all must be off ground
    front_forces = contact_sensor.data.net_forces_w_history[:, -1, front_contact_cfg.body_ids]
    front_norms = torch.norm(front_forces, dim=-1)
    front_max = front_norms.max(dim=1)[0]
    front_off = (front_max < max_front_contact).float()

    # Height gate (soft sigmoid instead of hard threshold)
    h = asset.data.root_pos_w[:, 2]
    height_gate = torch.sigmoid((h - min_height) / 0.02)

    g = asset.data.projected_gravity_b
    pitch = torch.atan2(-g[:, 0], -g[:, 2])
    target_pitch_rad = target_pitch_deg * 3.14159 / 180.0
    pitch_sigma_rad = pitch_sigma_deg * 3.14159 / 180.0
    pitch_gate = torch.exp(-((pitch - target_pitch_rad) / pitch_sigma_rad) ** 2)

    return rear_on * front_off * height_gate * pitch_gate

def rear_stand_duration(
    env: ManagerBasedRLEnv,
    target_height: float = 0.65,
    target_pitch_deg: float = 80.0,
    height_tolerance: float = 0.05,       # 对称容差：h ∈ [0.60, 0.70]
    pitch_tolerance_deg: float = 10.0,    # 对称容差：pitch ∈ [70°, 90°]
    alpha: float = 3.0,                   # 饱和 bonus，峰值 = 1 + α = 4.0
    tau: float = 2.0,                     # 时间常数（秒），~2τ 后接近饱和
    timer_decay: float = 0.95,            # 违反时计时器每步衰减率
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Duration-based reward for maintaining (target_height, target_pitch_deg).

    Robot is considered "in the balance zone" when BOTH:
        |h - target_height|          <= height_tolerance
        |pitch_deg - target_pitch_deg| <= pitch_tolerance_deg

    Timer dynamics:
        in zone:    timer += dt          (accumulates continuously)
        out of zone: timer *= timer_decay (soft decay, no hard reset)

    Reward (only while in zone):
        r_t = 1 + alpha * (1 - exp(-timer / tau))

    Out of zone: r_t = 0 (but timer is preserved via decay, so brief
    excursions—such as a stepping transient—don't wipe accumulated time).

    Pitch formula is IDENTICAL to `height_maintenance`:
        g = projected_gravity_b
        pitch = atan2(-g[:, 0], -g[:, 2])

    维持 (0.65, 80°) 的时长奖励。
    - 进入窗口：计时器每物理步累积 dt，奖励按饱和指数增长。
    - 短暂跳出窗口（迈腿瞬间）：计时器软衰减而不归零，回来仍能拿到接近之前的奖励。
    - 长期不在窗口：计时器逐步衰减到 0。
    """
    asset = env.scene[asset_cfg.name]

    # --- Height (identical to height_maintenance) ---
    h = asset.data.root_pos_w[:, 2]

    # --- Pitch (identical to height_maintenance) ---
    g = asset.data.projected_gravity_b
    pitch = torch.atan2(-g[:, 0], -g[:, 2])
    pitch_deg = pitch * 180.0 / 3.14159

    # --- Hard symmetric-tolerance window ---
    height_ok = (h >= target_height - height_tolerance) & \
                (h <= target_height + height_tolerance)
    pitch_ok = (pitch_deg >= target_pitch_deg - pitch_tolerance_deg) & \
               (pitch_deg <= target_pitch_deg + pitch_tolerance_deg)
    in_zone = height_ok & pitch_ok

    # --- Persistent timer with exponential decay on violation ---
    if not hasattr(env, "_rsd_timer"):
        env._rsd_timer = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.float32
        )
    reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
        env._rsd_timer[reset_ids] = 0.0

    dt = env.physics_dt * env.cfg.decimation
    env._rsd_timer = torch.where(
        in_zone,
        env._rsd_timer + dt,
        env._rsd_timer * timer_decay,
    )

    # --- Saturated-exponential reward, only when in zone ---
    t = env._rsd_timer
    bonus = 1.0 + alpha * (1.0 - torch.exp(-t / tau))
    reward = torch.where(in_zone, bonus, torch.zeros_like(bonus))

    return reward

def front_leg_posture_quadratic(
    env: ManagerBasedRLEnv,
    target_angles: list,
    scale: float = 5.0,
    min_height: float = 0.55,
    max_front_contact: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    front_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
) -> torch.Tensor:
    """Quadratic reward: 1 / (1 + scale * error^2). Peak at target, slow decrease away."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    targets = torch.tensor(target_angles, device=joint_pos.device).unsqueeze(0)

    error_sq = (joint_pos - targets) ** 2
    reward_per_joint = 1.0 / (1.0 + scale * error_sq)
    reward = reward_per_joint.mean(dim=1)

    h = asset.data.root_pos_w[:, 2]
    height_gate = torch.sigmoid((h - min_height) / 0.02)

    contact_sensor = env.scene[front_contact_cfg.name]
    front_forces = contact_sensor.data.net_forces_w_history[:, -1, front_contact_cfg.body_ids]
    front_norms = torch.norm(front_forces, dim=-1)
    front_max = front_norms.max(dim=1)[0]
    front_gate = (front_max < max_front_contact).float()

    return reward * height_gate * front_gate

def rear_joint_acc_penalty(
    env: ManagerBasedRLEnv,
    limit: float = 100.0,
    k: float = 0.001,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint acceleration exceeding limit. Returns 0 when below limit, increases toward 1 above."""
    asset = env.scene[asset_cfg.name]
    acc_sq = torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids])
    mean_acc = torch.sqrt(acc_sq.mean(dim=1))
    excess = (mean_acc - limit).clamp(min=0.0)
    return 1.0 - torch.exp(-k * excess)

def dynamic_balance_cp_switch(
    env: ManagerBasedRLEnv,
    # --- 几何 / 物理 ---
    foot_radius: float = 0.04,
    velocity_alpha: float = 1.0,
    # --- Mode A (inside): 静态稳定 权重 ---
    w_margin: float = 1.0,
    w_com_err_in: float = 0.8,
    w_vel_in: float = 0.5,
    w_angvel_in: float = 0.5,
    # --- Mode A: 形状参数 ---
    k_margin: float = 2.0,
    k_com_err: float = 3.0,
    sigma_v: float = 0.3,
    sigma_w: float = 1.5,
    # --- Mode B (outside): 迈步恢复 权重 ---
    w_outside_penalty: float = 1.5,
    w_cp_inward: float = 1.5,
    w_com_err_rec: float = 1.0,
    w_vel_rec: float = 0.7,
    w_angvel_rec: float = 0.5,
    w_reentry_bonus: float = 3.0,
    # --- Mode B: 特征速率（用来把一阶差分归一化到 [0,1]）---
    rate_d_norm: float = 1.0,    # 无量纲 / s
    rate_com_err: float = 1.0,   # 无量纲 / s
    rate_v: float = 0.5,         # m/s²
    rate_w: float = 2.0,         # rad/s²
    # --- Gates ---
    min_height: float = 0.55,
    max_front_contact: float = 1.0,
    # --- Entity configs ---
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["RL_calf", "RR_calf"]),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
    front_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
) -> torch.Tensor:
    """Dynamic balance reward with CP-based mode switching.

    Mode A (CP inside support region): reward static stabilization.
      - CP margin inside the anisotropic support ellipse
      - CoM close to support midpoint (anisotropic)
      - small CoM velocity
      - small roll/pitch angular velocity
    Mode B (CP outside support region): reward stepping-recovery behavior.
      - Always-on penalty proportional to how far CP is outside
      - Delta rewards (improvement) normalized by characteristic rates
      - Strong bonus when CP re-enters the support region

    模式 A（CP 在支撑区域内）：奖励原地稳定。
    模式 B（CP 超出支撑区域）：持续惩罚 "离得远" + 奖励 "正在变好" + 再入区 bonus。
    """
    asset = env.scene[asset_cfg.name]
    dt = env.physics_dt * env.cfg.decimation

    # =====================================================================
    # 1. Whole-body CoM (XY + height)
    # =====================================================================
    body_com = asset.data.body_com_pos_w                              # (N, B, 3)
    body_mass = asset.data.default_mass.to(body_com.device)           # (N, B)
    total_mass = body_mass.sum(dim=1, keepdim=True)
    com_world = (body_com * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass
    com_xy = com_world[:, :2]
    h_com = com_world[:, 2].clamp(min=0.3)

    # =====================================================================
    # 2. Capture Point   cp = com + α · v / ω,   ω = sqrt(g / h)
    # =====================================================================
    v_com = asset.data.root_lin_vel_w[:, :2]
    omega = torch.sqrt(9.81 / h_com)
    cp_xy = com_xy + velocity_alpha * v_com / omega.unsqueeze(-1)

    # =====================================================================
    # 3. Support region from the two rear feet (anisotropic ellipse)
    # =====================================================================
    foot_pos = _get_foot_pos_from_calf(asset, foot_cfg)               # (N, 2, 3)
    foot_xy = foot_pos[..., :2]                                       # (N, 2, 2)
    foot_L = foot_xy[:, 0, :]
    foot_R = foot_xy[:, 1, :]
    mid_xy = 0.5 * (foot_L + foot_R)
    support_vec = foot_R - foot_L
    support_len = torch.norm(support_vec, dim=1).clamp(min=1e-6)
    support_dir = support_vec / support_len.unsqueeze(-1)             # lateral axis
    half_lateral = (support_len * 0.5).clamp(min=0.02)
    half_sagittal = torch.full_like(half_lateral, foot_radius)

    # --- CP relative to support region (anisotropic normalized) ---
    cp_off = cp_xy - mid_xy
    cp_lat = (cp_off * support_dir).sum(dim=1)
    cp_sag_vec = cp_off - cp_lat.unsqueeze(-1) * support_dir
    cp_sag = torch.norm(cp_sag_vec, dim=1)
    d_norm = torch.sqrt(
        (cp_lat / half_lateral) ** 2 + (cp_sag / half_sagittal) ** 2
    )                                                                 # <1 inside, >=1 outside

    # --- CoM relative to support midpoint (same anisotropy) ---
    com_off = com_xy - mid_xy
    com_lat = (com_off * support_dir).sum(dim=1)
    com_sag_vec = com_off - com_lat.unsqueeze(-1) * support_dir
    com_sag = torch.norm(com_sag_vec, dim=1)
    com_err_norm = torch.sqrt(
        (com_lat / half_lateral) ** 2 + (com_sag / half_sagittal) ** 2
    )

    # --- Velocity / roll-pitch angular velocity (no yaw) ---
    v_norm = torch.norm(v_com, dim=1)
    ang_vel_b = asset.data.root_ang_vel_b                             # (N, 3)
    w_rp_norm = torch.norm(ang_vel_b[:, :2], dim=1)                   # roll + pitch only

    # =====================================================================
    # 4. Persistent state for first-order differences + re-entry detection
    # =====================================================================
    inside_mask = (d_norm < 1.0).float()

    if not hasattr(env, "_dbcp_prev_d_norm"):
        env._dbcp_prev_d_norm   = d_norm.clone()
        env._dbcp_prev_com_err  = com_err_norm.clone()
        env._dbcp_prev_v        = v_norm.clone()
        env._dbcp_prev_w        = w_rp_norm.clone()
        env._dbcp_prev_inside   = inside_mask.clone()

    reset_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    if reset_ids.numel() > 0:
        env._dbcp_prev_d_norm[reset_ids]  = d_norm[reset_ids]
        env._dbcp_prev_com_err[reset_ids] = com_err_norm[reset_ids]
        env._dbcp_prev_v[reset_ids]       = v_norm[reset_ids]
        env._dbcp_prev_w[reset_ids]       = w_rp_norm[reset_ids]
        env._dbcp_prev_inside[reset_ids]  = inside_mask[reset_ids]

    delta_d_norm  = env._dbcp_prev_d_norm  - d_norm                   # +: CP 向内走
    delta_com_err = env._dbcp_prev_com_err - com_err_norm             # +: CoM 偏差减小
    delta_v       = env._dbcp_prev_v       - v_norm                   # +: 线速度下降
    delta_w_rp    = env._dbcp_prev_w       - w_rp_norm                # +: 角速度下降
    reentry_event = ((env._dbcp_prev_inside < 0.5) & (inside_mask > 0.5)).float()

    env._dbcp_prev_d_norm  = d_norm.clone()
    env._dbcp_prev_com_err = com_err_norm.clone()
    env._dbcp_prev_v       = v_norm.clone()
    env._dbcp_prev_w       = w_rp_norm.clone()
    env._dbcp_prev_inside  = inside_mask.clone()

    # =====================================================================
    # 5. Mode A (inside): static stability
    # =====================================================================
    r_margin     = torch.exp(-k_margin * d_norm ** 2) * (1.0 - d_norm).clamp(min=0.0)
    r_com_err_in = torch.exp(-k_com_err * com_err_norm ** 2)
    r_vel_in     = torch.exp(-(v_norm / sigma_v) ** 2)
    r_angvel_in  = torch.exp(-(w_rp_norm / sigma_w) ** 2)

    reward_inside = (
        w_margin       * r_margin
        + w_com_err_in * r_com_err_in
        + w_vel_in     * r_vel_in
        + w_angvel_in  * r_angvel_in
    )

    # =====================================================================
    # 6. Mode B (outside): persistent penalty + improvement reward + re-entry
    # =====================================================================
    # 绝对惩罚：越深越痛，单调递增 (0 at boundary, -1 deep outside)
    r_absolute_penalty = -torch.tanh(0.7 * (d_norm - 1.0).clamp(min=0.0))

    # 改善奖励：用特征速率归一化 + 线性 clamp (梯度更干净、不易饱和)
    r_cp_inward_rec = (delta_d_norm  / dt / rate_d_norm ).clamp(min=0.0, max=1.0)
    r_com_err_rec   = (delta_com_err / dt / rate_com_err).clamp(min=0.0, max=1.0)
    r_vel_rec       = (delta_v       / dt / rate_v      ).clamp(min=0.0, max=1.0)
    r_angvel_rec    = (delta_w_rp    / dt / rate_w      ).clamp(min=0.0, max=1.0)

    reward_outside = (
        w_outside_penalty * r_absolute_penalty
        + w_cp_inward     * r_cp_inward_rec
        + w_com_err_rec   * r_com_err_rec
        + w_vel_rec       * r_vel_rec
        + w_angvel_rec    * r_angvel_rec
        + w_reentry_bonus * reentry_event
    )

    # ---- Probe cache for TensorBoard logging (read by zero-weight probe terms) ----
    env._dbcp_log_d_norm            = d_norm.detach()
    env._dbcp_log_inside_mask       = inside_mask.detach()
    env._dbcp_log_absolute_penalty  = r_absolute_penalty.detach()
    env._dbcp_log_reentry_event     = reentry_event.detach()

    # =====================================================================
    # 7. Hard mode switch
    # =====================================================================
    reward = inside_mask * reward_inside + (1.0 - inside_mask) * reward_outside

    # =====================================================================
    # 8. Gates: height + front feet off + at least one rear contact
    # =====================================================================
    contact_sensor = env.scene[contact_cfg.name]
    rear_forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]
    fz = rear_forces[..., 2].clamp(min=0.0)
    rear_any_contact = (fz > 1.0).any(dim=1).float()

    front_forces = contact_sensor.data.net_forces_w_history[:, -1, front_contact_cfg.body_ids]
    front_norms = torch.norm(front_forces, dim=-1)
    front_gate = (front_norms.max(dim=1)[0] < max_front_contact).float()

    h = asset.data.root_pos_w[:, 2]
    height_gate = torch.sigmoid((h - min_height) / 0.02)

    return reward * height_gate * front_gate * rear_any_contact

def base_height_task(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base height reward (Table I):  r = -(z - z^c)^2

    A pure quadratic penalty (<= 0). Maximum (= 0) is achieved when the base
    height z equals the target z^c; the reward decreases quadratically with
    the height error.

    Args:
        target_height: Target base height z^c, in meters.
        asset_cfg: Robot asset config.
    """
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    return torch.exp(-(h - target_height) ** 2)


def base_pitch_task(
    env: ManagerBasedRLEnv,
    target_pitch_deg: float = 90.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base pitch reward (Table I):  r = -cos(p^c - p)

    Pitch is extracted from the projected gravity in the body frame, using the
    SAME convention as `height_maintenance`:
        pitch = atan2(-g_x, -g_z)
    In this convention:
        - Flat quadrupedal stance  => pitch =  0 deg
        - Fully upright rearing    => pitch = +90 deg
    Therefore `target_pitch_deg` must be POSITIVE (e.g. 80-90 deg).

    Note:
        The formula follows Table I literally. If you prefer the "pure
        penalty" form (max = 0 at target, min = -2 at opposite), replace the
        return statement with:
            return torch.cos(target_pitch_rad - pitch) - 1.0
        Both forms share the same gradient direction in RL.

    Args:
        target_pitch_deg: Target pitch angle in degrees (positive = nose up).
        asset_cfg: Robot asset config.
    """
    asset = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    pitch = torch.atan2(-g[:, 0], -g[:, 2])
    target_pitch_rad = target_pitch_deg * math.pi / 180.0
    return torch.exp(-torch.cos(target_pitch_rad - pitch))


def upright_balance(
    env: ManagerBasedRLEnv,
    sigma_vz: float = 0.25,
    sigma_pitch_rate: float = 0.25,
    upright_pitch_deg: float = 60.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Upright balance reward (Table I):
        r = exp(-v_z^2 / sigma_vz) + exp(-p_dot^2 / sigma_pitch_rate)
            if upright, else 0

    Once the robot has reached a sufficiently upright posture, this term
    encourages minimizing:
      - v_z:     base vertical (world-z) linear velocity
      - p_dot:   base pitch angular velocity (body-frame y-axis component)

    The "is_upright" gate is based on the pitch angle extracted from projected
    gravity, using the same convention as `base_pitch_task`
    (upright corresponds to +90 deg).

    Args:
        sigma_vz: Gaussian scale for the vertical-velocity term.
        sigma_pitch_rate: Gaussian scale for the pitch-rate term.
        upright_pitch_deg: Threshold (deg) above which the reward becomes active.
        asset_cfg: Robot asset config.
    """
    asset = env.scene[asset_cfg.name]

    # Vertical (world-z) linear velocity of the base.
    v_z = asset.data.root_lin_vel_w[:, 2]
    # Pitch angular velocity: y-axis component of base angular velocity in body frame.
    pitch_rate = asset.data.root_ang_vel_b[:, 1]

    reward = (
        torch.exp(-(v_z * v_z) / sigma_vz)
        + torch.exp(-(pitch_rate * pitch_rate) / sigma_pitch_rate)
    )

    # Upright gate: pitch >= threshold (positive = nose up).
    g = asset.data.projected_gravity_b
    pitch = torch.atan2(-g[:, 0], -g[:, 2])
    pitch_deg = pitch * 180.0 / math.pi
    is_upright = (pitch_deg >= upright_pitch_deg).float()

    return reward * is_upright


def support_polygon(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["RL_calf", "RR_calf"]
    ),
) -> torch.Tensor:
    """Support polygon reward (Table I):
        r = -|v_x^c|^2 * (pi/2 - |arctan(dx_b / dz_b)|)^2
            if arctan(dx_b / dz_b) * v_x^c < 0, else 0

    Geometry:
        Let delta = base_pos - support_pos, expressed in the BODY frame,
        i.e. (dx_b, dy_b, dz_b). `support_pos` is the average world position
        of the two rear feet (approximating the support polygon center).
        `arctan(dx_b / dz_b)` is the angle between the support-to-base line
        and the body's vertical axis:
            0       : CoM directly above support (in body frame)
            > 0     : CoM leaning forward of support (in body frame)
            < 0     : CoM leaning backward of support (in body frame)
            +/- pi/2: base at the same body-z height as support (penalty = 0)

    Activation:
        The penalty is applied only when the lean direction CONFLICTS with the
        commanded forward velocity, i.e. angle * v_x^c < 0. Otherwise 0.

    Notes:
        - delta is transformed into the body frame by applying the CONJUGATE
          (= inverse, since the quaternion is unit) of the base quaternion.
          Using the world frame would be WRONG: when fully upright, dx_w ~= 0
          but dx_b ~= L, giving completely opposite angle values.
        - `_get_foot_pos_from_calf` already applies the 0.25 m offset along
          the calf's local -z axis (rotated into world frame via calf quat),
          so `foot_pos_w` is the true foot contact point, not the calf link
          center. See `rewards.py` top-of-file helper for details.
        - `torch.atan2` is well-defined at (0, 0) (returns 0); no clamp is
          needed. Clamping dz would destroy its sign and corrupt the angle.

    Args:
        command_name: Name of the velocity command term.
        asset_cfg: Robot asset config.
        foot_cfg: Config for the support feet (default: the two rear calves).
    """
    asset = env.scene[asset_cfg.name]

    # Commanded forward velocity v_x^c.
    v_x_cmd = env.command_manager.get_command(command_name)[:, 0]

    # Base pose and support center (average of two rear-foot contact points).
    base_pos_w = asset.data.root_pos_w                     # (N, 3)
    base_quat_w = asset.data.root_quat_w                   # (N, 4) wxyz, unit
    foot_pos_w = _get_foot_pos_from_calf(asset, foot_cfg)  # (N, 2, 3)
    support_pos_w = foot_pos_w.mean(dim=1)                 # (N, 3)

    # World-frame offset from support to base.
    delta_w = base_pos_w - support_pos_w                   # (N, 3)

    # Rotate delta into the body frame using the conjugate quaternion:
    # q* = (w, -x, -y, -z). For a unit quaternion this equals the inverse.
    base_quat_inv = torch.stack([
        base_quat_w[:, 0],
        -base_quat_w[:, 1],
        -base_quat_w[:, 2],
        -base_quat_w[:, 3],
    ], dim=-1)
    delta_b = _quat_apply_wxyz(base_quat_inv, delta_w)     # (N, 3)

    dx_b = delta_b[:, 0]
    dz_b = delta_b[:, 2]

    # atan2 natively handles (0, 0) -> 0; do NOT clamp dz, that breaks the sign.
    angle = torch.atan2(dx_b, dz_b)

    # Activate only when the lean direction opposes the commanded velocity.
    active = (angle * v_x_cmd < 0).float()

    # Quadratic penalty; magnitude grows with command speed and misalignment.
    penalty = -(v_x_cmd ** 2) * (math.pi / 2 - torch.abs(angle)) ** 2

    return penalty * active

def joint_acceleration_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint acceleration penalty:  r = -|q_ddot|^2
    Penalizes large joint accelerations to encourage smooth motion.
    Use a very small weight (e.g. 5e-9) since |q_ddot|^2 summed over
    12 joints can reach 1e6 ~ 1e8 (rad/s^2)^2.
    Args:
        asset_cfg: Robot asset config.
    """
    asset = env.scene[asset_cfg.name]
    return -torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)