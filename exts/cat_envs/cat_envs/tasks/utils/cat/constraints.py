# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

condThe functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_position(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = torch.abs(data.joint_pos[:, asset_cfg.joint_ids]) - limit
    return cstr


def joint_position_when_moving_forward(
    env: ManagerBasedRLEnv,
    limit: float,
    velocity_deadzone: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = (
        torch.abs(data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids])
        - limit
    )
    cstr *= (
        (
            torch.abs(env.command_manager.get_command("base_velocity")[:, 1])
            < velocity_deadzone
        )
        .float()
        .unsqueeze(1)
    )
    return cstr


def joint_torque(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr = torch.abs(data.applied_torque[:, asset_cfg.joint_ids]) - limit
    return cstr


def joint_velocity(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return torch.abs(data.joint_vel[:, asset_cfg.joint_ids]) - limit


def joint_acceleration(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return torch.abs(data.joint_acc[:, asset_cfg.joint_ids]) - limit


def upsidedown(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return data.projected_gravity_b[:, 2] > limit


def contact(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Check if any body has contact force exceeding threshold."""
    contact_sensor = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    current_frame_forces = net_contact_forces[:, -1, asset_cfg.body_ids]
    force_norms = torch.norm(current_frame_forces, dim=-1)
    return torch.any(force_norms > threshold, dim=1)


def base_orientation(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return torch.norm(data.projected_gravity_b[:, :2], dim=1) - limit


def air_time(
    env: ManagerBasedRLEnv,
    limit: float,
    velocity_deadzone: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, asset_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, asset_cfg.body_ids]
    # Like in CaT
    command_more_than_limit = (
        (
            torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
            >= velocity_deadzone
        )
        .float()
        .unsqueeze(1)
    )
    cstr = (limit - last_air_time) * touchdown.float() * command_more_than_limit
    return cstr


def n_foot_contact(
    env: ManagerBasedRLEnv,
    number_of_desired_feet: int,
    min_command_value: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact_cstr = torch.abs(
        (
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, asset_cfg.body_ids], dim=-1
                ),
                dim=1,
            )[0]
            > 1.0
        ).sum(1)
        - number_of_desired_feet
    )
    command_more_than_limit = (
        torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
        > min_command_value
    ).float()
    return contact_cstr * command_more_than_limit


def joint_range(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return (
        torch.abs(data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids])
        - limit
    )


def action_rate(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    return (
        torch.abs(
            env.action_manager._action[:, asset_cfg.joint_ids]
            - env.action_manager._prev_action[:, asset_cfg.joint_ids]
        )
        / env.step_dt
        - limit
    )


def foot_contact_force(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return (
        torch.max(torch.norm(net_contact_forces[:, :, asset_cfg.body_ids], dim=-1), dim=1)[0]
        - limit
    )


def min_base_height(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    return limit - robot.data.root_pos_w[:, 2]


def no_move(
    env: ManagerBasedRLEnv,
    velocity_deadzone: float,
    joint_vel_limit: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    data = env.scene[asset_cfg.name].data
    cstr_nomove = (torch.abs(data.joint_vel[:, asset_cfg.joint_ids]) - joint_vel_limit) * (
        torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
        < velocity_deadzone
    ).float().unsqueeze(1)
    return cstr_nomove

def min_foot_contact(
    env: ManagerBasedRLEnv,
    min_feet: int,
    min_command_value: float,
    asset_cfg: SceneEntityCfg,
    min_height: float = 0.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    contact_sensor = env.scene[asset_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    num_contact = (
        torch.max(
            torch.norm(net_contact_forces[:, :, asset_cfg.body_ids], dim=-1),
            dim=1,
        )[0]
        > 1.0
    ).sum(1)
    
    shortfall = (min_feet - num_contact).clamp(min=0.0) # at least one foot contact /至少有一条腿接触地面
    # shortfall = (num_contact - min_feet).clamp(min=0.0) # at most min_feet feet contact / 最多有 min_feet 条腿接触地面
    
    command_more_than_limit = (
        torch.norm(env.command_manager.get_command("base_velocity")[:, :3], dim=1)
        >= min_command_value
    ).float()

    h = env.scene[robot_cfg.name].data.root_pos_w[:, 2]
    standing = (h > min_height).float()

    return shortfall * command_more_than_limit * standing - (1.0 - standing) * 1.0

def height_below(
    env: ManagerBasedRLEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Constraint violation when base height drops below min_height.
    Returns (min_height - h) clamped to >= 0. Higher value = bigger violation.
    """
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    return (min_height - h).clamp(min=0.0)

def pitch_out_of_range(
    env: ManagerBasedRLEnv,
    upper_limit_deg: float,
    lower_limit_deg: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Constraint violation when pitch is outside [lower_limit, upper_limit].
    Returns the angular deviation (in degrees) from the nearest boundary, clamped >= 0.
    """
    asset = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    sin_pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0)
    pitch_deg = torch.asin(sin_pitch) * (180.0 / torch.pi)

    above = (pitch_deg - upper_limit_deg).clamp(min=0.0)
    below = (lower_limit_deg - pitch_deg).clamp(min=0.0)
    return above + below

def front_leg_pos_when_standing(
    env: ManagerBasedRLEnv,
    target_angles: list[float],
    limit: float,
    min_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize front leg joints deviating from target angles, only when standing.
    Below min_height, constraint is inactive."""
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    targets = torch.tensor(target_angles, device=joint_pos.device).unsqueeze(0)
    violation = torch.abs(joint_pos - targets) - limit

    standing = (h > min_height).float().unsqueeze(1)
    return violation * standing - (1.0 - standing) * 1.0

def base_ang_vel_z_when_standing(
    env: ManagerBasedRLEnv,
    limit: float,
    min_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize yaw rotation only when standing (above min_height).
    Below min_height, constraint is inactive."""
    data = env.scene[asset_cfg.name].data
    h = data.root_pos_w[:, 2]

    violation = torch.abs(data.root_ang_vel_b[:, 2:3]) - limit
    standing = (h > min_height).float().unsqueeze(1)
    return violation * standing - (1.0 - standing) * 1.0

def rear_leg_balance_response(
    env: ManagerBasedRLEnv,
    d_norm_threshold: float = 0.7,
    min_joint_vel: float = 2.0,
    foot_radius: float = 0.04,
    min_height: float = 0.50,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["RL_calf", "RR_calf"]),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
    front_contact_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
    rear_joint_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]),
) -> torch.Tensor:
    from cat_envs.tasks.utils.mdp.rewards import _get_foot_pos_from_calf
    asset = env.scene[asset_cfg.name]

    # --- CoM ---
    body_com = asset.data.body_com_pos_w
    body_mass = asset.data.default_mass.to(body_com.device)
    total_mass = body_mass.sum(dim=1, keepdim=True)
    com_xy = ((body_com * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass)[:, :2]

    # --- CoP ---
    contact_sensor = env.scene[contact_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, -1, contact_cfg.body_ids]
    fz = forces[..., 2:3].clamp(min=0.0)
    foot_pos = _get_foot_pos_from_calf(asset, foot_cfg)
    foot_xy = foot_pos[..., :2]
    cop_xy = (foot_xy * fz).sum(dim=1) / fz.sum(dim=1).clamp(min=1e-6)

    # --- 各向异性 d_norm ---
    foot_L = foot_xy[:, 0, :]
    foot_R = foot_xy[:, 1, :]
    support_vec = foot_R - foot_L
    support_len = torch.norm(support_vec, dim=1)
    support_dir = support_vec / support_len.clamp(min=1e-6).unsqueeze(-1)

    offset = com_xy - cop_xy
    lateral  = (offset * support_dir).sum(dim=1)
    sagittal_vec = offset - lateral.unsqueeze(-1) * support_dir
    sagittal = torch.norm(sagittal_vec, dim=1)

    d_norm = torch.sqrt(
        (lateral / (support_len / 2.0).clamp(min=0.02)) ** 2 +
        (sagittal / foot_radius) ** 2
    )

    # --- 违反量：d_norm 超过阈值 且 后腿不动 ---
    off_balance = (d_norm - d_norm_threshold).clamp(min=0.0)          # 超出安全域的程度
    rear_vel = asset.data.joint_vel[:, rear_joint_cfg.joint_ids]      # (N, 6)
    rear_vel_norm = torch.norm(rear_vel, dim=1)                        # (N,)
    vel_insufficient = (min_joint_vel - rear_vel_norm).clamp(min=0.0) / min_joint_vel  # [0,1]

    violation = off_balance * vel_insufficient

    # --- Gates ---
    h = asset.data.root_pos_w[:, 2]
    standing = (h > min_height).float()
    front_forces = contact_sensor.data.net_forces_w_history[:, -1, front_contact_cfg.body_ids]
    front_off = (torch.norm(front_forces, dim=-1).max(dim=1)[0] < 1.0).float()
    rear_on = (fz.squeeze(-1) > 1.0).any(dim=1).float()

    return violation * standing * front_off * rear_on

def knee_height(
    env: ManagerBasedRLEnv,
    min_z: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Trigger when any calf-link origin (= knee joint position in world frame)
    drops below `min_z` meters above ground.

    For Aliengo (calf length L = 0.25 m), setting `min_z = L / sqrt(2) ≈ 0.1768 m`
    geometrically forbids the calf from tilting more than 45° from vertical
    (i.e. forbids kneeling / squatting on the back of the calf).
    """
    robot = env.scene[asset_cfg.name]
    knee_z = robot.data.body_pos_w[:, asset_cfg.body_ids, 2]
    return torch.any(knee_z < min_z, dim=1)


def air_time_when_upright(
    env: ManagerBasedRLEnv,
    limit: float,
    upright_pitch_deg: float,
    min_height: float,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Air-time constraint that is only active once the robot is upright.

    Returns (limit - last_air_time) on the touchdown frame, identical in
    spirit to :func:`air_time` (CaT), but gated by:
        pitch_deg >= upright_pitch_deg   AND   base_z >= min_height

    Below the gate the term is identically 0, so it does not interfere
    with the rear-up learning phase. Above the gate, a touchdown whose
    preceding air time is shorter than ``limit`` produces a positive
    violation, which the CaT curriculum maps to a higher termination
    probability.

    Pitch convention (consistent with reward functions in this repo):
        pitch_deg = atan2(-g_x, -g_z) * 180/pi
        upright (fully reared) corresponds to pitch_deg = +90.

    Args:
        limit: Minimum desired air time per step (seconds).
        upright_pitch_deg: Pitch threshold above which the constraint activates.
        min_height: Base z threshold above which the constraint activates.
        asset_cfg: Contact sensor config selecting the feet to monitor
            (typically the two rear calves on Aliengo).
        robot_cfg: Robot articulation config (used for pitch + base height).
    """
    contact_sensor = env.scene[asset_cfg.name]
    touchdown = contact_sensor.compute_first_contact(env.step_dt)[:, asset_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, asset_cfg.body_ids]

    asset = env.scene[robot_cfg.name]
    g = asset.data.projected_gravity_b
    pitch_deg = torch.atan2(-g[:, 0], -g[:, 2]) * (180.0 / torch.pi)
    upright_gate = (
        (pitch_deg >= upright_pitch_deg)
        & (asset.data.root_pos_w[:, 2] >= min_height)
    ).float().unsqueeze(1)

    return (limit - last_air_time) * touchdown.float() * upright_gate

def base_xy_drift(
    env,
    free_half_extent: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """CaT 越界量 = max(0, |x|-r) + max(0, |y|-r)，单位米。
    自由区内整张返回 0，自由区外越远值越大。"""
    robot = env.scene[asset_cfg.name]
    pos_xy = robot.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    viol = (pos_xy.abs() - free_half_extent).clamp(min=0.0)  # (N, 2)
    return viol.sum(dim=1, keepdim=True)                      # (N, 1)