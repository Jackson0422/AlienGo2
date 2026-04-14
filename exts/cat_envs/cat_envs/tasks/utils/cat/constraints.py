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