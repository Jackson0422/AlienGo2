# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upside_down(
    env: ManagerBasedRLEnv,
    limit: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return torch.norm(data.projected_gravity_b[:, :2], dim=1) > limit


def illegal_contact_current_frame(
    env: ManagerBasedRLEnv, 
    threshold: float, 
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Terminate when the contact force on the sensor exceeds the force threshold.
    只检测当前帧，不检测历史帧，避免传感器初始化噪声导致的误触发。
    
    Args:
        env: The environment.
        threshold: The force threshold for contact detection.
        sensor_cfg: The sensor configuration.
    
    Returns:
        Boolean tensor indicating whether contact force exceeds threshold.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    
    # 只使用最后一帧（当前帧），索引=-1，避免历史帧的初始化噪声
    current_frame_forces = net_contact_forces[:, -1, sensor_cfg.body_ids]  # shape: (num_envs, num_bodies, 3)
    
    # 计算力的范数并检查是否超过阈值
    force_norms = torch.norm(current_frame_forces, dim=-1)  # shape: (num_envs, num_bodies)
    
    # 检查任何body是否超过阈值
    return torch.any(force_norms > threshold, dim=1)  # shape: (num_envs,)
