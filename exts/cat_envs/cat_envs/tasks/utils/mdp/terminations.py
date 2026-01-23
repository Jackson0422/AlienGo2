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

def base_height_below(
    env: ManagerBasedRLEnv,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when base height is below threshold.
    
    Args:
        env: The environment.
        min_height: Minimum allowed height.
        asset_cfg: The asset configuration.
    
    Returns:
        Boolean tensor indicating whether height is below threshold.
    """
    robot = env.scene[asset_cfg.name]
    return robot.data.root_pos_w[:, 2] < min_height

class BaseHeightBelowConsecutive:
    """检测连续 k 帧高度低于阈值时触发终止。
    
    使用环境级别的计数器来跟踪连续低高度帧数。
    """
    
    def __init__(self):
        self.counter: torch.Tensor | None = None
        self.num_envs: int = 0
    
    def __call__(
        self,
        env: ManagerBasedRLEnv,
        min_height: float,
        consecutive_frames: int = 5,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """检测连续 k 帧高度低于阈值。
        
        Args:
            env: 环境实例
            min_height: 最小高度阈值
            consecutive_frames: 需要连续满足条件的帧数
            asset_cfg: 资产配置
            
        Returns:
            布尔张量，指示是否应该终止
        """
        # 初始化计数器
        if self.counter is None or self.num_envs != env.num_envs:
            self.num_envs = env.num_envs
            self.counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        
        # 获取当前高度
        robot = env.scene[asset_cfg.name]
        current_height = robot.data.root_pos_w[:, 2]
        
        # 检查是否低于阈值
        is_below = current_height < min_height
        
        # 更新计数器
        # 如果低于阈值，计数器+1；否则重置为0
        self.counter = torch.where(is_below, self.counter + 1, torch.zeros_like(self.counter))
        
        # 当计数器达到阈值时触发终止
        return self.counter >= consecutive_frames
    
    def reset(self, env_ids: torch.Tensor | None = None):
        """重置指定环境的计数器。"""
        if self.counter is not None:
            if env_ids is None:
                self.counter.zero_()
            else:
                self.counter[env_ids] = 0


# 创建全局实例
_base_height_checker_global = BaseHeightBelowConsecutive()


# 包装函数：Isaac Lab 配置系统需要函数而不是类实例
def base_height_below_consecutive(
    env: ManagerBasedRLEnv,
    min_height: float,
    consecutive_frames: int = 5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when base height is below threshold for consecutive frames.
    
    This is a wrapper function that uses a global instance to track consecutive frames.
    
    Args:
        env: The environment instance.
        min_height: Minimum allowed height in meters.
        consecutive_frames: Number of consecutive frames below threshold to trigger termination.
        asset_cfg: The asset configuration.
    
    Returns:
        Boolean tensor indicating termination condition.
    """
    return _base_height_checker_global(env, min_height, consecutive_frames, asset_cfg)