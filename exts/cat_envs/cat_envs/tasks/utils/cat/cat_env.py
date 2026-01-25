# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import os
import json
import torch
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from cat_envs.tasks.utils.cat.constraint_manager import ConstraintManager


class CaTEnv(ManagerBasedRLEnv):
    def load_managers(self):
        """Load the managers for the environment.

        This function is responsible for creating the various managers (action, observation,
        events, etc.) for the environment. Since the managers require access to physics handles,
        they can only be created after the simulator is reset (i.e. played for the first time).

        .. note::
            In case of standalone application (when running simulator from Python), the function is called
            automatically when the class is initialized.

            However, in case of extension mode, the user must call this function manually after the simulator
            is reset. This is because the simulator is only reset when the user calls
            :meth:`SimulationContext.reset_async` and it isn't possible to call async functions in the constructor.

        """
        super().load_managers()
        # prepare the managers
        # -- constraint manager

        if hasattr(self.cfg, "constraints"):
            self.constraint_manager = ConstraintManager(self.cfg.constraints, self)
            print("[INFO] Constraint Manager: ", self.constraint_manager)
        
        # -- termination logging (JSONL + 统计摘要方案)
        self.log_dir = Path(os.path.expanduser("~")) / "termination_logs"
        self.log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 使用JSONL格式（每行一个JSON对象，可追加写入）
        self.log_file = self.log_dir / f"terminations_{timestamp}.jsonl"
        
        # 同时保存一个统计摘要文件
        self.summary_file = self.log_dir / f"summary_{timestamp}.json"
        self.summary_stats = {
            "start_time": timestamp,
            "total_resets": 0,
            "termination_counts": {},
        }
        
        print(f"[INFO] Termination logs will be saved to: {self.log_file}")
        print(f"[INFO] Summary stats will be saved to: {self.summary_file}")
        
        # -- reward tracking initialization
        self.reward_stats = {
            "total_reward": 0.0,
            "reward_term_sums": {},
            "reward_term_counts": {},
            "reward_term_contributions": {},
        }
        
        print(f"[INFO] Reward tracking initialized")

        # -- NaN/Inf debug logging (minimal, only first few occurrences)
        self.nan_log_file = self.log_dir / f"nan_debug_{timestamp}.jsonl"
        self.nan_log_count = 0

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        if self.nan_log_count < 10 and not torch.isfinite(action).all():
            with open(self.nan_log_file, "a") as f:
                f.write(json.dumps({
                    "tag": "action",
                    "step": self.common_step_counter,
                    "min": float(torch.nan_to_num(action, nan=0.0).min().item()),
                    "max": float(torch.nan_to_num(action, nan=0.0).max().item()),
                }) + "\n")
            self.nan_log_count += 1
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if (
                self._sim_step_counter % self.cfg.sim.render_interval == 0
                and is_rendering
            ):
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        
        # -- log termination reasons (优化：JSONL追加 + 统计摘要)
        if self.reset_buf.any():
            num_resets = self.reset_buf.sum().item()
            
            # 只存储统计信息，不存储完整的env_id列表
            termination_stats = {}
            for term_name in self.termination_manager.active_terms:
                term_value = self.termination_manager.get_term(term_name)
                if term_value.any():
                    count = term_value.sum().item()
                    termination_stats[term_name] = count
                    # 更新摘要统计
                    if term_name not in self.summary_stats["termination_counts"]:
                        self.summary_stats["termination_counts"][term_name] = 0
                    self.summary_stats["termination_counts"][term_name] += count
            
            # 计算训练进度
            current_iteration = self.common_step_counter // 24
            training_progress = (current_iteration / 2000) * 100
            
            if training_progress < 20:
                training_stage = "early"
            elif training_progress < 60:
                training_stage = "mid"
            elif training_progress < 90:
                training_stage = "late"
            else:
                training_stage = "final"
            
            # 创建轻量级日志条目（只有统计信息）
            log_entry = {
                "step": self.common_step_counter,
                "iteration": current_iteration,
                "progress": round(training_progress, 2),
                "stage": training_stage,
                "num_resets": num_resets,
                "termination_stats": termination_stats,  # 只存储计数，不存储ID列表
            }
            
            # 追加模式写入JSONL（O(1)操作，无需重写整个文件）
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            # 更新总计数
            self.summary_stats["total_resets"] += num_resets
            
            # 每1000次reset更新一次摘要文件（而不是每100次）
            if self.summary_stats["total_resets"] % 1000 == 0:
                self.summary_stats["last_step"] = self.common_step_counter
                self.summary_stats["last_update"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                with open(self.summary_file, 'w') as f:
                    json.dump(self.summary_stats, f, indent=2)
        
        # -- CaT constraints prob computation
        if hasattr(self.cfg, "constraints"):
            cstr_prob = self.constraint_manager.compute()
            # -- constrained reward computation
            self.reward_buf = torch.clip(
                self.reward_manager.compute(dt=self.step_dt) * (1.0 - cstr_prob),
                min=0.0,
                max=None,
            )
            dones = cstr_prob.clone()
        else:
            self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
            dones = torch.zeros(self.num_envs, device=self.device)

        # -- track reward statistics  
        # Track total reward
        self.reward_stats["total_reward"] += self.reward_buf.sum().item()

        if self.nan_log_count < 10 and not torch.isfinite(self.reward_buf).all():
            with open(self.nan_log_file, "a") as f:
                f.write(json.dumps({
                    "tag": "reward_buf",
                    "step": self.common_step_counter,
                    "min": float(torch.nan_to_num(self.reward_buf, nan=0.0).min().item()),
                    "max": float(torch.nan_to_num(self.reward_buf, nan=0.0).max().item()),
                }) + "\n")
            self.nan_log_count += 1
        
        # Count steps for all environments
        for term_name in self.reward_manager.active_terms:
            if term_name not in self.reward_stats["reward_term_counts"]:
                self.reward_stats["reward_term_counts"][term_name] = 0
            self.reward_stats["reward_term_counts"][term_name] += self.num_envs

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            dones[reset_env_ids] = 1.0
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            # this is needed to make joint positions set from reset events effective
            self.scene.write_data_to_sim()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        if self.nan_log_count < 10:
            if isinstance(self.obs_buf, dict):
                for obs_key, obs_val in self.obs_buf.items():
                    if torch.is_tensor(obs_val) and not torch.isfinite(obs_val).all():
                        with open(self.nan_log_file, "a") as f:
                            f.write(json.dumps({
                                "tag": f"obs_buf:{obs_key}",
                                "step": self.common_step_counter,
                                "min": float(torch.nan_to_num(obs_val, nan=0.0).min().item()),
                                "max": float(torch.nan_to_num(obs_val, nan=0.0).max().item()),
                            }) + "\n")
                        self.nan_log_count += 1
                        if self.nan_log_count >= 10:
                            break
            elif torch.is_tensor(self.obs_buf) and not torch.isfinite(self.obs_buf).all():
                with open(self.nan_log_file, "a") as f:
                    f.write(json.dumps({
                        "tag": "obs_buf",
                        "step": self.common_step_counter,
                        "min": float(torch.nan_to_num(self.obs_buf, nan=0.0).min().item()),
                        "max": float(torch.nan_to_num(self.obs_buf, nan=0.0).max().item()),
                    }) + "\n")
                self.nan_log_count += 1

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, dones, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(
                mode="reset", env_ids=env_ids, global_env_step_count=env_step_count
            )

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        
        # Track reward statistics from episode summaries
        for key, value in info.items():
            # Keys are like "Episode_Reward/track_lin_vel_xy_exp"
            if key.startswith("Episode_Reward/"):
                term_name = key.replace("Episode_Reward/", "")
                if term_name not in self.reward_stats["reward_term_sums"]:
                    self.reward_stats["reward_term_sums"][term_name] = 0.0
                # Add the episode reward sum for the reset environments
                # value is the mean reward across reset environments, multiply by count
                if isinstance(value, torch.Tensor):
                    reward_value = value.item()
                else:
                    reward_value = float(value)
                self.reward_stats["reward_term_sums"][term_name] += reward_value * len(env_ids)
        # -- constraints manager
        if hasattr(self.cfg, "constraints"):
            info = self.constraint_manager.reset(env_ids)
            self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0
    
    def close(self):
        """Save final termination stats and reward stats before closing the environment."""
        if hasattr(self, 'summary_stats'):
            # 保存最终摘要
            self.summary_stats["end_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.summary_stats["final_step"] = self.common_step_counter
            
            # Add reward statistics to summary
            if hasattr(self, 'reward_stats'):
                # Calculate contribution percentage for each reward term
                total_abs_reward = sum(abs(v) for v in self.reward_stats["reward_term_sums"].values())
                if total_abs_reward > 0:
                    for term_name, term_sum in self.reward_stats["reward_term_sums"].items():
                        contribution = (abs(term_sum) / total_abs_reward) * 100
                        self.reward_stats["reward_term_contributions"][term_name] = contribution
                
                self.summary_stats["reward_stats"] = self.reward_stats
            
            with open(self.summary_file, 'w') as f:
                json.dump(self.summary_stats, f, indent=2)
            
            print(f"\n[INFO] Termination logs saved to: {self.log_file}")
            print(f"[INFO] Summary stats saved to: {self.summary_file}")
            
            # Print termination reason summary
            print(f"\n[INFO] Termination Reason Summary:")
            print(f"  Total resets: {self.summary_stats['total_resets']}")
            for reason, count in sorted(self.summary_stats['termination_counts'].items(), 
                                       key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count} occurrences")
            
            # Print reward summary
            if hasattr(self, 'reward_stats'):
                print(f"\n[INFO] Reward Summary:")
                print(f"  Total reward accumulated: {self.reward_stats['total_reward']:.2f}")
                print(f"\n[INFO] Reward Term Contributions:")
                
                # Sort by absolute contribution
                sorted_rewards = sorted(
                    self.reward_stats["reward_term_sums"].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                
                for term_name, term_sum in sorted_rewards:
                    count = self.reward_stats["reward_term_counts"][term_name]
                    contribution = self.reward_stats["reward_term_contributions"].get(term_name, 0.0)
                    avg_reward = term_sum / count if count > 0 else 0.0
                    print(f"  {term_name}:")
                    print(f"    Total: {term_sum:.4f}")
                    print(f"    Triggered: {count} times")
                    print(f"    Average: {avg_reward:.6f}")
                    print(f"    Contribution: {contribution:.2f}%")
        
        super().close()
