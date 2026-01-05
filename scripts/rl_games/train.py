# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
import torch
from datetime import datetime

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import TerminationTermCfg as DoneTerm
from cat_envs.tasks.utils.cat.manager_constraint_cfg import ConstraintTermCfg as ConstraintTerm
from isaaclab.managers import SceneEntityCfg 
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rl_games import RlGamesGpuEnv
from cat_envs.tasks.utils.rl_games.rl_games import RlGamesVecEnvWrapperCaT as RlGamesVecEnvWrapper
from cat_envs.tasks.utils.rl_games.build_alg_runner import build_alg_runner
from cat_envs.tasks.utils.cat.manager_constraint_cfg import ConstraintTermCfg as ConstraintTerm
from isaaclab.managers import SceneEntityCfg
import cat_envs.tasks  # noqa: F401
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import cat_envs.tasks.utils.cat.constraints as constraints 

from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1 or (args_cli.seed is None and agent_cfg["params"]["seed"] == -1):
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = (args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"])

    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg["params"]["config"]["max_epochs"]
    )
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    # multi-gpu training config
    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs
    log_dir = agent_cfg["params"]["config"].get("full_experiment_name", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    # set directory into agent config
    # logging directory path: <train_dir>/<full_experiment_name>
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    if "env" in agent_cfg["params"]:
        clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
        clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    else:
        clip_obs = math.inf
        clip_actions = math.inf

    # configure terminations - 只保留 time_out
    # env_cfg.terminations = type(env_cfg.terminations)( time_out=DoneTerm(func=mdp.time_out, time_out=True) )
    # print("[DEBUG] terminations cfg AFTER override:", env_cfg.terminations)

    # print("[DEBUG] constraints BEFORE:", env_cfg.constraints) 
    # for k in list(vars(env_cfg.constraints).keys()): 
    #     setattr(env_cfg.constraints, k, None) 
    
    
    # # 禁掉 curriculum 
    # env_cfg.constraints.joint_torque = ConstraintTerm( func=constraints.joint_torque, max_p=0.0, params={ "limit": 1e9, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]) },)
    
    # for k in list(vars(env_cfg.curriculum).keys()):
    #     setattr(env_cfg.curriculum, k, None)
        
    # print("[DEBUG] constraints AFTER:", env_cfg.constraints)
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ================== DEBUG: 检查初始高度是否正确 ==================
    print("\n" + "="*70)
    print("[DEBUG] 正在检查机器人初始姿态...")
    obs, info = env.reset()
    # 使用 unwrapped 访问底层 Isaac Lab 环境
    root_z = env.unwrapped.scene["robot"].data.root_state_w[:, 2]
    print(f"[DEBUG] Reset后的机器人高度:")
    print(f"        平均: {root_z.mean().item():.4f}m")
    print(f"        最小: {root_z.min().item():.4f}m")
    print(f"        最大: {root_z.max().item():.4f}m")
    
    # 执行一步物理模拟，看是否会掉落
    sample_action = torch.from_numpy(env.action_space.sample()).to(env.unwrapped.device)
    obs, rew, terminated, truncated, info = env.step(sample_action)
    root_z2 = env.unwrapped.scene["robot"].data.root_state_w[:, 2]
    height_change = (root_z2.mean() - root_z.mean()).item()
    print(f"[DEBUG] 第1步后的机器人高度:")
    print(f"        平均: {root_z2.mean().item():.4f}m")
    print(f"        最小: {root_z2.min().item():.4f}m")
    print(f"        最大: {root_z2.max().item():.4f}m")
    print(f"[DEBUG] 高度变化: {height_change:+.4f}m")
    
    # 检查有多少环境立即触发了termination
    done = (terminated > 0.5) | (truncated > 0.5)
    if done.any():
        num_terminated = done.sum().item()
        total_envs = done.numel()
        percent = (num_terminated / total_envs) * 100
        print(f"[DEBUG] ⚠️  第1步就有 {num_terminated}/{total_envs} ({percent:.1f}%) 个环境终止了！")
        
        if height_change < -0.02:
            print(f"[DEBUG] ⚠️  机器人快速下落 ({height_change:.4f}m)，姿态可能不稳定！")
    else:
        print(f"[DEBUG] ✓ 没有环境在第1步终止")
        if abs(height_change) < 0.01:
            print(f"[DEBUG] ✓ 高度变化很小，姿态稳定！")
    print("="*70 + "\n")

    print("[DEBUG] 检查接触传感器Body名称...")
    
    import re
    contact_sensor = env.unwrapped.scene["contact_forces"]
    
    print(f"\n[INFO] 所有可用的Body名称 (共{len(contact_sensor.body_names)}个):")
    for idx, name in enumerate(contact_sensor.body_names):
        print(f"  [{idx:2d}] {name}")
    
    # 检查 "base" 是否存在
    print("\n[检查1] body_names=['base']:")
    if "base" in contact_sensor.body_names:
        print("  ✓ 找到 'base'")
    else:
        print("  ✗ 未找到 'base'")
        base_candidates = [name for name in contact_sensor.body_names 
                          if any(keyword in name.lower() for keyword in ['base', 'trunk', 'body'])]
        if base_candidates:
            print(f"  → 建议使用: {base_candidates}")
    
    # 检查 ".*_thigh" 模式
    print("\n[检查2] body_names=['.*_thigh']:")
    thigh_pattern = re.compile(r".*_thigh")
    thigh_bodies = [name for name in contact_sensor.body_names if thigh_pattern.match(name)]
    if thigh_bodies:
        print(f"  ✓ 匹配到 {len(thigh_bodies)} 个: {thigh_bodies}")
    else:
        print("  ✗ 未匹配到任何body")
        thigh_candidates = [name for name in contact_sensor.body_names if 'thigh' in name.lower()]
        if thigh_candidates:
            print(f"  → 可能的名称: {thigh_candidates}")
    
    # 检查 ".*_calf" 模式
    print("\n[检查3] body_names='.*_calf':")
    calf_pattern = re.compile(r".*_calf")
    calf_bodies = [name for name in contact_sensor.body_names if calf_pattern.match(name)]
    if calf_bodies:
        print(f"  ✓ 匹配到 {len(calf_bodies)} 个: {calf_bodies}")
    else:
        print("  ✗ 未匹配到任何body")
        calf_candidates = [name for name in contact_sensor.body_names if 'calf' in name.lower()]
        if calf_candidates:
            print(f"  → 可能的名称: {calf_candidates}")
    
    # 显示当前的接触力数据
    print("\n[INFO] 当前各body的接触力 (>0.1N):")
    net_forces = contact_sensor.data.net_forces_w_history
    force_norms = torch.norm(net_forces[0, -1, :, :], dim=-1)
    has_contact = False
    for idx, (name, force) in enumerate(zip(contact_sensor.body_names, force_norms)):
        if force > 0.1:
            print(f"  {name}: {force.item():.2f} N")
            has_contact = True
    if not has_contact:
        print("  (无明显接触)")
    
    # ================== DEBUG: 检查传感器数据延迟和噪声 ==================
    print("\n" + "="*70)
    print("[DEBUG] 详细检查传感器数据...")
    
    net_forces = contact_sensor.data.net_forces_w_history
    print(f"\n[传感器配置]")
    print(f"  History length: {contact_sensor.cfg.history_length}")
    print(f"  Update period: {contact_sensor.cfg.update_period}")
    print(f"  数据形状: {net_forces.shape} (num_envs, history, num_bodies, 3)")
    
    # 检查每个body在所有历史帧的接触力
    print(f"\n[所有历史帧的接触力详情]")
    for body_idx, body_name in enumerate(contact_sensor.body_names):
        print(f"\n  Body [{body_idx}] {body_name}:")
        for hist_idx in range(net_forces.shape[1]):
            force = net_forces[0, hist_idx, body_idx, :]
            force_norm = torch.norm(force).item()
            if force_norm > 0.01:  # 显示所有 > 0.01N 的力
                print(f"    历史帧 {hist_idx}: {force_norm:.4f} N  (x={force[0].item():.3f}, y={force[1].item():.3f}, z={force[2].item():.3f})")
            else:
                print(f"    历史帧 {hist_idx}: {force_norm:.4f} N")
    
    # 检查关键body的接触力（base和thigh）
    print(f"\n[关键Body接触力分析]")
    
    # Base
    base_idx = contact_sensor.body_names.index('base')
    base_forces_all = net_forces[0, :, base_idx, :]
    base_norms = torch.norm(base_forces_all, dim=-1)
    print(f"\n  Base 接触力:")
    print(f"    当前帧: {base_norms[-1].item():.6f} N")
    print(f"    最大值(历史): {base_norms.max().item():.6f} N")
    print(f"    平均值(历史): {base_norms.mean().item():.6f} N")
    print(f"    是否 > 1.0N (termination阈值): {(base_norms > 1.0).any().item()}")
    
    # Thighs
    thigh_indices = [contact_sensor.body_names.index(name) 
                     for name in ['FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh']]
    print(f"\n  Thigh 接触力:")
    for thigh_name, thigh_idx in zip(['FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh'], thigh_indices):
        thigh_forces = net_forces[0, :, thigh_idx, :]
        thigh_norms = torch.norm(thigh_forces, dim=-1)
        max_force = thigh_norms.max().item()
        has_contact = (thigh_norms > 1.0).any().item()
        print(f"    {thigh_name}: 当前={thigh_norms[-1].item():.6f}N, "
              f"最大={max_force:.6f}N, >1.0N={has_contact}")
    
    # Calfs (脚部)
    calf_indices = [contact_sensor.body_names.index(name) 
                    for name in ['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf']]
    print(f"\n  Calf (脚部) 接触力:")
    for calf_name, calf_idx in zip(['FL_calf', 'FR_calf', 'RL_calf', 'RR_calf'], calf_indices):
        calf_forces = net_forces[0, :, calf_idx, :]
        calf_norms = torch.norm(calf_forces, dim=-1)
        max_force = calf_norms.max().item()
        exceeds_limit = (calf_norms.max() > 50.0).item()
        print(f"    {calf_name}: 当前={calf_norms[-1].item():.4f}N, "
              f"最大={max_force:.4f}N, >50N={exceeds_limit}")
    
    # 检查是否有异常的噪声峰值
    print(f"\n[噪声检测]")
    all_forces = net_forces[0, :, :, :]  # (history, bodies, 3)
    all_norms = torch.norm(all_forces, dim=-1)  # (history, bodies)
    
    # 检查每个body的力是否有突变
    for body_idx, body_name in enumerate(contact_sensor.body_names):
        body_force_history = all_norms[:, body_idx]
        if body_force_history.max() > 0.1:
            force_diff = torch.abs(body_force_history[1:] - body_force_history[:-1])
            max_change = force_diff.max().item() if len(force_diff) > 0 else 0
            print(f"  {body_name}: 最大变化={max_change:.4f}N/步")
    
    print("="*70 + "\n")
    # ================================================================

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = build_alg_runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # reset the agent and env
    runner.reset()
    # train the agent
    if args_cli.checkpoint is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
