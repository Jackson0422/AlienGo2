"""Sim-to-sim rollout: take the policy trained in Isaac Lab and run it in MuJoCo,
recording an offscreen video to `logs_mujoco/`.

Designed to mirror, as closely as possible, the rollout produced by:

    python scripts/rl_games/play.py \
        --task=Isaac-Velocity-CaT-Flat-AlienGo-Play-v0 \
        --headless --video --video_length 200

Same policy network, same observation pipeline (after sim-to-sim
re-implementation), same control rate (50 Hz on top of 200 Hz physics), same
default joint pose, same JointPositionAction scale (0.3) + default offset.

Designed for headless (SSH) operation: MuJoCo is rendered with EGL, video is
written via imageio (ffmpeg backend). No on-screen viewer.

Usage (from project root):

    conda activate torch251tf2170-py310-cuda124
    python mujoco/sim_to_sim.py \
        --run-dir logs/rl_games/solo_cat/2026-05-07_21-28-14 \
        --video-length 200

Outputs:

    logs_mujoco/<run-id>/rl-video-step-0.mp4

where `<run-id>` mirrors the Isaac Lab run directory name.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

# Make sibling modules importable. Note: the parent directory `mujoco/` does
# NOT have an __init__.py, so it does NOT shadow the PyPI `mujoco` package.
# Python's stdlib import machinery prefers the installed package because it
# carries a regular __init__.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Make sure EGL is selected before mujoco imports. The user can override.
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import mujoco  # PyPI; see note above.

import model_builder as mb
from model_builder import (
    ISAAC_DEFAULT_JOINT_POS,
    ISAAC_JOINT_NAMES,
    ACTION_SCALE,
    JOINT_KP,
    JOINT_KD,
    JOINT_EFFORT_LIMIT,
    SIM_DT,
    DECIMATION,
    POLICY_DT,
)
import obs_builder as ob
import policy_loader as pl


DEFAULT_URDF = "exts/cat_envs/cat_envs/assets/Robots/odri/AlienGo_description/aliengo.urdf"
DEFAULT_RUN = "logs/rl_games/solo_cat/2026-05-07_21-28-14"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run an Isaac Lab CaT policy in MuJoCo (headless).")
    p.add_argument("--run-dir", default=DEFAULT_RUN,
                   help="Isaac Lab rl_games run directory (must contain params/agent.yaml and nn/solo_cat.pth).")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit .pth path; defaults to <run-dir>/nn/solo_cat.pth (= what play.py loads).")
    p.add_argument("--urdf", default=DEFAULT_URDF,
                   help="AlienGo URDF (same one Isaac Lab spawns from USD that is generated from).")
    p.add_argument("--video-length", type=int, default=200,
                   help="Number of policy steps (50 Hz) to record. 200 matches the default in play.py.")
    p.add_argument("--output-dir", default=None,
                   help="Output directory. Defaults to logs_mujoco/<run-id>/.")
    p.add_argument("--video-name", default="rl-video-step-0.mp4",
                   help="Output video filename.")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--device", default=None, help="torch device for the policy; default auto.")
    p.add_argument("--cmd-vx", type=float, default=0.0, help="Forward velocity command (m/s).")
    p.add_argument("--cmd-vy", type=float, default=0.0, help="Lateral velocity command (m/s).")
    p.add_argument("--cmd-wz", type=float, default=0.0, help="Yaw rate command (rad/s).")
    p.add_argument("--seed", type=int, default=0, help="Numpy seed for any future randomization.")
    p.add_argument("--print-every", type=int, default=50,
                   help="Print debug snapshot every N policy steps (0 to disable).")
    return p.parse_args()


def compute_pd_torque(
    q_des_isaac: np.ndarray,
    q_isaac: np.ndarray,
    qd_isaac: np.ndarray,
) -> np.ndarray:
    """Replicates `IdealPDActuator.compute()` from Isaac Lab.

    tau = kp * (q_des - q) + kd * (0 - qd)
    tau_applied = clip(tau, ±effort_limit)

    Note IdealPDActuator does NOT enforce velocity_limit at the torque level;
    `effort_limit` is the only saturation. See odri.py:148-151 for the gain
    values used during training.
    """
    err_pos = q_des_isaac - q_isaac
    # joint_vel_target is 0 for JointPositionAction.
    tau = JOINT_KP * err_pos - JOINT_KD * qd_isaac
    return np.clip(tau, -JOINT_EFFORT_LIMIT, JOINT_EFFORT_LIMIT)


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    # ---- output paths
    run_id = os.path.basename(os.path.normpath(args.run_dir))
    if args.output_dir is None:
        args.output_dir = os.path.join("logs_mujoco", run_id)
    os.makedirs(args.output_dir, exist_ok=True)
    out_video = os.path.join(args.output_dir, args.video_name)

    # ---- build model + policy
    print(f"[sim_to_sim] Building MuJoCo model from URDF: {args.urdf}")
    model, data, info = mb.build_model(args.urdf)
    print(f"[sim_to_sim] Model OK: nq={model.nq} nv={model.nv} nu={model.nu} njnt={model.njnt}")
    print(f"[sim_to_sim] Total robot mass = {info.total_mass:.3f} kg")

    print(f"[sim_to_sim] Loading policy from run: {args.run_dir}")
    policy = pl.load_policy(args.run_dir, checkpoint=args.checkpoint, device=args.device)
    print(f"[sim_to_sim] Policy device = {policy.device}, checkpoint = {policy.checkpoint_path}")

    # ---- reset
    mb.reset_robot_to_default(model, data, info)
    print(f"[sim_to_sim] Reset: base z = {data.qpos[info.base_qpos_addr + 2]:.4f} m")

    # ---- renderer (headless EGL)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    frames: List[np.ndarray] = []

    # ---- rollout loop
    last_action_raw = np.zeros(12, dtype=np.float64)
    velocity_command = np.array([args.cmd_vx, args.cmd_vy, args.cmd_wz], dtype=np.float64)

    t0 = time.time()
    pd_torque_log: List[float] = []
    for step in range(args.video_length):
        obs = ob.build_obs(model, data, info, last_action_raw, velocity_command)

        action = policy.act(obs).astype(np.float64)
        last_action_raw = action.copy()

        # Isaac Lab JointPositionAction.process_actions:
        #   q_des = default_joint_pos + scale * raw_action
        # (use_default_offset=True, scale=0.3) — cat_flat_env_cfg.py:119-138 and odri.py defaults.
        q_des_isaac = ISAAC_DEFAULT_JOINT_POS + ACTION_SCALE * action

        # Inner physics loop: decimation=4 substeps per policy step.
        for _ in range(DECIMATION):
            q_isaac = data.qpos[info.isaac_joint_qpos_addr]
            qd_isaac = data.qvel[info.isaac_joint_qvel_addr]
            tau_isaac = compute_pd_torque(q_des_isaac, q_isaac, qd_isaac)
            # Write to ctrl in MuJoCo actuator order via the lookup table.
            data.ctrl[info.isaac_joint_actuator_id] = tau_isaac
            mujoco.mj_step(model, data)

        pd_torque_log.append(float(np.abs(tau_isaac).max()))

        # Render exactly one frame per policy step → video fps = 1/POLICY_DT = 50 Hz,
        # matching the Isaac Lab gym.wrappers.RecordVideo output (which records at the
        # env step rate, i.e. after each decimation-grouped step).
        renderer.update_scene(data, camera="track")
        frames.append(renderer.render())

        if args.print_every and step % args.print_every == 0:
            base_z = float(data.qpos[info.base_qpos_addr + 2])
            base_quat = data.qpos[info.base_qpos_addr + 3 : info.base_qpos_addr + 7]
            # pitch (deg) using projected gravity, same convention as scripts/rl_games/play.py
            g_b = ob._quat_rotate_inverse_wxyz(base_quat, np.array([0.0, 0.0, -1.0]))
            pitch_deg = float(np.degrees(np.arctan2(-g_b[0], -g_b[2])))
            print(
                f"[step {step:4d}] base_z={base_z:+.3f} pitch={pitch_deg:+.1f}deg "
                f"|act|max={float(np.abs(action).max()):.2f} |tau|max={pd_torque_log[-1]:.2f}"
            )

    elapsed = time.time() - t0
    print(f"[sim_to_sim] Rolled out {args.video_length} steps in {elapsed:.2f}s "
          f"(target wallclock = {args.video_length * POLICY_DT:.2f}s of simulated time)")

    # ---- save video
    import imageio.v2 as iio
    iio.mimsave(out_video, frames, fps=int(round(1.0 / POLICY_DT)), macro_block_size=1)
    size_kb = os.path.getsize(out_video) // 1024
    print(f"[sim_to_sim] Wrote video: {out_video} ({size_kb} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
