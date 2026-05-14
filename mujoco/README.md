# Sim-to-Sim: Isaac Lab → MuJoCo (AlienGo CaT policy)

This folder contains a self-contained, headless rollout of the rl-games policy
trained in Isaac Lab, replayed inside MuJoCo. The goal is to verify the
trained checkpoint outside of Isaac Sim by visualising it in MuJoCo, with the
video written to `logs_mujoco/` (so it can be inspected on a remote SSH host
that cannot run a viewer).

The loaded checkpoint is **exactly** the one that
`scripts/rl_games/play.py --task=Isaac-Velocity-CaT-Flat-AlienGo-Play-v0`
loads (i.e. `<run-dir>/nn/solo_cat.pth` plus the RunningMeanStd normalizer
stored inside the same `.pth`). Nothing else is fine-tuned.

## Quick start

```bash
conda activate torch251tf2170-py310-cuda124
# from the project root:
python mujoco/sim_to_sim.py \
    --run-dir logs/rl_games/solo_cat/2026-05-07_21-28-14 \
    --video-length 200
```

Output:

```
logs_mujoco/<run-id>/rl-video-step-0.mp4    # 50 fps, matches Isaac Lab play.py
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--run-dir DIR` | Isaac Lab rl_games training directory (must contain `params/agent.yaml` and `nn/solo_cat.pth`). Defaults to `logs/rl_games/solo_cat/2026-05-07_21-28-14`. |
| `--checkpoint PATH` | Explicit `.pth` path. Default is `<run-dir>/nn/solo_cat.pth` (the best ckpt — same as `play.py` without `--use_last_checkpoint`). |
| `--video-length N` | Number of policy steps to record. Default 200 (= 4 s at 50 Hz, matches `play.py --video_length 200`). |
| `--cmd-vx`, `--cmd-vy`, `--cmd-wz` | Velocity command sent to the policy. Defaults to 0 (matches the `_PLAY` config of the task, which freezes the command at zero). |
| `--width`, `--height` | Render resolution. Default 1280×720 (= `play.py` viewer.resolution). |
| `--output-dir DIR` | Where to write the mp4. Default `logs_mujoco/<run-id>/`. |
| `--device` | torch device for the policy. Default `cuda:0` if available, else `cpu`. |
| `--print-every N` | Print a one-line debug snapshot every N policy steps. 0 to disable. |

## Files

| File | Purpose |
|------|---------|
| `model_builder.py` | Builds the MuJoCo model in-memory from the URDF used by Isaac Lab. Adds a freejoint, motor actuators, floor + lighting + tracking camera, and bookkeeping tables so the rest of the pipeline never does string lookups. Also disables self-collisions to match `enabled_self_collisions=False` from `exts/cat_envs/cat_envs/assets/odri.py`. |
| `policy_loader.py` | Standalone rl-games loader. Bypasses `cat_envs` (which transitively imports `omni.kit`) by injecting `env_info` directly and inlining the single algo registration line from `cat_envs.tasks.utils.rl_games.build_alg_runner`. Restores the RunningMeanStd observation normalizer alongside the actor MLP. |
| `obs_builder.py` | Reconstructs the 57-d policy observation byte-for-byte from MuJoCo state. Mirrors the `ObservationsCfg.PolicyCfg` block in `exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/aliengo/cat_flat_env_cfg.py` (lines 142-213). |
| `sim_to_sim.py` | Main script: build, reset, roll out, render, save. |

## Correctness checklist (every issue the user pointed out, and how it is handled)

| Issue | Where it is enforced |
|-------|----------------------|
| **Observation order must match exactly** | `obs_builder.build_obs` concatenates in the same order as `PolicyCfg` and applies the same scales (0.25, 2.0, 2.0, (2,2,0.25), 0.7, 1.0, 0.05, 1.0, 0.005, 5.0, 1.0). |
| **Action scale + default joint position** | `sim_to_sim.main` does `q_des = ISAAC_DEFAULT_JOINT_POS + 0.3 * action`, with constants taken from `odri.py:114-127` and `cat_flat_env_cfg.py:135-136`. |
| **MuJoCo actuator must match Isaac Lab control** | Each joint has a `<motor>` actuator (gear=1), and `compute_pd_torque` reproduces `IdealPDActuator.compute()` verbatim: `τ = kp*(q_des - q) - kd*qd`, clipped to `±effort_limit`. Joint armature is also propagated. |
| **Simulation frequency / decimation** | `SIM_DT=0.005`, `DECIMATION=4` in `model_builder.py`. Each policy step performs 4 `mj_step` calls and writes exactly one video frame, so `fps=50` ⇔ policy rate. |
| **Joint order consistency** | `ISAAC_JOINT_NAMES` lists the Isaac Lab order `FL, FR, RR, RL`; `info.isaac_joint_qpos_addr` and `info.isaac_joint_actuator_id` permute between that order and MuJoCo’s natural URDF order `FR, FL, RR, RL`. Every tensor crossing the policy boundary is in Isaac order; every write to `data.ctrl` is reordered via the table. |
| **Do NOT subtract default joint pos in `joint_pos` obs** | `cat_flat_env_cfg.py:172` uses `mdp.joint_pos` (ABSOLUTE), not `mdp.joint_pos_rel`. `build_obs` therefore feeds raw `data.qpos[...]` values, scaled by 1.0. |
| **Don’t forget the action scale (0.3)** | `ACTION_SCALE = 0.3` is the single source of truth in `model_builder.py`; the main loop multiplies the raw policy output by this before adding the default offset. |
| **Don’t forget previous action in obs** | `sim_to_sim` keeps `last_action_raw` and passes it to `build_obs`. The very first step uses zeros, matching `JointAction.reset()` in Isaac Lab. The value is the RAW policy output (before scale + offset), not `q_des`. |
| **Step the policy at the right rate** | The policy is queried once per `POLICY_DT = 0.02 s` outer step; the same action is held across 4 inner physics substeps. |
| **Isaac is position target, MuJoCo must implement PD** | Each MuJoCo actuator is a torque source (`<motor>`); the PD is computed in Python with `kp=25, kd=1.5` from `odri.py:148-151` and applied via `data.ctrl`. No MuJoCo `<position>` actuator is used (those have P only). |
| **Handle observation normalization** | `policy_loader.load_policy` calls `player.restore(checkpoint)`, which loads `model.running_mean_std` from the same `.pth`. The forward pass of `continuous_a2c_logstd` applies `norm_obs` internally on each call (see `rl_games/algos_torch/models.py:51-52, 86, 142, 208, 261, 321`). So we feed unnormalised observations and the normalisation happens inside the network. |
| **Base frame & quaternion order** | MuJoCo stores quaternions as `(w, x, y, z)`, identical to Isaac Lab. `_quat_rotate_inverse_wxyz` is a verbatim numpy port of `isaaclab.utils.math.quat_rotate_inverse` (sanity-checked to 1e-16 in a unit test). `base_lin_vel`/`base_ang_vel` are produced by rotating world-frame `qvel[0:3]` / `qvel[3:6]` through that function — matching Isaac Lab’s `root_lin_vel_b` / `root_ang_vel_b`. `projected_gravity_b` uses `GRAVITY_VEC_W = (0, 0, -1)`. |

## Expected behaviour

The Isaac Lab task (`Isaac-Velocity-CaT-Flat-AlienGo-Play-v0`) trains the
robot to **stand on its hind legs from a quadruped stance**. The MuJoCo
rollout reproduces this: at `t = 0` the robot is in the standard four-foot
stance (joints `0/0.9/-1.7`), and around `t = 1 s` it reaches the upright
bipedal pose (pitch ≈ +85 deg, base z ≈ 0.68 m).

Beyond that, the MuJoCo dynamics differ from PhysX — solver type, contact
stiffness, joint friction model — so without further tuning the robot loses
balance noticeably earlier than it does in Isaac Lab. The policy itself is
byte-identical; the behavior delta is pure physics.

### Step-A physics tuning (applied by default)

To close the PhysX -> MuJoCo gap on the rear-stand task, `model_builder.py`
applies a conservative set of fixes during model construction. All of them
are exposed as module-level constants near the top of the file so you can
tune them without touching `build_model`:

| Constant | Default | What it does |
|---|---|---|
| `JOINT_DAMPING` | `0.05` N·m·s/rad | Light viscous damping on every hinge joint. Mimics the small numerical damping PhysX gets for free from its low-iteration implicit solver; kills the high-frequency joint chatter that otherwise destabilises the rear-stand policy in MuJoCo. |
| `JOINT_FRICTIONLOSS` | `0.05` N·m | Light Coulomb friction at the joints. Same motivation as above. |
| `FLOOR_TORSIONAL_FRICTION` | `0.05` | Bumped from MuJoCo's default `0.005`. At the default the rear-foot contact patch offers essentially no resistance to body-yaw rotation, which never showed up during PhysX training but is unique to bipedal stance. |
| `CONTACT_SOLREF` | `[2*SIM_DT, 1.0]` | Contact time-constant set to 2 physics steps (=0.01 s), critically damped. Applied to floor and foot geoms; brings MuJoCo's response time closer to PhysX's "few-iteration soft contact". |
| `CONTACT_SOLIMP` | `[0.9, 0.92, 0.001, 0.5, 2.0]` | Slightly loosened `dmax` compared to MuJoCo's default `0.95`. |
| `SOLVER_ITERATIONS` / `SOLVER_TOLERANCE` | `50` / `1e-8` | Switch to Newton solver. Converges in <10 iterations on this smooth contact geometry and produces lower frame-to-frame contact-force noise, which matters because the policy is open-loop driven by `data.qpos`/`data.qvel`. |

These values are intentionally small. Raising them further starts to
diverge from the Isaac Lab training distribution; if you need bigger
changes, the right move is to add (matching) domain randomization on the
Isaac Lab side and retrain, rather than push MuJoCo further away from
PhysX.

If you want to recover the original "load policy verbatim into vanilla
MuJoCo" behaviour for a baseline comparison, set:

```python
JOINT_DAMPING = 0.0
JOINT_FRICTIONLOSS = 0.0
FLOOR_TORSIONAL_FRICTION = 0.005
# and remove the floor.solref / floor.solimp / foot.solref / foot.solimp
# assignments inside build_model().
```

## Headless rendering note

We use EGL (`MUJOCO_GL=egl`) by default — works over SSH without an X
display. If EGL is not available, set `MUJOCO_GL=osmesa` before launching
the script.

## Versions tested

- Python 3.10 / conda env `torch251tf2170-py310-cuda124`
- mujoco 3.8
- torch 2.5.1, CUDA 12.4
- rl_games installed at the same version Isaac Lab’s training used
- imageio 2.37 + imageio-ffmpeg
