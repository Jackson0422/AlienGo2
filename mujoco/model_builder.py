"""Build a MuJoCo model that mirrors the Isaac Lab AlienGo training setup.

Single source of truth for joint order, default joint angles, and PD gains is
`exts/cat_envs/cat_envs/assets/odri.py` and
`exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/aliengo/cat_flat_env_cfg.py`.

We intentionally generate the MjModel programmatically from the URDF rather
than serialising to XML on disk, because `MjSpec.to_xml()` loses the
foot-body fixed-joint offset (the foot sphere ends up at body origin if
re-loaded). Loading the URDF + post-processing via MjSpec preserves the merge
correctly at compile time.

Conventions matched with Isaac Lab:
- Free joint on the robot base, initial pose pos=(0,0,0.4) quat=(1,0,0,0).
- 12 hinge joints in MuJoCo's natural FR/FL/RR/RL ordering. The training-time
  Isaac Lab order is FL/FR/RR/RL; we expose explicit permutation tables to
  remap between the two so the policy never sees a wrong index.
- Hinge armature = 0.00036207 (matches IdealPDActuator armature).
- Motor actuators with gear=1: we compute torque in Python (PD law identical
  to IdealPDActuator) and write directly to `data.ctrl`.
- Physics dt = 0.005 s, control runs at decimation=4 → 50 Hz policy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

import mujoco
import numpy as np


# Joint order used by Isaac Lab observations & actions (preserve_order=True in
# cat_flat_env_cfg.py). The policy was trained with THIS exact ordering; any
# tensor crossing the policy boundary must be in this order.
ISAAC_JOINT_NAMES: List[str] = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]

# Default joint positions from odri.py:114-127. Same value 0/0.9/-1.7 per leg,
# in the Isaac Lab joint order above.
ISAAC_DEFAULT_JOINT_POS: np.ndarray = np.array(
    [0.0, 0.9, -1.7] * 4, dtype=np.float64
)

# Action / actuator settings from cat_flat_env_cfg.py:119-138 (ActionsCfg) and
# odri.py:131-153 (IdealPDActuatorCfg).
ACTION_SCALE: float = 0.3
JOINT_KP: float = 25.0
JOINT_KD: float = 1.5
JOINT_EFFORT_LIMIT: float = 33.5      # IdealPDActuator effort_limit
JOINT_VELOCITY_LIMIT: float = 21.0    # informational; IdealPDActuator does NOT clip on velocity
JOINT_ARMATURE: float = 0.00036207

# Per-joint URDF-declared effort limits (used by MuJoCo `actuatorfrcrange`)
URDF_EFFORT_HIP_THIGH: float = 35.278
URDF_EFFORT_CALF: float = 44.4

# Simulation timing (matches env.yaml).
SIM_DT: float = 0.005
DECIMATION: int = 4
POLICY_DT: float = SIM_DT * DECIMATION  # 0.02 s, 50 Hz

# Base init pose (env.yaml, init_state.pos / .rot).
BASE_INIT_POS: np.ndarray = np.array([0.0, 0.0, 0.4], dtype=np.float64)
BASE_INIT_QUAT_WXYZ: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

# Foot offset along calf body z. Matches `observations.py:_FOOT_OFFSET_LOCAL = (0,0,-0.25)`
# and the URDF foot_fixed joint origin. The compiled MuJoCo model already
# places the foot sphere at this offset within the calf body, so we just need
# the geom ids to read its world position.
FOOT_LOCAL_OFFSET_Z: float = -0.25

# Foot body / geom names. Order is FL, FR, RL, RR — matches cat_flat_env_cfg.py
# observation `foot_contact` order on line 191.
FEET_ORDER_FOR_CONTACT: List[str] = ["FL", "FR", "RL", "RR"]
# Rear feet only, for rear_fz / com_cop observation terms.
REAR_FEET_ORDER: List[str] = ["RL", "RR"]


@dataclass
class ModelInfo:
    """Pre-computed id tables so the simulation loop never does string lookups."""

    # Base body
    base_body_id: int = -1
    base_qpos_addr: int = -1     # qpos start index of free joint (always 0 in
    base_qvel_addr: int = -1     # our setup but kept explicit)

    # 12 actuated joints, indexed in ISAAC_JOINT_NAMES order
    isaac_joint_qpos_addr: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.int32))
    isaac_joint_qvel_addr: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.int32))
    isaac_joint_actuator_id: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.int32))

    # Foot geom ids in the four-foot order used by obs (FL, FR, RL, RR)
    foot_geom_id_all: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int32))
    # Calf body ids in the same order
    calf_body_id_all: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.int32))
    # Rear (RL, RR) subset of foot geoms (for rear_fz / com_cop)
    rear_foot_geom_id: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.int32))

    # Floor geom id (for contact pair filtering, optional)
    floor_geom_id: int = -1

    # Total robot mass (= sum of body masses excluding world). Pre-computed because
    # IsaacLab uses `default_mass.sum()` rather than dynamic mass.
    total_mass: float = 0.0
    # Body ids that contribute to the CoM (all robot bodies, i.e. body_id >= 1).
    com_body_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    com_body_mass: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))


def _foot_geom_name(leg: str) -> str:
    return f"{leg}_foot_geom"


def build_model(urdf_path: str) -> tuple[mujoco.MjModel, mujoco.MjData, ModelInfo]:
    """Compile MuJoCo model from the Isaac Lab AlienGo URDF + sim-to-sim glue.

    Returns:
        (model, data, info). `info` holds the lookup tables the rest of the
        sim-to-sim code needs.
    """
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    spec = mujoco.MjSpec.from_file(urdf_path)

    # ---- name the foot collision geoms BEFORE compile so we can find them later.
    # Each `<leg>_foot` body in the URDF has one collision sphere; after fixed-joint
    # merging it ends up on the corresponding `<leg>_calf` body at z=-0.25.
    for body in spec.bodies:
        name = body.name or ""
        if name.endswith("_foot") and name != "_foot":
            for g in body.geoms:
                g.name = _foot_geom_name(name.removesuffix("_foot"))

    # ---- disable self-collisions on the robot (matches odri.py:104
    # `enabled_self_collisions=False`). Strategy: put all robot geoms on
    # `contype=2, conaffinity=1`, and the floor we'll add later on
    # `contype=1, conaffinity=2`. Two robot geoms then share neither a
    # contype-vs-conaffinity bit, so they never collide; floor-vs-robot
    # collides on both sides.
    for body in spec.bodies:
        for g in body.geoms:
            g.contype = 2
            g.conaffinity = 1

    # ---- add a free joint at the base body so the robot can move in the world.
    # URDF root link "base" is at the top of worldbody children.
    spec.worldbody.bodies[0].add_freejoint()

    # ---- armature on every hinge joint (IdealPDActuator armature in Isaac Lab).
    for body in spec.bodies:
        for j in body.joints:
            if j.type == mujoco.mjtJoint.mjJNT_HINGE:
                j.armature = JOINT_ARMATURE
                # Damping was 0 in URDF; keep it that way — Isaac Lab's IdealPDActuator
                # implements damping as a control-side term, not joint-side viscous.

    # ---- simulation options.
    spec.option.timestep = SIM_DT
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    # Offscreen framebuffer size — needed for headless EGL rendering at 1280x720.
    # Set above the largest render resolution we ever request.
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    # ---- world: floor plane + light + tracking camera.
    wb = spec.worldbody
    floor = wb.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[20.0, 20.0, 0.1],
        rgba=[0.45, 0.5, 0.55, 1.0],
    )
    # Match Isaac Lab terrain friction (static=1.0, dynamic=1.0).
    floor.friction = [1.0, 0.005, 0.0001]
    # See self-collision discussion above: floor uses contype=1, conaffinity=2
    # so it collides only with robot geoms (contype=2, conaffinity=1), never
    # with other floor-like geoms.
    floor.contype = 1
    floor.conaffinity = 2

    wb.add_light(
        pos=[0.0, 0.0, 3.0],
        castshadow=False,
        diffuse=[0.7, 0.7, 0.7],
        ambient=[0.4, 0.4, 0.4],
    )
    wb.add_camera(
        name="track",
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
        pos=[1.6, -1.6, 0.9],
        xyaxes=[1.0, 1.0, 0.0, -0.35, 0.35, 1.3],
    )

    # ---- one motor actuator per actuated hinge joint, in MuJoCo's natural
    # joint order (FR, FL, RR, RL). We compute PD torque in Python and write
    # to data.ctrl, so gear=1 motors are appropriate.
    mj_joint_order_in_urdf = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    for jn in mj_joint_order_in_urdf:
        eff = URDF_EFFORT_CALF if jn.endswith("_calf_joint") else URDF_EFFORT_HIP_THIGH
        # Apply IsaacLab IdealPDActuator effort_limit (33.5 N·m), not URDF's 35.278/44.4.
        # We clip in Python anyway, but constrain at actuator level too to be safe.
        eff = min(eff, JOINT_EFFORT_LIMIT) if jn.endswith("_calf_joint") else min(eff, JOINT_EFFORT_LIMIT)
        spec.add_actuator(
            name=jn,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=jn,
            gear=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ctrllimited=1,
            ctrlrange=[-eff, eff],
            forcelimited=1,
            forcerange=[-eff, eff],
        )

    model = spec.compile()
    data = mujoco.MjData(model)

    # ---- build the lookup tables.
    info = ModelInfo()
    info.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if info.base_body_id < 0:
        raise RuntimeError("base body not found in compiled model")
    # The free joint is the first (and only) free joint we added; it lives on `base`.
    # qposaddr / dofadr are 0 in this layout, but we read them via the joint id for safety.
    base_jid = -1
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            base_jid = j
            break
    if base_jid < 0:
        raise RuntimeError("free joint not found in compiled model")
    info.base_qpos_addr = int(model.jnt_qposadr[base_jid])
    info.base_qvel_addr = int(model.jnt_dofadr[base_jid])

    for i, name in enumerate(ISAAC_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if jid < 0 or aid < 0:
            raise RuntimeError(f"joint/actuator missing: {name}")
        info.isaac_joint_qpos_addr[i] = model.jnt_qposadr[jid]
        info.isaac_joint_qvel_addr[i] = model.jnt_dofadr[jid]
        info.isaac_joint_actuator_id[i] = aid

    for i, leg in enumerate(FEET_ORDER_FOR_CONTACT):
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, _foot_geom_name(leg))
        if gid < 0:
            raise RuntimeError(f"foot geom missing: {_foot_geom_name(leg)}")
        info.foot_geom_id_all[i] = gid
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
        if cid < 0:
            raise RuntimeError(f"calf body missing: {leg}_calf")
        info.calf_body_id_all[i] = cid

    rear_idx = [FEET_ORDER_FOR_CONTACT.index(l) for l in REAR_FEET_ORDER]
    info.rear_foot_geom_id = info.foot_geom_id_all[rear_idx].astype(np.int32)

    info.floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # All robot bodies except world (body 0).
    info.com_body_ids = np.arange(1, model.nbody, dtype=np.int32)
    info.com_body_mass = np.array(
        [model.body_mass[b] for b in info.com_body_ids], dtype=np.float64
    )
    info.total_mass = float(info.com_body_mass.sum())

    return model, data, info


def reset_robot_to_default(
    model: mujoco.MjModel, data: mujoco.MjData, info: ModelInfo
) -> None:
    """Set qpos / qvel to the Isaac Lab `init_state` (matches reset_robot_joints
    with position_range=(1.0, 1.0))."""
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[info.base_qpos_addr + 0 : info.base_qpos_addr + 3] = BASE_INIT_POS
    data.qpos[info.base_qpos_addr + 3 : info.base_qpos_addr + 7] = BASE_INIT_QUAT_WXYZ
    # Default joint angles, written through the qpos addresses (Isaac order
    # -> MuJoCo qpos by addr table).
    for i in range(12):
        data.qpos[info.isaac_joint_qpos_addr[i]] = ISAAC_DEFAULT_JOINT_POS[i]
    mujoco.mj_forward(model, data)
