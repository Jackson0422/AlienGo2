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

# Simulation timing (matches env.yaml). Declared early so the step-A
# `CONTACT_SOLREF` below can express its time-constant in units of SIM_DT.
SIM_DT: float = 0.005
DECIMATION: int = 4
POLICY_DT: float = SIM_DT * DECIMATION  # 0.02 s, 50 Hz

# ---------------------------------------------------------------------------
# Step-A sim-to-sim tuning (close the PhysX -> MuJoCo physics gap).
#
# These do NOT exist in Isaac Lab's IdealPDActuator / PhysX setup, but PhysX
# silently introduces a small amount of numerical damping via its low-iteration
# implicit solver, and PhysX's friction model has effective torsional friction
# at the contact patch. MuJoCo with default settings has neither, which is the
# main reason the rear-stand policy goes unstable in MuJoCo even though the
# observation pipeline is byte-identical.
#
# The three knobs below are intentionally small. Larger values will start to
# diverge from the Isaac Lab training distribution. Tune in this order:
#   1) JOINT_DAMPING / JOINT_FRICTIONLOSS  (kills high-frequency joint chatter)
#   2) FLOOR_TORSIONAL_FRICTION            (stops the rear feet from spinning
#                                           freely about z while bipedal-standing)
#   3) CONTACT_SOLREF / CONTACT_SOLIMP     (soften ground contact toward
#                                           PhysX's "few-iteration" feel)
# ---------------------------------------------------------------------------
JOINT_DAMPING: float = 0.05         # N·m·s/rad, mimics PhysX implicit damping
JOINT_FRICTIONLOSS: float = 0.05    # N·m, light Coulomb friction at the joints

# Floor friction (slide, torsion, roll). Isaac Lab terrain friction = 1.0
# (slide). MuJoCo default torsion of 0.005 is essentially zero for a
# bipedal-stance robot: the rear feet pivot freely about the body z-axis,
# producing a yaw-drift the policy never had to handle in training.
FLOOR_SLIDE_FRICTION: float = 1.0
FLOOR_TORSIONAL_FRICTION: float = 0.05
FLOOR_ROLLING_FRICTION: float = 0.0001

# Contact compliance (solref = [time_const, damp_ratio]; solimp = [dmin, dmax,
# width, midpoint, power]). Defaults are [0.02, 1.0] and [0.9, 0.95, 0.001,
# 0.5, 2]. We loosen dmax slightly and set time_const = 2*SIM_DT = 0.01s
# explicitly so the contact response time matches the policy step. This
# brings MuJoCo closer to PhysX's "soft" few-iteration contact feel.
CONTACT_SOLREF: List[float] = [2.0 * SIM_DT, 1.0]
CONTACT_SOLIMP: List[float] = [0.9, 0.92, 0.001, 0.5, 2.0]

# Solver: Newton + tighter tolerance + capped iterations.
# Default solver is CG with iterations=100, tolerance=1e-8.
# Newton converges in fewer iterations for our smooth contact geometry and
# produces a more deterministic contact force per step (smaller frame-to-frame
# noise), which helps the open-loop replay of a deterministic policy.
SOLVER_ITERATIONS: int = 50
SOLVER_TOLERANCE: float = 1.0e-8

# Per-joint URDF-declared effort limits (used by MuJoCo `actuatorfrcrange`)
URDF_EFFORT_HIP_THIGH: float = 35.278
URDF_EFFORT_CALF: float = 44.4

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
    # In this MJCF (aliengo.xml) every `<leg>_calf` body carries TWO sphere
    # geoms at z=-0.25: a visually-tagged `<leg>_foot` sphere with
    # `contype=0 conaffinity=0` (purely cosmetic, the dark ball you see in
    # renders) and an **unnamed** sphere of size 0.0265 that uses the file
    # default `contype=1 conaffinity=1` and is the one that actually
    # collides with the floor. We pick the latter (sphere + contype!=0) and
    # give it the canonical name `<leg>_foot_geom`, so the look-up table
    # `info.foot_geom_id_all` further down keeps working unchanged.
    #
    # Step-A: apply matched friction + solref/solimp to the foot geoms so the
    # contact parameters are symmetric with the floor side. MuJoCo's per-pair
    # mixing rule picks one value from each geom (defaults: max for friction,
    # min for solref), so leaving the foot at MJCF defaults
    # (`friction="1 0.3 0.3"`) would silently undo the floor-side change. We
    # set the same numbers on both sides to make the effective contact
    # behavior independent of the mixing rule.
    foot_friction = [
        FLOOR_SLIDE_FRICTION,
        FLOOR_TORSIONAL_FRICTION,
        FLOOR_ROLLING_FRICTION,
    ]
    for body in spec.bodies:
        name = body.name or ""
        if name.endswith("_calf") and name != "_calf":
            leg = name.removesuffix("_calf")
            for g in body.geoms:
                if (
                    g.type == mujoco.mjtGeom.mjGEOM_SPHERE
                    and (g.contype != 0 or g.conaffinity != 0)
                ):
                    g.name = _foot_geom_name(leg)
                    g.friction = foot_friction
                    g.solref = CONTACT_SOLREF
                    g.solimp = CONTACT_SOLIMP
                    break

    # ---- disable self-collisions on the robot (matches odri.py:104
    # `enabled_self_collisions=False`). Strategy: put all robot collision
    # geoms on `contype=2, conaffinity=1`, and the floor we patch below on
    # `contype=1, conaffinity=2`. Two robot geoms then share neither a
    # contype-vs-conaffinity bit, so they never collide; floor-vs-robot
    # collides on both sides.
    #
    # IMPORTANT (MJCF specific): aliengo.xml explicitly declares the visual
    # meshes (trunk/hip/thigh/calf) and the visual `<leg>_foot` ball as
    # `contype=0 conaffinity=0` -- they are decorative and must not enter
    # contact. Blindly overwriting their bits to (2,1) would silently
    # promote them into the collision set and produce phantom contacts
    # (e.g. the trunk mesh colliding with the floor at its bounding-box
    # extents). We therefore leave any geom already declared as
    # non-colliding alone.
    for body in spec.bodies:
        for g in body.geoms:
            if g.contype == 0 and g.conaffinity == 0:
                continue
            g.contype = 2
            g.conaffinity = 1

    # ---- The MJCF (aliengo.xml line 52) already declares
    # `<joint type="free"/>` on the root `trunk` body, so we do NOT add
    # another freejoint here. MjSpec compile would otherwise raise
    # "free joint not allowed: body already has a free joint".

    # ---- armature + step-A damping/frictionloss on every hinge joint.
    #
    # Armature matches IdealPDActuator armature in Isaac Lab.
    #
    # Damping/frictionloss are NOT present in IdealPDActuator (Isaac Lab
    # implements damping as a control-side -kd*qd term, not joint-side
    # viscous). However, PhysX's low-iteration implicit solver silently
    # introduces a small numerical damping that MuJoCo's IMPLICITFAST
    # integrator does not reproduce. The values below are tuned to be small
    # enough that the gap to Isaac Lab's training distribution stays narrow,
    # but large enough to suppress the high-frequency joint chatter that
    # makes the rear-stand policy diverge in MuJoCo.
    #
    # Note on shapes (MuJoCo 3.8 MjSpec):
    #   - `armature` and `frictionloss` are exposed as scalar floats.
    #   - `damping` (and `stiffness`) are exposed as length-3 ndarrays, with
    #     elements [0],[1],[2] mapping to the (up to) 3 DoFs of the joint.
    #     For a 1-DoF hinge, only element [0] is meaningful; we leave [1],[2]
    #     at zero. Assigning a scalar raises TypeError.
    damping_vec = np.array([JOINT_DAMPING, 0.0, 0.0], dtype=np.float64)
    for body in spec.bodies:
        for j in body.joints:
            if j.type == mujoco.mjtJoint.mjJNT_HINGE:
                j.armature = JOINT_ARMATURE
                j.damping = damping_vec
                j.frictionloss = JOINT_FRICTIONLOSS

    # ---- simulation options.
    spec.option.timestep = SIM_DT
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    # Step-A: use Newton with a tight tolerance. Newton typically converges
    # in <10 iterations for our smooth contact geometry, yielding lower
    # frame-to-frame contact-force noise than CG. This matters because the
    # rear-stand policy is open-loop driven by `data.qpos`/`data.qvel`; noisy
    # contact forces propagate into joint velocities and into the next
    # observation.
    spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON
    spec.option.iterations = SOLVER_ITERATIONS
    spec.option.tolerance = SOLVER_TOLERANCE

    # Offscreen framebuffer size — needed for headless EGL rendering at 1280x720.
    # Set above the largest render resolution we ever request.
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080

    # ---- world: patch the existing floor / keep the existing light /
    # tracking camera. The MJCF already provides:
    #   - <geom name="floor" type="plane" .../>            (line 48)
    #   - <light directional="true" .../>                  (line 46)
    #   - <camera name="track" mode="trackcom" .../>       (line 47)
    # We do NOT add duplicates (would just shadow/overlap). We just patch
    # the floor's friction / solref / solimp / contype-conaffinity to the
    # sim-to-sim Step-A values, mirroring what we set on the foot side
    # above so that MuJoCo's per-pair mixing rule is a no-op.
    floor_patched = False
    for g in spec.worldbody.geoms:
        if g.name == "floor":
            g.friction = [
                FLOOR_SLIDE_FRICTION,
                FLOOR_TORSIONAL_FRICTION,
                FLOOR_ROLLING_FRICTION,
            ]
            g.solref = CONTACT_SOLREF
            g.solimp = CONTACT_SOLIMP
            g.contype = 1
            g.conaffinity = 2
            floor_patched = True
            break
    if not floor_patched:
        raise RuntimeError(
            "MJCF must define a worldbody geom named 'floor' "
            "(expected for sim-to-sim contact tuning)."
        )

    # ---- The MJCF (aliengo.xml line 142-156) already declares 12
    # <motor gear="1" joint="..."/> actuators, one per hinge joint. Their
    # `name=` attributes exactly match `ISAAC_JOINT_NAMES` (the lookup at
    # the end of build_model uses `mj_name2id(mjOBJ_ACTUATOR, name)`), so
    # we do NOT add a second set here -- doing so would compile-time fail
    # with duplicate transmission targets.
    #
    # The MJCF default sets ctrlrange="-44.4 44.4" (wider than IsaacLab's
    # IdealPDActuator effort_limit=33.5). This is harmless because
    # `compute_pd_torque()` in sim_to_sim.py already clips to
    # ±JOINT_EFFORT_LIMIT before writing to data.ctrl, so the MJCF
    # ctrlrange is never reached and the effective behavior is identical
    # to URDF-built models that had forcerange/ctrlrange set to ±33.5
    # at actuator level.

    model = spec.compile()
    data = mujoco.MjData(model)

    # ---- build the lookup tables.
    info = ModelInfo()
    # Robot root body: MJCF calls it "trunk", the legacy URDF called it
    # "base". Rather than hard-coding either name, find the first body
    # whose parent is world (body_parentid == 0). For our setup that is
    # unambiguously the robot root.
    info.base_body_id = -1
    for b in range(1, model.nbody):
        if model.body_parentid[b] == 0:
            info.base_body_id = b
            break
    if info.base_body_id < 0:
        raise RuntimeError("robot root body not found in compiled model")
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

    # Build a joint_id -> actuator_id reverse-lookup table by reading the
    # compiled `actuator_trnid`. We do this instead of
    # `mj_name2id(mjOBJ_ACTUATOR, joint_name)` because the MJCF names its
    # motors "FR_hip" / "FR_thigh" / "FR_calf" while the joints they drive
    # are named "FR_hip_joint" etc. (aliengo.xml line 144-155 vs 61-71).
    # The transmission target is the joint, so trnid is always reliable.
    joint_to_actuator: Dict[int, int] = {}
    for aid in range(model.nu):
        if model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
            jid_for_aid = int(model.actuator_trnid[aid, 0])
            joint_to_actuator[jid_for_aid] = aid

    for i, name in enumerate(ISAAC_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"joint missing: {name}")
        aid = joint_to_actuator.get(jid, -1)
        if aid < 0:
            raise RuntimeError(f"no motor actuator drives joint: {name}")
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
