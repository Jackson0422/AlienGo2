"""Construct the 57-dim policy observation in MuJoCo, byte-for-byte matching
the Isaac Lab `ObservationsCfg.PolicyCfg` defined in
`exts/cat_envs/cat_envs/tasks/locomotion/velocity/config/aliengo/cat_flat_env_cfg.py`
(lines 142-213).

Observation layout (after concatenation, scales applied):

    [0:3]   base_ang_vel   * 0.25                                    (body frame)
    [3:6]   base_lin_vel   * 2.0                                     (body frame)
    [6:7]   base_height    * 2.0                                     (world z of base)
    [7:10]  velocity_cmd   * (2.0, 2.0, 0.25)                        (zeros in PLAY)
    [10:13] projected_grav * 0.7                                     (body frame, ||g||=1)
    [13:25] joint_pos      * 1.0                                     (ABSOLUTE, Isaac order)
    [25:37] joint_vel      * 0.05                                    (absolute, Isaac order)
    [37:41] foot_contact   * 1.0                                     (FL, FR, RL, RR binary)
    [41:43] rear_fz        * 0.005                                   (RL, RR world Fz clamp(0))
    [43:45] com_cop_xy     * 5.0                                     (com_xy - cop_xy in world)
    [45:57] last_action    * 1.0                                     (raw policy output, Isaac order)

PLAY-mode caveats (cat_flat_env_cfg.py:694-696):
- enable_corruption=False → no observation noise.
- velocity command range collapses to zero, so the commands vector is all zeros.
- push_robot disabled.

Sim-to-sim alignment checklist (each item is enforced below):
1. Joint order matches ISAAC_JOINT_NAMES. ✔
2. joint_pos is ABSOLUTE (mdp.joint_pos, not joint_pos_rel). ✔
3. action scale 0.3 + default offset is applied by sim_to_sim.py (NOT here).
4. last_action is the previous raw policy output (mu), not q_des. ✔
5. base velocities expressed in body (root) frame, exactly as
   `quat_rotate_inverse(root_quat_w, root_lin_vel_w)`. ✔
6. projected_gravity uses GRAVITY_VEC_W = (0, 0, -1). ✔
7. Quaternion order is (w, x, y, z) for both Isaac Lab and MuJoCo qpos. ✔
8. RunningMeanStd normalization is done INSIDE the policy network on the
   GPU; we feed unnormalized observation here. ✔
"""

from __future__ import annotations

from typing import Sequence

import mujoco
import numpy as np

from model_builder import ModelInfo, FOOT_LOCAL_OFFSET_Z


OBS_DIM = 57


# Scales as written in PolicyCfg (cat_flat_env_cfg.py:142-213). Keep the
# numeric duplication here so we don't accidentally drift if Isaac Lab evolves.
SCALE_BASE_ANG_VEL = 0.25
SCALE_BASE_LIN_VEL = 2.0
SCALE_BASE_HEIGHT = 2.0
SCALE_VEL_CMD = np.array([2.0, 2.0, 0.25], dtype=np.float64)
SCALE_PROJECTED_GRAV = 0.7
SCALE_JOINT_POS = 1.0
SCALE_JOINT_VEL = 0.05
SCALE_FOOT_CONTACT = 1.0
SCALE_REAR_FZ = 0.005
SCALE_COM_COP = 5.0
SCALE_LAST_ACTION = 1.0

FOOT_CONTACT_THRESHOLD = 1.0  # N, matches observations.foot_contact_bool threshold


def _quat_rotate_inverse_wxyz(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by the INVERSE of quaternion q.

    Verbatim numpy port of `isaaclab.utils.math.quat_rotate_inverse`
    (see IsaacLab/source/isaaclab/isaaclab/utils/math.py:606-625). The
    quaternion is (w, x, y, z), matching MuJoCo's `data.qpos[3:7]` layout.

    Formula:
        a = v * (2 * w^2 - 1)
        b = 2 * w * cross(q_xyz, v)
        c = 2 * q_xyz * (q_xyz . v)
        result = a - b + c

    This is the body-frame projection of a world-frame vector when q is
    the body's orientation (world←body): for example, Isaac Lab uses it
    to compute `root_lin_vel_b` and `projected_gravity_b`.
    """
    q_w = q_wxyz[0]
    q_xyz = np.asarray(q_wxyz[1:4], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_xyz, v) * (q_w * 2.0)
    c = q_xyz * (np.dot(q_xyz, v) * 2.0)
    return a - b + c


def _contact_force_on_geom_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> np.ndarray:
    """Sum of contact forces ACTING ON the given geom, in world frame.

    Mirrors Isaac Lab `ContactSensor.data.net_forces_w` semantics: force applied
    by everything else onto this geom's body, as a 3-vector in world frame.

    MuJoCo sign convention (verified empirically with a free-fall box):
      * `contact.frame` rows are (normal, tan1, tan2) in world coords.
      * `mj_contactForce` returns the wrench in the contact frame, where the
        force component is the action of GEOM1 on GEOM2 (i.e. +normal force
        means geom1 pushes geom2 along the normal direction).
      * For a box settling on floor: floor=g1, box=g2; result[0] is the
        positive normal force, and `R.T @ result[:3]` is the upward force ON
        the box — exactly what Isaac Lab's net_forces_w would report.
    """
    f_world = np.zeros(3, dtype=np.float64)
    if data.ncon == 0:
        return f_world

    result = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g2 == geom_id:
            sign = 1.0       # result already expresses the force on geom2
        elif g1 == geom_id:
            sign = -1.0      # force on geom1 is the reaction (-result)
        else:
            continue
        mujoco.mj_contactForce(model, data, i, result)
        R = np.array(c.frame, dtype=np.float64).reshape(3, 3)
        f_world += sign * (R.T @ result[:3])
    return f_world


def build_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    info: ModelInfo,
    last_action_raw: np.ndarray,
    velocity_command: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Compose the 57-d observation vector.

    Args:
        model, data: MuJoCo state, AFTER the latest mj_step (so that derived
            quantities like xipos/geom_xpos are up to date — `mj_step` does
            an implicit `mj_forward` at the start of the next step, but for
            the very first observation make sure to call `mj_forward` after
            reset).
        info: id tables from `model_builder.build_model`.
        last_action_raw: shape (12,) numpy array of the previous policy
            output BEFORE the action scale + default offset. Pass zeros on
            the first step (matches Isaac Lab `last_action` semantics at
            reset).
        velocity_command: (vx, vy, wz) command in BODY frame. Defaults to
            zero to mirror the _PLAY config; pass nonzero values to test
            tracking behavior in MuJoCo without retraining.

    Returns:
        obs: shape (57,) float32 numpy array.
    """
    # --- base pose & velocities ---
    base_qpos = data.qpos[info.base_qpos_addr : info.base_qpos_addr + 7]
    base_pos_w = base_qpos[:3]
    base_quat_wxyz = base_qpos[3:7]

    base_qvel = data.qvel[info.base_qvel_addr : info.base_qvel_addr + 6]

    # MuJoCo free joint:
    # qvel[:3]  = base linear velocity, usually treated as world-frame
    # qvel[3:6] = base angular velocity in body frame
    base_lin_vel_w = base_qvel[:3]
    base_ang_vel_b = base_qvel[3:6]

    base_lin_vel_b = _quat_rotate_inverse_wxyz(base_quat_wxyz, base_lin_vel_w)
    projected_grav_b = _quat_rotate_inverse_wxyz(
        base_quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float64)
    )

    # --- joint state, in Isaac Lab order ---
    qpos_isaac = data.qpos[info.isaac_joint_qpos_addr]
    qvel_isaac = data.qvel[info.isaac_joint_qvel_addr]

    # --- foot contact (FL, FR, RL, RR), threshold 1.0 N ---
    foot_force_w = np.zeros((4, 3), dtype=np.float64)
    for i in range(4):
        foot_force_w[i] = _contact_force_on_body_world(model, data, int(info.calf_body_id_all[i]))
    foot_force_mag = np.linalg.norm(foot_force_w, axis=1)
    foot_contact = (foot_force_mag > FOOT_CONTACT_THRESHOLD).astype(np.float64)

    # --- rear Fz clamped to 0 (RL, RR) ---
    rear_idx = np.array([2, 3])  # RL, RR within FL,FR,RL,RR order
    rear_fz = np.clip(foot_force_w[rear_idx, 2], a_min=0.0, a_max=None)

    # --- com / cop offset in xy (world frame), using rear feet ---
    # CoM = mass-weighted sum of body_ipos (Cartesian inertial position in world).
    com_w = (data.xipos[info.com_body_ids] * info.com_body_mass[:, None]).sum(axis=0) / info.total_mass
    com_xy = com_w[:2]

    # Foot world xy is taken from the foot geom centers (= calf + (0,0,-0.25) in calf frame
    # because the URDF foot_fixed joint offset was preserved at compile time).
    rear_foot_w_xy = data.geom_xpos[info.rear_foot_geom_id, :2]
    fz_rear = np.clip(foot_force_w[rear_idx, 2:3], a_min=0.0, a_max=None)  # (2, 1)

    fz_sum = float(fz_rear.sum())
    if fz_sum > 1.0:
        cop_xy = (rear_foot_w_xy * fz_rear).sum(axis=0) / fz_sum
    else:
        # no reliable rear contact: do NOT fall back to world origin
        cop_xy = rear_foot_w_xy.mean(axis=0)

    com_cop_xy = com_xy - cop_xy

    # keep policy input inside a sane range before scale=5.0
    com_cop_xy = np.clip(com_cop_xy, -0.25, 0.25)

    # --- assemble in the exact PolicyCfg ordering ---
    cmd = np.asarray(velocity_command, dtype=np.float64)
    if cmd.shape != (3,):
        raise ValueError(f"velocity_command must be 3-d, got shape {cmd.shape}")

    obs = np.concatenate([
        base_ang_vel_b * SCALE_BASE_ANG_VEL,                              # 3
        base_lin_vel_b * SCALE_BASE_LIN_VEL,                              # 3
        np.array([base_pos_w[2]], dtype=np.float64) * SCALE_BASE_HEIGHT,  # 1
        cmd * SCALE_VEL_CMD,                                              # 3
        projected_grav_b * SCALE_PROJECTED_GRAV,                          # 3
        qpos_isaac * SCALE_JOINT_POS,                                     # 12
        qvel_isaac * SCALE_JOINT_VEL,                                     # 12
        foot_contact * SCALE_FOOT_CONTACT,                                # 4
        rear_fz * SCALE_REAR_FZ,                                          # 2
        com_cop_xy * SCALE_COM_COP,                                       # 2
        np.asarray(last_action_raw, dtype=np.float64) * SCALE_LAST_ACTION,  # 12
    ]).astype(np.float32)

    assert obs.shape == (OBS_DIM,), f"obs has wrong shape {obs.shape}"
    return obs

def _contact_force_on_body_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> np.ndarray:
    f_world = np.zeros(3, dtype=np.float64)
    if data.ncon == 0:
        return f_world

    result = np.zeros(6, dtype=np.float64)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])

        if b2 == body_id:
            sign = 1.0
        elif b1 == body_id:
            sign = -1.0
        else:
            continue

        mujoco.mj_contactForce(model, data, i, result)
        R = np.array(c.frame, dtype=np.float64).reshape(3, 3)
        f_world += sign * (R.T @ result[:3])

    return f_world