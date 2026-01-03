# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`SOLO12_CFG`: SOLO12 robot
* :obj:`SOLO12_MINIMAL_CFG`: SOLO12 robot with minimal collision bodies

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


SOLO12_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"exts/cat_envs/cat_envs/assets/Robots/odri/solo12_description/usd/solo12_description.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_contact_impulse=1e32,
            max_depenetration_velocity=100.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02, rest_offset=0.0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        joint_pos={
            "FL_HAA": 0.05,
            "FL_HFE": 0.4,
            "FL_KFE": -0.8,
            "FR_HAA": -0.05,
            "FR_HFE": 0.4,
            "FR_KFE": -0.8,
            "HR_HAA": -0.05,
            "HR_HFE": 0.4,
            "HR_KFE": -0.8,
            "HL_HAA": 0.05,
            "HL_HFE": 0.4,
            "HL_KFE": -0.8,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                "FL_HAA",
                "FL_HFE",
                "FL_KFE",
                "FR_HAA",
                "FR_HFE",
                "FR_KFE",
                "HR_HAA",
                "HR_HFE",
                "HR_KFE",
                "HL_HAA",
                "HL_HFE",
                "HL_KFE",
            ],
            armature=0.00036207,
            effort_limit=10.0,
            velocity_limit=100.0,
            stiffness={".*": 4.0},
            damping={".*": 0.2},
        ),
    },
)
SOLO12_MINIMAL_CFG = SOLO12_CFG.copy()


ALIENGO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"exts/cat_envs/cat_envs/assets/Robots/odri/AlienGo_description/usd/aliengo_description.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_contact_impulse=1e32,
            max_depenetration_velocity=100.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02, rest_offset=0.0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.9,
            "FL_calf_joint": -1.7,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.9,
            "FR_calf_joint": -1.7,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.9,
            "RR_calf_joint": -1.7,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.9,
            "RL_calf_joint": -1.7,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
            ],
            armature=0.00036207,
            effort_limit=33.5,
            velocity_limit=21.0,
            stiffness={".*": 25.0},
            damping={".*": 0.5},
        ),
    },
)
ALIENGO_MINIMAL_CFG = ALIENGO_CFG.copy()
