# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from cat_envs.tasks.utils.cat.manager_constraint_cfg import (
    ConstraintTermCfg as ConstraintTerm,
)
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import cat_envs.tasks.utils.cat.constraints as constraints
import cat_envs.tasks.utils.cat.curriculums as curriculums

import cat_envs.tasks.utils.mdp.terminations as terminations
import cat_envs.tasks.utils.mdp.events as events
import cat_envs.tasks.utils.mdp.commands as commands
import cat_envs.tasks.utils.mdp.rewards as custom_rewards

import cat_envs.tasks.utils.mdp.observations as custom_obs

base_height_checker = terminations.BaseHeightBelowConsecutive()

##
# Pre-defined configs
##
from cat_envs.assets.odri import ALIENGO_MINIMAL_CFG


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ALIENGO_MINIMAL_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = commands.UniformVelocityCommandWithDeadzoneCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        velocity_deadzone=1.0,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0) # (-0.78, 0.78) is the default target velocity range for the robot
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
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
        scale=0.3, # 0.5 is the default scale for the robot like a weight for the output action
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.001, n_max=0.001), scale=0.25
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            scale=2.0,
        )
        base_height = ObsTerm(
            func=mdp.base_pos_z,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=2.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=(2.0, 2.0, 0.25),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), scale=0.7
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"], preserve_order=True)
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"], preserve_order=True)
            },
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.05,
        )
        foot_contact = ObsTerm(
            func=custom_obs.foot_contact_bool,
            params={
                "threshold": 1.0,
                "contact_cfg": SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf", "RL_calf", "RR_calf"]),
            },
            scale=1.0,
        )   
        rear_fz = ObsTerm(
            func=custom_obs.foot_contact_fz,
            params={
                "contact_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
            },
            noise=Unoise(n_min=-5.0, n_max=5.0),
            scale=0.005,  # 归一化：200N * 0.005 = 1.0
        )
        com_cop_xy = ObsTerm(
            func=custom_obs.com_cop_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "foot_cfg": SceneEntityCfg("robot", body_names=["RL_calf", "RR_calf"]),
                "contact_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=5.0,  # 归一化：0.2m * 5.0 = 1.0
        )
        actions = ObsTerm(func=mdp.last_action, scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 100,
        },
    )
    # give different position and velocity range for different environments
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (0, 0), # the default yaw range is (-1.57, 1.57)
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval

    # set pushing every step, as only some of the environments are chosen
    # as in the isaacgym cat version
    push_robot = EventTerm(
        # Standard push_by_setting_velocity also works, but interestingly results
        # in a different gait
        func=events.push_by_setting_velocity_with_random_envs,
        mode="interval",
        is_global_time=True,
        interval_range_s=(1.5, 3.0), # 0.005 is the default interval for the robot
        params={"velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}}, # (-0.5, 0.5) is the default velocity range for the robot
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task: velocity tracking (minimal weight for standing task)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,  # Reduced from 0.1 - we want standing, not walking
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    height_maintenance = RewTerm(
        func=custom_rewards.height_maintenance,
        weight=1.0,
        params={
            "target_height": 0.65,
            "sigma": 0.05,
            "target_pitch_deg": 80.0,
            "pitch_sigma_deg": 25.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # (C) CoM-CoP Horizontal Alignment - Balance Core / CoM-CoP 水平对齐 - 平衡核心
    com_cop_align = RewTerm(
        func=custom_rewards.com_cop_correction,
        weight=0.7,
        params={
            "d_max": 0.15,
            "k": 2.0,
            "correction_scale": 2.0,
            "min_height": 0.60,
            "max_front_contact": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_cfg": SceneEntityCfg("robot", body_names=["RL_calf", "RR_calf"]),
            "contact_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
            "front_contact_cfg": SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
        },
    )

    rear_leg_alive = RewTerm(
        func=custom_rewards.rear_stand_alive,
        weight=0.5,  # 根据需要调整权重
        params={
            "max_front_contact": 1.0,   # 前脚接触力低于此值视为离地
            "min_rear_contact": 1.0,    # 后脚接触力高于此值视为着地
            "min_height": 0.45,         # 最低高度门控
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
            "front_contact_cfg": SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
        },
    )

@configclass
class ConstraintsCfg:
    # Safety Soft constraints
    joint_torque_hip_thigh = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25,
        params={
            "limit": 35.0,  # Hip and Thigh 35 N·m / Hip 和 Thigh 35 N·m
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint"])
        },
    )

    # Calf limitation
    joint_torque_calf = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25,
        params={
            "limit": 45.0,  # Calf 45 N·m
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint"])
        },
    )

    joint_velocity_hip_thigh = ConstraintTerm(
        func=constraints.joint_velocity,
        max_p=0.25,
        params={"limit": 20.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint"])},
    )

    joint_velocity_calf = ConstraintTerm(
        func=constraints.joint_velocity,
        max_p=0.25,
        params={"limit": 16.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint"])},
    )
    joint_acceleration_hip_thigh = ConstraintTerm(
        func=constraints.joint_acceleration,
        max_p=0.25,
        params={"limit": 200.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint"])},
    )
    joint_acceleration_calf = ConstraintTerm(
        func=constraints.joint_acceleration,
        max_p=0.25,
        params={"limit": 150.0,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint"])},
    )
    action_rate_rear_legs   = ConstraintTerm(
        func=constraints.action_rate,
        max_p=0.25,
        params={"limit": 150.0, # 150.0 is the default limit for the robot
                "asset_cfg": SceneEntityCfg("robot", joint_names=["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"])},
    )
    action_rate_front_legs = ConstraintTerm(
        func=constraints.action_rate,
        max_p=0.25,
        params={"limit": 30.0, # 80.0 is the default limit for the robot
                "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"])},
    )

    # Safety Hard constraints
    # Knee and base
    contact = ConstraintTerm(
        func=constraints.contact,
        max_p=1.0,
        params={
            "threshold": 1.0,
            "asset_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_thigh"])},
    )
    # The leg of AlienGo is hip - thigh - calf, so the foot contact force is the calf contact force
    # foot_contact_force = ConstraintTerm(
    #     func=constraints.foot_contact_force,
    #     max_p=1.0,
    #     params={"limit": 210.0,  # 50.0 is the default limit for the robot
    #             "asset_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf")},
    # )

    # Front feet should not have contact force (should be airborne when standing) / 前脚不应该有接触力（直立时应悬空）
    front_foot_contact = ConstraintTerm(
        func=constraints.contact,  # Contact constraint, no contact allowed / 使用接触约束，不允许接触
        max_p = 0.25, # 1.0 is the default max_p for the contact constraint
        params={
            "threshold": 1.0,
            "asset_cfg": SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"])
        },
    )

    upsidedown = ConstraintTerm(
        func=constraints.upsidedown, 
        max_p=1.0,
        params={"limit": 0.2,
                "asset_cfg": SceneEntityCfg("robot")}
    )

    # Style constraints
    air_time = ConstraintTerm(
        func=constraints.air_time,
        max_p=0.25,
        params={
            "limit": 0.2, 
            "velocity_deadzone": 0.0,
            "asset_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"])},
    )

    one_foot_contact = ConstraintTerm(
        func=constraints.min_foot_contact,
        max_p=0.25,
        params={
            "min_feet": 1,
            "min_command_value": 0.0,
            "asset_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
            "min_height": 0.50,
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

    hip_position = ConstraintTerm(
        func=constraints.joint_position,
        max_p=0.25,
        params={
            "limit": 0.40,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])
        },
    )

    height_below = ConstraintTerm(
        func=constraints.height_below,
        max_p=0.25,
        params={
            "min_height": 0.60,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # yaw_rate = ConstraintTerm(
    #     func=constraints.base_ang_vel_z_when_standing,
    #     max_p=0.25,
    #     params={
    #         "limit": 0.3,
    #         "min_height": 0.60,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=terminations.illegal_contact_current_frame,  # Custom function, only checks current frame / 使用自定义函数，只检测当前帧
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base"]
            ),
            "threshold": 1.0,
        },
    )
    
    # thigh_contact = DoneTerm(
    #     func=terminations.illegal_contact_current_frame,  # Custom function, only checks current frame / 使用自定义函数，只检测当前帧
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", body_names=[".*_thigh"]  # Only detect thigh / 只检测大腿
    #         ),
    #         "threshold": 1.0,
    #     },
    # )

    base_height_low = DoneTerm(
        func=terminations.base_height_below_consecutive,  # Wrapper function / 使用包装函数
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": 0.30,
            "consecutive_frames": 40,  # Terminate only after 40 consecutive frames below threshold / 连续40帧低于阈值才终止
        },
    )

    # front_foot_contact = DoneTerm(
    #     func=terminations.illegal_contact_current_frame,  # Custom function, only checks current frame / 使用自定义函数，只检测当前帧
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", body_names=["FL_calf", "FR_calf"]
    #         ),
    #         "threshold": 1.0,
    #     },
    # )

    upside_down = DoneTerm(
        func=terminations.upside_down,
        params={
            "limit": 1.2,
        },
    )


MAX_CURRICULUM_ITERATIONS = 1000


@configclass
class CurriculumCfg:
    # Safety Soft constraints
    # joint_torque = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "joint_torque",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.25,
    #     },
    # )

    joint_torque_hip_thigh = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_torque_hip_thigh",  # New constraint term name / 新的约束项名称
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    
    joint_torque_calf = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_torque_calf",  # New constraint term name / 新的约束项名称
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )

    joint_velocity_hip_thigh = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_velocity_hip_thigh",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )

    joint_velocity_calf = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_velocity_calf",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )

    joint_acceleration_hip_thigh = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_acceleration_hip_thigh",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    joint_acceleration_calf = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_acceleration_calf",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    action_rate_rear_legs = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "action_rate_rear_legs",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    action_rate_front_legs = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "action_rate_front_legs",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    # Style constraints
    air_time = CurrTerm(
        func=curriculums.modify_constraint_p_custom,
        params={
            "term_name": "air_time",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS, # 24 is the default num_steps for the air time constraint
            "start_max_p": 0.1,
            "end_max_p": 0.25,
        },
    )
    one_foot_contact = CurrTerm(
        func=curriculums.modify_constraint_p_custom,
        params={
            "term_name": "one_foot_contact",
            "num_steps": 12 * MAX_CURRICULUM_ITERATIONS, # 24 is the default num_steps for the one foot contact constraint
            "start_max_p": 0.1,
            "end_max_p": 1.0,
        },
    )
    # Progressively tighten front foot contact constraint / 渐进式收紧前脚接触约束
    front_foot_contact = CurrTerm(
        func=curriculums.modify_constraint_p_custom,
        params={
            "term_name": "front_foot_contact",
            "num_steps": 6 * MAX_CURRICULUM_ITERATIONS,
            "start_max_p": 0.1,   # 从 0.1 开始
            "end_max_p": 1.0,     # 涨到 1.0
        },
    )

    height_below = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "height_below",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,  # Low penalty early on / 早期低惩罚
        },
    )
    # tarhet height
    height_maintenance_target = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "height_maintenance",
            "param_name": "target_height",
            "start_value": 0.45,
            "end_value": 0.65,
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
        },
    )
    # height_maintenance: sigma 0.20 -> 0.05
    height_maintenance_sigma = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "height_maintenance",
            "param_name": "sigma",
            "start_value": 0.20,   # 早期宽松，远处也有梯度
            "end_value": 0.05,     # 后期收紧，精确定位
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
        },
    )
    # min_pitch_deg curriculum
    height_maintenance_min_pitch_deg = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "height_maintenance",
            "param_name": "target_pitch_deg",
            "start_value": 10.0,
            "end_value": 80.0,
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
        },
    )

    height_maintenance_pitch_sigma_deg = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "height_maintenance",
            "param_name": "pitch_sigma_deg",
            "start_value": 25.0,
            "end_value": 5.0,
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
        },
    )

    # com_cop_align: min_height 0.45 -> 0.60
    com_cop_min_height = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "com_cop_align",
            "param_name": "min_height",
            "start_value": 0.40,
            "end_value": 0.60,
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
        },
    )

    # com_cop_align: max_front_contact 100 -> 1
    com_cop_max_front = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "com_cop_align",
            "param_name": "max_front_contact",
            "start_value": 40.0,
            "end_value": 1.0,
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
        },
    )

    # com_cop_align: d_max 0.25 -> 0.30
    com_cop_d_max = CurrTerm(
        func = curriculums.modify_reward_param,
        params = {
            "term_name": "com_cop_align",
            "param_name": "d_max",
            "start_value": 0.30,
            "end_value": 0.10,
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
        },
    )

    # rear_leg_alive: min_height 0.45 -> 0.60
    rear_leg_alive_min_height = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "rear_leg_alive",
            "param_name": "min_height",
            "start_value": 0.45,
            "end_value": 0.60,
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
        },
    )

    

##
# Environment configuration
##


@configclass
class AlienGoFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0

        # simulation settings
        self.sim.solver_type = 0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.max_position_iteration_count = 4
        self.sim.max_velocity_iteration_count = 1
        self.sim.bounce_threshold_velocity = 0.2
        self.sim.gpu_max_rigid_contact_count = 33554432
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


class AlienGoFlatEnvCfg_PLAY(AlienGoFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 3.0

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None

        # set velocity command
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
