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
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), scale=0.1
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
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
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
    # push_robot = EventTerm(
    #     # Standard push_by_setting_velocity also works, but interestingly results
    #     # in a different gait
    #     func=events.push_by_setting_velocity_with_random_envs,
    #     mode="interval",
    #     is_global_time=True,
    #     interval_range_s=(0.8, 1.5), # 0.005 is the default interval for the robot
    #     params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}}, # (-0.5, 0.5) is the default velocity range for the robot
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task: velocity tracking (minimal weight for standing task)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=0.0,  # Reduced from 0.1 - we want standing, not walking
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_exp,
    #     weight=0.5,
    #     params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    # )

    # ============ Height Control (Core Rewards for Biped Standing) ============
    
    # Continuous height progress: encourages exploration from 0.35m to 0.65m
    # This provides gradient at every step, avoiding "plateau problem"
    # height_progress = RewTerm(
    #     func=custom_rewards.base_height_progress,
    #     weight=1.5,
    #     params={
    #         "h0": 0.40,
    #         "h1": 0.60,
    #         "max_front_contact": 1.0,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "front_contact_cfg": SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
    #         "rear_contact_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
    #     },
    # )

    height_maintenance = RewTerm(
        func=custom_rewards.height_maintenance,
        weight=2.0,
        params={
            "target_height": 0.60,
            "sigma": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


    # Bonus reward when reaching target height
    upright_alive = RewTerm(
        func=custom_rewards.base_height_above,
        weight=1.0,  # Large bonus for achieving biped stance
        params={
            "min_height": 0.60,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Bonus reward for standing duration
    standing_duration_bonus = RewTerm(
        func=custom_rewards.standing_time_bonus_exponential,
        weight=1.0,
        params={
            "min_height": 0.50,
            "max_height": 0.60,
            "max_front_foot_contact": 1.0,
            "alpha": 2.0,
            "tau": 2.0,
            "delay": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "contact_cfg": SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
            "rear_contact_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
        },
    )

    # (A) 机身直立 - 重力向量一致性 Upright fuselage - Consistent gravity vector
    # upright_gravity = RewTerm(
    #     func=custom_rewards.upright_gravity_alignment,
    #     weight=1.5,
    #     params={
    #         "k_o": 5.0,
    #         "target_pitch_deg": 75.0,  # 75.0 is the default target pitch degree for the robot
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # (B) Roll 稳定 - 抑制侧翻 Roll stability - inhibits rollover
    roll_stable = RewTerm(
        func=custom_rewards.roll_stability,
        weight=0.3,
        params={
            "k_r": 10.0,
            "min_height": 0.50,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # (C) CoM-CoP 水平对齐 - 平衡核心 CoM-CoP Horizontal Alignment - Balanced Core
    com_cop_align = RewTerm(
        func=custom_rewards.com_cop_progress,
        weight=1.5,
        params={
            "d_max": 0.30,
            "min_height": 0.50,
            "max_front_contact": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_cfg": SceneEntityCfg("robot", body_names=["RL_calf", "RR_calf"]),
            "contact_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"]),
            "front_contact_cfg": SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
        },
    )

    # (D) CoP 居中 - 后足负载均衡  CoP Centering - Rear Foot Load Balancing
    cop_center = RewTerm(
        func=custom_rewards.cop_midpoint,
        weight=1.0,
        params={
            "k_cop": 50.0,
            "min_height": 0.50,
            "max_front_contact": 1.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_cfg": SceneEntityCfg("robot", body_names=["RL_calf", "RR_calf"]),
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
            "limit": 35.0,  # ✅ Hip 和 Thigh  35 N·m
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint"])
        },
    )

    # Calf limitation
    joint_torque_calf = ConstraintTerm(
        func=constraints.joint_torque,
        max_p=0.25,
        params={
            "limit": 45.0,  # ✅ Calf 45 N·m
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
    joint_acceleration = ConstraintTerm(
        func=constraints.joint_acceleration,
        max_p=0.25,
        params={"limit": 800.0, 
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
    )
    action_rate = ConstraintTerm(
        func=constraints.action_rate,
        max_p=0.25,
        params={"limit": 80.0, # 80.0 is the default limit for the robot
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])},
    )

    # Safety Hard constraints
    # Knee and base
    contact = ConstraintTerm(
        func=constraints.contact,
        max_p=1.0,
        params={
                "asset_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_thigh"])},
    )
    # The leg of AlienGo is hip - thigh - calf, so the foot contact force is the calf contact force
    # foot_contact_force = ConstraintTerm(
    #     func=constraints.foot_contact_force,
    #     max_p=1.0,
    #     params={"limit": 210.0,  # 50.0 is the default limit for the robot
    #             "asset_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf")},
    # )

    # 前脚不应该有接触力（直立时应悬空）
    front_foot_contact = ConstraintTerm(
        func=constraints.contact,  # 使用接触约束，不允许接触
        max_p = 1.0, # 1.0 is the default max_p for the contact constraint
        params={
            "asset_cfg": SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"])
        },
    )

    # 后脚接触力限制
    # rear_foot_contact_force = ConstraintTerm(
    #     func=constraints.foot_contact_force,
    #     max_p=1.0,
    #     params={
    #         "limit": 250.0,  # 后脚需要承受更大的力
    #         "asset_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"])
    #     },
    # )

    # front_hfe_position = ConstraintTerm(
    #     func=constraints.joint_position,
    #     max_p=1.0,
    #     params={"limit": 1.3, 
    #             "asset_cfg": SceneEntityCfg("robot", joint_names=["FL_thigh_joint", "FR_thigh_joint"])},
    # )

    upsidedown = ConstraintTerm(
        func=constraints.upsidedown, 
        max_p=1.0,
        params={"limit": 0.0,
                "asset_cfg": SceneEntityCfg("robot")}
    )

    # Style constraints
    hip_position = ConstraintTerm(
        func=constraints.joint_position_when_moving_forward,
        max_p=0.25,
        params={
            "limit": 0.4,  # 0.2 is the default limit for the robot
            "velocity_deadzone": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    # base_orientation = ConstraintTerm(
    #     func=constraints.base_orientation, 
    #     max_p=0.25, 
    #     params={
    #         "limit": 0.5,  # 0.26 is the default limit for the robot
    #         "asset_cfg": SceneEntityCfg("robot")}
    # )

    air_time = ConstraintTerm(
        func=constraints.air_time,
        max_p=0.25,
        params={
            "limit": 0.1, 
            "velocity_deadzone": 0.0,
            "asset_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"])},
    )

    # no_move = ConstraintTerm(
    #     func=constraints.no_move,
    #     max_p=0.05,
    #     params={
    #         "velocity_deadzone": 0.1,
    #         "joint_vel_limit": 10.0,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"])
    #     },
    # )

    one_foot_contact = ConstraintTerm(
        func=constraints.min_foot_contact,
        max_p=0.25,
        params={
            "min_feet": 1,
            "min_command_value": 0.0,
            "asset_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"])
        },
    )

    height_below = ConstraintTerm(
        func=constraints.height_below,
        max_p=1.0,
        params={
            "min_height": 0.55,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    base_contact = DoneTerm(
        func=terminations.illegal_contact_current_frame,  # 使用自定义函数，只检测当前帧
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base"]
            ),
            "threshold": 1.0,
        },
    )
    
    # thigh_contact = DoneTerm(
    #     func=terminations.illegal_contact_current_frame,  # 使用自定义函数，只检测当前帧
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces", body_names=[".*_thigh"]  # 只检测大腿
    #         ),
    #         "threshold": 1.0,
    #     },
    # )

    base_height_low = DoneTerm(
        func=terminations.base_height_below_consecutive,  # 使用包装函数
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": 0.30,
            "consecutive_frames": 40,  # 连续5帧低于阈值才终止
        },
    )

    # front_foot_contact = DoneTerm(
    #     func=terminations.illegal_contact_current_frame,  # 使用自定义函数，只检测当前帧
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
            "limit": 0.8,
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
            "term_name": "joint_torque_hip_thigh",  # ✅ 新的约束项名称
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    
    joint_torque_calf = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_torque_calf",  # ✅ 新的约束项名称
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

    joint_acceleration = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "joint_acceleration",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    action_rate = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "action_rate",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )

    # Style constraints
    hip_position = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "hip_position",
            "num_steps": 48 * MAX_CURRICULUM_ITERATIONS, # 24 is the default num_steps for the hip position constraint
            "init_max_p": 0.03, # 0.25 is the default init_max_p for the hip position constraint
        },
    )
    # base_orientation = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={
    #         "term_name": "base_orientation",
    #         "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
    #         "init_max_p": 0.25,
    #     },
    # )
    air_time = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "air_time",
            "num_steps": 48 * MAX_CURRICULUM_ITERATIONS, # 24 is the default num_steps for the air time constraint
            "init_max_p": 0.25,
        },
    )
    one_foot_contact = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "one_foot_contact",
            "num_steps": 48 * MAX_CURRICULUM_ITERATIONS, # 24 is the default num_steps for the one foot contact constraint
            "init_max_p": 0.25,
        },
    )
    # 渐进式收紧前脚接触约束
    front_foot_contact = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "front_foot_contact",
            "num_steps": 48 * MAX_CURRICULUM_ITERATIONS,  # 更长的渐进时间
            "init_max_p": 0.1,  # 初始不惩罚
            # 最终会到达 max_p=1.0（ConstraintsCfg 中设置的初始值）
        },
    )

    height_below = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "height_below",
            "num_steps": 48 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.01,  # 早期几乎不惩罚
        },
    )

    standing_min_height = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "standing_duration_bonus",
            "param_name": "min_height",
            "start_value": 0.45,
            "end_value": 0.50,
            "num_steps": 48 * MAX_CURRICULUM_ITERATIONS,
        },
    )

    # standing_duration_bonus: max_front_foot_contact 100 -> 1
    standing_max_front = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "standing_duration_bonus",
            "param_name": "max_front_foot_contact",
            "start_value": 40.0,
            "end_value": 1.0,
            "num_steps": 48 * MAX_CURRICULUM_ITERATIONS,
        },
    )

    # com_cop_align: min_height 0.45 -> 0.60
    com_cop_min_height = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "com_cop_align",
            "param_name": "min_height",
            "start_value": 0.40,
            "end_value": 0.50,
            "num_steps": 48 * MAX_CURRICULUM_ITERATIONS,
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
            "num_steps": 48 * MAX_CURRICULUM_ITERATIONS,
        },
    )

    # cop_center: 同理
    cop_center_min_height = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "cop_center",
            "param_name": "min_height",
            "start_value": 0.45,
            "end_value": 0.50,
            "num_steps": 48 * MAX_CURRICULUM_ITERATIONS,
        },
    )

    cop_center_max_front = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "cop_center",
            "param_name": "max_front_contact",
            "start_value": 40.0,
            "end_value": 1.0,
            "num_steps": 48 * MAX_CURRICULUM_ITERATIONS,
        },
    )

    # height_progress: max_front_contact 100 -> 1
    # height_progress_max_front = CurrTerm(
    #     func=curriculums.modify_reward_param,
    #     params={
    #         "term_name": "height_progress",
    #         "param_name": "max_front_contact",
    #         "start_value": 100.0,
    #         "end_value": 1.0,
    #         "num_steps": 48 * MAX_CURRICULUM_ITERATIONS,
    #     },
    # )

    # roll_stable: min_height 0.45 -> 0.60
    roll_stable_min_height = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "roll_stable",
            "param_name": "min_height",
            "start_value": 0.45,
            "end_value": 0.50,
            "num_steps": 48 * MAX_CURRICULUM_ITERATIONS,
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
            "num_steps": 48 * MAX_CURRICULUM_ITERATIONS,
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

        # set velocity command
        self.commands.base_velocity.ranges.lin_vel_x = (-0.3, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.7, 0.7)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.78, 0.78)
