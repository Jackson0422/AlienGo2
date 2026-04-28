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

    # Base height:  r = -(z - z^c)^2
    base_height = RewTerm(
        func=custom_rewards.base_height_task,
        weight=1.0,
        params={
            "target_height": 0.65,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
   # Base pitch:   r = 1 + cos(p^c - p)   (max = 2 at target, min = 0 at opposite)
    base_pitch = RewTerm(
        func=custom_rewards.base_pitch_task,
        weight=1.0,
        params={
            "target_pitch_deg": 90.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Upright balance:  r = exp(-v_z^2/sigma) + exp(-p_dot^2/sigma)  if upright else 0
    upright_balance = RewTerm(
        func=custom_rewards.upright_balance,
        weight=0.5,
        params={
            "sigma_vz": 0.25,
            "sigma_pitch_rate": 0.25,
            "upright_pitch_deg": 60.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Support polygon:  r = -|v_x^c|^2 * (pi/2 - |atan2(dx_b, dz_b)|)^2  (conditional)
    support_polygon = RewTerm(
        func=custom_rewards.support_polygon,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
            "foot_cfg": SceneEntityCfg(
                "robot", body_names=["RL_calf", "RR_calf"]
            ),
        },
    )

    joint_acc_penalty = RewTerm(
        func=custom_rewards.joint_acceleration_penalty,
        weight=5e-9,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
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
    action_rate_rear_legs   = ConstraintTerm(
        func=constraints.action_rate,
        max_p=0.25,
        params={"limit": 150.0, # 150.0 is the default limit for the robot
                "asset_cfg": SceneEntityCfg("robot", joint_names=["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"])},
    )
    action_rate_front_legs = ConstraintTerm(
        func=constraints.action_rate,
        max_p=0.25,
        params={"limit": 150.0, # 80.0 is the default limit for the robot
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

    upsidedown = ConstraintTerm(
        func=constraints.upsidedown, 
        max_p=1.0,
        params={"limit": 0.2,
                "asset_cfg": SceneEntityCfg("robot")}
    )

    knee_contact = ConstraintTerm(
        func=constraints.knee_height,
        max_p=1.0,
        params={
            "min_z": 0.25 / 2 ** 0.5,   # ≈ 0.1768 m, 即 calf 倾斜 ≤ 45°
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[".*_calf"]
            ),
        },
    )

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

    base_height_low = DoneTerm(
        func=terminations.base_height_below_consecutive,  # Wrapper function / 使用包装函数
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": 0.30,
            "consecutive_frames": 40,  # Terminate only after 40 consecutive frames below threshold / 连续40帧低于阈值才终止
        },
    )

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

    # Add these inside CurriculumCfg:

    base_pitch_target = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "base_pitch",
            "param_name": "target_pitch_deg",
            "start_value": 10.0,   # early stage: only mildly lift the nose
            "end_value": 90.0,     # final target: fully upright
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
        },
    )

    base_height_target = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "base_height",
            "param_name": "target_height",
            "start_value": 0.40,   # start around natural stance height
            "end_value": 0.65,     # final target
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
        },
    )

    upright_balance_threshold = CurrTerm(
        func=curriculums.modify_reward_param,
        params={
            "term_name": "upright_balance",
            "param_name": "upright_pitch_deg",
            "start_value": 20.0,   # early: easy to enter upright gate
            "end_value": 75.0,     # final: only truly upright gets the bonus
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