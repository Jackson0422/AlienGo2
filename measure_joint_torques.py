#!/usr/bin/env python3
"""
Measure AlienGo joint torques in a specific pose using Isaac Sim simulation

This is a completely independent script that doesn't depend on any training environment or other code.
Only requires Isaac Sim and the basic robot model.
"""

import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

# Create argument parser
parser = argparse.ArgumentParser(description="Measure AlienGo joint torques")
parser.add_argument("--headless", action="store_true", help="Run in headless mode")

# Parse arguments and launch the application
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import Isaac Lab modules (must be after AppLauncher)
import omni
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import SimulationContext
from pxr import UsdPhysics


def main():
    """Main function"""
    
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device="cuda:0", gravity=(0.0, 0.0, -9.81))
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    
    # Add ground plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)
    
    # Add light source - dome light (ambient light)
    cfg_dome_light = sim_utils.DomeLightCfg(
        intensity=1000.0,
        color=(1.0, 1.0, 1.0),
    )
    cfg_dome_light.func("/World/DomeLight", cfg_dome_light)
    
    # Add directional light 1 (main light, from above)
    cfg_distant_light1 = sim_utils.DistantLightCfg(
        intensity=600.0,
        color=(1.0, 1.0, 1.0),
        angle=0.5,
    )
    cfg_distant_light1.func("/World/DistantLight1", cfg_distant_light1, translation=(0.0, 0.0, 10.0))
    
    # Add directional light 2 (fill light, from side)
    cfg_distant_light2 = sim_utils.DistantLightCfg(
        intensity=400.0,
        color=(1.0, 0.98, 0.95),
        angle=0.5,
    )
    cfg_distant_light2.func("/World/DistantLight2", cfg_distant_light2, translation=(5.0, 5.0, 5.0))
    
    # Add directional light 3 (fill light, from other side)
    cfg_distant_light3 = sim_utils.DistantLightCfg(
        intensity=300.0,
        color=(0.95, 0.98, 1.0),
        angle=0.5,
    )
    cfg_distant_light3.func("/World/DistantLight3", cfg_distant_light3, translation=(-5.0, 5.0, 5.0))
    
    # Create AlienGo robot configuration
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path="exts/cat_envs/cat_envs/assets/Robots/odri/AlienGo_description/usd/aliengo_description.usd",
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
            pos=(0.0, 0.0, 0.4),  # Height 0.4m (more stable height)
            joint_pos={
                "FL_hip_joint": 0.0,
                "FL_thigh_joint": 0.9,   # 0.9 rad (approx 51.6°, thigh more vertical)
                "FL_calf_joint": -1.7,    # -1.7 rad (approx -97.4°, moderate bend)
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
                    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                ],
                effort_limit=33.5,
                velocity_limit=21.0,
                stiffness={".*": 80.0},  # Increase stiffness
                damping={".*": 2.0},     # Increase damping
            ),
        },
    )
    
    # Create robot
    print("Creating AlienGo robot...")
    robot = Articulation(robot_cfg)
    
    # Start simulation
    sim.reset()
    print("Scene reset complete")
    
    # Get joint names
    joint_names = robot.data.joint_names
    print(f"\nNumber of robot joints: {len(joint_names)}")
    print(f"Joint name order: {joint_names}")
    
    # Build target position based on actual joint order
    target_angles = {
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
    }
    
    # Create target position tensor according to actual joint order
    target_list = [target_angles[name] for name in joint_names]
    target_joint_pos = torch.tensor([target_list], device=robot.device)
    
    print("\nTarget joint positions (rad):")
    for i, name in enumerate(joint_names):
        angle_rad = target_joint_pos[0, i].item()
        angle_deg = np.degrees(angle_rad)
        print(f"  {name:20s}: {angle_rad:7.3f} rad ({angle_deg:8.2f}°)")
    
    print("\nStarting simulation, waiting for system to stabilize...")
    print("=" * 80)
    
    # Run simulation to let system stabilize
    num_steps = 300  # Increased to 300 steps
    torque_history = []
    stable_count = 0
    stable_threshold = 50  # Need 50 consecutive stable steps
    
    for step in range(num_steps):
        # Use PD controller to maintain pose
        robot.set_joint_position_target(target_joint_pos)
        
        # Write data to simulation
        robot.write_data_to_sim()
        
        # Step simulation
        sim.step()
        
        # Update robot state
        robot.update(dt=sim.get_physics_dt())
        
        # Get current state
        current_pos = robot.data.joint_pos[0]
        current_vel = robot.data.joint_vel[0]
        base_pos = robot.data.root_pos_w[0]
        
        # Check if stable
        pos_error = torch.abs(current_pos - target_joint_pos[0]).max().item()
        vel_max = torch.abs(current_vel).max().item()
        
        if pos_error < 0.1 and vel_max < 0.5:  # Position error < 0.1rad, velocity < 0.5rad/s
            stable_count += 1
        else:
            stable_count = 0
        
        # Record torque (last 50 steps for averaging)
        if step >= num_steps - 50 or (stable_count > 20):
            applied_torque = robot.data.applied_torque.clone()
            torque_history.append(applied_torque)
        
        # Print status every 20 steps
        if (step + 1) % 20 == 0:
            current_torque = robot.data.applied_torque[0]
            
            print(f"\nStep {step + 1}/{num_steps}:")
            print(f"  Base height: {base_pos[2]:.4f} m")
            print(f"  Position error (max): {pos_error:.6f} rad")
            print(f"  Velocity (max): {vel_max:.6f} rad/s")
            print(f"  Current max torque: {torch.abs(current_torque).max():.3f} Nm")
            print(f"  Stable count: {stable_count}")
        
        # If stable for long enough, end early
        if stable_count >= stable_threshold:
            print(f"\n✓ Robot is stable! Ending early at step {step + 1}.")
            break
    
    # Calculate average torque
    if len(torque_history) == 0:
        print("\n⚠️  Warning: No torque data collected!")
        simulation_app.close()
        return
    
    torque_history = torch.stack(torque_history)
    print(f"\nCollected {len(torque_history)} steps of torque data")
    
    avg_torque = torque_history.mean(dim=0)[0]
    std_torque = torque_history.std(dim=0)[0]
    max_torque_per_joint = torque_history.abs().max(dim=0)[0][0]
    max_torque_overall = torque_history.abs().max().item()
    
    # Final state
    final_pos = robot.data.joint_pos[0]
    final_vel = robot.data.joint_vel[0]
    final_base_pos = robot.data.root_pos_w[0]
    
    print("\n" + "=" * 80)
    print("Measurement Results")
    print("=" * 80)
    
    print(f"\nFinal base position: [{final_base_pos[0]:.4f}, {final_base_pos[1]:.4f}, {final_base_pos[2]:.4f}] m")
    print(f"Target base height: 0.35 m")
    print(f"Height deviation: {final_base_pos[2] - 0.35:.4f} m")
    
    print("\nJoint States:")
    print("-" * 80)
    print(f"{'Joint Name':<20s} | {'Target(rad)':<10s} | {'Actual(rad)':<10s} | {'Error(rad)':<10s} | {'Velocity(rad/s)':<12s}")
    print("-" * 80)
    for i, name in enumerate(joint_names):
        target = target_joint_pos[0, i].item()
        actual = final_pos[i].item()
        error = actual - target
        vel = final_vel[i].item()
        print(f"{name:<20s} | {target:10.4f} | {actual:10.4f} | {error:10.6f} | {vel:12.6f}")
    
    print("\n" + "=" * 80)
    print("Joint Torque Measurements (Last 20 steps average)")
    print("=" * 80)
    print(f"{'Joint Name':<20s} | {'Avg Torque(Nm)':<15s} | {'Std Dev(Nm)':<12s} | {'Abs Avg':<12s}")
    print("-" * 80)
    
    torque_by_type = {'hip': [], 'thigh': [], 'calf': []}
    
    for i, name in enumerate(joint_names):
        avg = avg_torque[i].item()
        std = std_torque[i].item()
        abs_avg = torque_history[:, 0, i].abs().mean().item()
        
        print(f"{name:<20s} | {avg:15.6f} | {std:12.6f} | {abs_avg:12.6f}")
        
        # Classify statistics
        if 'hip' in name:
            torque_by_type['hip'].append(abs_avg)
        elif 'thigh' in name:
            torque_by_type['thigh'].append(abs_avg)
        elif 'calf' in name:
            torque_by_type['calf'].append(abs_avg)
    
    print("\n" + "=" * 80)
    print("Statistics by Joint Type")
    print("=" * 80)
    print(f"{'Joint Type':<10s} | {'Max(Nm)':<12s} | {'Min(Nm)':<12s} | {'Avg(Nm)':<12s}")
    print("-" * 80)
    
    for joint_type, torques in torque_by_type.items():
        if torques:
            print(f"{joint_type.upper():<10s} | {max(torques):12.6f} | {min(torques):12.6f} | {np.mean(torques):12.6f}")
    
    print("\n" + "=" * 80)
    print("Actuator Capability Comparison")
    print("=" * 80)
    print(f"Actuator torque limit: 33.5 Nm")
    print(f"Measured max torque: {max_torque_overall:.6f} Nm")
    print(f"Max torque percentage: {max_torque_overall / 33.5 * 100:.2f}%")
    
    overall_max = max(max(torques) for torques in torque_by_type.values() if torques)
    safety_factor = 33.5 / overall_max
    
    print(f"Safety factor: {safety_factor:.2f}x")
    
    if overall_max > 33.5:
        print("\n⚠️  Warning: Required torque exceeds actuator limit!")
    elif safety_factor < 2.0:
        print("\n⚠️  Note: Safety factor is low, may be unstable.")
    else:
        print("\n✓ Torque requirements are within safe range.")
    
    print("\n" + "=" * 80)
    print("Test Configuration:")
    print("=" * 80)
    print(f"  Initial height: 0.4 m")
    print(f"  Joint angles: thigh=0.9 rad (51.6°), calf=-1.7 rad (-97.4°)")
    print(f"  PD controller stiffness: 80.0")
    print(f"  PD controller damping: 2.0")
    print(f"  Actuator torque limit: 33.5 Nm")
    
    print("\n" + "=" * 80)
    print("Notes:")
    print("  - Positive torque: Joint applies moment in positive direction")
    print("  - Negative torque: Joint applies moment in negative direction")
    print("  - This measurement is from actual physics calculation in simulation")
    print("  - Includes effects of gravity, inertia and ground contact forces")
    print("=" * 80)
    
    # Close simulation
    simulation_app.close()


if __name__ == "__main__":
    main()
