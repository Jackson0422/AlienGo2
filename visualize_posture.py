from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.articulations import Articulation

world = World(stage_units_in_meters=1.0)

# Load Aliengo - use the same USD as in Isaac Sim
add_reference_to_stage(
    usd_path="https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Robots/Unitree/aliengo/aliengo.usd",
    prim_path="/World/aliengo",
)

world.scene.add_default_ground_plane()
world.reset()

robot = world.scene.add(Articulation(prim_path="/World/aliengo", name="aliengo"))
world.reset()

# Print all joint names to verify
print("Joint names:", robot.dof_names)

# Set joint positions
joint_positions = robot.get_joint_positions()
for name, val in [
    ("FL_hip_joint", 0.0),
    ("FL_thigh_joint", 2.20),
    ("FL_calf_joint", -1.7),
    ("FR_hip_joint", 0.0),
    ("FR_thigh_joint", 2.20),
    ("FR_calf_joint", -1.7),
]:
    idx = list(robot.dof_names).index(name)
    joint_positions[idx] = val

robot.set_joint_positions(joint_positions)

# Keep window open
while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()