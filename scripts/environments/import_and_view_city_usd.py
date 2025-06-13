import argparse
import sys

from isaacsim import SimulationApp

# This sample loads a usd stage and starts simulation
CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}

# Set up command line arguments
parser = argparse.ArgumentParser("Usd Load sample")
parser.add_argument(
    "--usd_path", type=str, help="Path to usd file, should be relative to your default assets folder", required=True
)
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")

args, unknown = parser.parse_known_args()
# Start the omniverse application
usd_path = args.usd_path
kit = SimulationApp(launch_config=CONFIG)

import carb
import omni

# Locate Isaac Sim assets folder to load sample
from isaacsim.storage.native import get_assets_root_path, is_file

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    kit.close()
    sys.exit()
# make sure the file exists before we try to open it
try:
    result = is_file(usd_path)
except:
    result = False

if result:
    omni.usd.get_context().open_stage(usd_path)
else:
    carb.log_error(
        f"the usd path {usd_path} could not be opened, please make sure that {args.usd_path} is a valid usd file in {assets_root_path}"
    )
    kit.close()
    sys.exit()
# Wait two frames so that stage starts loading
kit.update()
kit.update()

print("Loading stage...")

from isaacsim.core.utils.stage import is_stage_loading
from isaacsim.sensors.camera import Camera
from isaacsim.core.api import World
import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import torch
from pxr import Gf, Sdf, Semantics, Usd, UsdGeom, Vt
import isaacsim.core.utils.stage as stage_utils

while is_stage_loading():
    kit.update()
print("Loading Complete")

from isaaclab.devices import Se3Keyboard
from isaaclab.managers import TerminationTermCfg as DoneTerm
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

def main():
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance():
        """Reset the environment to its initial state.

        This callback is triggered when the user presses the reset key (typically 'R').
        It's useful when:
        - The robot gets into an undesirable configuration
        - The user wants to start over with the task
        - Objects in the scene need to be reset to their initial positions

        The environment will be reset on the next simulation step.
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    def start_teleoperation():
        """Activate teleoperation control of the robot.

        This callback enables active control of the robot through the input device.
        It's typically triggered by a specific gesture or button press and is used when:
        - Beginning a new teleoperation session
        - Resuming control after temporarily pausing
        - Switching from observation mode to control mode

        While active, all commands from the device will be applied to the robot.
        """
        nonlocal teleoperation_active
        teleoperation_active = True

    def stop_teleoperation():
        """Deactivate teleoperation control of the robot.

        This callback temporarily suspends control of the robot through the input device.
        It's typically triggered by a specific gesture or button press and is used when:
        - Taking a break from controlling the robot
        - Repositioning the input device without moving the robot
        - Pausing to observe the scene without interference

        While inactive, the simulation continues to render but device commands are ignored.
        """
        nonlocal teleoperation_active
        teleoperation_active = False

    def pre_process_actions(
        teleop_data: tuple[np.ndarray, bool] | list[tuple[np.ndarray, np.ndarray, np.ndarray]], num_envs: int, device: str
    ) -> torch.Tensor:
        """Convert teleop data to the format expected by the environment action space.

        Args:
            teleop_data: Data from the teleoperation device.
            num_envs: Number of environments.
            device: Device to create tensors on.

        Returns:
            Processed actions as a tensor.
        """
        # compute actions
        delta_pose, gripper_command = teleop_data
        # convert to torch
        delta_pose = torch.tensor(delta_pose, dtype=torch.float, device=device).repeat(num_envs, 1)
        gripper_vel = torch.zeros((delta_pose.shape[0], 1), dtype=torch.float, device=device)
        gripper_vel[:] = -1 if gripper_command else 1
        # compute actions
        return torch.concat([delta_pose, gripper_vel], dim=1)

    # spawn keyboard teleoperation device
    sensitivity = 1
    teleop_interface = Se3Keyboard(
        pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity
    )
    teleop_interface.add_callback("R", reset_recording_instance)
    print(teleop_interface)
    teleop_interface.reset()
    # Run in test mode, exit after a fixed number of steps
    if args.test is True:
        for i in range(10):
            # Run in realtime mode, we don't specify the step size
            kit.update()
    else:
        while kit.is_running():
            # Run in realtime mode, we don't specify the step size
            stage = stage_utils.get_current_stage()
            
            # get teleoperation data
            with torch.inference_mode():
                # get device command
                teleop_data = teleop_interface.advance()
                
                if teleoperation_active:
                    # compute actions based on environment
                    actions = pre_process_actions(teleop_data, 1, 'cpu')
                    print(actions)
if __name__ == "__main__":
    # run the application
    main()
    # close the application      
    kit.close()
            