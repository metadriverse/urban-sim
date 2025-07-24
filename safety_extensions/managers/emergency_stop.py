# Copyright (c) 2025, Simulation Toolkit Project
# Emergency Stop System

import torch
from typing import Optional
from isaaclab.assets import Articulation


class EmergencyStopSystem:
    """Manages emergency stop functionality for robots."""
    
    def __init__(self, robot: Articulation):
        """Initialize emergency stop system.
        
        Args:
            robot: The robot articulation to control
        """
        self.robot = robot
        self._is_stopped = False
        self._saved_joint_targets = None
        self._saved_joint_velocities = None
        
    def trigger(self) -> bool:
        """Trigger emergency stop - immediately halt all motion.
        
        Returns:
            True if stop was successful
        """
        if self._is_stopped:
            return True
        
        try:
            # Save current targets for potential recovery
            self._saved_joint_targets = self.robot.data.joint_pos.clone()
            self._saved_joint_velocities = self.robot.data.joint_vel.clone()
            
            # Set all joint velocities to zero
            zero_vel = torch.zeros_like(self.robot.data.joint_vel)
            self.robot.set_joint_velocity_target(zero_vel)
            
            # Set position targets to current positions (hold in place)
            self.robot.set_joint_position_target(self.robot.data.joint_pos)
            
            # Apply maximum damping for quick stop
            # This would interface with actual motor controllers
            
            self._is_stopped = True
            return True
            
        except Exception as e:
            print(f"Emergency stop failed: {e}")
            return False
    
    def reset(self) -> bool:
        """Reset emergency stop and return to normal operation.
        
        Returns:
            True if reset was successful
        """
        if not self._is_stopped:
            return True
            
        try:
            # Gradually resume motion if saved states exist
            if self._saved_joint_targets is not None:
                # In a real system, this would ramp up slowly
                self.robot.set_joint_position_target(self._saved_joint_targets)
            
            self._is_stopped = False
            self._saved_joint_targets = None
            self._saved_joint_velocities = None
            
            return True
            
        except Exception as e:
            print(f"Emergency stop reset failed: {e}")
            return False
    
    @property
    def is_stopped(self) -> bool:
        """Check if emergency stop is currently active."""
        return self._is_stopped