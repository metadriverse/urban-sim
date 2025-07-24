# Copyright (c) 2025, Simulation Toolkit Project
# Safety Manager - Central safety coordination system

import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation

from .emergency_stop import EmergencyStopSystem
from .collision_detector import CollisionDetector
from .safety_zones import SafetyZoneManager
from ..metrics.safety_metrics import SafetyMetricsCollector


@dataclass
class SafetyViolation:
    """Represents a safety violation event."""
    violation_type: str  # collision, near_miss, zone_violation, hardware_limit
    severity: str  # critical, warning, info
    distance: float
    robot_id: int
    object_id: Optional[int] = None
    timestamp: float = 0.0
    details: Dict = None


@configclass
class SafetyManagerCfg:
    """Configuration for the safety manager."""
    # Safety thresholds
    collision_threshold: float = 0.0  # meters
    near_miss_threshold: float = 0.5  # meters
    comfort_zone_radius: float = 1.0  # meters
    
    # Response times
    max_reaction_time: float = 0.1  # seconds
    max_stopping_distance: float = 0.5  # meters
    
    # Hardware limits (will be overridden by actual specs)
    max_velocity: float = 2.0  # m/s
    max_acceleration: float = 5.0  # m/s^2
    max_angular_velocity: float = 3.14  # rad/s
    
    # Safety zone configuration
    enable_dynamic_zones: bool = True
    zone_update_frequency: float = 10.0  # Hz


class SafetyManager:
    """Central safety management system for robotics simulation."""
    
    def __init__(self, cfg: SafetyManagerCfg, robot: Articulation, scene: InteractiveScene):
        """Initialize the safety manager.
        
        Args:
            cfg: Safety manager configuration
            robot: The robot articulation to monitor
            scene: The interactive scene containing humans and obstacles
        """
        self.cfg = cfg
        self.robot = robot
        self.scene = scene
        
        # Initialize subsystems
        self.emergency_stop = EmergencyStopSystem(robot)
        self.collision_detector = CollisionDetector(
            collision_threshold=cfg.collision_threshold,
            near_miss_threshold=cfg.near_miss_threshold
        )
        self.zone_manager = SafetyZoneManager(
            robot=robot,
            comfort_radius=cfg.comfort_zone_radius,
            dynamic_zones=cfg.enable_dynamic_zones
        )
        self.metrics_collector = SafetyMetricsCollector()
        
        # Internal state
        self._violations: List[SafetyViolation] = []
        self._last_update_time = 0.0
        self._emergency_stop_active = False
        
        # Get device
        self.device = robot.device
        
    def update(self, dt: float) -> Dict[str, any]:
        """Update safety systems and check for violations.
        
        Args:
            dt: Time step in seconds
            
        Returns:
            Dictionary containing safety status and metrics
        """
        self._last_update_time += dt
        
        # Get current states
        robot_state = self._get_robot_state()
        human_positions = self._get_human_positions()
        obstacle_positions = self._get_obstacle_positions()
        
        # Check for violations
        violations = self.check_all_violations(
            robot_state, human_positions, obstacle_positions
        )
        
        # Handle critical violations
        if any(v.severity == "critical" for v in violations):
            self.trigger_emergency_stop()
        
        # Update metrics
        self.metrics_collector.update(violations, dt)
        
        # Update safety zones if enabled
        if self.cfg.enable_dynamic_zones:
            self.zone_manager.update(robot_state, human_positions)
        
        return {
            "violations": violations,
            "emergency_stop_active": self._emergency_stop_active,
            "metrics": self.metrics_collector.get_current_metrics(),
            "safety_zones": self.zone_manager.get_active_zones()
        }
    
    def check_all_violations(
        self, 
        robot_state: Dict,
        human_positions: torch.Tensor,
        obstacle_positions: torch.Tensor
    ) -> List[SafetyViolation]:
        """Check for all types of safety violations.
        
        Args:
            robot_state: Current robot state dictionary
            human_positions: Tensor of human positions [N, 3]
            obstacle_positions: Tensor of obstacle positions [M, 3]
            
        Returns:
            List of detected violations
        """
        violations = []
        
        # Check collisions and near-misses with humans
        for i, human_pos in enumerate(human_positions):
            distance = self._calculate_minimum_distance(
                robot_state["position"], human_pos
            )
            
            if distance <= self.cfg.collision_threshold:
                violations.append(SafetyViolation(
                    violation_type="collision",
                    severity="critical",
                    distance=distance,
                    robot_id=0,
                    object_id=i,
                    timestamp=self._last_update_time,
                    details={"object_type": "human"}
                ))
            elif distance <= self.cfg.near_miss_threshold:
                violations.append(SafetyViolation(
                    violation_type="near_miss",
                    severity="warning",
                    distance=distance,
                    robot_id=0,
                    object_id=i,
                    timestamp=self._last_update_time,
                    details={"object_type": "human"}
                ))
            elif distance <= self.cfg.comfort_zone_radius:
                violations.append(SafetyViolation(
                    violation_type="zone_violation",
                    severity="info",
                    distance=distance,
                    robot_id=0,
                    object_id=i,
                    timestamp=self._last_update_time,
                    details={"zone": "comfort", "object_type": "human"}
                ))
        
        # Check hardware limit violations
        hw_violations = self._check_hardware_limits(robot_state)
        violations.extend(hw_violations)
        
        # Check safety zone violations
        zone_violations = self.zone_manager.check_violations(
            robot_state["position"], human_positions, obstacle_positions
        )
        violations.extend(zone_violations)
        
        self._violations = violations
        return violations
    
    def trigger_emergency_stop(self) -> bool:
        """Trigger emergency stop for the robot.
        
        Returns:
            True if emergency stop was successfully triggered
        """
        self._emergency_stop_active = True
        success = self.emergency_stop.trigger()
        
        if success:
            self.metrics_collector.record_emergency_stop(self._last_update_time)
            
        return success
    
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop and allow normal operation.
        
        Returns:
            True if emergency stop was successfully reset
        """
        if not self._emergency_stop_active:
            return True
            
        # Check if it's safe to reset
        current_violations = [v for v in self._violations if v.severity == "critical"]
        if current_violations:
            return False
            
        self._emergency_stop_active = False
        return self.emergency_stop.reset()
    
    def _get_robot_state(self) -> Dict[str, torch.Tensor]:
        """Get current robot state including position, velocity, etc."""
        return {
            "position": self.robot.data.root_pos_w,
            "orientation": self.robot.data.root_quat_w,
            "linear_velocity": self.robot.data.root_lin_vel_w,
            "angular_velocity": self.robot.data.root_ang_vel_w,
            "joint_positions": self.robot.data.joint_pos,
            "joint_velocities": self.robot.data.joint_vel,
        }
    
    def _get_human_positions(self) -> torch.Tensor:
        """Get positions of all humans in the scene."""
        # This would interface with URBAN-SIM's human tracking
        # For now, return placeholder
        return torch.zeros((0, 3), device=self.device)
    
    def _get_obstacle_positions(self) -> torch.Tensor:
        """Get positions of all obstacles in the scene."""
        # This would interface with URBAN-SIM's obstacle tracking
        # For now, return placeholder
        return torch.zeros((0, 3), device=self.device)
    
    def _calculate_minimum_distance(
        self, 
        robot_pos: torch.Tensor, 
        object_pos: torch.Tensor
    ) -> float:
        """Calculate minimum distance between robot and object.
        
        This is a simplified version - in production, use mesh-based collision detection.
        """
        distance = torch.norm(robot_pos - object_pos).item()
        return distance
    
    def _check_hardware_limits(self, robot_state: Dict) -> List[SafetyViolation]:
        """Check if robot is violating hardware limits."""
        violations = []
        
        # Check velocity limits
        linear_vel = torch.norm(robot_state["linear_velocity"]).item()
        if linear_vel > self.cfg.max_velocity:
            violations.append(SafetyViolation(
                violation_type="hardware_limit",
                severity="warning",
                distance=0.0,
                robot_id=0,
                timestamp=self._last_update_time,
                details={
                    "limit_type": "velocity",
                    "current": linear_vel,
                    "limit": self.cfg.max_velocity
                }
            ))
        
        # Check angular velocity limits
        angular_vel = torch.norm(robot_state["angular_velocity"]).item()
        if angular_vel > self.cfg.max_angular_velocity:
            violations.append(SafetyViolation(
                violation_type="hardware_limit",
                severity="warning",
                distance=0.0,
                robot_id=0,
                timestamp=self._last_update_time,
                details={
                    "limit_type": "angular_velocity",
                    "current": angular_vel,
                    "limit": self.cfg.max_angular_velocity
                }
            ))
        
        return violations
    
    def get_safety_status(self) -> Dict[str, any]:
        """Get current safety status summary."""
        return {
            "emergency_stop_active": self._emergency_stop_active,
            "active_violations": len(self._violations),
            "critical_violations": len([v for v in self._violations if v.severity == "critical"]),
            "metrics": self.metrics_collector.get_summary(),
            "last_update": self._last_update_time
        }