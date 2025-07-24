# Copyright (c) 2025, Simulation Toolkit Project
# Safety Zone Management

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
from isaaclab.assets import Articulation


@dataclass
class SafetyZone:
    """Represents a safety zone around the robot."""
    zone_type: str  # detection, warning, critical, emergency
    radius: float  # meters
    height: float  # meters
    active: bool = True
    color: Tuple[float, float, float, float] = (1.0, 1.0, 0.0, 0.3)  # RGBA


class SafetyZoneManager:
    """Manages dynamic safety zones around the robot."""
    
    def __init__(
        self,
        robot: Articulation,
        comfort_radius: float = 1.0,
        dynamic_zones: bool = True
    ):
        """Initialize safety zone manager.
        
        Args:
            robot: The robot to create zones around
            comfort_radius: Base comfort zone radius
            dynamic_zones: Whether to adjust zones based on robot state
        """
        self.robot = robot
        self.comfort_radius = comfort_radius
        self.dynamic_zones = dynamic_zones
        
        # Define zone hierarchy (from outer to inner)
        self.zones = {
            "detection": SafetyZone(
                zone_type="detection",
                radius=comfort_radius * 2.0,
                height=2.0,
                color=(0.0, 1.0, 0.0, 0.2)  # Green
            ),
            "warning": SafetyZone(
                zone_type="warning",
                radius=comfort_radius * 1.5,
                height=2.0,
                color=(1.0, 1.0, 0.0, 0.3)  # Yellow
            ),
            "critical": SafetyZone(
                zone_type="critical",
                radius=comfort_radius,
                height=2.0,
                color=(1.0, 0.5, 0.0, 0.4)  # Orange
            ),
            "emergency": SafetyZone(
                zone_type="emergency",
                radius=comfort_radius * 0.5,
                height=2.0,
                color=(1.0, 0.0, 0.0, 0.5)  # Red
            )
        }
    
    def update(self, robot_state: Dict, human_positions: torch.Tensor):
        """Update safety zones based on robot state and environment.
        
        Args:
            robot_state: Current robot state
            human_positions: Positions of humans in environment
        """
        if not self.dynamic_zones:
            return
        
        # Adjust zones based on robot velocity
        velocity = torch.norm(robot_state["linear_velocity"]).item()
        
        # Increase zone sizes with velocity
        velocity_factor = 1.0 + velocity * 0.5  # 50% increase per m/s
        
        for zone in self.zones.values():
            base_radius = self._get_base_radius(zone.zone_type)
            zone.radius = base_radius * velocity_factor
    
    def check_violations(
        self,
        robot_pos: torch.Tensor,
        human_positions: torch.Tensor,
        obstacle_positions: torch.Tensor
    ) -> List:
        """Check for zone violations.
        
        Args:
            robot_pos: Robot position
            human_positions: Human positions
            obstacle_positions: Obstacle positions
            
        Returns:
            List of zone violations
        """
        violations = []
        
        # Check each object against zones
        all_positions = torch.cat([human_positions, obstacle_positions], dim=0)
        
        for i, pos in enumerate(all_positions):
            distance = torch.norm(robot_pos - pos).item()
            
            # Check which zones are violated (inner zones are more severe)
            for zone_name in ["emergency", "critical", "warning", "detection"]:
                zone = self.zones[zone_name]
                if zone.active and distance <= zone.radius:
                    # Create violation for innermost violated zone only
                    from ..managers.safety_manager import SafetyViolation
                    violations.append(SafetyViolation(
                        violation_type="zone_violation",
                        severity=self._get_zone_severity(zone_name),
                        distance=distance,
                        robot_id=0,
                        object_id=i,
                        details={"zone": zone_name}
                    ))
                    break  # Only report innermost violation
        
        return violations
    
    def get_active_zones(self) -> Dict[str, SafetyZone]:
        """Get currently active safety zones."""
        return {k: v for k, v in self.zones.items() if v.active}
    
    def _get_base_radius(self, zone_type: str) -> float:
        """Get base radius for a zone type."""
        base_radii = {
            "detection": self.comfort_radius * 2.0,
            "warning": self.comfort_radius * 1.5,
            "critical": self.comfort_radius,
            "emergency": self.comfort_radius * 0.5
        }
        return base_radii.get(zone_type, self.comfort_radius)
    
    def _get_zone_severity(self, zone_type: str) -> str:
        """Map zone type to violation severity."""
        severity_map = {
            "emergency": "critical",
            "critical": "warning",
            "warning": "info",
            "detection": "info"
        }
        return severity_map.get(zone_type, "info")