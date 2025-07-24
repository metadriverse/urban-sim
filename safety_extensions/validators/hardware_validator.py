# Copyright (c) 2025, Simulation Toolkit Project
# Hardware Validation System

import torch
from typing import Dict, List, Optional, Tuple
import yaml
from dataclasses import dataclass


@dataclass
class HardwareSpecs:
    """Hardware specifications for validation."""
    max_velocity: float
    max_acceleration: float
    max_angular_velocity: float
    max_torque: Dict[str, float]  # per joint
    sensor_rates: Dict[str, float]  # Hz
    safety_limits: Dict[str, float]


class HardwareValidator:
    """Validates simulation against actual hardware constraints."""
    
    def __init__(self, specs_file: Optional[str] = None, specs: Optional[HardwareSpecs] = None):
        """Initialize hardware validator.
        
        Args:
            specs_file: Path to hardware specifications YAML file
            specs: HardwareSpecs object (alternative to file)
        """
        if specs_file:
            self.specs = self._load_specs_from_file(specs_file)
        elif specs:
            self.specs = specs
        else:
            # Default specs for testing
            self.specs = HardwareSpecs(
                max_velocity=2.0,
                max_acceleration=5.0,
                max_angular_velocity=3.14,
                max_torque={"wheel_left": 50.0, "wheel_right": 50.0, "blade": 25.0},
                sensor_rates={"lidar": 10.0, "camera": 30.0, "ultrasonic": 20.0},
                safety_limits={"min_ground_clearance": 0.05, "max_slope": 0.52}  # 30 degrees
            )
        
        self.violation_count = 0
        self.violation_history = []
    
    def clamp_to_limits(self, actions: torch.Tensor) -> torch.Tensor:
        """Clamp actions to hardware limits.
        
        Args:
            actions: Raw control actions
            
        Returns:
            Actions clamped to hardware limits
        """
        clamped_actions = actions.clone()
        
        # This is simplified - in production, would map actions to specific joints
        # and apply joint-specific limits
        
        # Example: Clamp velocity commands
        velocity_limit = self.specs.max_velocity
        clamped_actions = torch.clamp(
            clamped_actions, 
            -velocity_limit, 
            velocity_limit
        )
        
        # Track violations
        if not torch.allclose(actions, clamped_actions):
            self.violation_count += 1
            self.violation_history.append({
                "type": "action_clamping",
                "original": actions.tolist(),
                "clamped": clamped_actions.tolist(),
                "timestamp": torch.cuda.Event().record() if torch.cuda.is_available() else 0
            })
        
        return clamped_actions
    
    def validate_sensor_rates(self, sensor_data: Dict[str, Dict]) -> Dict[str, bool]:
        """Validate sensor sampling rates against hardware specs.
        
        Args:
            sensor_data: Dictionary of sensor name -> sensor info
            
        Returns:
            Dictionary of sensor validations
        """
        validations = {}
        
        for sensor_name, data in sensor_data.items():
            if sensor_name in self.specs.sensor_rates:
                expected_rate = self.specs.sensor_rates[sensor_name]
                actual_rate = data.get("rate", 0.0)
                
                # Allow some tolerance (10%)
                tolerance = expected_rate * 0.1
                validations[sensor_name] = abs(actual_rate - expected_rate) <= tolerance
        
        return validations
    
    def validate_robot_state(self, robot_state: Dict) -> List[Dict]:
        """Validate current robot state against hardware limits.
        
        Args:
            robot_state: Current robot state dictionary
            
        Returns:
            List of validation violations
        """
        violations = []
        
        # Check velocity limits
        if "linear_velocity" in robot_state:
            velocity = torch.norm(robot_state["linear_velocity"]).item()
            if velocity > self.specs.max_velocity:
                violations.append({
                    "type": "velocity_limit",
                    "current": velocity,
                    "limit": self.specs.max_velocity,
                    "severity": "warning"
                })
        
        # Check angular velocity limits
        if "angular_velocity" in robot_state:
            ang_velocity = torch.norm(robot_state["angular_velocity"]).item()
            if ang_velocity > self.specs.max_angular_velocity:
                violations.append({
                    "type": "angular_velocity_limit",
                    "current": ang_velocity,
                    "limit": self.specs.max_angular_velocity,
                    "severity": "warning"
                })
        
        # Check joint torques (if available)
        if "joint_efforts" in robot_state:
            joint_efforts = robot_state["joint_efforts"]
            for i, effort in enumerate(joint_efforts):
                joint_name = f"joint_{i}"  # Simplified joint naming
                if joint_name in self.specs.max_torque:
                    max_torque = self.specs.max_torque[joint_name]
                    if abs(effort.item()) > max_torque:
                        violations.append({
                            "type": "torque_limit",
                            "joint": joint_name,
                            "current": effort.item(),
                            "limit": max_torque,
                            "severity": "critical"
                        })
        
        return violations
    
    def get_validation_summary(self) -> Dict:
        """Get summary of validation results."""
        return {
            "total_violations": self.violation_count,
            "recent_violations": len([v for v in self.violation_history[-100:]]),
            "hardware_specs": {
                "max_velocity": self.specs.max_velocity,
                "max_acceleration": self.specs.max_acceleration,
                "max_angular_velocity": self.specs.max_angular_velocity,
            }
        }
    
    def _load_specs_from_file(self, filepath: str) -> HardwareSpecs:
        """Load hardware specifications from YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return HardwareSpecs(
            max_velocity=data.get("max_velocity", 2.0),
            max_acceleration=data.get("max_acceleration", 5.0),
            max_angular_velocity=data.get("max_angular_velocity", 3.14),
            max_torque=data.get("max_torque", {}),
            sensor_rates=data.get("sensor_rates", {}),
            safety_limits=data.get("safety_limits", {})
        )