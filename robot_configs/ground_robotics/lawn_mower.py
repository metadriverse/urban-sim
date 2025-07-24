# Copyright (c) 2025, Simulation Toolkit Project
# Ground Robotics Autonomous Lawn Mower Configuration

from __future__ import annotations
from dataclasses import MISSING
import torch
import numpy as np

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg, RayCasterCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from safety_extensions.validators.hardware_validator import HardwareSpecs


@configclass
class GroundRoboticsLawnMowerCfg:
    """Configuration for Ground Robotics Autonomous Lawn Mower."""
    
    # Physical specifications (placeholders - replace with actual specs)
    wheelbase: float = 0.8  # meters
    track_width: float = 0.6  # meters
    mass: float = 45.0  # kg
    max_velocity: float = 2.0  # m/s
    max_angular_velocity: float = 1.5  # rad/s
    
    # Motor specifications
    wheel_motor_max_torque: float = 50.0  # Nm
    blade_motor_power: float = 1500.0  # Watts
    
    # Safety specifications
    emergency_stop_time: float = 0.5  # seconds
    min_obstacle_distance: float = 0.3  # meters
    cliff_detection_threshold: float = 0.1  # meters
    
    # Sensor specifications
    lidar_range: float = 10.0  # meters
    lidar_frequency: float = 10.0  # Hz
    camera_fps: float = 30.0
    ultrasonic_range: float = 3.0  # meters
    ultrasonic_frequency: float = 20.0  # Hz


# Create hardware specifications from configuration
def create_hardware_specs(cfg: GroundRoboticsLawnMowerCfg) -> HardwareSpecs:
    """Create hardware specs from mower configuration."""
    return HardwareSpecs(
        max_velocity=cfg.max_velocity,
        max_acceleration=cfg.max_velocity / 2.0,  # Assume 2s to max speed
        max_angular_velocity=cfg.max_angular_velocity,
        max_torque={
            "wheel_left": cfg.wheel_motor_max_torque,
            "wheel_right": cfg.wheel_motor_max_torque,
            "blade_motor": cfg.blade_motor_power / 1000.0  # Approximate torque
        },
        sensor_rates={
            "lidar": cfg.lidar_frequency,
            "camera": cfg.camera_fps,
            "ultrasonic": cfg.ultrasonic_frequency
        },
        safety_limits={
            "min_obstacle_distance": cfg.min_obstacle_distance,
            "cliff_threshold": cfg.cliff_detection_threshold,
            "emergency_stop_time": cfg.emergency_stop_time
        }
    )


# Robot articulation configuration
GROUND_ROBOTICS_MOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Using COCO robot as placeholder until actual URDF is available
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=2.0,
            max_angular_velocity=1.5,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        # Copy from COCO - will be replaced with actual mower materials
        mass_props=sim_utils.MassPropertiesCfg(mass=45.0),  # Typical lawn mower mass
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),  # Start slightly above ground
        joint_pos={
            # These will be replaced with actual joint names from URDF
            ".*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Drive motors (differential drive)
        "drive_motors": DCMotorCfg(
            joint_names_expr=[".*wheel.*"],  # Match wheel joints
            effort_limit=50.0,  # Nm
            velocity_limit=10.0,  # rad/s
            stiffness=80.0,
            damping=4.0,
        ),
        # Blade motor
        "blade_motor": DCMotorCfg(
            joint_names_expr=[".*blade.*"],  # Match blade joint
            effort_limit=25.0,  # Nm (estimated from 1.5kW)
            velocity_limit=200.0,  # rad/s (high RPM for cutting)
            stiffness=40.0,
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)


class GroundRoboticsLawnMower(Articulation):
    """Ground Robotics Autonomous Lawn Mower robot with safety features."""
    
    def __init__(self, cfg: ArticulationCfg):
        """Initialize the lawn mower robot.
        
        Args:
            cfg: Articulation configuration
        """
        super().__init__(cfg)
        
        # Create default configuration
        self.mower_cfg = GroundRoboticsLawnMowerCfg()
        self.hardware_specs = create_hardware_specs(self.mower_cfg)
        
        # Safety systems will be initialized by SafetyManager
        self._emergency_stop_active = False
        self._blade_enabled = True
        
    def set_emergency_stop(self, active: bool):
        """Set emergency stop state.
        
        Args:
            active: Whether emergency stop should be active
        """
        self._emergency_stop_active = active
        
        if active:
            # Stop all motors immediately
            zero_vel = torch.zeros_like(self.data.joint_vel)
            self.set_joint_velocity_target(zero_vel)
            
            # Disable blade
            self._blade_enabled = False
        
    def enable_blade(self, enable: bool = True):
        """Enable or disable the cutting blade.
        
        Args:
            enable: Whether to enable the blade
        """
        if not self._emergency_stop_active:
            self._blade_enabled = enable
    
    def get_safety_state(self) -> dict:
        """Get current safety-related state information."""
        return {
            "emergency_stop_active": self._emergency_stop_active,
            "blade_enabled": self._blade_enabled,
            "position": self.data.root_pos_w.clone(),
            "linear_velocity": self.data.root_lin_vel_w.clone(),
            "angular_velocity": self.data.root_ang_vel_w.clone(),
            "hardware_specs": self.hardware_specs
        }
    
    def apply_safety_constraints(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply safety constraints to control actions.
        
        Args:
            actions: Raw control actions
            
        Returns:
            Safety-constrained actions
        """
        if self._emergency_stop_active:
            # Return zero actions during emergency stop
            return torch.zeros_like(actions)
        
        # Apply velocity limits
        constrained_actions = torch.clamp(
            actions,
            -self.hardware_specs.max_velocity,
            self.hardware_specs.max_velocity
        )
        
        return constrained_actions


# Configuration for safety testing environment
@configclass
class GroundRoboticsSceneCfg:
    """Scene configuration for Ground Robotics lawn mower testing."""
    
    # Robot configuration
    robot: GroundRoboticsLawnMowerCfg = GroundRoboticsLawnMowerCfg()
    
    # Environment settings
    num_envs: int = 1
    env_spacing: float = 10.0  # meters between environments
    
    # Terrain settings
    terrain_type: str = "plane"  # plane, rough, slope
    terrain_size: tuple = (20.0, 20.0)  # meters
    
    # Human agents
    num_humans: int = 3
    human_spawn_radius: float = 8.0  # meters from robot
    
    # Obstacles
    num_static_obstacles: int = 5
    obstacle_types: list = ["tree", "fence", "shed"]
    
    # Safety test scenarios
    test_scenarios: list = [
        "child_approach",
        "adult_crossing",
        "pet_interference",
        "slope_operation",
        "crowded_area"
    ]