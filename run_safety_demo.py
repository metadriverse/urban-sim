#!/usr/bin/env python3
# Copyright (c) 2025, Simulation Toolkit Project
# Safety Demonstration Script

"""
Demonstration script for simulation toolkit safety features.
Shows Ground Robotics lawn mower with safety systems in urban environment.
"""

import sys
import argparse
import torch
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# URBAN-SIM imports
from urbansim.envs.separate_envs.random_env import RandomEnv
from urbansim.scene.urban_scene import UrbanScene

# Safety extensions
from safety_extensions.managers.safety_manager import SafetyManager, SafetyManagerCfg
from robot_configs.ground_robotics import GroundRoboticsLawnMower, GROUND_ROBOTICS_MOWER_CFG


class SafetyDemoEnv:
    """Demo environment showcasing safety features."""
    
    def __init__(self, num_envs: int = 1, enable_cameras: bool = True):
        """Initialize safety demo environment.
        
        Args:
            num_envs: Number of parallel environments
            enable_cameras: Whether to enable camera observations
        """
        self.num_envs = num_envs
        self.enable_cameras = enable_cameras
        
        print(f"[INFO] Initializing Safety Demo with {num_envs} environment(s)")
        
        # Create base URBAN-SIM environment
        # This uses the existing random environment as a foundation
        try:
            # Import required modules
            import omni
            from isaacsim import SimulationApp
            
            # Initialize simulation
            self.simulation_app = SimulationApp({
                "headless": False,  # Enable GUI for demo
                "width": 1920,
                "height": 1080,
            })
            
            print("✅ Isaac Sim initialized successfully")
            
        except Exception as e:
            print(f"❌ Failed to initialize Isaac Sim: {e}")
            print("Make sure Isaac Sim is properly installed and configured")
            raise
        
        # Initialize safety systems
        self._setup_safety_systems()
        
        # Demo state
        self.step_count = 0
        self.demo_scenarios = [
            "normal_operation",
            "human_approach", 
            "emergency_stop_test",
            "hardware_limit_test"
        ]
        self.current_scenario = 0
        
    def _setup_safety_systems(self):
        """Set up safety management systems."""
        print("[INFO] Setting up safety systems...")
        
        # Create safety manager configuration
        self.safety_cfg = SafetyManagerCfg()
        self.safety_cfg.collision_threshold = 0.0
        self.safety_cfg.near_miss_threshold = 0.5
        self.safety_cfg.comfort_zone_radius = 1.5
        self.safety_cfg.max_velocity = 1.0  # Slower for demo
        
        # Robot will be created when scene is set up
        self.robot = None
        self.safety_manager = None
        
    def run_demo(self, duration_seconds: float = 60.0):
        """Run the safety demonstration.
        
        Args:
            duration_seconds: How long to run the demo
        """
        print(f"[INFO] Starting safety demonstration ({duration_seconds}s)")
        
        try:
            # Set up scene (simplified for demo)
            self._setup_demo_scene()
            
            # Main demo loop
            start_time = time.time()
            
            while (time.time() - start_time) < duration_seconds:
                self._step_demo()
                time.sleep(0.1)  # 10Hz update rate
                
                # Switch scenarios every 15 seconds
                if self.step_count % 150 == 0 and self.step_count > 0:
                    self._next_scenario()
            
            print("[INFO] Demo completed successfully")
            
        except KeyboardInterrupt:
            print("[INFO] Demo interrupted by user")
        except Exception as e:
            print(f"[ERROR] Demo failed: {e}")
            raise
        finally:
            self._cleanup()
    
    def _setup_demo_scene(self):
        """Set up demonstration scene with robot and safety systems."""
        print("[INFO] Setting up demo scene...")
        
        # For now, create a simplified setup
        # In production, this would use URBAN-SIM's full scene system
        
        # Create robot (placeholder - using basic robot for demo)
        print("[INFO] Loading Ground Robotics lawn mower...")
        self.robot = self._create_demo_robot()
        
        # Initialize safety manager
        if self.robot:
            self.safety_manager = SafetyManager(
                self.safety_cfg, 
                self.robot, 
                None  # Scene will be added later
            )
            print("✅ Safety manager initialized")
        
        # Add some demo humans/obstacles (simulated positions)
        self.demo_humans = [
            torch.tensor([3.0, 2.0, 0.0]),  # Person in distance
            torch.tensor([5.0, -1.0, 0.0]), # Another person
        ]
        
        print("✅ Demo scene setup complete")
    
    def _create_demo_robot(self):
        """Create demo robot (simplified version)."""
        try:
            # Create a simplified robot for demo purposes
            # This is a placeholder - in production would use full URDF
            
            class DemoRobot:
                def __init__(self):
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"
                    
                    # Mock robot data
                    self.data = type('obj', (object,), {
                        'root_pos_w': torch.tensor([0.0, 0.0, 0.0]),
                        'root_quat_w': torch.tensor([1.0, 0.0, 0.0, 0.0]),
                        'root_lin_vel_w': torch.tensor([0.0, 0.0, 0.0]),
                        'root_ang_vel_w': torch.tensor([0.0, 0.0, 0.0]),
                        'joint_pos': torch.tensor([0.0, 0.0]),
                        'joint_vel': torch.tensor([0.0, 0.0]),
                    })
                    
                    self._emergency_stop = False
                    
                def set_joint_velocity_target(self, vel):
                    pass
                    
                def set_joint_position_target(self, pos):
                    pass
            
            return DemoRobot()
            
        except Exception as e:
            print(f"[WARNING] Could not create full robot: {e}")
            print("[INFO] Running in simplified demo mode")
            return None
    
    def _step_demo(self):
        """Step the demonstration."""
        self.step_count += 1
        
        if self.step_count % 50 == 0:  # Print status every 5 seconds
            scenario_name = self.demo_scenarios[self.current_scenario]
            print(f"[DEMO] Step {self.step_count}, Scenario: {scenario_name}")
        
        # Simulate robot movement based on current scenario
        robot_actions = self._get_scenario_actions()
        
        # Update safety systems (if available)
        if self.safety_manager and self.robot:
            # Update robot position (simulate movement)
            self._update_robot_position(robot_actions)
            
            # Run safety checks
            safety_status = self.safety_manager.update(0.1)  # 0.1s timestep
            
            # Display safety events
            if safety_status["violations"]:
                print(f"[SAFETY] {len(safety_status['violations'])} violations detected:")
                for violation in safety_status["violations"][:3]:  # Show first 3
                    print(f"  - {violation.violation_type}: {violation.severity}")
            
            if safety_status["emergency_stop_active"]:
                print("[EMERGENCY] 🚨 Emergency stop active!")
    
    def _get_scenario_actions(self):
        """Get robot actions based on current scenario."""
        scenario = self.demo_scenarios[self.current_scenario]
        
        if scenario == "normal_operation":
            # Gentle forward movement
            return torch.tensor([0.5, 0.1])  # [linear, angular]
            
        elif scenario == "human_approach":
            # Move toward humans to test collision avoidance
            return torch.tensor([0.8, 0.0])
            
        elif scenario == "emergency_stop_test":
            # Trigger emergency stop
            if self.step_count % 100 == 50:  # Trigger partway through scenario
                if self.safety_manager:
                    self.safety_manager.trigger_emergency_stop()
            return torch.tensor([0.3, 0.0])
            
        elif scenario == "hardware_limit_test":
            # Try to exceed hardware limits
            return torch.tensor([5.0, 3.0])  # Excessive speeds
            
        return torch.tensor([0.0, 0.0])  # Stop
    
    def _update_robot_position(self, actions):
        """Update robot position based on actions (simplified simulation)."""
        if not self.robot:
            return
            
        # Simple kinematic update
        dt = 0.1
        linear_vel = actions[0].item()
        angular_vel = actions[1].item()
        
        # Update position (very simplified)
        current_pos = self.robot.data.root_pos_w
        current_pos[0] += linear_vel * dt  # Move in X direction
        
        # Update velocities
        self.robot.data.root_lin_vel_w = torch.tensor([linear_vel, 0.0, 0.0])
        self.robot.data.root_ang_vel_w = torch.tensor([0.0, 0.0, angular_vel])
    
    def _next_scenario(self):
        """Switch to next demo scenario."""
        self.current_scenario = (self.current_scenario + 1) % len(self.demo_scenarios)
        scenario_name = self.demo_scenarios[self.current_scenario]
        print(f"\n[DEMO] 🎬 Switching to scenario: {scenario_name}\n")
        
        # Reset emergency stop for new scenario
        if self.safety_manager:
            self.safety_manager.reset_emergency_stop()
    
    def _cleanup(self):
        """Clean up demo resources."""
        print("[INFO] Cleaning up demo resources...")
        
        if hasattr(self, 'simulation_app') and self.simulation_app:
            self.simulation_app.close()


def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(description="Safety Demonstration for Simulation Toolkit")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
    parser.add_argument("--duration", type=float, default=60.0, help="Demo duration in seconds")
    parser.add_argument("--enable_cameras", action="store_true", help="Enable camera observations")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    
    args = parser.parse_args()
    
    print("🤖 Simulation Toolkit - Safety Demonstration")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  - Environments: {args.num_envs}")
    print(f"  - Duration: {args.duration}s")
    print(f"  - Cameras: {args.enable_cameras}")
    print(f"  - Headless: {args.headless}")
    print()
    
    try:
        # Create and run demo
        demo = SafetyDemoEnv(
            num_envs=args.num_envs,
            enable_cameras=args.enable_cameras
        )
        
        demo.run_demo(duration_seconds=args.duration)
        
        print("\n✅ Demo completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)