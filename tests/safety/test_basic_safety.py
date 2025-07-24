#!/usr/bin/env python3
# Copyright (c) 2025, Simulation Toolkit Project
# Basic Safety Test - Validates core safety functionality

"""
Basic safety test script for simulation toolkit.
Tests fundamental safety features with Ground Robotics lawn mower.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# URBAN-SIM imports
from urbansim.envs.abstract_env import AbstractEnv
from urbansim.scene.urban_scene import UrbanScene
from isaaclab.utils import configclass
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from isaaclab.scene.interactive_scene_cfg import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import SimulationCfg

# Safety extensions
from safety_extensions.managers.safety_manager import SafetyManager, SafetyManagerCfg
from robot_configs.ground_robotics import GroundRoboticsLawnMower, GROUND_ROBOTICS_MOWER_CFG


@configclass
class SafetyTestEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for safety testing environment."""
    
    # Simulation settings
    decimation = 10  # Run physics at 100Hz, control at 10Hz
    episode_length_s = 30.0  # 30 second episodes
    
    # Enable rendering for visualization
    sim: SimulationCfg = SimulationCfg(
        dt=1/100.0,  # 100Hz physics
        render_interval=10,  # Render every 10 physics steps
        disable_contact_processing=False,
    )
    
    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,  # Single environment for testing
        env_spacing=10.0,
        replicate_physics=False,
    )
    
    # No RL-specific configurations needed for safety testing
    actions = None
    observations = None
    rewards = None
    terminations = None
    events = None
    curriculum = None


class SafetyTestEnv(AbstractEnv):
    """Environment for testing safety features."""
    
    def __init__(self, cfg: SafetyTestEnvCfg, **kwargs):
        """Initialize safety test environment."""
        super().__init__(cfg, **kwargs)
        
        # Add robot to scene
        self.robot = GroundRoboticsLawnMower(GROUND_ROBOTICS_MOWER_CFG)
        
        # Initialize safety manager
        safety_cfg = SafetyManagerCfg()
        safety_cfg.collision_threshold = 0.0
        safety_cfg.near_miss_threshold = 0.5
        safety_cfg.comfort_zone_radius = 1.0
        
        self.safety_manager = SafetyManager(safety_cfg, self.robot, self.scene)
        
        # Test state
        self.test_step = 0
        self.test_results = {}
        
    def _setup_scene(self):
        """Set up the test scene."""
        # Add ground plane
        self.scene.add_default_ground_plane()
        
        # Spawn robot at origin
        self.robot.spawn("/World/Robot", translation=(0.0, 0.0, 0.5))
        
        print("[INFO] Safety test scene setup complete")
    
    def step(self, actions):
        """Step the environment with safety monitoring."""
        self.test_step += 1
        
        # Apply safety constraints to actions
        if actions is not None:
            safe_actions = self.robot.apply_safety_constraints(actions)
        else:
            safe_actions = torch.zeros((1, 2))  # [linear_vel, angular_vel]
        
        # Apply actions (simplified for testing)
        if not self.safety_manager._emergency_stop_active:
            # In a real implementation, this would control actual motors
            pass
        
        # Update physics
        self.sim.step()
        
        # Update safety systems
        dt = self.physics_dt * self.cfg.decimation
        safety_status = self.safety_manager.update(dt)
        
        # Log safety events
        if safety_status["violations"]:
            print(f"[SAFETY] Step {self.test_step}: {len(safety_status['violations'])} violations detected")
            for violation in safety_status["violations"]:
                print(f"  - {violation.violation_type}: {violation.severity} (distance: {violation.distance:.3f}m)")
        
        # Check for emergency stops
        if safety_status["emergency_stop_active"]:
            print(f"[EMERGENCY] Emergency stop triggered at step {self.test_step}")
        
        # Store test results
        self.test_results[self.test_step] = {
            "safety_status": safety_status,
            "robot_state": self.robot.get_safety_state()
        }
        
        # Create dummy return values (not needed for safety testing)
        obs = torch.zeros((1, 10))  # Dummy observation
        reward = torch.zeros((1, 1))
        done = torch.tensor([self.test_step >= 1000])  # 10 seconds at 100Hz
        info = {"safety_status": safety_status}
        
        return obs, reward, done, info
    
    def reset(self):
        """Reset the environment for a new test."""
        print(f"[INFO] Resetting safety test environment")
        
        # Reset robot state
        self.robot.set_emergency_stop(False)
        self.robot.enable_blade(True)
        
        # Reset safety manager
        self.safety_manager.reset_emergency_stop()
        
        # Reset test state
        self.test_step = 0
        self.test_results.clear()
        
        # Reset scene
        super().reset()
        
        return torch.zeros((1, 10))  # Dummy observation


def run_emergency_stop_test():
    """Test emergency stop functionality."""
    print("\n=== EMERGENCY STOP TEST ===")
    
    cfg = SafetyTestEnvCfg()
    env = SafetyTestEnv(cfg)
    
    try:
        # Initialize environment
        env.reset()
        
        # Run for a few steps
        for i in range(10):
            actions = torch.tensor([[0.5, 0.0]])  # Move forward slowly
            obs, reward, done, info = env.step(actions)
            
            # Trigger emergency stop at step 5
            if i == 5:
                print("[TEST] Triggering emergency stop...")
                success = env.safety_manager.trigger_emergency_stop()
                print(f"[TEST] Emergency stop {'successful' if success else 'failed'}")
        
        # Verify emergency stop is active
        safety_status = env.safety_manager.get_safety_status()
        assert safety_status["emergency_stop_active"], "Emergency stop should be active"
        
        # Try to reset emergency stop
        print("[TEST] Attempting to reset emergency stop...")
        reset_success = env.safety_manager.reset_emergency_stop()
        print(f"[TEST] Emergency stop reset {'successful' if reset_success else 'failed'}")
        
        print("✅ Emergency stop test passed")
        return True
        
    except Exception as e:
        print(f"❌ Emergency stop test failed: {e}")
        return False
    
    finally:
        env.close()


def run_collision_detection_test():
    """Test collision detection system."""
    print("\n=== COLLISION DETECTION TEST ===")
    
    cfg = SafetyTestEnvCfg()
    env = SafetyTestEnv(cfg)
    
    try:
        # Initialize environment
        env.reset()
        
        # Simulate approaching obstacle by manually creating violation
        print("[TEST] Simulating obstacle approach...")
        
        # Create fake human positions for testing
        human_positions = torch.tensor([[1.0, 0.0, 0.0]])  # 1 meter away
        
        # Check safety violations
        robot_state = env.robot.get_safety_state()
        violations = env.safety_manager.check_all_violations(
            robot_state, human_positions, torch.zeros((0, 3))
        )
        
        # Should have zone violation but no collision
        zone_violations = [v for v in violations if v.violation_type == "zone_violation"]
        collisions = [v for v in violations if v.violation_type == "collision"]
        
        print(f"[TEST] Found {len(zone_violations)} zone violations, {len(collisions)} collisions")
        
        # Test collision scenario
        close_human = torch.tensor([[0.01, 0.0, 0.0]])  # Very close (collision)
        violations = env.safety_manager.check_all_violations(
            robot_state, close_human, torch.zeros((0, 3))
        )
        
        collisions = [v for v in violations if v.violation_type == "collision"]
        print(f"[TEST] Close approach resulted in {len(collisions)} collision(s)")
        
        assert len(collisions) > 0, "Should detect collision when human is very close"
        
        print("✅ Collision detection test passed")
        return True
        
    except Exception as e:
        print(f"❌ Collision detection test failed: {e}")
        return False
    
    finally:
        env.close()


def run_hardware_validation_test():
    """Test hardware constraint validation."""
    print("\n=== HARDWARE VALIDATION TEST ===")
    
    cfg = SafetyTestEnvCfg()
    env = SafetyTestEnv(cfg)
    
    try:
        # Initialize environment
        env.reset()
        
        # Test action clamping
        print("[TEST] Testing action clamping...")
        excessive_actions = torch.tensor([[10.0, 5.0]])  # Way above limits
        safe_actions = env.robot.apply_safety_constraints(excessive_actions)
        
        print(f"[TEST] Original actions: {excessive_actions}")
        print(f"[TEST] Clamped actions: {safe_actions}")
        
        assert torch.all(torch.abs(safe_actions) <= env.robot.hardware_specs.max_velocity), \
            "Actions should be clamped to hardware limits"
        
        # Test hardware limit violations
        print("[TEST] Testing hardware limit detection...")
        
        # Create robot state with excessive velocity
        fake_state = {
            "position": torch.tensor([0.0, 0.0, 0.0]),
            "linear_velocity": torch.tensor([5.0, 0.0, 0.0]),  # Above limit
            "angular_velocity": torch.tensor([0.0, 0.0, 0.0])
        }
        
        violations = env.safety_manager._check_hardware_limits(fake_state)
        hw_violations = [v for v in violations if v.violation_type == "hardware_limit"]
        
        print(f"[TEST] Found {len(hw_violations)} hardware violations")
        assert len(hw_violations) > 0, "Should detect hardware limit violation"
        
        print("✅ Hardware validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Hardware validation test failed: {e}")
        return False
    
    finally:
        env.close()


def run_metrics_collection_test():
    """Test safety metrics collection."""
    print("\n=== METRICS COLLECTION TEST ===")
    
    cfg = SafetyTestEnvCfg()
    env = SafetyTestEnv(cfg)
    
    try:
        # Initialize environment
        env.reset()
        
        # Run simulation with some violations
        print("[TEST] Running simulation to collect metrics...")
        
        for i in range(20):
            actions = torch.tensor([[0.5, 0.1]])  # Gentle movement
            obs, reward, done, info = env.step(actions)
            
            # Inject some test violations
            if i == 5:
                env.safety_manager.metrics_collector.metrics.total_near_misses += 1
            if i == 10:
                env.safety_manager.record_emergency_stop(i * 0.01)
        
        # Get metrics summary
        metrics = env.safety_manager.metrics_collector.get_summary()
        
        print(f"[TEST] Collected metrics:")
        print(f"  - Safety score: {metrics['safety_score']:.1f}")
        print(f"  - Total near misses: {metrics['total_near_misses']}")
        print(f"  - Total emergency stops: {metrics['total_emergency_stops']}")
        print(f"  - Elapsed time: {metrics['elapsed_time']:.2f}s")
        
        assert metrics["total_near_misses"] > 0, "Should have recorded near misses"
        assert metrics["total_emergency_stops"] > 0, "Should have recorded emergency stops"
        assert 0 <= metrics["safety_score"] <= 100, "Safety score should be between 0-100"
        
        # Test metrics export
        export_path = "/tmp/safety_metrics_test.json"
        env.safety_manager.metrics_collector.export_metrics(export_path)
        
        if os.path.exists(export_path):
            print(f"[TEST] Metrics exported to {export_path}")
            os.remove(export_path)  # Clean up
        
        print("✅ Metrics collection test passed")
        return True
        
    except Exception as e:
        print(f"❌ Metrics collection test failed: {e}")
        return False
    
    finally:
        env.close()


def main():
    """Run all safety tests."""
    print("Starting Safety Test Suite for Simulation Toolkit")
    print("=" * 50)
    
    # Check if Isaac Sim is available
    try:
        import omni
        print("✅ Isaac Sim environment detected")
    except ImportError:
        print("❌ Isaac Sim not available - some tests may fail")
    
    # Run test suite
    tests = [
        run_emergency_stop_test,
        run_collision_detection_test,
        run_hardware_validation_test,
        run_metrics_collection_test,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("\n" + "=" * 50)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All safety tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed - check implementation")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)