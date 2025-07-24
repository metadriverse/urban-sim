# Action Plan: Adapting URBAN-SIM for Robotics Safety Simulation Toolkit

## Executive Summary

This action plan outlines how to leverage URBAN-SIM's infrastructure to build the robotics safety simulation toolkit described in the PRD. URBAN-SIM provides a strong foundation with GPU-accelerated physics, parallel simulation capabilities, and urban environment generation that aligns well with the safety testing requirements.

## 1. Foundation Analysis

### What URBAN-SIM Provides (Can Reuse)
- ✅ **NVIDIA Isaac Sim Integration** - Already configured and operational
- ✅ **GPU-Accelerated Physics** - PhysX with parallel environment support
- ✅ **Robot Model Framework** - URDF/USD loading and articulation support
- ✅ **Human Simulation** - Basic pedestrian models and behaviors
- ✅ **Sensor Simulation** - Camera, LiDAR infrastructure
- ✅ **RL Training Pipeline** - rl_games and RSL-RL integration
- ✅ **Cloud Deployment Ready** - Docker and parallel execution support

### What Needs to be Built
- ❌ **Safety-Specific Features** - Emergency stops, safety zones, violation detection
- ❌ **Edge-Case Generation** - OpenSCENARIO integration, Monte Carlo scenarios
- ❌ **Hardware Validation** - Actuator limits, sensor constraints
- ❌ **Safety Metrics System** - Collision tracking, near-miss detection
- ❌ **Ground Robotics Integration** - Lawn mower specific behaviors
- ❌ **Gritt Robot Support** - Solar panel installer configurations

## 2. Implementation Roadmap (4 Weeks)

### Week 1: Foundation & Robot Integration
**Days 1-2: Project Setup**
```bash
# Fork and adapt URBAN-SIM structure
simulation_toolkit/
├── robots/             # Ground Robotics & Gritt models
├── safety/             # Safety systems and metrics
├── scenarios/          # Edge-case generation
├── deployment/         # GCP/Docker configs
└── tests/             # Safety validation tests
```

**Days 3-4: Ground Robotics URDF Integration**
- Adapt `urbansim/primitives/robot/` structure for lawn mower
- Create `GroundRoboticsLawnMower` class extending `Articulation`
- Implement sensor configurations (360° LiDAR, ultrasonics, cliff sensors)
- Add safety-specific properties (emergency stop, lift detection)

**Days 5-7: Human Behavior Enhancement**
- Extend URBAN-SIM's pedestrian models with:
  - Child behavior (curious, unpredictable)
  - Pet behavior (erratic movement)
  - Worker behavior (predictable paths)
- Implement PedSim_ROS integration for social force modeling

### Week 2: Safety Systems & Gritt Integration
**Days 8-9: Safety Infrastructure**
- Create `SafetyManager` class for:
  - Real-time collision detection
  - Safety zone monitoring
  - Emergency stop triggering
  - Hardware constraint validation

**Days 10-12: Gritt Site Visit & Integration**
- Prepare integration framework
- Document requirements during visit
- Create placeholder for Gritt robot configuration

**Days 13-14: Unified Safety Metrics**
- Implement metric collection system:
  - Collision events
  - Near-miss tracking
  - Response time measurement
  - Safety zone violations

### Week 3: Edge-Case Generation & Scaling
**Days 15-17: Scenario Generation System**
- Integrate OpenSCENARIO for standardized scenarios
- Implement Monte Carlo edge-case generator
- Create scenario validation system
- Build on URBAN-SIM's scene generation

**Days 18-19: Cloud Deployment**
- Adapt URBAN-SIM's Docker configuration
- Create GCP deployment scripts
- Implement batch simulation orchestration
- Set up result aggregation pipeline

**Days 20-21: Performance Optimization**
- Leverage URBAN-SIM's parallel environment system
- Optimize for 100+ simultaneous simulations
- Implement GPU memory management
- Profile and tune physics timestep

### Week 4: Testing & Documentation
**Days 22-23: Automated Testing**
- Create safety scenario test suite
- Implement hardware validation tests
- Build regression testing framework

**Days 24-25: Documentation & Examples**
- Write API documentation
- Create example scenarios
- Document safety metrics
- Prepare deployment guide

**Days 26-28: Integration & Handoff**
- Final testing and validation
- Performance benchmarking
- Deployment to GCP
- Knowledge transfer

## 3. Technical Implementation Details

### 3.1 Safety System Architecture
```python
# safety/safety_manager.py
class SafetyManager:
    def __init__(self, robot_config, hardware_specs):
        self.safety_zones = self._calculate_safety_zones(hardware_specs)
        self.emergency_stop = EmergencyStopSystem()
        self.collision_detector = CollisionDetector()
        self.metrics_collector = SafetyMetricsCollector()
        
    def step(self, robot_state, environment_state):
        # Check all safety conditions
        violations = self._check_safety_violations()
        if violations:
            self.emergency_stop.trigger()
        self.metrics_collector.record(violations)
```

### 3.2 Robot Integration Pattern
```python
# robots/ground_robotics_mower.py
from urbansim.primitives.robot import RobotBase

class GroundRoboticsMower(RobotBase):
    def __init__(self, urdf_path, sensor_specs, actuator_specs):
        super().__init__()
        self.load_hardware_specs(sensor_specs, actuator_specs)
        self.safety_system = MowerSafetySystem()
        
    def configure_sensors(self):
        # Configure actual hardware sensors
        self.lidar_360 = self._setup_lidar(self.specs['lidar'])
        self.ultrasonics = self._setup_ultrasonic_array(self.specs['ultrasonic'])
        self.cliff_sensors = self._setup_cliff_detection(self.specs['cliff'])
```

### 3.3 Scenario Generation Integration
```python
# scenarios/edge_case_generator.py
class EdgeCaseGenerator:
    def __init__(self, urban_scene):
        self.scene = urban_scene
        self.scenario_templates = load_openscenario_templates()
        
    def generate_safety_scenarios(self, robot_type, count=1000):
        scenarios = []
        for _ in range(count):
            # Leverage URBAN-SIM's scene generation
            base_scene = self.scene.generate_random_layout()
            # Add safety-specific elements
            scenario = self._add_safety_challenges(base_scene, robot_type)
            scenarios.append(scenario)
        return scenarios
```

## 4. Leveraging URBAN-SIM Components

### 4.1 Scene Generation
- Use URBAN-SIM's procedural generation for base environments
- Extend with safety-specific obstacles and scenarios
- Leverage existing road/sidewalk generation

### 4.2 Human Simulation
- Build on existing pedestrian models
- Add safety-critical behaviors (sudden movements, curiosity)
- Integrate with robot safety zones

### 4.3 Training Infrastructure
- Use URBAN-SIM's RL pipeline for safety policy training
- Adapt reward functions for safety objectives
- Leverage parallel training capabilities

### 4.4 Sensor Simulation
- Extend camera/LiDAR systems for safety sensors
- Add ultrasonic and cliff sensor models
- Integrate with safety detection algorithms

## 5. Key Modifications to URBAN-SIM

### 5.1 New Modules
```
simulation_toolkit/
├── safety/
│   ├── __init__.py
│   ├── safety_manager.py
│   ├── emergency_stop.py
│   ├── collision_detection.py
│   └── metrics_collector.py
├── robots/
│   ├── ground_robotics/
│   │   ├── lawn_mower.py
│   │   └── configs/
│   └── gritt/
│       ├── solar_installer.py
│       └── configs/
├── scenarios/
│   ├── openscenario_integration.py
│   ├── monte_carlo_generator.py
│   └── templates/
└── hardware_validation/
    ├── sensor_validator.py
    └── actuator_validator.py
```

### 5.2 Extended Components
- Extend `UrbanScene` with safety-aware scene generation
- Modify `AbstractEnv` to include safety manager
- Enhance human behaviors for safety testing
- Add hardware constraint checking to robot models

## 6. Risk Mitigation

### Technical Risks
- **Isaac Sim Complexity**: Use existing URBAN-SIM abstractions
- **Hardware Integration**: Start with simplified models, iterate
- **Performance**: Leverage URBAN-SIM's optimization work

### Schedule Risks
- **4-Week Timeline**: Focus on core safety features first
- **Gritt Integration**: Prepare flexible framework pre-visit
- **Testing Coverage**: Prioritize critical safety scenarios

## 7. Success Metrics

### Development Milestones
- Week 1: ✓ Basic lawn mower simulation with safety zones
- Week 2: ✓ Safety metrics collection working
- Week 3: ✓ 100+ parallel simulations running
- Week 4: ✓ Full safety test suite operational

### Technical Achievements
- Generate 1000+ safety scenarios automatically
- Support 100+ parallel robot simulations
- Achieve <10ms safety response latency
- Validate against hardware specifications

## 8. Next Steps

1. **Immediate Actions**:
   - Set up simulation_toolkit repository structure
   - Begin Ground Robotics URDF integration
   - Start safety system design

2. **Week 1 Priorities**:
   - Get basic robot simulation running
   - Implement initial safety zones
   - Test sensor integration

3. **Coordination**:
   - Schedule Gritt site visit for Week 2
   - Prepare hardware specification templates
   - Set up GCP infrastructure

## Conclusion

URBAN-SIM provides an excellent foundation for the robotics safety simulation toolkit. By leveraging its GPU-accelerated physics, parallel simulation capabilities, and urban environment generation, we can focus on implementing safety-specific features rather than building core infrastructure from scratch. This approach significantly reduces development time while ensuring robust performance and scalability.

The modular architecture allows for easy integration of new robots and safety scenarios, making it ideal for the iterative development approach outlined in the PRD. With careful planning and focused implementation, the 4-week timeline is achievable.