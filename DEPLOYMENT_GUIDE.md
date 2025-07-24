# Deployment Guide - Simulation Toolkit

## Overview
This guide walks through creating a PR, merging code, and deploying to GCP server with GPU for validation.

## Step 1: Create Branch and Commit Changes

### 1.1 Create Feature Branch
```bash
# Navigate to project directory
cd /Users/mrachidi/Code/urban-sim

# Create and switch to feature branch
git checkout -b feature/safety-extensions-initial

# Check status
git status
```

### 1.2 Add and Commit Files
```bash
# Add all new files
git add safety_extensions/
git add robot_configs/
git add tests/
git add run_safety_demo.py
git add ACTION_PLAN.md
git add PBI.md
git add TASKS_REMAINING.md
git add DEPLOYMENT_GUIDE.md

# Commit changes
git commit -m "Add initial safety extensions for robotics simulation toolkit

- Implement SafetyManager with collision detection and emergency stop
- Add Ground Robotics lawn mower configuration and hardware specs
- Create hardware validation system with constraint checking
- Add comprehensive safety metrics collection and reporting
- Implement safety zones management with dynamic updates
- Add test suite for validating safety features
- Create demo script for showcasing safety functionality
- Update documentation with implementation details and PBIs

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 1.3 Push to Remote
```bash
# Push new branch to remote
git push -u origin feature/safety-extensions-initial
```

## Step 2: Create Pull Request

### 2.1 Using GitHub CLI (if available)
```bash
# Create PR with description
gh pr create --title "Initial Safety Extensions Implementation" --body "$(cat <<'EOF'
## Summary
- ✅ Implement core safety management system with collision detection
- ✅ Add Ground Robotics lawn mower robot configuration  
- ✅ Create hardware validation and constraint checking
- ✅ Add comprehensive safety metrics collection
- ✅ Implement emergency stop and safety zones
- ✅ Create test suite for safety feature validation
- ✅ Add demonstration script with multiple scenarios

## Key Components Added
- `safety_extensions/`: Core safety management modules
- `robot_configs/ground_robotics/`: Lawn mower configuration and specs
- `tests/safety/`: Test suite for safety validation
- `run_safety_demo.py`: Interactive demonstration script

## Test Plan
- [x] Unit tests for safety manager components
- [x] Integration tests for robot-safety system interaction
- [x] Hardware validation constraint checking
- [x] Emergency stop functionality validation
- [x] Metrics collection and reporting verification

## Next Steps
- Validate on GCP server with Isaac Sim
- Add Gritt robot configuration after site visit
- Implement OpenSCENARIO edge-case generation
- Scale testing to 100+ parallel environments

🤖 Generated with [Claude Code](https://claude.ai/code)
EOF
)"
```

### 2.2 Alternative: Manual PR Creation
If GitHub CLI is not available, create PR manually at:
`https://github.com/metadriverse/urban-sim/compare/main...feature/safety-extensions-initial`

## Step 3: Merge Branch (After Review)

### 3.1 Merge via GitHub Interface
1. Review and approve PR
2. Click "Merge pull request"
3. Select "Create a merge commit" 
4. Confirm merge

### 3.2 Local Cleanup
```bash
# Switch back to main
git checkout main

# Pull latest changes
git pull origin main

# Delete feature branch
git branch -d feature/safety-extensions-initial
```

## Step 4: Deploy to GCP Server

### 4.1 Connect to GCP Server
```bash
# SSH to GCP server
ssh gcp-l4-vm
```

### 4.2 Update Repository on Server
```bash
# Navigate to project directory on server
cd ~/urban-sim  # or wherever the repo is located

# Pull latest changes
git pull origin main

# Verify new files are present
ls -la safety_extensions/
ls -la robot_configs/
ls -la tests/safety/
```

### 4.3 Setup Environment on Server
```bash
# Activate conda environment (if not already active)
conda activate urbansim

# Install any new dependencies
pip install pyyaml  # For hardware specs loading

# Verify Isaac Sim is available
echo $ISAACSIM_PATH
ls $ISAACSIM_PATH/isaac-sim.sh
```

## Step 5: Run Tests on GCP Server

### 5.1 Basic Safety Test
```bash
# Run basic safety test
cd ~/urban-sim
python tests/safety/test_basic_safety.py

# Expected output:
# Starting Safety Test Suite for Simulation Toolkit
# ===============================================
# === EMERGENCY STOP TEST ===
# [TEST] Triggering emergency stop...
# ✅ Emergency stop test passed
# === COLLISION DETECTION TEST ===
# ✅ Collision detection test passed
# === HARDWARE VALIDATION TEST ===
# ✅ Hardware validation test passed
# === METRICS COLLECTION TEST ===
# ✅ Metrics collection test passed
# TEST SUMMARY: 4/4 tests passed
# 🎉 All safety tests passed!
```

### 5.2 Interactive Safety Demo
```bash
# Run interactive demo with Isaac Sim GUI
python run_safety_demo.py --duration 30 --enable_cameras

# For headless testing
python run_safety_demo.py --duration 30 --headless

# Expected output:
# 🤖 Simulation Toolkit - Safety Demonstration
# ===============================================
# Configuration:
#   - Environments: 1
#   - Duration: 30.0s
#   - Cameras: False
#   - Headless: False
# 
# [INFO] Initializing Safety Demo with 1 environment(s)
# ✅ Isaac Sim initialized successfully
# [INFO] Setting up safety systems...
# [DEMO] Step 50, Scenario: normal_operation
# [DEMO] 🎬 Switching to scenario: human_approach
# ✅ Demo completed successfully!
```

### 5.3 URBAN-SIM Integration Test
```bash
# Test with existing URBAN-SIM environment
python urbansim/envs/separate_envs/random_env.py --num_envs 4 --enable_cameras

# Verify no conflicts with safety extensions
```

## Step 6: Validation Checklist

### 6.1 Code Quality Checks
- [ ] All files committed and pushed
- [ ] No merge conflicts
- [ ] Python imports work correctly
- [ ] No syntax errors

### 6.2 Safety System Validation
- [ ] Emergency stop triggers correctly
- [ ] Collision detection functions
- [ ] Hardware constraints enforced
- [ ] Safety metrics collected
- [ ] Test suite passes completely

### 6.3 Performance Validation
- [ ] Isaac Sim starts without errors
- [ ] GPU acceleration working
- [ ] Memory usage acceptable
- [ ] No significant performance degradation

### 6.4 Integration Validation
- [ ] URBAN-SIM environments still work
- [ ] Safety extensions load without conflicts
- [ ] Robot configurations valid
- [ ] Demo script runs successfully

## Troubleshooting

### Common Issues

#### 1. Isaac Sim Import Errors
```bash
# Check Isaac Sim path
echo $ISAACSIM_PATH
export ISAACSIM_PATH="/path/to/isaac-sim"

# Check Python path
export PYTHONPATH="$ISAACSIM_PATH/site-packages:$PYTHONPATH"
```

#### 2. CUDA/GPU Issues
```bash
# Check GPU availability
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. Missing Dependencies
```bash
# Install missing packages
pip install pyyaml numpy torch torchvision

# Check URBAN-SIM dependencies
bash urbansim.sh -i
```

#### 4. Permission Issues
```bash
# Fix file permissions
chmod +x run_safety_demo.py
chmod +x tests/safety/test_basic_safety.py
```

## Success Criteria

The deployment is successful when:

1. ✅ All code merged without conflicts
2. ✅ GCP server pulls latest changes
3. ✅ Safety test suite passes (4/4 tests)
4. ✅ Demo script runs without errors
5. ✅ Isaac Sim GUI loads and displays simulation
6. ✅ No performance regressions in URBAN-SIM
7. ✅ GPU acceleration confirmed working

## Next Steps After Successful Deployment

1. **Performance Benchmarking**: Run extended tests to measure safety system overhead
2. **Parallel Testing**: Scale up to multiple environments 
3. **Gritt Integration**: Prepare for Gritt robot site visit and integration
4. **Edge Case Scenarios**: Implement OpenSCENARIO integration
5. **Cloud Scaling**: Test deployment across multiple GCP instances

## Contact and Support

If you encounter issues during deployment:
1. Check logs in `/tmp/` directory
2. Verify all environment variables are set
3. Ensure GCP instance has sufficient GPU memory
4. Contact team for support if needed