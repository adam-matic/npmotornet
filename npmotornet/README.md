# npmotornet

A NumPy-based biomechanical simulation library for motor control research.

## Overview

`npmotornet` is a pure NumPy conversion of [MotorNet](https://github.com/OlivierCodol/MotorNet), a Python toolbox for controlling biomechanically realistic effectors with artificial neural networks. This version removes the PyTorch dependency and provides the same biomechanical simulation capabilities using only NumPy.

## Features

- **Pure NumPy**: No PyTorch or GPU dependencies
- **Biomechanically Realistic Models**: Multiple skeleton types (PointMass, TwoDofArm) and muscle models
- **Muscle Models**:
  - ReluMuscle: Simple linear force production
  - RigidTendonHillMuscle: Hill-type muscle with rigid tendon
  - RigidTendonHillMuscleThelen: Thelen's formulation
  - CompliantTendonHillMuscle: Full elastic tendon model
- **Pre-built Effectors**: Ready-to-use biomechanical systems (ReluPointMass24, RigidTendonArm26, CompliantTendonArm26)
- **Numerical Integration**: Euler and Runge-Kutta 4th order methods

## Installation

```bash
# Clone or copy the npmotornet directory
# No additional dependencies beyond NumPy
pip install numpy
```

## Quick Start

```python
import numpy as np
import npmotornet.skeleton as skeleton
import npmotornet.muscle as muscle
import npmotornet.effector as effector

# Create a 2D point-mass with ReLu muscles
pm_skeleton = skeleton.PointMass(space_dim=2)
relu_muscle = muscle.ReluMuscle()
pm_effector = effector.Effector(skeleton=pm_skeleton, muscle=relu_muscle)

# Add muscles
pm_effector.add_muscle(
    path_fixation_body=[0, 1],
    path_coordinates=[[2, 2], [0, 0]],
    name='UpperRight',
    max_isometric_force=500
)

# Initialize and simulate
pm_effector.reset(options={"joint_state": np.zeros((1, 4))})
action = np.array([[1.0]])  # Full activation
pm_effector.step(action)

print("Position:", pm_effector.states["joint"][0, :2])
```

## Examples

See the `examples/` directory for detailed demonstrations:
- `01_basic_pointmass_relu.py`: Basic point-mass simulation
- `02_hill_muscle_demo.py`: Hill-type muscle mechanics
- `03_two_dof_arm.py`: 2DOF planar arm kinematics
- `04_compliant_tendon_muscle.py`: Elastic tendon dynamics

## Documentation

The API closely matches the original MotorNet. Key differences:
- All `torch.tensor` operations replaced with `np.array`
- No device management (`.to()`, `.device`, etc.)
- Classes are plain Python objects (no `torch.nn.Module`)
- All numerical operations use NumPy

For detailed documentation on the biomechanics and usage patterns, see:
- Original MotorNet docs: https://www.motornet.org/
- MotorNet paper: https://doi.org/10.7554/eLife.88591

## Citation

If you use this software, please cite the original MotorNet work:

```
Codol, O., Michaels, J.A., Kashefi, M. et al. MotorNet, a Python toolbox
for controlling differentiable biomechanical effectors with artificial
neural networks. eLife 12:RP88591 (2024).
https://doi.org/10.7554/eLife.88591
```

## License

This is a derivative work of MotorNet, licensed under the GNU General Public License v3.0, the same license as the original project. See `LICENSE` for the full license text and `NOTICE` for attribution details.

**Original MotorNet:**
- Copyright (c) 2021-2024 Olivier Codol and contributors
- Repository: https://github.com/OlivierCodol/MotorNet
- License: GPL-3.0

**npmotornet (this derivative work):**
- Copyright (c) 2025
- Licensed under GPL-3.0

## Contributing

This is a converted version of MotorNet. For contributions to the core biomechanics:
1. Consider contributing to the original MotorNet project
2. For NumPy-specific improvements, modifications follow GPL v3 requirements

## Acknowledgments

This work is based on MotorNet by Olivier Codol and contributors. All biomechanical models, algorithms, and core functionality credit goes to the original MotorNet team.
