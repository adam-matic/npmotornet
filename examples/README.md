# npmotornet Examples

This directory contains example scripts demonstrating the use of the `npmotornet` library - a NumPy-based conversion of the MotorNet biomechanical simulation library.

## Overview

These examples showcase various components of the npmotornet library:
- Different skeleton types (PointMass, TwoDofArm)
- Different muscle models (ReLu, RigidTendonHill, CompliantTendonHill)
- Building custom effectors
- Running simulations

## Examples

### 01_basic_pointmass_relu.py
**Basic PointMass with ReLu Muscles**

The simplest example demonstrating:
- Creating a 2D point-mass skeleton
- Adding 4 ReLu muscles in an "X" configuration
- Inspecting muscle states
- Simulating directed motion (single muscle activation)
- Simulating isometric co-contraction (all muscles activated)

**Run:** `python 01_basic_pointmass_relu.py`

### 02_hill_muscle_demo.py
**Hill-Type Muscle Demonstration**

Demonstrates biomechanically realistic Hill-type muscles:
- Force-length relationship curves
- Active and passive force production
- Muscle activation dynamics
- Dynamic simulations with Hill muscles
- Comparison with ReLu muscles

**Run:** `python 02_hill_muscle_demo.py`

### 03_two_dof_arm.py
**Two Degrees-of-Freedom Arm**

Shows a more complex skeletal system:
- Creating a 2DOF planar arm with realistic biomechanics
- Joint-to-cartesian coordinate transformations
- Adding muscles to specific bones
- Simulating shoulder and elbow movements
- Using pre-built effector models (RigidTendonArm26)

**Run:** `python 03_two_dof_arm.py`

### 04_compliant_tendon_muscle.py
**Compliant Tendon Hill Muscle**

Demonstrates the most realistic muscle model with elastic tendons:
- Force production curves for compliant tendon muscles
- Tendon vs. passive muscle force relationships
- Tug-of-war configuration showing tendon stretching
- Isometric co-contraction with compliant tendons
- Pre-built CompliantTendonArm26 effector
- Small timesteps and RK4 integration for stability

**Run:** `python 04_compliant_tendon_muscle.py`

## Requirements

All examples require:
- Python 3.8+
- NumPy
- The `npmotornet` package (in the parent directory)

For visualization examples (if created):
- Matplotlib

## Running the Examples

From the examples directory:

```bash
python 01_basic_pointmass_relu.py
python 02_hill_muscle_demo.py
python 03_two_dof_arm.py
```

Or from the parent directory:

```bash
python examples/01_basic_pointmass_relu.py
```

## Example Output

Each example prints detailed information about:
- Object creation and initialization
- State dimensions and configurations
- Simulation progress
- Final results and verification

## Key Concepts Demonstrated

### Skeletons
- **PointMass**: Simplest skeleton, mass in 2D/3D space
- **TwoDofArm**: Planar arm with shoulder and elbow joints

### Muscles
- **ReluMuscle**: Simple linear force production
- **RigidTendonHillMuscle**: Hill-type model with rigid tendon (Kistemaker)
- **RigidTendonHillMuscleThelen**: Hill-type model (Thelen formulation)
- **CompliantTendonHillMuscle**: Hill-type with elastic tendon

### Effectors
Combine skeletons and muscles to create complete biomechanical systems:
- Custom effectors built from components
- Pre-built effectors (ReluPointMass24, RigidTendonArm26, CompliantTendonArm26)

### States
- **Joint state**: Position and velocity in joint space
- **Cartesian state**: Position and velocity in Cartesian space
- **Muscle state**: Activation, length, velocity, forces
- **Geometry state**: Musculotendon lengths, velocities, moment arms

## Differences from Original MotorNet

These examples are converted from PyTorch-based MotorNet to NumPy-based npmotornet:

**Key changes:**
- `torch.tensor()` → `np.array()`
- `torch.cat()` → `np.concatenate()`
- No GPU/device management needed
- Pure NumPy operations throughout

**API compatibility:**
- Same class names and structure
- Same method signatures
- Same state organization
- Results should be numerically equivalent

## Further Learning

For more advanced usage:
- Check the original MotorNet documentation: https://www.motornet.org/
- See the MotorNet paper: [MotorNet: a Python toolbox for controlling differentiable biomechanical effectors with artificial neural networks](https://elifesciences.org/articles/88591)
- Explore the source code in `../npmotornet/`

## Contributing

To add more examples:
1. Follow the existing structure
2. Include clear docstrings
3. Add entry to this README
4. Test with `python <example_name>.py`

## License

These examples follow the same license as npmotornet and the original MotorNet project.
