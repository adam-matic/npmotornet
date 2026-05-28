"""
Basic PointMass with ReLu Muscles Example
==========================================
This example demonstrates how to create a simple 2D point-mass effector with
4 ReLu muscles in an "X" configuration, and simulate its motion.

Converted from MotorNet PyTorch to npmotornet (NumPy).
"""

import numpy as np
import sys
import os

# Add parent directory to path to import npmotornet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import npmotornet.skeleton as skeleton
import npmotornet.muscle as muscle
import npmotornet.effector as effector


def simulate_motion(effector_obj, action, movement_duration):
    """Simulate motion of the effector with constant muscle activation.

    Args:
        effector_obj: The effector object to simulate
        action: Array of muscle activations (one per muscle)
        movement_duration: Duration of simulation in seconds

    Returns:
        Tuple of (joint_states, muscle_states, geometry_states)
    """
    # Initialize with zero position
    effector_obj.reset(options={"joint_state": np.zeros((1, 4))})

    # Get initial states
    j_state = effector_obj.states["joint"]
    m_state = effector_obj.states["muscle"]
    g_state = effector_obj.states["geometry"]

    # Time vector
    n_steps = int(movement_duration / effector_obj.dt)
    action = np.ones((1, effector_obj.n_muscles)) * np.array(action)

    # Run simulation
    for _ in range(n_steps):
        effector_obj.step(action)

        j_state = np.concatenate([j_state, effector_obj.states["joint"]], axis=0)
        m_state = np.concatenate([m_state, effector_obj.states["muscle"]], axis=0)
        g_state = np.concatenate([g_state, effector_obj.states["geometry"]], axis=0)

    return j_state, m_state, g_state


def main():
    print("=" * 60)
    print("Basic PointMass with ReLu Muscles Example")
    print("=" * 60)

    # Create the point-mass skeleton and ReLu muscle type
    print("\n1. Creating skeleton and muscle type...")
    pm_skeleton = skeleton.PointMass(space_dim=2, mass=1.0)
    relu_muscle = muscle.ReluMuscle()

    # Create the effector
    print("2. Creating effector...")
    relu_effector = effector.Effector(
        skeleton=pm_skeleton,
        muscle=relu_muscle,
        timestep=0.01
    )

    # Add muscles in an "X" configuration
    print("3. Adding muscles...")
    max_force = 500.0

    # Upper right muscle
    relu_effector.add_muscle(
        path_fixation_body=[0, 1],
        path_coordinates=[[2, 2], [0, 0]],
        name='UpRight',
        max_isometric_force=max_force
    )

    # Upper left muscle
    relu_effector.add_muscle(
        path_fixation_body=[0, 1],
        path_coordinates=[[-2, 2], [0, 0]],
        name='UpLeft',
        max_isometric_force=max_force
    )

    # Lower left muscle
    relu_effector.add_muscle(
        path_fixation_body=[0, 1],
        path_coordinates=[[-2, -2], [0, 0]],
        name='DownLeft',
        max_isometric_force=max_force
    )

    # Lower right muscle (3-point muscle for testing)
    relu_effector.add_muscle(
        path_fixation_body=[0, 0, 1],
        path_coordinates=[[4, -2], [2, -2], [0, 0]],
        name='DownRight',
        max_isometric_force=max_force
    )

    print(f"   Added {relu_effector.n_muscles} muscles")
    print(f"   Muscle names: {relu_effector.muscle_name}")

    # Inspect initial muscle states
    print("\n4. Inspecting initial muscle states...")
    relu_effector.reset(options={"joint_state": np.zeros((1, 4))})
    muscle_state = relu_effector.states["muscle"]

    print(f"   Muscle state features: {relu_effector.muscle.state_name}")
    print(f"   Muscle state shape: {muscle_state.shape}")
    print(f"   (batch_size={muscle_state.shape[0]}, n_features={muscle_state.shape[1]}, n_muscles={muscle_state.shape[2]})")

    print("\n   Initial muscle lengths:")
    for i, name in enumerate(relu_effector.muscle_name):
        length = muscle_state[0, 1, i]  # index 1 is muscle length
        print(f"     {name}: {length:.4f} m")

    # Simulate motion - activate only upper right muscle
    print("\n5. Simulating motion (upper right muscle only)...")
    movement_duration = 0.15  # 150 ms
    joint_state, muscle_state, geometry_state = simulate_motion(
        relu_effector,
        [1.0, 0.0, 0.0, 0.0],  # Only activate UpRight
        movement_duration
    )

    final_pos = joint_state[-1, :2]
    initial_pos = joint_state[0, :2]
    print(f"   Initial position: [{initial_pos[0]:.4f}, {initial_pos[1]:.4f}] m")
    print(f"   Final position: [{final_pos[0]:.4f}, {final_pos[1]:.4f}] m")
    print(f"   Displacement: [{final_pos[0]-initial_pos[0]:.4f}, {final_pos[1]-initial_pos[1]:.4f}] m")

    # Simulate isometric co-contraction
    print("\n6. Simulating isometric co-contraction (all muscles)...")
    relu_effector.reset(options={"joint_state": np.zeros((1, 4))})
    joint_state, muscle_state, geometry_state = simulate_motion(
        relu_effector,
        [1.0, 1.0, 1.0, 1.0],  # Activate all muscles
        movement_duration
    )

    final_pos = joint_state[-1, :2]
    initial_pos = joint_state[0, :2]
    print(f"   Initial position: [{initial_pos[0]:.4f}, {initial_pos[1]:.4f}] m")
    print(f"   Final position: [{final_pos[0]:.4f}, {final_pos[1]:.4f}] m")
    print(f"   Displacement: [{final_pos[0]-initial_pos[0]:.4f}, {final_pos[1]-initial_pos[1]:.4f}] m")
    print("   (Should be near zero for isometric contraction)")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
