"""
Hill-Type Muscle Demonstration
===============================
This example demonstrates the use of Hill-type muscles (Kistemaker version)
with a 2D point-mass effector. It shows muscle force production curves and
dynamic simulations.

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
    """Simulate motion of the effector with constant muscle activation."""
    effector_obj.reset(options={"joint_state": np.zeros((1, 4))})

    j_state = effector_obj.states["joint"]
    m_state = effector_obj.states["muscle"]
    g_state = effector_obj.states["geometry"]

    n_steps = int(movement_duration / effector_obj.dt)
    action = np.ones((1, effector_obj.n_muscles)) * np.array(action)

    for _ in range(n_steps):
        effector_obj.step(action)
        j_state = np.concatenate([j_state, effector_obj.states["joint"]], axis=0)
        m_state = np.concatenate([m_state, effector_obj.states["muscle"]], axis=0)
        g_state = np.concatenate([g_state, effector_obj.states["geometry"]], axis=0)

    return j_state, m_state, g_state


def demonstrate_force_curves():
    """Demonstrate force-length relationships of Hill-type muscle."""
    print("\n" + "=" * 60)
    print("Force-Length Curves Demonstration")
    print("=" * 60)

    # Create simple effector with one muscle
    pm_skeleton = skeleton.PointMass(space_dim=2)
    hill_muscle = muscle.RigidTendonHillMuscle()
    test_effector = effector.Effector(skeleton=pm_skeleton, muscle=hill_muscle)

    test_effector.add_muscle(
        path_fixation_body=[0, 1],
        path_coordinates=[[1, 1], [0, 0]],
        max_isometric_force=1.0,
        tendon_length=0.0,
        optimal_muscle_length=0.4
    )

    print(f"\nMuscle state features: {test_effector.muscle.state_name}")

    # Test force production at different muscle lengths
    n_points = 20
    muscle_lengths = np.linspace(0.2, 0.8, n_points)

    print("\nActive Force-Length Relationship:")
    print("  Length (m) | Active FL | Passive FL")
    print("  " + "-" * 40)

    for muscle_len in muscle_lengths[::4]:  # Show every 4th point
        # Create muscle state with zero activation
        m = np.zeros((1, test_effector.muscle_state_dim, test_effector.n_muscles))
        deriv = test_effector.muscle.ode(action=np.array(0.0), muscle_state=m)

        # Create geometry state with specific muscle length
        g = np.zeros((1, test_effector.geometry_state_dim, test_effector.n_muscles))
        g[0, 0, 0] = muscle_len  # musculotendon length

        # Integrate to get forces
        states = test_effector.muscle.integrate(
            dt=test_effector.dt,
            state_derivative=deriv,
            muscle_state=m,
            geometry_state=g
        )

        active_fl = states[0, 4, 0]  # force-length CE
        passive_fl = states[0, 3, 0]  # force-length PE

        print(f"    {muscle_len:.4f}   |  {active_fl:.4f}   |  {passive_fl:.4f}")


def main():
    print("=" * 60)
    print("Hill-Type Muscle Demonstration")
    print("=" * 60)

    # Demonstrate force curves
    demonstrate_force_curves()

    # Create effector with 4 Hill muscles
    print("\n" + "=" * 60)
    print("Dynamic Simulation with Hill Muscles")
    print("=" * 60)

    pm_skeleton = skeleton.PointMass(space_dim=2)
    hill_muscle = muscle.RigidTendonHillMuscle()
    hill_effector = effector.Effector(
        skeleton=pm_skeleton,
        muscle=hill_muscle,
        timestep=0.001
    )

    # Add 4 muscles in X configuration
    optimal_length = np.sqrt(2 * (2 ** 2))
    max_force = 50.0

    print("\n1. Building effector with 4 Hill-type muscles...")

    hill_effector.add_muscle(
        path_fixation_body=[0, 1],
        path_coordinates=[[2, 2], [0, 0]],
        max_isometric_force=max_force,
        tendon_length=0.0,
        optimal_muscle_length=optimal_length,
        name='UpRight'
    )

    hill_effector.add_muscle(
        path_fixation_body=[0, 1],
        path_coordinates=[[-2, 2], [0, 0]],
        max_isometric_force=max_force,
        tendon_length=0.0,
        optimal_muscle_length=optimal_length,
        name='UpLeft'
    )

    hill_effector.add_muscle(
        path_fixation_body=[0, 1],
        path_coordinates=[[2, -2], [0, 0]],
        max_isometric_force=max_force,
        tendon_length=0.0,
        optimal_muscle_length=optimal_length,
        name='DownRight'
    )

    hill_effector.add_muscle(
        path_fixation_body=[0, 1],
        path_coordinates=[[-2, -2], [0, 0]],
        max_isometric_force=max_force,
        tendon_length=0.0,
        optimal_muscle_length=optimal_length,
        name='DownLeft'
    )

    print(f"   Added {hill_effector.n_muscles} muscles")

    # Inspect initial states
    print("\n2. Inspecting initial muscle states...")
    hill_effector.reset(options={"joint_state": np.zeros((1, 4))})
    initial_state = hill_effector.states["muscle"]

    print(f"   Initial state shape: {initial_state.shape}")
    print("\n   Initial muscle values (at center position):")
    for i, name in enumerate(hill_effector.muscle_name):
        activation = initial_state[0, 0, i]
        length = initial_state[0, 1, i]
        velocity = initial_state[0, 2, i]
        force = initial_state[0, 6, i]  # total force
        print(f"     {name:12s}: act={activation:.4f}, len={length:.4f}m, vel={velocity:.4f}m/s, F={force:.4f}N")

    # Simulate motion
    print("\n3. Simulating rightward movement (UpRight muscle only)...")
    movement_duration = 0.25
    joint_state, muscle_state, _ = simulate_motion(
        hill_effector,
        [1.0, 0.0, 0.0, 0.0],
        movement_duration
    )

    final_pos = joint_state[-1, :2]
    initial_pos = joint_state[0, :2]
    print(f"   Initial position: [{initial_pos[0]:.4f}, {initial_pos[1]:.4f}] m")
    print(f"   Final position: [{final_pos[0]:.4f}, {final_pos[1]:.4f}] m")
    print(f"   Displacement: [{final_pos[0]-initial_pos[0]:.4f}, {final_pos[1]-initial_pos[1]:.4f}] m")

    # Check muscle activation development
    final_activation = muscle_state[-1, 0, 0]
    print(f"   UpRight final activation: {final_activation:.4f}")

    # Isometric co-contraction
    print("\n4. Simulating isometric co-contraction...")
    hill_effector.reset(options={"joint_state": np.zeros((1, 4))})
    joint_state, muscle_state, _ = simulate_motion(
        hill_effector,
        [1.0, 1.0, 1.0, 1.0],
        movement_duration
    )

    final_pos = joint_state[-1, :2]
    displacement = np.linalg.norm(final_pos - joint_state[0, :2])
    print(f"   Total displacement: {displacement:.6f} m (should be near zero)")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
