"""
Compliant Tendon Hill Muscle Demonstration
==========================================
This example demonstrates the CompliantTendonHillMuscle class, which includes
an elastic tendon component. This is the full Kistemaker model with compliant
tendon mechanics.

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
    """Simulate motion with constant muscle activation."""
    effector_obj.reset(options={"joint_state": np.zeros((1, 4))})

    j_state = effector_obj.states["joint"]
    m_state = effector_obj.states["muscle"]

    n_steps = int(movement_duration / effector_obj.dt)
    action = np.ones((1, effector_obj.n_muscles)) * np.array(action)

    for _ in range(n_steps):
        effector_obj.step(action)
        j_state = np.concatenate([j_state, effector_obj.states["joint"]], axis=0)
        m_state = np.concatenate([m_state, effector_obj.states["muscle"]], axis=0)

    return j_state, m_state


def test_force_curves():
    """Test force production curves for compliant tendon muscle."""
    print("\n" + "=" * 60)
    print("Compliant Tendon Force Curves")
    print("=" * 60)

    # Create effector with one compliant muscle
    pm_skeleton = skeleton.PointMass(space_dim=2)
    compliant_muscle = muscle.CompliantTendonHillMuscle()
    test_effector = effector.Effector(
        skeleton=pm_skeleton,
        muscle=compliant_muscle,
        timestep=0.0001
    )

    # Long tendon to reduce stiffness and see force curves better
    test_effector.add_muscle(
        path_fixation_body=[0, 1],
        path_coordinates=[[6, 0], [0, 0]],
        max_isometric_force=1.0,
        tendon_length=5.0,
        optimal_muscle_length=0.8
    )

    print(f"\nMuscle state features: {test_effector.muscle.state_name}")
    print("(Note: Compliant muscles track muscle length separately from musculotendon length)")

    # Test force production at different muscle lengths
    n_points = 20
    muscle_lengths = np.linspace(0.75, 1.2, n_points)

    print("\nForce Production at Different Muscle Lengths:")
    print("  Muscle Len | Tendon FL | Passive FL | Active F | Total F")
    print("  " + "-" * 60)

    for muscle_len in muscle_lengths[::4]:  # Show every 4th point
        # Create muscle state with specific muscle length
        m = np.zeros((1, test_effector.muscle_state_dim, test_effector.n_muscles))
        m[0, 1, 0] = muscle_len  # Set muscle length
        deriv = test_effector.muscle.ode(action=np.array(0.0), muscle_state=m)

        # Create geometry state (musculotendon length is 6m)
        g = np.zeros((1, test_effector.geometry_state_dim, test_effector.n_muscles))
        g[0, 0, 0] = 6.0

        # Integrate to get forces
        states = test_effector.muscle.integrate(
            dt=test_effector.dt,
            state_derivative=deriv,
            muscle_state=m,
            geometry_state=g
        )

        tendon_fl = states[0, 4, 0]  # force-length SE (tendon)
        passive_fl = states[0, 3, 0]  # force-length PE (passive)
        active_f = states[0, 5, 0]  # active force
        total_f = states[0, 6, 0]  # total force

        print(f"    {muscle_len:.4f}   |  {tendon_fl:.4f}    |  {passive_fl:.4f}   |  {active_f:.4f} |  {total_f:.4f}")


def test_tug_of_war():
    """Test tug-of-war configuration with compliant tendons."""
    print("\n" + "=" * 60)
    print("Tug-of-War with Compliant Tendons")
    print("=" * 60)

    pm_skeleton = skeleton.PointMass(space_dim=2, mass=1.0)
    compliant_muscle = muscle.CompliantTendonHillMuscle()
    tow_effector = effector.Effector(
        skeleton=pm_skeleton,
        muscle=compliant_muscle,
        timestep=1e-4,
        integration_method='rk4'
    )

    # Add two opposing muscles
    tow_effector.add_muscle(
        path_fixation_body=[0, 1],
        path_coordinates=[[6, 0], [0, 0]],
        max_isometric_force=100,
        tendon_length=5,
        optimal_muscle_length=0.8,
        name='RightPuller'
    )

    tow_effector.add_muscle(
        path_fixation_body=[0, 1],
        path_coordinates=[[-6, 0], [0, 0]],
        max_isometric_force=100,
        tendon_length=5,
        optimal_muscle_length=0.8,
        name='LeftPuller'
    )

    print(f"\nCreated tug-of-war effector with {tow_effector.n_muscles} muscles")

    # Test 1: Activate right muscle only
    print("\n1. Activating right muscle only...")
    joint_state, muscle_state = simulate_motion(
        tow_effector,
        [0.5, 0.0],
        0.2
    )

    initial_pos = joint_state[0, 0]
    final_pos = joint_state[-1, 0]
    print(f"   Initial x position: {initial_pos:.4f} m")
    print(f"   Final x position: {final_pos:.4f} m")
    print(f"   Displacement: {final_pos - initial_pos:.4f} m (should be positive)")

    # Check muscle length changes
    initial_muscle_len = muscle_state[0, 1, :]
    final_muscle_len = muscle_state[-1, 1, :]
    print(f"\n   Right muscle: {initial_muscle_len[0]:.4f} -> {final_muscle_len[0]:.4f} m")
    print(f"   Left muscle:  {initial_muscle_len[1]:.4f} -> {final_muscle_len[1]:.4f} m")

    # Test 2: Isometric co-contraction
    print("\n2. Isometric co-contraction...")
    tow_effector.reset(options={"joint_state": np.zeros((1, 4))})
    joint_state, muscle_state = simulate_motion(
        tow_effector,
        [0.5, 0.5],
        0.2
    )

    initial_pos = joint_state[0, 0]
    final_pos = joint_state[-1, 0]
    displacement = abs(final_pos - initial_pos)
    print(f"   Initial x position: {initial_pos:.4f} m")
    print(f"   Final x position: {final_pos:.4f} m")
    print(f"   Total displacement: {displacement:.6f} m (should be near zero)")

    # Check that muscles are tensing even though position doesn't change
    initial_muscle_len = muscle_state[0, 1, :]
    final_muscle_len = muscle_state[-1, 1, :]
    initial_muscle_vel = muscle_state[0, 2, :]
    final_muscle_vel = muscle_state[-1, 2, :]

    print(f"\n   Right muscle length: {initial_muscle_len[0]:.4f} -> {final_muscle_len[0]:.4f} m")
    print(f"   Left muscle length:  {initial_muscle_len[1]:.4f} -> {final_muscle_len[1]:.4f} m")
    print(f"   (Muscle lengths change as tendons stretch)")


def test_prebuild_arm():
    """Test pre-built CompliantTendonArm26."""
    print("\n" + "=" * 60)
    print("Pre-built CompliantTendonArm26 Effector")
    print("=" * 60)

    # Create pre-built arm
    arm = effector.CompliantTendonArm26(timestep=0.0002)

    print(f"\nNumber of muscles: {arm.n_muscles}")
    print(f"Muscle names: {arm.muscle_name}")
    print(f"Integration method: {arm.integration_method}")
    print(f"Timestep: {arm.dt*1000:.4f} ms")

    # Initialize at 45 degrees for both joints
    initial_angles = np.array([[np.pi/4, np.pi/4]])
    arm.reset(options={"joint_state": initial_angles})

    print("\nInitial muscle states:")
    initial_muscle_state = arm.states["muscle"]

    print("  Muscle Name          | Activation | Length (m) | Velocity (m/s) | Force (N)")
    print("  " + "-" * 75)

    for i, name in enumerate(arm.muscle_name):
        activation = initial_muscle_state[0, 0, i]
        length = initial_muscle_state[0, 1, i]
        velocity = initial_muscle_state[0, 2, i]
        force = initial_muscle_state[0, 6, i]
        print(f"  {name:20s} |   {activation:.4f}   |   {length:.4f}   |    {velocity:.4f}    | {force:.4f}")

    print("\nNote: CompliantTendonArm26 uses very small timesteps (0.2ms) and RK4 integration")
    print("      for stability with elastic tendons.")


def main():
    print("=" * 60)
    print("Compliant Tendon Hill Muscle Demonstration")
    print("=" * 60)

    # Test force curves
    test_force_curves()

    # Test tug-of-war dynamics
    test_tug_of_war()

    # Test pre-built arm
    test_prebuild_arm()

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
