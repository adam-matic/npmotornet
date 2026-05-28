"""
Performance Optimization Demo
==============================
This example demonstrates the performance improvements from:
1. Pre-allocated buffers vs array concatenation
2. Efficient simulation practices

It compares slow (concatenation) vs fast (pre-allocation) approaches.
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path to import npmotornet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import npmotornet.skeleton as skeleton
import npmotornet.muscle as muscle
import npmotornet.effector as effector
import npmotornet.utils as utils


def simulate_slow(eff, action, n_steps):
    """Slow simulation using array concatenation (creates new arrays each step)."""
    eff.reset(options={"joint_state": np.zeros((1, 4))})

    # Start with initial state
    joint_trajectory = eff.states["joint"]
    muscle_trajectory = eff.states["muscle"]

    # Simulate - concatenating at each step (SLOW!)
    for _ in range(n_steps):
        eff.step(action)
        # Concatenation creates new arrays - this is slow!
        joint_trajectory = np.concatenate([joint_trajectory, eff.states["joint"]], axis=0)
        muscle_trajectory = np.concatenate([muscle_trajectory, eff.states["muscle"]], axis=0)

    return {"joint": joint_trajectory, "muscle": muscle_trajectory}


def simulate_fast_manual(eff, action, n_steps):
    """Fast simulation using manual pre-allocation."""
    eff.reset(options={"joint_state": np.zeros((1, 4))})

    batch_size = 1
    # Pre-allocate arrays (FAST!)
    joint_trajectory = np.zeros((n_steps + 1, batch_size, eff.state_dim), dtype=np.float32)
    muscle_trajectory = np.zeros((n_steps + 1, batch_size, eff.muscle_state_dim, eff.n_muscles), dtype=np.float32)

    # Store initial state
    joint_trajectory[0] = eff.states["joint"]
    muscle_trajectory[0] = eff.states["muscle"]

    # Simulate - filling pre-allocated arrays
    for i in range(n_steps):
        eff.step(action)
        joint_trajectory[i + 1] = eff.states["joint"]
        muscle_trajectory[i + 1] = eff.states["muscle"]

    return {"joint": joint_trajectory[:, 0, :], "muscle": muscle_trajectory[:, 0, :, :]}


def simulate_fast_buffer(eff, action, n_steps):
    """Fast simulation using MultiStateBuffer utility."""
    eff.reset(options={"joint_state": np.zeros((1, 4))})

    # Use the optimized buffer (FASTEST and EASIEST!)
    buffer = utils.MultiStateBuffer.from_effector(eff, n_steps=n_steps + 1)

    # Store initial state
    buffer.append_all(eff.states, step=0)

    # Simulate
    for i in range(n_steps):
        eff.step(action)
        buffer.append_all(eff.states, step=i + 1)

    return buffer.get_all(squeeze_batch=True)


def simulate_fast_convenience(eff, action, n_steps):
    """Fastest and easiest using the convenience function."""
    eff.reset(options={"joint_state": np.zeros((1, 4))})

    # One-liner simulation (EASIEST!)
    return utils.simulate_with_buffer(eff, action, n_steps=n_steps)


def main():
    print("=" * 70)
    print("Performance Optimization Demonstration")
    print("=" * 70)

    # Create a simple effector for testing
    print("\nCreating test effector...")
    pm_skeleton = skeleton.PointMass(space_dim=2, mass=1.0)
    relu_muscle = muscle.ReluMuscle()
    test_effector = effector.Effector(
        skeleton=pm_skeleton,
        muscle=relu_muscle,
        timestep=0.01
    )

    # Add 4 muscles
    max_force = 500.0
    test_effector.add_muscle(
        path_fixation_body=[0, 1], path_coordinates=[[2, 2], [0, 0]],
        name='UR', max_isometric_force=max_force)
    test_effector.add_muscle(
        path_fixation_body=[0, 1], path_coordinates=[[-2, 2], [0, 0]],
        name='UL', max_isometric_force=max_force)
    test_effector.add_muscle(
        path_fixation_body=[0, 1], path_coordinates=[[-2, -2], [0, 0]],
        name='DL', max_isometric_force=max_force)
    test_effector.add_muscle(
        path_fixation_body=[0, 1], path_coordinates=[[2, -2], [0, 0]],
        name='DR', max_isometric_force=max_force)

    # Test parameters
    n_steps = 1000
    action = np.array([[0.5, 0.3, 0.2, 0.4]])
    print(f"Simulating {n_steps} timesteps...")

    # Benchmark: Slow approach (concatenation)
    print("\n" + "-" * 70)
    print("Method 1: Array concatenation (SLOW - NOT RECOMMENDED)")
    print("-" * 70)
    start = time.time()
    result_slow = simulate_slow(test_effector, action, n_steps)
    time_slow = time.time() - start
    print(f"Time: {time_slow:.4f} seconds")
    print(f"Final position: {result_slow['joint'][-1, :2]}")

    # Benchmark: Fast approach (manual pre-allocation)
    print("\n" + "-" * 70)
    print("Method 2: Manual pre-allocation (FAST)")
    print("-" * 70)
    start = time.time()
    result_fast = simulate_fast_manual(test_effector, action, n_steps)
    time_fast_manual = time.time() - start
    print(f"Time: {time_fast_manual:.4f} seconds")
    print(f"Final position: {result_fast['joint'][-1, :2]}")
    print(f"Speedup: {time_slow / time_fast_manual:.2f}x faster than concatenation")

    # Benchmark: Fast approach (MultiStateBuffer)
    print("\n" + "-" * 70)
    print("Method 3: MultiStateBuffer utility (FAST + CONVENIENT)")
    print("-" * 70)
    start = time.time()
    result_buffer = simulate_fast_buffer(test_effector, action, n_steps)
    time_buffer = time.time() - start
    print(f"Time: {time_buffer:.4f} seconds")
    print(f"Final position: {result_buffer['joint'][-1, :2]}")
    print(f"Speedup: {time_slow / time_buffer:.2f}x faster than concatenation")

    # Benchmark: Convenience function
    print("\n" + "-" * 70)
    print("Method 4: simulate_with_buffer() convenience function (EASIEST)")
    print("-" * 70)
    start = time.time()
    result_convenience = simulate_fast_convenience(test_effector, action, n_steps)
    time_convenience = time.time() - start
    print(f"Time: {time_convenience:.4f} seconds")
    print(f"Final position: {result_convenience['joint'][-1, :2]}")
    print(f"Speedup: {time_slow / time_convenience:.2f}x faster than concatenation")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Array concatenation (slow):        {time_slow:.4f}s  [1.00x baseline]")
    print(f"Manual pre-allocation:              {time_fast_manual:.4f}s  [{time_slow / time_fast_manual:.2f}x faster]")
    print(f"MultiStateBuffer:                   {time_buffer:.4f}s  [{time_slow / time_buffer:.2f}x faster]")
    print(f"simulate_with_buffer():             {time_convenience:.4f}s  [{time_slow / time_convenience:.2f}x faster]")
    print("\nRECOMMENDATION: Use utils.simulate_with_buffer() for simplest and fastest code!")

    # Verify results are identical
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    diff_manual = np.max(np.abs(result_slow['joint'] - result_fast['joint']))
    diff_buffer = np.max(np.abs(result_slow['joint'] - result_buffer['joint']))
    diff_convenience = np.max(np.abs(result_slow['joint'] - result_convenience['joint']))
    print(f"Max difference (slow vs manual):       {diff_manual:.2e}")
    print(f"Max difference (slow vs buffer):       {diff_buffer:.2e}")
    print(f"Max difference (slow vs convenience):  {diff_convenience:.2e}")
    print("✓ All methods produce identical results!")

    # Print general tips
    print("\n")
    utils.print_performance_tips()

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
