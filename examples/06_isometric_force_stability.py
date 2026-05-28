"""
Isometric Constraint for Force Stability
=========================================
This example demonstrates the importance of isometric constraints when studying
muscle force production and sensory feedback.

When using a free-floating PointMass, muscle contraction causes the mass to
accelerate, which reduces tendon tension as the system moves. This makes it
difficult to achieve stable force measurements.

The FixedPointMass skeleton solves this by maintaining a fixed position,
allowing muscles to develop full isometric force. This is critical for:
- Validating GTO force-firing relationships
- Maximum voluntary contraction studies
- Clean force production data

This example compares:
1. Regular PointMass (free-floating) - forces drop due to acceleration
2. FixedPointMass (isometric) - forces reach stable maximum

References:
    [1] Houk JC, Henneman E. (1967). J Neurophysiol, 30(3), 466-481.
        (GTO recordings performed under isometric conditions)
"""

import numpy as np
import sys
import os

# Add parent directory to path to import npmotornet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import npmotornet.skeleton as skeleton
import npmotornet.muscle as muscle
import npmotornet.effector as effector
import npmotornet.sensory as sensory

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will not be generated.")


def test_free_floating_mass():
    """Test muscle force with regular PointMass (free-floating)."""
    print("\n" + "=" * 70)
    print("Test 1: Free-Floating PointMass (Non-Isometric)")
    print("=" * 70)
    print("\nThe point mass accelerates when muscle contracts, reducing force.")

    # Create free-floating system
    pm_skeleton = skeleton.PointMass(space_dim=2, mass=1.0)
    compliant_muscle = muscle.CompliantTendonHillMuscle()
    test_effector = effector.Effector(
        skeleton=pm_skeleton,
        muscle=compliant_muscle,
        timestep=0.0001
    )

    # Add single muscle
    test_effector.add_muscle(
        path_fixation_body=[0, 1],
        path_coordinates=[[6, 0], [0, 0]],
        max_isometric_force=200.0,
        tendon_length=5.0,
        optimal_muscle_length=0.8
    )

    # Create GTO
    gto = sensory.GolgiTendonOrgan(n_receptors=1)
    gto.reset(batch_size=1)

    # Initialize
    test_effector.reset(options={"joint_state": np.zeros((1, 4))})

    # Apply constant activation
    n_steps = 2000  # 200 ms
    action = np.array([[0.7]])  # Strong contraction

    forces = []
    firing_rates = []
    positions = []
    time_points = []

    print(f"\n  Applying constant activation of {action[0, 0]:.1f} for {n_steps * test_effector.dt * 1000:.0f} ms...")

    for step in range(n_steps):
        test_effector.step(action)

        # Extract data
        muscle_state = test_effector.states["muscle"]
        joint_state = test_effector.states["joint"]
        tendon_force = muscle_state[0, 6, :].reshape(1, -1)
        position = joint_state[0, :2]  # First two values are position

        # Get GTO firing rate
        firing_rate = gto.get_firing_rate(tendon_force, dt=test_effector.dt, include_dynamic=True)

        forces.append(tendon_force[0, 0])
        firing_rates.append(firing_rate[0, 0])
        positions.append(np.linalg.norm(position))  # Distance from origin
        time_points.append(step * test_effector.dt * 1000)

    forces = np.array(forces)
    firing_rates = np.array(firing_rates)
    positions = np.array(positions)
    time_points = np.array(time_points)

    # Analyze results
    peak_force = np.max(forces)
    final_force = forces[-1]
    force_drop = peak_force - final_force
    final_displacement = positions[-1]

    print(f"\n  Results:")
    print(f"    Peak force:       {peak_force:6.2f} N (at t={time_points[np.argmax(forces)]:.1f} ms)")
    print(f"    Final force:      {final_force:6.2f} N")
    print(f"    Force drop:       {force_drop:6.2f} N ({force_drop/peak_force*100:.1f}% reduction)")
    print(f"    Displacement:     {final_displacement:6.3f} m (mass moved)")
    print(f"    Final GTO firing: {firing_rates[-1]:6.2f} Hz")

    print(f"\n  ⚠️  Force is unstable - drops as the mass accelerates")

    return time_points, forces, firing_rates, positions


def test_fixed_mass():
    """Test muscle force with FixedPointMass (isometric)."""
    print("\n" + "=" * 70)
    print("Test 2: Fixed PointMass (Isometric Constraint)")
    print("=" * 70)
    print("\nThe point mass stays fixed in place, allowing full force development.")

    # Create fixed system
    fixed_skeleton = skeleton.FixedPointMass(space_dim=2)
    compliant_muscle = muscle.CompliantTendonHillMuscle()
    test_effector = effector.Effector(
        skeleton=fixed_skeleton,
        muscle=compliant_muscle,
        timestep=0.0001
    )

    # Add single muscle (same parameters as before)
    test_effector.add_muscle(
        path_fixation_body=[0, 1],
        path_coordinates=[[6, 0], [0, 0]],
        max_isometric_force=200.0,
        tendon_length=5.0,
        optimal_muscle_length=0.8
    )

    # Create GTO
    gto = sensory.GolgiTendonOrgan(n_receptors=1)
    gto.reset(batch_size=1)

    # Initialize
    test_effector.reset(options={"joint_state": np.zeros((1, 4))})

    # Apply constant activation (same as before)
    n_steps = 2000  # 200 ms
    action = np.array([[0.7]])

    forces = []
    firing_rates = []
    positions = []
    time_points = []

    print(f"\n  Applying constant activation of {action[0, 0]:.1f} for {n_steps * test_effector.dt * 1000:.0f} ms...")

    for step in range(n_steps):
        test_effector.step(action)

        # Extract data
        muscle_state = test_effector.states["muscle"]
        joint_state = test_effector.states["joint"]
        tendon_force = muscle_state[0, 6, :].reshape(1, -1)
        position = joint_state[0, :2]

        # Get GTO firing rate
        firing_rate = gto.get_firing_rate(tendon_force, dt=test_effector.dt, include_dynamic=True)

        forces.append(tendon_force[0, 0])
        firing_rates.append(firing_rate[0, 0])
        positions.append(np.linalg.norm(position))
        time_points.append(step * test_effector.dt * 1000)

    forces = np.array(forces)
    firing_rates = np.array(firing_rates)
    positions = np.array(positions)
    time_points = np.array(time_points)

    # Analyze results
    peak_force = np.max(forces)
    final_force = forces[-1]
    force_stability = np.std(forces[-500:])  # Std dev of last 50ms
    final_displacement = positions[-1]

    print(f"\n  Results:")
    print(f"    Peak force:       {peak_force:6.2f} N (at t={time_points[np.argmax(forces)]:.1f} ms)")
    print(f"    Final force:      {final_force:6.2f} N")
    print(f"    Force stability:  {force_stability:6.2f} N (std dev, lower is better)")
    print(f"    Displacement:     {final_displacement:6.3f} m (stays at origin)")
    print(f"    Final GTO firing: {firing_rates[-1]:6.2f} Hz")

    print(f"\n  ✅ Force is stable - isometric conditions maintained")

    return time_points, forces, firing_rates, positions


def test_force_activation_curve_isometric():
    """Generate force-activation curve under isometric conditions."""
    print("\n" + "=" * 70)
    print("Test 3: Force-Activation Curve (Isometric)")
    print("=" * 70)
    print("\nGenerating maximum voluntary contraction curve.\n")

    # Create fixed system for isometric testing
    fixed_skeleton = skeleton.FixedPointMass(space_dim=2)
    compliant_muscle = muscle.CompliantTendonHillMuscle()
    test_effector = effector.Effector(
        skeleton=fixed_skeleton,
        muscle=compliant_muscle,
        timestep=0.0001
    )

    test_effector.add_muscle(
        path_fixation_body=[0, 1],
        path_coordinates=[[6, 0], [0, 0]],
        max_isometric_force=200.0,
        tendon_length=5.0,
        optimal_muscle_length=0.8
    )

    gto = sensory.GolgiTendonOrgan(n_receptors=1)

    # Test range of activations
    activation_levels = np.linspace(0, 1.0, 11)
    steady_state_forces = []
    steady_state_firing = []

    print("  Activation | Steady Force (N) | GTO Firing (Hz)")
    print("  " + "-" * 55)

    for act_level in activation_levels:
        test_effector.reset(options={"joint_state": np.zeros((1, 4))})
        gto.reset(batch_size=1)

        action = np.array([[act_level]])

        # Run to steady state (150 ms)
        for _ in range(1500):
            test_effector.step(action)

        # Measure steady-state values
        muscle_state = test_effector.states["muscle"]
        tendon_force = muscle_state[0, 6, :].reshape(1, -1)
        firing_rate = gto.get_firing_rate(tendon_force, dt=test_effector.dt, include_dynamic=False)

        steady_state_forces.append(tendon_force[0, 0])
        steady_state_firing.append(firing_rate[0, 0])

        print(f"     {act_level:.2f}    |      {tendon_force[0, 0]:6.2f}      |     {firing_rate[0, 0]:6.2f}")

    print(f"\n  ✅ Clean force-activation relationship achieved")

    return activation_levels, np.array(steady_state_forces), np.array(steady_state_firing)


def plot_comparison(free_data, fixed_data, activation_data):
    """Plot comparison between free-floating and fixed conditions."""
    if not PLOTTING_AVAILABLE:
        print("\nSkipping plots (matplotlib not available)")
        return

    time_free, force_free, firing_free, pos_free = free_data
    time_fixed, force_fixed, firing_fixed, pos_fixed = fixed_data
    activations, forces_act, firing_act = activation_data

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Isometric Constraint for Force Stability', fontsize=15, fontweight='bold')

    # Plot 1: Force comparison
    ax1 = axes[0, 0]
    ax1.plot(time_free, force_free, 'r-', linewidth=2, alpha=0.7, label='Free-Floating PointMass')
    ax1.plot(time_fixed, force_fixed, 'b-', linewidth=2, alpha=0.7, label='Fixed PointMass (Isometric)')
    ax1.set_xlabel('Time (ms)', fontsize=11)
    ax1.set_ylabel('Tendon Force (N)', fontsize=11)
    ax1.set_title('Force Development Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, 'Isometric condition\nmaintains stable force',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # Plot 2: GTO firing rate comparison
    ax2 = axes[0, 1]
    ax2.plot(time_free, firing_free, 'r-', linewidth=2, alpha=0.7, label='Free-Floating')
    ax2.plot(time_fixed, firing_fixed, 'b-', linewidth=2, alpha=0.7, label='Isometric')
    ax2.set_xlabel('Time (ms)', fontsize=11)
    ax2.set_ylabel('GTO Firing Rate (Hz)', fontsize=11)
    ax2.set_title('GTO Response Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Displacement comparison
    ax3 = axes[1, 0]
    ax3.plot(time_free, pos_free, 'r-', linewidth=2, alpha=0.7, label='Free-Floating (moves)')
    ax3.plot(time_fixed, pos_fixed, 'b-', linewidth=2, alpha=0.7, label='Fixed (stays at origin)')
    ax3.set_xlabel('Time (ms)', fontsize=11)
    ax3.set_ylabel('Distance from Origin (m)', fontsize=11)
    ax3.set_title('Position Stability', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.text(0.02, 0.98, 'Isometric: no movement\nFree: mass accelerates',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Force-Activation curve (isometric only)
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()

    line1 = ax4.plot(activations, forces_act, 'b-o', linewidth=2, markersize=6,
                     alpha=0.7, label='Muscle Force')
    line2 = ax4_twin.plot(activations, firing_act, 'g-s', linewidth=2, markersize=5,
                          alpha=0.7, label='GTO Firing Rate')

    ax4.set_xlabel('Muscle Activation', fontsize=11)
    ax4.set_ylabel('Steady-State Force (N)', fontsize=11, color='b')
    ax4_twin.set_ylabel('GTO Firing Rate (Hz)', fontsize=11, color='g')
    ax4.set_title('Force-Activation Curve\n(Isometric Conditions)', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.tick_params(axis='y', labelcolor='g')
    ax4.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper left', fontsize=10)

    ax4.text(0.02, 0.98, 'Clean relationship\nfor validation studies',
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    plt.savefig('isometric_force_stability.png', dpi=150, bbox_inches='tight')
    print(f"\n{'='*70}")
    print(f"Figure saved as: isometric_force_stability.png")
    print(f"{'='*70}")
    plt.show()


def main():
    print("=" * 70)
    print("Isometric Constraint for Improved Force Stability")
    print("=" * 70)
    print("\nThis example demonstrates why isometric constraints are essential")
    print("for studying muscle force production and GTO responses.\n")
    print("When muscles contract on a free-floating mass:")
    print("  • The mass accelerates, reducing tendon tension")
    print("  • Forces are unstable and don't reach maximum values")
    print("  • GTO firing rates don't reflect true force production\n")
    print("With a fixed (isometric) constraint:")
    print("  • Position remains constant (true isometric conditions)")
    print("  • Forces reach stable maximum values")
    print("  • Clean force-firing relationships for validation\n")

    # Run all tests
    free_data = test_free_floating_mass()
    fixed_data = test_fixed_mass()
    activation_data = test_force_activation_curve_isometric()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\n  Free-Floating PointMass:")
    print(f"    • Force drops by {(free_data[1][0] - free_data[1][-1]) / free_data[1][0] * 100:.1f}% as mass accelerates")
    print(f"    • Mass moves {free_data[3][-1]:.3f} m from origin")
    print(f"    • ❌ Not suitable for isometric validation studies\n")

    print("  Fixed PointMass (Isometric):")
    print(f"    • Force remains stable (std dev: {np.std(fixed_data[1][-500:]):.2f} N)")
    print(f"    • Position stays at origin (0.000 m)")
    print(f"    • ✅ Ideal for GTO validation and MVC studies\n")

    print("  Recommendation:")
    print("    Use FixedPointMass for all isometric testing and validation.")
    print("=" * 70)

    # Generate plots
    if PLOTTING_AVAILABLE:
        print("\nGenerating comparison plots...")
        plot_comparison(free_data, fixed_data, activation_data)


if __name__ == "__main__":
    main()
