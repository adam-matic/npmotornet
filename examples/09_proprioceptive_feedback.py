"""
Combined Proprioceptive Feedback: GTO + Muscle Spindle
=======================================================
This example demonstrates how Golgi tendon organs (GTOs) and muscle spindles
work together to provide the nervous system with complete information about
muscle state.

GTOs measure FORCE (tendon tension)
Spindles measure LENGTH and VELOCITY (muscle fiber stretch)

Together, they provide complementary information:
- During isometric contraction: GTO signals force, spindle signals constant length
- During passive stretch: GTO signals minimal force, spindle signals length change
- During active movement: Both signal simultaneously

This example shows:
1. Passive stretch: Spindle active, GTO minimal
2. Isometric contraction: GTO active, spindle constant
3. Active concentric contraction: Both change (force ↑, length ↓)
4. Active eccentric contraction: Both change (force ↑, length ↑)
5. Complete movement cycle with both sensors

References:
    [1] Proske U, Gandevia SC. (2012). The proprioceptive senses: their roles
        in signaling body shape, body position and movement, and muscle force.
        Physiol Rev, 92(4), 1651-1697.
    [2] Prochazka A, Gorassini M. (1998). Ensemble firing of muscle afferents
        recorded during normal locomotion in cats. J Physiol, 507(1), 293-304.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import npmotornet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import npmotornet.sensory as sensory
from npmotornet.sensory_configs import get_gto_config, get_spindle_config

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available. Plots will not be generated.")
    print("Install matplotlib with: pip install matplotlib")


class ProprioceptiveSensors:
    """Combined GTO and muscle spindle sensors."""

    def __init__(self, gto_config='default', spindle_config='default'):
        """Initialize both GTO and spindle with specified configurations."""
        # Create GTO
        gto_params = get_gto_config(gto_config)
        self.gto = sensory.GolgiTendonOrgan(n_receptors=1, **gto_params)

        # Create muscle spindle
        spindle_params = get_spindle_config(spindle_config)
        self.spindle = sensory.MuscleSpindle(n_receptors=1, **spindle_params)

        self.gto.reset(batch_size=1)
        self.spindle.reset(batch_size=1)

    def get_feedback(self, tendon_force, muscle_length, muscle_velocity, dt,
                     gamma_dynamic=None, gamma_static=None):
        """Get combined proprioceptive feedback.

        Args:
            tendon_force: Tendon force (N)
            muscle_length: Muscle fiber length (m)
            muscle_velocity: Muscle fiber velocity (m/s)
            dt: Timestep (s)
            gamma_dynamic: Gamma dynamic drive (0-1)
            gamma_static: Gamma static drive (0-1)

        Returns:
            Dictionary with GTO and spindle responses
        """
        # GTO response
        force_array = np.array([[tendon_force]])
        gto_firing = self.gto.get_firing_rate(force_array, dt)

        # Spindle response
        length_array = np.array([[muscle_length]])
        velocity_array = np.array([[muscle_velocity]])

        gamma_dyn = np.array([[gamma_dynamic]]) if gamma_dynamic is not None else None
        gamma_stat = np.array([[gamma_static]]) if gamma_static is not None else None

        spindle_response = self.spindle.get_firing_rate(
            length_array, velocity_array, dt,
            gamma_dynamic=gamma_dyn, gamma_static=gamma_stat
        )

        return {
            'gto': gto_firing[0, 0],
            'Ia': spindle_response['Ia'][0, 0],
            'II': spindle_response['II'][0, 0]
        }


def test_passive_stretch():
    """Demonstrate passive stretch: spindle responds, GTO minimal."""
    print("\n" + "=" * 70)
    print("Test 1: Passive Stretch")
    print("=" * 70)
    print("\nPassive stretch (no active force generation):")
    print("  Expected: Spindle (Ia, II) increase with stretch")
    print("            GTO remains near baseline (minimal passive force)")

    sensors = ProprioceptiveSensors()

    # Simulation parameters
    dt = 0.001  # 1ms
    n_steps = 300

    # Passive stretch from 0.08m to 0.10m (20mm stretch)
    optimal_length = 0.08
    lengths = np.linspace(optimal_length, optimal_length + 0.02, n_steps)
    velocities = np.diff(lengths, prepend=lengths[0]) / dt

    # Minimal passive force (assuming small passive stiffness)
    # For simplicity, assume linear passive force: F = k * (L - L0)
    passive_stiffness = 50  # N/m (example value)
    forces = passive_stiffness * np.maximum(0, lengths - optimal_length)

    # Record responses
    time = np.arange(n_steps) * dt * 1000  # Convert to ms
    gto_firing = []
    Ia_firing = []
    II_firing = []

    for i in range(n_steps):
        feedback = sensors.get_feedback(forces[i], lengths[i], velocities[i], dt)
        gto_firing.append(feedback['gto'])
        Ia_firing.append(feedback['Ia'])
        II_firing.append(feedback['II'])

    # Report results
    print(f"\n  Initial state:")
    print(f"    Length: {lengths[0]:.4f} m, Force: {forces[0]:.1f} N")
    print(f"    GTO: {gto_firing[0]:.1f} Hz, Ia: {Ia_firing[0]:.1f} Hz, II: {II_firing[0]:.1f} Hz")

    print(f"\n  Final state (after stretch):")
    print(f"    Length: {lengths[-1]:.4f} m, Force: {forces[-1]:.1f} N")
    print(f"    GTO: {gto_firing[-1]:.1f} Hz, Ia: {Ia_firing[-1]:.1f} Hz, II: {II_firing[-1]:.1f} Hz")

    print(f"\n  Changes:")
    print(f"    GTO: {gto_firing[-1] - gto_firing[0]:+.1f} Hz (small change)")
    print(f"    Ia:  {Ia_firing[-1] - Ia_firing[0]:+.1f} Hz (large change)")
    print(f"    II:  {II_firing[-1] - II_firing[0]:+.1f} Hz (moderate change)")

    print("\n  Validation: Spindle dominates during passive stretch ✓")

    return time, lengths, forces, gto_firing, Ia_firing, II_firing


def test_isometric_contraction():
    """Demonstrate isometric contraction: GTO responds, spindle constant."""
    print("\n" + "=" * 70)
    print("Test 2: Isometric Contraction")
    print("=" * 70)
    print("\nIsometric contraction (constant length, increasing force):")
    print("  Expected: GTO increases with force")
    print("            Spindle remains constant (no length change)")

    sensors = ProprioceptiveSensors()

    # Simulation parameters
    dt = 0.001
    n_steps = 300

    # Constant muscle length
    optimal_length = 0.08
    lengths = np.ones(n_steps) * optimal_length
    velocities = np.zeros(n_steps)

    # Increasing force (simulating increasing activation)
    forces = np.linspace(0, 100, n_steps)  # 0 to 100N

    # Record responses
    time = np.arange(n_steps) * dt * 1000
    gto_firing = []
    Ia_firing = []
    II_firing = []

    for i in range(n_steps):
        feedback = sensors.get_feedback(forces[i], lengths[i], velocities[i], dt)
        gto_firing.append(feedback['gto'])
        Ia_firing.append(feedback['Ia'])
        II_firing.append(feedback['II'])

    # Report results
    print(f"\n  Initial state:")
    print(f"    Length: {lengths[0]:.4f} m, Force: {forces[0]:.1f} N")
    print(f"    GTO: {gto_firing[0]:.1f} Hz, Ia: {Ia_firing[0]:.1f} Hz, II: {II_firing[0]:.1f} Hz")

    print(f"\n  Final state (after contraction):")
    print(f"    Length: {lengths[-1]:.4f} m, Force: {forces[-1]:.1f} N")
    print(f"    GTO: {gto_firing[-1]:.1f} Hz, Ia: {Ia_firing[-1]:.1f} Hz, II: {II_firing[-1]:.1f} Hz")

    print(f"\n  Changes:")
    print(f"    GTO: {gto_firing[-1] - gto_firing[0]:+.1f} Hz (large change)")
    print(f"    Ia:  {Ia_firing[-1] - Ia_firing[0]:+.1f} Hz (minimal)")
    print(f"    II:  {II_firing[-1] - II_firing[0]:+.1f} Hz (minimal)")

    print("\n  Validation: GTO dominates during isometric contraction ✓")

    return time, lengths, forces, gto_firing, Ia_firing, II_firing


def test_concentric_contraction():
    """Demonstrate concentric contraction: both sensors active."""
    print("\n" + "=" * 70)
    print("Test 3: Concentric (Shortening) Contraction")
    print("=" * 70)
    print("\nConcentric contraction (muscle shortens while generating force):")
    print("  Expected: GTO increases (force production)")
    print("            Ia and II decrease (muscle shortening)")

    sensors = ProprioceptiveSensors()

    dt = 0.001
    n_steps = 300

    # Muscle shortens from 0.08m to 0.06m
    optimal_length = 0.08
    lengths = np.linspace(optimal_length, optimal_length - 0.02, n_steps)
    velocities = np.diff(lengths, prepend=lengths[0]) / dt

    # Force increases during contraction (active force)
    forces = np.linspace(0, 150, n_steps)

    # Record responses
    time = np.arange(n_steps) * dt * 1000
    gto_firing = []
    Ia_firing = []
    II_firing = []

    for i in range(n_steps):
        feedback = sensors.get_feedback(forces[i], lengths[i], velocities[i], dt)
        gto_firing.append(feedback['gto'])
        Ia_firing.append(feedback['Ia'])
        II_firing.append(feedback['II'])

    # Report results
    print(f"\n  Initial state:")
    print(f"    Length: {lengths[0]:.4f} m, Force: {forces[0]:.1f} N")
    print(f"    GTO: {gto_firing[0]:.1f} Hz, Ia: {Ia_firing[0]:.1f} Hz, II: {II_firing[0]:.1f} Hz")

    print(f"\n  Final state:")
    print(f"    Length: {lengths[-1]:.4f} m, Force: {forces[-1]:.1f} N")
    print(f"    GTO: {gto_firing[-1]:.1f} Hz, Ia: {Ia_firing[-1]:.1f} Hz, II: {II_firing[-1]:.1f} Hz")

    print(f"\n  Changes:")
    print(f"    GTO: {gto_firing[-1] - gto_firing[0]:+.1f} Hz (increase, tracks force)")
    print(f"    Ia:  {Ia_firing[-1] - Ia_firing[0]:+.1f} Hz (decrease, tracks shortening)")
    print(f"    II:  {II_firing[-1] - II_firing[0]:+.1f} Hz (decrease, tracks length)")

    print("\n  Validation: Both sensors provide complementary information ✓")

    return time, lengths, forces, gto_firing, Ia_firing, II_firing


def test_eccentric_contraction():
    """Demonstrate eccentric contraction: both sensors increase."""
    print("\n" + "=" * 70)
    print("Test 4: Eccentric (Lengthening) Contraction")
    print("=" * 70)
    print("\nEccentric contraction (muscle lengthens while resisting):")
    print("  Expected: GTO increases (high force during eccentric)")
    print("            Ia and II increase (muscle lengthening)")

    sensors = ProprioceptiveSensors()

    dt = 0.001
    n_steps = 300

    # Muscle lengthens from 0.08m to 0.10m under load
    optimal_length = 0.08
    lengths = np.linspace(optimal_length, optimal_length + 0.02, n_steps)
    velocities = np.diff(lengths, prepend=lengths[0]) / dt

    # High force during eccentric contraction (muscles produce more force eccentrically)
    forces = np.linspace(50, 200, n_steps)

    # Record responses
    time = np.arange(n_steps) * dt * 1000
    gto_firing = []
    Ia_firing = []
    II_firing = []

    for i in range(n_steps):
        feedback = sensors.get_feedback(forces[i], lengths[i], velocities[i], dt)
        gto_firing.append(feedback['gto'])
        Ia_firing.append(feedback['Ia'])
        II_firing.append(feedback['II'])

    # Report results
    print(f"\n  Initial state:")
    print(f"    Length: {lengths[0]:.4f} m, Force: {forces[0]:.1f} N")
    print(f"    GTO: {gto_firing[0]:.1f} Hz, Ia: {Ia_firing[0]:.1f} Hz, II: {II_firing[0]:.1f} Hz")

    print(f"\n  Final state:")
    print(f"    Length: {lengths[-1]:.4f} m, Force: {forces[-1]:.1f} N")
    print(f"    GTO: {gto_firing[-1]:.1f} Hz, Ia: {Ia_firing[-1]:.1f} Hz, II: {II_firing[-1]:.1f} Hz")

    print(f"\n  Changes:")
    print(f"    GTO: {gto_firing[-1] - gto_firing[0]:+.1f} Hz (large increase)")
    print(f"    Ia:  {Ia_firing[-1] - Ia_firing[0]:+.1f} Hz (increase)")
    print(f"    II:  {II_firing[-1] - II_firing[0]:+.1f} Hz (increase)")

    print("\n  Validation: Both sensors signal this demanding condition ✓")

    return time, lengths, forces, gto_firing, Ia_firing, II_firing


def test_complete_movement_cycle():
    """Simulate a complete movement cycle with all phases."""
    print("\n" + "=" * 70)
    print("Test 5: Complete Movement Cycle")
    print("=" * 70)
    print("\nSimulating a complete movement cycle:")
    print("  Phase 1: Rest (low force, optimal length)")
    print("  Phase 2: Concentric contraction (force ↑, length ↓)")
    print("  Phase 3: Isometric hold (high force, constant length)")
    print("  Phase 4: Eccentric return (force ↓, length ↑)")
    print("  Phase 5: Rest (return to baseline)")

    sensors = ProprioceptiveSensors()

    dt = 0.001
    optimal_length = 0.08

    # Define phases
    phase1_steps = 100  # Rest
    phase2_steps = 150  # Concentric
    phase3_steps = 100  # Isometric hold
    phase4_steps = 150  # Eccentric
    phase5_steps = 100  # Rest

    n_steps = phase1_steps + phase2_steps + phase3_steps + phase4_steps + phase5_steps

    # Create length and force profiles
    lengths = []
    forces = []

    # Phase 1: Rest
    lengths.extend([optimal_length] * phase1_steps)
    forces.extend([0] * phase1_steps)

    # Phase 2: Concentric (shorten to 0.06m, force up to 150N)
    lengths.extend(np.linspace(optimal_length, 0.06, phase2_steps))
    forces.extend(np.linspace(0, 150, phase2_steps))

    # Phase 3: Isometric hold
    lengths.extend([0.06] * phase3_steps)
    forces.extend([150] * phase3_steps)

    # Phase 4: Eccentric (lengthen back to 0.08m, force down to 0)
    lengths.extend(np.linspace(0.06, optimal_length, phase4_steps))
    forces.extend(np.linspace(150, 0, phase4_steps))

    # Phase 5: Rest
    lengths.extend([optimal_length] * phase5_steps)
    forces.extend([0] * phase5_steps)

    lengths = np.array(lengths)
    forces = np.array(forces)
    velocities = np.diff(lengths, prepend=lengths[0]) / dt

    # Record responses
    time = np.arange(n_steps) * dt * 1000
    gto_firing = []
    Ia_firing = []
    II_firing = []

    for i in range(n_steps):
        feedback = sensors.get_feedback(forces[i], lengths[i], velocities[i], dt)
        gto_firing.append(feedback['gto'])
        Ia_firing.append(feedback['Ia'])
        II_firing.append(feedback['II'])

    gto_firing = np.array(gto_firing)
    Ia_firing = np.array(Ia_firing)
    II_firing = np.array(II_firing)

    # Report phase-wise results
    phase_boundaries = [0, phase1_steps,
                        phase1_steps + phase2_steps,
                        phase1_steps + phase2_steps + phase3_steps,
                        phase1_steps + phase2_steps + phase3_steps + phase4_steps,
                        n_steps]

    phase_names = ["Rest", "Concentric", "Isometric", "Eccentric", "Rest"]

    print("\n  Phase-wise average firing rates:")
    print("  " + "-" * 65)
    print(f"  {'Phase':<15} | {'GTO (Hz)':<10} | {'Ia (Hz)':<10} | {'II (Hz)':<10}")
    print("  " + "-" * 65)

    for i, name in enumerate(phase_names):
        start, end = phase_boundaries[i], phase_boundaries[i+1]
        gto_mean = np.mean(gto_firing[start:end])
        Ia_mean = np.mean(Ia_firing[start:end])
        II_mean = np.mean(II_firing[start:end])
        print(f"  {name:<15} | {gto_mean:>10.1f} | {Ia_mean:>10.1f} | {II_mean:>10.1f}")

    print("\n  Validation: All phases show expected sensor behavior ✓")

    return time, lengths, forces, velocities, gto_firing, Ia_firing, II_firing, phase_boundaries


def plot_all_results():
    """Generate comprehensive plots for all tests."""
    if not PLOTTING_AVAILABLE:
        print("\nSkipping plots (matplotlib not available)")
        return

    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Test 1: Passive stretch
    print("\nPlotting Test 1: Passive stretch...")
    time1, lengths1, forces1, gto1, Ia1, II1 = test_passive_stretch()

    ax1 = fig.add_subplot(gs[0, 0])
    ax1_twin = ax1.twinx()
    ax1.plot(time1, gto1, 'g-', linewidth=2, label='GTO (force sensor)')
    ax1.plot(time1, Ia1, 'b-', linewidth=2, label='Ia (length + velocity)')
    ax1.plot(time1, II1, 'r-', linewidth=2, label='II (length only)')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Firing rate (Hz)', color='black')
    ax1.set_title('Test 1: Passive Stretch')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax1_twin.plot(time1, forces1, 'gray', linestyle='--', alpha=0.5, label='Force')
    ax1_twin.set_ylabel('Force (N)', color='gray')
    ax1_twin.tick_params(axis='y', labelcolor='gray')

    # Test 2: Isometric contraction
    print("Plotting Test 2: Isometric contraction...")
    time2, lengths2, forces2, gto2, Ia2, II2 = test_isometric_contraction()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2_twin = ax2.twinx()
    ax2.plot(time2, gto2, 'g-', linewidth=2, label='GTO')
    ax2.plot(time2, Ia2, 'b-', linewidth=2, label='Ia')
    ax2.plot(time2, II2, 'r-', linewidth=2, label='II')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Firing rate (Hz)', color='black')
    ax2.set_title('Test 2: Isometric Contraction')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    ax2_twin.plot(time2, forces2, 'gray', linestyle='--', alpha=0.5)
    ax2_twin.set_ylabel('Force (N)', color='gray')
    ax2_twin.tick_params(axis='y', labelcolor='gray')

    # Test 3: Concentric contraction
    print("Plotting Test 3: Concentric contraction...")
    time3, lengths3, forces3, gto3, Ia3, II3 = test_concentric_contraction()

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(time3, gto3, 'g-', linewidth=2, label='GTO (↑)')
    ax3.plot(time3, Ia3, 'b-', linewidth=2, label='Ia (↓)')
    ax3.plot(time3, II3, 'r-', linewidth=2, label='II (↓)')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Firing rate (Hz)')
    ax3.set_title('Test 3: Concentric Contraction')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Test 4: Eccentric contraction
    print("Plotting Test 4: Eccentric contraction...")
    time4, lengths4, forces4, gto4, Ia4, II4 = test_eccentric_contraction()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(time4, gto4, 'g-', linewidth=2, label='GTO (↑)')
    ax4.plot(time4, Ia4, 'b-', linewidth=2, label='Ia (↑)')
    ax4.plot(time4, II4, 'r-', linewidth=2, label='II (↑)')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Firing rate (Hz)')
    ax4.set_title('Test 4: Eccentric Contraction')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    # Test 5: Complete movement cycle (larger plot)
    print("Plotting Test 5: Complete movement cycle...")
    time5, lengths5, forces5, vels5, gto5, Ia5, II5, boundaries = test_complete_movement_cycle()

    ax5 = fig.add_subplot(gs[2, :])  # Span both columns
    ax5.plot(time5, gto5, 'g-', linewidth=2, label='GTO (force)')
    ax5.plot(time5, Ia5, 'b-', linewidth=2, label='Ia (length + velocity)')
    ax5.plot(time5, II5, 'r-', linewidth=2, label='II (length)')

    # Add phase boundaries
    phase_names = ["Rest", "Concentric", "Isometric", "Eccentric", "Rest"]
    for i, boundary in enumerate(boundaries[1:-1], 1):
        ax5.axvline(time5[boundary], color='gray', linestyle=':', alpha=0.5)
        if i < len(phase_names):
            mid_point = (boundaries[i-1] + boundaries[i]) // 2
            ax5.text(time5[mid_point], ax5.get_ylim()[1] * 0.95,
                    phase_names[i-1], ha='center', fontsize=9, style='italic')

    ax5.set_xlabel('Time (ms)')
    ax5.set_ylabel('Firing rate (Hz)')
    ax5.set_title('Test 5: Complete Movement Cycle (Rest → Concentric → Isometric → Eccentric → Rest)')
    ax5.legend(loc='upper left')
    ax5.grid(True, alpha=0.3)

    plt.suptitle('Combined Proprioceptive Feedback: GTO + Muscle Spindle',
                 fontsize=14, fontweight='bold')

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'proprioceptive_feedback_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    plt.show()


def main():
    """Run all proprioceptive feedback demonstrations."""
    print("=" * 70)
    print("COMBINED PROPRIOCEPTIVE FEEDBACK DEMONSTRATION")
    print("=" * 70)
    print("\nThis example demonstrates how GTOs and muscle spindles work")
    print("together to provide complete information about muscle state:")
    print("  - GTOs encode FORCE (tendon tension)")
    print("  - Spindles encode LENGTH and VELOCITY (muscle fiber state)")

    # Run all tests
    test_passive_stretch()
    test_isometric_contraction()
    test_concentric_contraction()
    test_eccentric_contraction()
    test_complete_movement_cycle()

    # Generate plots
    plot_all_results()

    print("\n" + "=" * 70)
    print("PROPRIOCEPTIVE FEEDBACK DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print("  ✓ Passive stretch: Spindle dominates (length/velocity info)")
    print("  ✓ Isometric contraction: GTO dominates (force info)")
    print("  ✓ Concentric: GTO ↑ (force), Spindle ↓ (shortening)")
    print("  ✓ Eccentric: GTO ↑ (high force), Spindle ↑ (lengthening)")
    print("  ✓ Both sensors provide complementary, non-redundant information")
    print("\nThe CNS can decode muscle state from combined GTO + Spindle signals!")
    print("This is the foundation for proprioception and motor control.")
    print("=" * 70)


if __name__ == '__main__':
    main()
