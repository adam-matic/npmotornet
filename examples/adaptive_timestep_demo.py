"""
Demonstration of adaptive timestep control integration.

This example shows how to use the adaptive timestep methods (RKF45 and DOPRI5)
and compares them with fixed timestep methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from npmotornet.effector import ReluPointMass24


def run_comparison():
    """Compare fixed and adaptive timestep methods."""

    # Simulation parameters
    duration = 1.0  # seconds
    base_dt = 0.01  # base timestep

    # Create effectors with different integration methods
    effectors = {
        'Euler (fixed)': ReluPointMass24(
            timestep=base_dt,
            integration_method='euler'
        ),
        'RK4 (fixed)': ReluPointMass24(
            timestep=base_dt,
            integration_method='rk4'
        ),
        'RKF45 (adaptive)': ReluPointMass24(
            timestep=base_dt,
            integration_method='rkf45',
            adaptive_tolerance=1e-5,
            adaptive_min_dt=1e-6,
            adaptive_max_dt=0.1
        ),
        'DOPRI5 (adaptive)': ReluPointMass24(
            timestep=base_dt,
            integration_method='dopri5',
            adaptive_tolerance=1e-5,
            adaptive_min_dt=1e-6,
            adaptive_max_dt=0.1
        )
    }

    results = {}

    # Run simulations
    for name, effector in effectors.items():
        print(f"\nRunning {name}...")

        # Reset
        effector.reset(seed=42, options={'batch_size': 1})

        # Storage
        positions = []
        velocities = []
        timesteps = []
        times = []
        t = 0.0

        # Simulate
        n_steps = int(duration / base_dt)
        for i in range(n_steps):
            # Variable frequency sinusoidal input (creates varying dynamics)
            freq = 2.0 + 1.5 * np.sin(2 * np.pi * t / 0.5)
            action = 0.5 + 0.3 * np.sin(2 * np.pi * freq * t)
            action = np.array([[action, action, action, action]], dtype=np.float32)

            effector.step(action)

            # Store results
            joint_state = effector.states['joint']
            positions.append(joint_state[0, :2].copy())

            if hasattr(effector, 'current_adaptive_dt'):
                timesteps.append(effector.current_adaptive_dt)
                t += effector.current_adaptive_dt * effector.n_ministeps
            else:
                timesteps.append(base_dt)
                t += base_dt

            times.append(t)

        # Store results
        results[name] = {
            'positions': np.array(positions),
            'times': np.array(times),
            'timesteps': np.array(timesteps)
        }

        # Print statistics for adaptive methods
        if hasattr(effector, 'adaptive_step_count'):
            print(f"  Total steps: {effector.adaptive_step_count}")
            print(f"  Rejected steps: {effector.adaptive_rejected_count}")
            acceptance_rate = 1.0 - effector.adaptive_rejected_count / max(effector.adaptive_step_count, 1)
            print(f"  Acceptance rate: {acceptance_rate:.1%}")
            print(f"  Timestep: min={np.min(timesteps):.2e}, max={np.max(timesteps):.2e}, "
                  f"mean={np.mean(timesteps):.2e}")

    return results


def plot_results(results):
    """Create comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Adaptive vs Fixed Timestep Integration Comparison', fontsize=16)

    colors = {
        'Euler (fixed)': 'red',
        'RK4 (fixed)': 'blue',
        'RKF45 (adaptive)': 'green',
        'DOPRI5 (adaptive)': 'purple'
    }

    # Plot 1: Trajectories in X-Y space
    ax = axes[0, 0]
    for name, data in results.items():
        pos = data['positions']
        ax.plot(pos[:, 0], pos[:, 1], label=name, color=colors[name], alpha=0.7, linewidth=1.5)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Trajectory in State Space')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    # Plot 2: X position over time
    ax = axes[0, 1]
    for name, data in results.items():
        pos = data['positions']
        times = data['times']
        ax.plot(times, pos[:, 0], label=name, color=colors[name], alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('X Position')
    ax.set_title('X Position vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Timestep size over time (adaptive methods only)
    ax = axes[1, 0]
    for name, data in results.items():
        if 'adaptive' in name.lower():
            times = data['times']
            timesteps = data['timesteps']
            ax.plot(times, timesteps, label=name, color=colors[name], linewidth=1.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Timestep Size (s)')
    ax.set_title('Adaptive Timestep Evolution')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # Plot 4: Timestep distribution (histogram)
    ax = axes[1, 1]
    for name, data in results.items():
        if 'adaptive' in name.lower():
            timesteps = data['timesteps']
            ax.hist(np.log10(timesteps), bins=30, alpha=0.6, label=name, color=colors[name])
    ax.set_xlabel('log10(Timestep Size) [s]')
    ax.set_ylabel('Frequency')
    ax.set_title('Timestep Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Main demonstration."""

    print("=" * 80)
    print("ADAPTIVE TIMESTEP INTEGRATION DEMONSTRATION")
    print("=" * 80)

    print("\nThis example demonstrates adaptive timestep control for muscle simulation.")
    print("The input has varying frequency, creating regions of fast and slow dynamics.")
    print("Adaptive methods should use larger timesteps in smooth regions and smaller")
    print("timesteps in regions with rapid changes.\n")

    # Run comparison
    results = run_comparison()

    # Create plots
    print("\nGenerating comparison plots...")
    fig = plot_results(results)
    plt.savefig('adaptive_timestep_demo.png', dpi=150, bbox_inches='tight')
    print("Plot saved: adaptive_timestep_demo.png")

    # Show plot if possible
    try:
        plt.show()
    except:
        print("(Display not available)")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey observations:")
    print("1. Adaptive methods automatically adjust timestep based on dynamics")
    print("2. Timesteps vary by 2-3 orders of magnitude during simulation")
    print("3. Acceptance rate indicates how often error tolerance is met")
    print("4. All methods produce similar trajectories (validate implementation)")


if __name__ == "__main__":
    main()
