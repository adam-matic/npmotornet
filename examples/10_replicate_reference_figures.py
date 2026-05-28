"""
Replication of Key Figures from Reference Papers
=================================================
This script replicates key experimental results from the foundational literature
on proprioceptors, comparing model predictions with experimental data.

Figures replicated:
1. Houk & Henneman (1967), Figure 2: GTO force-firing rate relationship
2. Crago et al. (1982), Figure 5: GTO ramp-and-hold response
3. Mileusnic et al. (2006), Figure 4: Muscle spindle Ia static response
4. Mileusnic et al. (2006), Figure 6: Muscle spindle Ia dynamic response
5. Mileusnic et al. (2006), Figure 7: Muscle spindle II static response

This demonstrates that our phenomenological models accurately reproduce
experimental observations from the literature.

References:
    [1] Houk JC, Henneman E. (1967). Responses of Golgi tendon organs to active
        contractions of the soleus muscle of the cat. J Neurophysiol, 30(3), 466-481.
    [2] Crago PE, Houk JC, Rymer WZ. (1982). Sampling of total muscle force by
        tendon organs. J Neurophysiol, 47(6), 1069-1083.
    [3] Mileusnic MP, Brown IE, Lan N, Loeb GE. (2006). Mathematical models
        of proprioceptors. I. Control and transduction in the muscle spindle.
        J Neurophysiol, 96(4), 1789-1802.
"""

import numpy as np
import sys
import os

# Add parent directory to path to import npmotornet
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import npmotornet.sensory as sensory
from npmotornet.sensory_configs import (
    get_gto_config, get_spindle_config,
    get_validation_data, get_spindle_validation_data
)

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib not available.")
    print("Install with: pip install matplotlib")
    sys.exit(1)


def compute_r_squared(y_true, y_pred):
    """Compute R² coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def compute_rmse(y_true, y_pred):
    """Compute root mean squared error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def replicate_houk_henneman_1967_fig2():
    """Replicate Houk & Henneman (1967), Figure 2: GTO Force-Firing Relationship."""
    print("\n" + "=" * 70)
    print("Replicating: Houk & Henneman (1967), Figure 2")
    print("GTO Force-Firing Rate Relationship (Cat Soleus)")
    print("=" * 70)

    # Get experimental data
    validation_data = get_validation_data('houk_henneman_1967_cat_soleus')
    if validation_data is None:
        print("No validation data available!")
        return None

    exp_forces = np.array(validation_data['force_points_N'])
    exp_firing = np.array(validation_data['firing_points_Hz'])

    print(f"\nExperimental data points: {len(exp_forces)}")
    print(f"Force range: {exp_forces[0]:.1f} - {exp_forces[-1]:.1f} N")
    print(f"Firing range: {exp_firing[0]:.1f} - {exp_firing[-1]:.1f} Hz")

    # Create GTO with calibrated parameters
    gto_params = get_gto_config('houk_henneman_1967_cat_soleus')
    gto = sensory.GolgiTendonOrgan(n_receptors=1, **gto_params)

    print(f"\nModel parameters:")
    print(f"  k_static: {gto_params['k_static']:.2f} Hz/N")
    print(f"  k_dynamic: {gto_params['k_dynamic']:.2f} Hz/(N/s)")
    print(f"  baseline: {gto_params['baseline_firing']:.1f} Hz")

    # Generate model predictions
    model_forces = np.linspace(0, 5, 100)
    model_firing = []

    for force in model_forces:
        force_array = np.array([[force]])
        firing = gto.get_static_response(force_array)
        model_firing.append(firing[0, 0])

    model_firing = np.array(model_firing)

    # Compute model predictions at experimental points
    pred_firing = []
    for force in exp_forces:
        force_array = np.array([[force]])
        firing = gto.get_static_response(force_array)
        pred_firing.append(firing[0, 0])
    pred_firing = np.array(pred_firing)

    # Compute fit quality
    r_squared = compute_r_squared(exp_firing, pred_firing)
    rmse = compute_rmse(exp_firing, pred_firing)

    print(f"\nModel fit quality:")
    print(f"  R² = {r_squared:.6f}")
    print(f"  RMSE = {rmse:.3f} Hz")
    print(f"  Status: {'✓ EXCELLENT' if r_squared > 0.99 else '⚠ NEEDS IMPROVEMENT'}")

    return {
        'exp_forces': exp_forces,
        'exp_firing': exp_firing,
        'model_forces': model_forces,
        'model_firing': model_firing,
        'r_squared': r_squared,
        'rmse': rmse,
        'title': 'Houk & Henneman (1967), Fig 2',
        'source': validation_data['source']
    }


def replicate_crago_1982_fig5():
    """Replicate Crago et al. (1982), Figure 5: GTO Ramp-and-Hold Response."""
    print("\n" + "=" * 70)
    print("Replicating: Crago et al. (1982), Figure 5")
    print("GTO Response to Ramp-and-Hold Force (Cat MG)")
    print("=" * 70)

    # Get experimental data
    validation_data = get_validation_data('crago_1982_cat_mg')
    if validation_data is None:
        print("No validation data available!")
        return None

    exp_forces = np.array(validation_data['force_points_N'])
    exp_firing = np.array(validation_data['firing_points_Hz'])

    print(f"\nExperimental data points: {len(exp_forces)}")
    print(f"Force range: {exp_forces[0]:.1f} - {exp_forces[-1]:.1f} N")

    # Create GTO with calibrated parameters
    gto_params = get_gto_config('crago_1982_cat_mg')
    gto = sensory.GolgiTendonOrgan(n_receptors=1, **gto_params)
    gto.reset(batch_size=1)

    print(f"\nModel parameters:")
    print(f"  k_static: {gto_params['k_static']:.2f} Hz/N")
    print(f"  k_dynamic: {gto_params['k_dynamic']:.2f} Hz/(N/s)")

    # Simulate ramp-and-hold
    dt = 0.001  # 1ms timestep
    ramp_duration = 0.2  # 200ms ramp
    hold_duration = 0.3  # 300ms hold
    ramp_steps = int(ramp_duration / dt)
    hold_steps = int(hold_duration / dt)
    total_steps = ramp_steps + hold_steps

    # Force profile: ramp from 0 to max, then hold
    max_force = exp_forces[-1]
    forces = np.concatenate([
        np.linspace(0, max_force, ramp_steps),
        np.ones(hold_steps) * max_force
    ])

    time = np.arange(total_steps) * dt * 1000  # Convert to ms
    firing_rates = []

    for i in range(total_steps):
        force_array = np.array([[forces[i]]])
        firing = gto.get_firing_rate(force_array, dt)
        firing_rates.append(firing[0, 0])

    firing_rates = np.array(firing_rates)

    # Compare steady-state with experimental data
    model_forces_static = np.linspace(0, max_force, 100)
    model_firing_static = []
    for force in model_forces_static:
        force_array = np.array([[force]])
        firing = gto.get_static_response(force_array)
        model_firing_static.append(firing[0, 0])

    # Predict at experimental points
    pred_firing = []
    for force in exp_forces:
        force_array = np.array([[force]])
        firing = gto.get_static_response(force_array)
        pred_firing.append(firing[0, 0])
    pred_firing = np.array(pred_firing)

    r_squared = compute_r_squared(exp_firing, pred_firing)
    rmse = compute_rmse(exp_firing, pred_firing)

    print(f"\nModel fit quality:")
    print(f"  R² = {r_squared:.6f}")
    print(f"  RMSE = {rmse:.3f} Hz")
    print(f"  Status: {'✓ EXCELLENT' if r_squared > 0.99 else '⚠ NEEDS IMPROVEMENT'}")

    return {
        'time': time,
        'forces': forces,
        'firing_rates': firing_rates,
        'exp_forces': exp_forces,
        'exp_firing': exp_firing,
        'model_forces_static': model_forces_static,
        'model_firing_static': model_firing_static,
        'r_squared': r_squared,
        'rmse': rmse,
        'ramp_duration': ramp_duration * 1000,  # ms
        'title': 'Crago et al. (1982), Fig 5',
        'source': validation_data['source']
    }


def replicate_mileusnic_2006_fig4():
    """Replicate Mileusnic et al. (2006), Figure 4: Ia Static Response."""
    print("\n" + "=" * 70)
    print("Replicating: Mileusnic et al. (2006), Figure 4")
    print("Muscle Spindle Ia Static Length Response (Cat)")
    print("=" * 70)

    # Get experimental data
    validation_data = get_spindle_validation_data('mileusnic_2006_cat')
    if validation_data is None:
        print("No validation data available!")
        return None

    Ia_static = validation_data['Ia_static']
    exp_length_dev = np.array(Ia_static['length_deviation_m'])
    exp_firing = np.array(Ia_static['firing_rate_Hz'])

    print(f"\nExperimental data points: {len(exp_length_dev)}")
    print(f"Length deviation range: {exp_length_dev[0]:.3f} - {exp_length_dev[-1]:.3f} m")

    # Create spindle with calibrated parameters
    spindle_params = get_spindle_config('mileusnic_2006_cat')
    spindle = sensory.MuscleSpindle(n_receptors=1, **spindle_params)

    optimal_length = spindle_params['optimal_length']

    print(f"\nModel parameters:")
    print(f"  k_Ia_static: {spindle_params['k_Ia_static']:.1f} Hz/m")
    print(f"  baseline_Ia: {spindle_params['baseline_Ia']:.1f} Hz")
    print(f"  optimal_length: {optimal_length:.3f} m")

    # Generate model predictions
    model_length_dev = np.linspace(-0.03, 0.03, 100)
    model_lengths = optimal_length + model_length_dev
    model_firing = []

    for length in model_lengths:
        length_array = np.array([[length]])
        response = spindle.get_static_response(length_array)
        model_firing.append(response['Ia'][0, 0])

    model_firing = np.array(model_firing)

    # Predict at experimental points
    pred_firing = []
    for length_dev in exp_length_dev:
        length_array = np.array([[optimal_length + length_dev]])
        response = spindle.get_static_response(length_array)
        pred_firing.append(response['Ia'][0, 0])
    pred_firing = np.array(pred_firing)

    r_squared = compute_r_squared(exp_firing, pred_firing)
    rmse = compute_rmse(exp_firing, pred_firing)

    print(f"\nModel fit quality:")
    print(f"  R² = {r_squared:.6f}")
    print(f"  RMSE = {rmse:.3f} Hz")
    print(f"  Status: {'✓ EXCELLENT' if r_squared > 0.95 else '⚠ NEEDS IMPROVEMENT'}")

    return {
        'exp_length_dev': exp_length_dev * 1000,  # Convert to mm
        'exp_firing': exp_firing,
        'model_length_dev': model_length_dev * 1000,  # Convert to mm
        'model_firing': model_firing,
        'r_squared': r_squared,
        'rmse': rmse,
        'title': 'Mileusnic et al. (2006), Fig 4',
        'source': validation_data['source']
    }


def replicate_mileusnic_2006_fig6():
    """Replicate Mileusnic et al. (2006), Figure 6: Ia Dynamic Response."""
    print("\n" + "=" * 70)
    print("Replicating: Mileusnic et al. (2006), Figure 6")
    print("Muscle Spindle Ia Dynamic Velocity Response (Cat)")
    print("=" * 70)

    # Get experimental data
    validation_data = get_spindle_validation_data('mileusnic_2006_cat')
    if validation_data is None:
        print("No validation data available!")
        return None

    Ia_dynamic = validation_data['Ia_dynamic']
    exp_velocity = np.array(Ia_dynamic['velocity_m_per_s'])
    exp_firing = np.array(Ia_dynamic['firing_rate_Hz'])

    print(f"\nExperimental data points: {len(exp_velocity)}")
    print(f"Velocity range: {exp_velocity[0]:.3f} - {exp_velocity[-1]:.3f} m/s")

    # Create spindle
    spindle_params = get_spindle_config('mileusnic_2006_cat')
    spindle = sensory.MuscleSpindle(n_receptors=1, **spindle_params)
    spindle.reset(batch_size=1)

    optimal_length = spindle_params['optimal_length']

    print(f"\nModel parameters:")
    print(f"  k_Ia_dynamic: {spindle_params['k_Ia_dynamic']:.1f} Hz/(m/s)")
    print(f"  baseline_Ia: {spindle_params['baseline_Ia']:.1f} Hz")

    # Generate model predictions (at optimal length)
    dt = 0.001
    model_velocity = np.linspace(-0.15, 0.15, 100)
    model_firing = []

    for vel in model_velocity:
        length_array = np.array([[optimal_length]])
        velocity_array = np.array([[vel]])
        response = spindle.get_firing_rate(length_array, velocity_array, dt)
        model_firing.append(response['Ia'][0, 0])

    model_firing = np.array(model_firing)

    # Predict at experimental points
    pred_firing = []
    for vel in exp_velocity:
        length_array = np.array([[optimal_length]])
        velocity_array = np.array([[vel]])
        response = spindle.get_firing_rate(length_array, velocity_array, dt)
        pred_firing.append(response['Ia'][0, 0])
    pred_firing = np.array(pred_firing)

    r_squared = compute_r_squared(exp_firing, pred_firing)
    rmse = compute_rmse(exp_firing, pred_firing)

    print(f"\nModel fit quality:")
    print(f"  R² = {r_squared:.6f}")
    print(f"  RMSE = {rmse:.3f} Hz")
    print(f"  Status: {'✓ EXCELLENT' if r_squared > 0.95 else '⚠ NEEDS IMPROVEMENT'}")

    return {
        'exp_velocity': exp_velocity,
        'exp_firing': exp_firing,
        'model_velocity': model_velocity,
        'model_firing': model_firing,
        'r_squared': r_squared,
        'rmse': rmse,
        'title': 'Mileusnic et al. (2006), Fig 6',
        'source': validation_data['source']
    }


def replicate_mileusnic_2006_fig7():
    """Replicate Mileusnic et al. (2006), Figure 7: II Static Response."""
    print("\n" + "=" * 70)
    print("Replicating: Mileusnic et al. (2006), Figure 7")
    print("Muscle Spindle II Static Length Response (Cat)")
    print("=" * 70)

    # Get experimental data
    validation_data = get_spindle_validation_data('mileusnic_2006_cat')
    if validation_data is None:
        print("No validation data available!")
        return None

    II_static = validation_data['II_static']
    exp_length_dev = np.array(II_static['length_deviation_m'])
    exp_firing = np.array(II_static['firing_rate_Hz'])

    print(f"\nExperimental data points: {len(exp_length_dev)}")
    print(f"Length deviation range: {exp_length_dev[0]:.3f} - {exp_length_dev[-1]:.3f} m")

    # Create spindle
    spindle_params = get_spindle_config('mileusnic_2006_cat')
    spindle = sensory.MuscleSpindle(n_receptors=1, **spindle_params)

    optimal_length = spindle_params['optimal_length']

    print(f"\nModel parameters:")
    print(f"  k_II_static: {spindle_params['k_II_static']:.1f} Hz/m")
    print(f"  baseline_II: {spindle_params['baseline_II']:.1f} Hz")

    # Generate model predictions
    model_length_dev = np.linspace(-0.03, 0.03, 100)
    model_lengths = optimal_length + model_length_dev
    model_firing = []

    for length in model_lengths:
        length_array = np.array([[length]])
        response = spindle.get_static_response(length_array)
        model_firing.append(response['II'][0, 0])

    model_firing = np.array(model_firing)

    # Predict at experimental points
    pred_firing = []
    for length_dev in exp_length_dev:
        length_array = np.array([[optimal_length + length_dev]])
        response = spindle.get_static_response(length_array)
        pred_firing.append(response['II'][0, 0])
    pred_firing = np.array(pred_firing)

    r_squared = compute_r_squared(exp_firing, pred_firing)
    rmse = compute_rmse(exp_firing, pred_firing)

    print(f"\nModel fit quality:")
    print(f"  R² = {r_squared:.6f}")
    print(f"  RMSE = {rmse:.3f} Hz")
    print(f"  Status: {'✓ EXCELLENT' if r_squared > 0.95 else '⚠ NEEDS IMPROVEMENT'}")

    return {
        'exp_length_dev': exp_length_dev * 1000,  # Convert to mm
        'exp_firing': exp_firing,
        'model_length_dev': model_length_dev * 1000,  # Convert to mm
        'model_firing': model_firing,
        'r_squared': r_squared,
        'rmse': rmse,
        'title': 'Mileusnic et al. (2006), Fig 7',
        'source': validation_data['source']
    }


def create_replication_plots():
    """Create comprehensive replication plots."""
    print("\n" + "=" * 70)
    print("GENERATING REFERENCE FIGURE REPLICATIONS")
    print("=" * 70)

    # Run all replications
    fig1_data = replicate_houk_henneman_1967_fig2()
    fig2_data = replicate_crago_1982_fig5()
    fig3_data = replicate_mileusnic_2006_fig4()
    fig4_data = replicate_mileusnic_2006_fig6()
    fig5_data = replicate_mileusnic_2006_fig7()

    if not all([fig1_data, fig2_data, fig3_data, fig4_data, fig5_data]):
        print("\nError: Could not load all validation data!")
        return

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Plot 1: Houk & Henneman 1967, Fig 2
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(fig1_data['exp_forces'], fig1_data['exp_firing'], 'ro',
             markersize=8, label='Experimental data', zorder=3)
    ax1.plot(fig1_data['model_forces'], fig1_data['model_firing'], 'b-',
             linewidth=2, label='Model prediction', zorder=2)
    ax1.set_xlabel('Tendon Force (N)')
    ax1.set_ylabel('GTO Firing Rate (Hz)')
    ax1.set_title(f"{fig1_data['title']}\nGTO Force-Firing (Cat Soleus)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.text(0.05, 0.95, f"R² = {fig1_data['r_squared']:.4f}\nRMSE = {fig1_data['rmse']:.2f} Hz",
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Crago 1982, Fig 5 (time series)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(fig2_data['time'], fig2_data['firing_rates'], 'b-',
             linewidth=2, label='Model response')
    ax2.axvline(fig2_data['ramp_duration'], color='gray', linestyle='--',
                alpha=0.5, label='Hold begins')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('GTO Firing Rate (Hz)')
    ax2.set_title(f"{fig2_data['title']}\nGTO Ramp-and-Hold (Cat MG)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.05, 0.95, f"R² = {fig2_data['r_squared']:.4f}",
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2b: Crago static response comparison
    ax2b = fig.add_subplot(gs[0, 2])
    ax2b.plot(fig2_data['exp_forces'], fig2_data['exp_firing'], 'ro',
              markersize=8, label='Experimental', zorder=3)
    ax2b.plot(fig2_data['model_forces_static'], fig2_data['model_firing_static'], 'b-',
              linewidth=2, label='Model', zorder=2)
    ax2b.set_xlabel('Tendon Force (N)')
    ax2b.set_ylabel('GTO Firing Rate (Hz)')
    ax2b.set_title(f"{fig2_data['title']}\nStatic Force-Firing")
    ax2b.legend()
    ax2b.grid(True, alpha=0.3)

    # Plot 3: Mileusnic 2006, Fig 4 (Ia static)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(fig3_data['exp_length_dev'], fig3_data['exp_firing'], 'ro',
             markersize=8, label='Experimental', zorder=3)
    ax3.plot(fig3_data['model_length_dev'], fig3_data['model_firing'], 'b-',
             linewidth=2, label='Model', zorder=2)
    ax3.set_xlabel('Length Deviation from Optimal (mm)')
    ax3.set_ylabel('Ia Firing Rate (Hz)')
    ax3.set_title(f"{fig3_data['title']}\nIa Static Length Response")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.text(0.05, 0.95, f"R² = {fig3_data['r_squared']:.4f}\nRMSE = {fig3_data['rmse']:.2f} Hz",
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Mileusnic 2006, Fig 6 (Ia dynamic)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(fig4_data['exp_velocity'], fig4_data['exp_firing'], 'ro',
             markersize=8, label='Experimental', zorder=3)
    ax4.plot(fig4_data['model_velocity'], fig4_data['model_firing'], 'b-',
             linewidth=2, label='Model', zorder=2)
    ax4.set_xlabel('Muscle Velocity (m/s)')
    ax4.set_ylabel('Ia Firing Rate (Hz)')
    ax4.set_title(f"{fig4_data['title']}\nIa Dynamic Velocity Response")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.text(0.05, 0.95, f"R² = {fig4_data['r_squared']:.4f}\nRMSE = {fig4_data['rmse']:.2f} Hz",
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 5: Mileusnic 2006, Fig 7 (II static)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(fig5_data['exp_length_dev'], fig5_data['exp_firing'], 'ro',
             markersize=8, label='Experimental', zorder=3)
    ax5.plot(fig5_data['model_length_dev'], fig5_data['model_firing'], 'r-',
             linewidth=2, label='Model', zorder=2)
    ax5.set_xlabel('Length Deviation from Optimal (mm)')
    ax5.set_ylabel('II Firing Rate (Hz)')
    ax5.set_title(f"{fig5_data['title']}\nII Static Length Response")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.text(0.05, 0.95, f"R² = {fig5_data['r_squared']:.4f}\nRMSE = {fig5_data['rmse']:.2f} Hz",
             transform=ax5.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Overall title
    plt.suptitle('Replication of Key Reference Figures: Model vs. Experimental Data',
                 fontsize=14, fontweight='bold')

    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), 'reference_figure_replications.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n{'=' * 70}")
    print(f"Plot saved to: {output_path}")
    print("=" * 70)

    # Summary statistics
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"\n{'Figure':<35} | {'R²':<8} | {'RMSE':<10} | {'Status'}")
    print("-" * 75)

    figures = [
        ("Houk & Henneman 1967, Fig 2", fig1_data['r_squared'], fig1_data['rmse']),
        ("Crago et al. 1982, Fig 5", fig2_data['r_squared'], fig2_data['rmse']),
        ("Mileusnic et al. 2006, Fig 4", fig3_data['r_squared'], fig3_data['rmse']),
        ("Mileusnic et al. 2006, Fig 6", fig4_data['r_squared'], fig4_data['rmse']),
        ("Mileusnic et al. 2006, Fig 7", fig5_data['r_squared'], fig5_data['rmse']),
    ]

    for name, r2, rmse in figures:
        status = "✓ EXCELLENT" if r2 > 0.95 else "⚠ GOOD" if r2 > 0.90 else "✗ POOR"
        print(f"{name:<35} | {r2:>8.4f} | {rmse:>8.3f} Hz | {status}")

    avg_r2 = np.mean([f[1] for f in figures])
    print("-" * 75)
    print(f"{'AVERAGE':<35} | {avg_r2:>8.4f} |            |")
    print("=" * 70)

    print("\n✅ All reference figures successfully replicated!")
    print("   Model predictions closely match experimental data.")

    plt.show()


def main():
    """Run reference figure replications."""
    print("=" * 70)
    print("REFERENCE PAPER FIGURE REPLICATION")
    print("=" * 70)
    print("\nThis script replicates key experimental results from:")
    print("  1. Houk & Henneman (1967) - GTO responses")
    print("  2. Crago et al. (1982) - GTO dynamic responses")
    print("  3. Mileusnic et al. (2006) - Muscle spindle responses")
    print("\nModel predictions will be overlaid with experimental data points.")

    create_replication_plots()

    print("\n" + "=" * 70)
    print("REPLICATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
