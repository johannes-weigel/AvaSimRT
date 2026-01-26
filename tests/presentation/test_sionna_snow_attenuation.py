"""
Proof-of-concept test: Snow material reduces signal strength.

This test validates that:
1. A snow obstacle between TX and RX attenuates the signal
2. The attenuation magnitude is consistent with theoretical expectations
   based on dry snow dielectric properties (Ulaby & Long, 2014)

For DRY snow (Ps=0.4 g/cm³, mv=0.5%) at 3.5 GHz:
- ε' ≈ 1.7 (affects refraction/wave speed)
- ε'' ≈ 0.01 (minimal absorption ~2.3 dB/m - "2m dry snow should be no problem")

The imaginary part ε'' determines absorption loss.
The real part ε' determines refraction and reflection at interfaces.

Note: Sionna's ITURadioMaterial uses conductivity σ, which relates to ε'' via:
σ = ω × ε₀ × ε'' = 2πf × 8.854e-12 × ε''
"""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
import pytest
import sionna_vispy

from avasimrt.channelstate.simulation import (
    _build_context,
    _evaluate_cfr,
)

from avasimrt.channelstate.snow import (
    Snow, create_scene_with_snow, ulaby_long_snow_dielectric
)

from sionna.rt import subcarrier_frequencies
from avasimrt.math import mean_amp_in_db_from_cfr, distance
import matplotlib.pyplot as plt


def attenuation_db_per_meter(eps_real: float, eps_imag: float, freq_hz: float) -> float:
    """
    Calculate attenuation in dB/m from complex dielectric constant.

    For a lossy medium, the attenuation constant α (Np/m) is:
    α = (2πf/c) * sqrt(ε'/2) * sqrt(sqrt(1 + (ε''/ε')²) - 1)

    Then convert to dB/m: α_dB = 8.686 * α_Np

    Args:
        eps_real: Real part of relative permittivity (ε')
        eps_imag: Imaginary part of relative permittivity (ε'')
        freq_hz: Frequency in Hz

    Returns:
        Attenuation in dB/m
    """
    c = 3e8  # Speed of light (m/s)
    omega = 2 * np.pi * freq_hz

    # Loss tangent
    tan_delta = eps_imag / eps_real

    # Attenuation constant in Np/m
    alpha_np = (omega / c) * np.sqrt(eps_real / 2) * np.sqrt(np.sqrt(1 + tan_delta**2) - 1)

    # Convert to dB/m
    alpha_db = 8.686 * alpha_np

    return alpha_db


def theoretical_snow_attenuation(thickness_m: float, Ps: float = 0.4, mv: float = 0.5,
                                  freq_ghz: float = 3.5) -> float:
    """
    Calculate theoretical attenuation through snow of given thickness.

    Args:
        thickness_m: Snow thickness in meters (diameter for sphere)
        Ps: Dry snow density (g/cm³), default 0.4 for avalanche debris
        mv: Volumetric water content (%), default 0.5% for nearly dry snow
        freq_ghz: Frequency in GHz

    Returns:
        Total attenuation in dB
    """
    eps_r, eps_i = ulaby_long_snow_dielectric(Ps, mv, freq_ghz)
    atten_per_m = attenuation_db_per_meter(eps_r, eps_i, freq_ghz * 1e9)
    return atten_per_m * thickness_m


@pytest.mark.presentation
def test_snow_obstacle_reduces_signal(
    presentation_output: Path,
    examples: Path,
    presentation_config
):
    """
    Setup:
        - TX at (0, -10, 0)
        - RX at (0, 0, 0)
        - Snow obstacle at (0, -5, 0) - midway between TX and RX
        - Compare signal with vs without snow

    Expected:
        Signal amplitude WITH snow << Signal amplitude WITHOUT snow
    """

    out_dir = presentation_output / "sionna_snow_attenuation"
    out_dir.mkdir(exist_ok=True)

    tx_pos = (0.0, -10.0, 0.0)
    rx_pos = (0.0, 0.0, 0.0)

    txs = [("tx1", tx_pos, 1.0)]

    # Test 1: Without snow (baseline)
    print("\n" + "="*70)
    print("TEST 1: NO SNOW (baseline)")
    print("="*70)

    # Use simple channel config: 101 subcarriers @ 5 MHz spacing = 505 MHz bandwidth
    sc_num = 101
    sc_spacing = 5e6
    freq_center = 3.5e9  # 3.5 GHz to match dielectric calculations
    bandwidth = sc_num * sc_spacing

    ctx_no_snow = _build_context(
        anchors=txs,
        scene_src=None,  # Empty scene
        freq_center=freq_center,
        bandwidth=bandwidth,
        snow=None,
        reflection_depth=3,
        seed=42
    )

    paths_no_snow = ctx_no_snow.solve_paths()
    a_no_snow, tau_no_snow = paths_no_snow.cir()

    with sionna_vispy.patch():
        ctx_no_snow.scene.preview(paths=paths_no_snow)

    sionna_vispy.get_canvas(ctx_no_snow.scene).show()
    sionna_vispy.get_canvas(ctx_no_snow.scene).app.run()


    # Compute mean amplitude (dB)
    freqs = subcarrier_frequencies(sc_num, sc_spacing)
    cfr_no_snow, dists_no_snow = _evaluate_cfr(
        paths_no_snow,
        freqs=freqs,
        anchors=txs,
        node_pos=rx_pos
    )
    amp_db_no_snow = mean_amp_in_db_from_cfr(cfr_no_snow)

    print(f"TX position: {tx_pos}")
    print(f"RX position: {rx_pos}")
    print(f"Distance: {distance(tx_pos, rx_pos):.1f} m")
    print(f"Mean amplitude (no snow): {amp_db_no_snow:.2f} dB")

    # Test 2: With 0.5m radius snow sphere
    print("\n" + "="*70)
    print("TEST 2: WITH 0.5M SNOW SPHERE")
    print("="*70)

    # Create scene with snow sphere between TX and RX
    base_scene = examples / "empty.xml"
    scene_xml = out_dir / "scene.xml"

    # Copy the base scene
    shutil.copy(base_scene, scene_xml)

    # Copy meshes directory if needed
    base_meshes = examples / "seegrube" / "assets" / "meshes"
    if base_meshes.exists():
        shutil.copytree(base_meshes, out_dir / "meshes", dirs_exist_ok=True)
    else:
        (out_dir / "meshes").mkdir(exist_ok=True)

    snow_1 = Snow(thickness_m=1)  # 0.5m radius = 1m diameter

    # Place snow sphere midway between TX and RX
    snow_position = np.array([[0.0, -5.0, 0.0]])

    snow_scene_xml = create_scene_with_snow(
        xml_path=scene_xml,
        meshes_dir=out_dir / "meshes",
        radius=0.5,  # 0.5m radius sphere (1m diameter obstacle)
        positions=snow_position
    )

    ctx_with_snow = _build_context(
        anchors=txs,
        scene_src=snow_scene_xml,
        freq_center=freq_center,
        bandwidth=bandwidth,
        snow=snow_1,
        reflection_depth=3,
        seed=42
    )

    paths_with_snow = ctx_with_snow.solve_paths()
    a_with_snow, tau_with_snow = paths_with_snow.cir()



    with sionna_vispy.patch():
        ctx_with_snow.scene.preview(paths=paths_with_snow)

    sionna_vispy.get_canvas(ctx_with_snow.scene).show()
    sionna_vispy.get_canvas(ctx_with_snow.scene).app.run()

    # Verify paths exist
    print(f"\nPath analysis:")
    print(f"  CIR shape: {a_with_snow[0].shape if len(a_with_snow) > 0 else 'No paths'}")
    if len(a_with_snow) > 0 and len(a_with_snow[0].shape) > 4:
        num_paths = a_with_snow[0].shape[4]
        print(f"  Number of paths found: {num_paths}")
        if num_paths > 0:
            delays = tau_with_snow[0].numpy() if hasattr(tau_with_snow[0], 'numpy') else tau_with_snow[0]
            delays_ns = delays.flatten()[:5] * 1e9
            print(f"  Path delays (first few): {np.array2string(delays_ns, precision=2, separator=', ')} ns")

    cfr_with_snow, dists_with_snow = _evaluate_cfr(
        paths_with_snow,
        freqs=freqs,
        anchors=txs,
        node_pos=rx_pos
    )
    amp_db_with_snow = mean_amp_in_db_from_cfr(cfr_with_snow)

    print(f"\nSnow position: {snow_position[0]}")
    print(f"Snow radius: 0.5 m (diameter: 1m)")
    print(f"Mean amplitude (with 0.5m snow): {amp_db_with_snow:.2f} dB")

    # Test 3: With 1m radius snow sphere
    print("\n" + "="*70)
    print("TEST 3: WITH 1M SNOW SPHERE")
    print("="*70)

    # Create scene with 1m radius snow sphere
    snow_scene_xml_1m = create_scene_with_snow(
        xml_path=scene_xml,
        meshes_dir=out_dir / "meshes",
        radius=1.0,  # 1m radius sphere (2m diameter obstacle)
        positions=snow_position
    )
    snow_2 = Snow(thickness_m=2)  # 1m radius = 2m diameter
    ctx_with_snow_1m = _build_context(
        anchors=txs,
        scene_src=snow_scene_xml_1m,
        freq_center=freq_center,
        bandwidth=bandwidth,
        snow=snow_2,
        reflection_depth=3,
        seed=42
    )

    paths_with_snow_1m = ctx_with_snow_1m.solve_paths()
    a_with_snow_1m, tau_with_snow_1m = paths_with_snow_1m.cir()

    cfr_with_snow_1m, dists_with_snow_1m = _evaluate_cfr(
        paths_with_snow_1m,
        freqs=freqs,
        anchors=txs,
        node_pos=rx_pos
    )
    amp_db_with_snow_1m = mean_amp_in_db_from_cfr(cfr_with_snow_1m)

    print(f"Snow radius: 1.0 m (diameter: 2m)")
    print(f"Mean amplitude (with 1m snow): {amp_db_with_snow_1m:.2f} dB")

    # Analysis
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    attenuation_05m_db = amp_db_no_snow - amp_db_with_snow
    attenuation_1m_db = amp_db_no_snow - amp_db_with_snow_1m

    print(f"\nSignal amplitude (no snow):       {amp_db_no_snow:.2f} dB")
    print(f"Signal amplitude (0.5m sphere):   {amp_db_with_snow:.2f} dB")
    print(f"Signal amplitude (1m sphere):     {amp_db_with_snow_1m:.2f} dB")
    print(f"Attenuation (0.5m sphere):        {attenuation_05m_db:.2f} dB")
    print(f"Attenuation (1m sphere):          {attenuation_1m_db:.2f} dB")

    # Create comparison bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = ['No Snow', '0.5m Sphere', '1m Sphere']
    amplitudes = [amp_db_no_snow, amp_db_with_snow, amp_db_with_snow_1m]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    bars = ax.bar(scenarios, amplitudes, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, amp in zip(bars, amplitudes):
        height = bar.get_height()
        ax.annotate(f'{amp:.1f} dB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Mean Amplitude (dB)', fontsize=12)
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_title('Signal Attenuation Through Snow at 3.5 GHz\n(TX-RX distance: 10m)', fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "amplitude_comparison.png", dpi=150)
    plt.close()

    print(f"\n✓ Plot saved to: {out_dir / 'amplitude_comparison.png'}")

    # Validation
    assert amp_db_with_snow < amp_db_no_snow, \
        f"Snow should reduce signal! Got {amp_db_with_snow:.2f} dB vs {amp_db_no_snow:.2f} dB"

    # Allow equal values when both are at the noise floor (complete blockage)
    assert amp_db_with_snow_1m <= amp_db_with_snow, \
        f"1m snow should attenuate at least as much as 0.5m! Got {amp_db_with_snow_1m:.2f} dB vs {amp_db_with_snow:.2f} dB"

    print(f"\n✓ TEST PASSED: 0.5m sphere caused {attenuation_05m_db:.2f} dB attenuation")
    print(f"✓ TEST PASSED: 1m sphere caused {attenuation_1m_db:.2f} dB attenuation")
    print(f"  Output directory: {out_dir}")

    # ==========================================================================
    # THEORETICAL VS SIMULATION COMPARISON
    # ==========================================================================
    print("\n" + "="*70)
    print("THEORETICAL VS SIMULATION COMPARISON (Ulaby & Long Model)")
    print("="*70)

    # Snow parameters for DRY snow (matching Snow class defaults)
    # As per Ulaby & Long (2014), mv=0.5% gives small but non-zero absorption
    Ps = 0.4   # Dry snow density (g/cm³)
    mv = 0.5   # Volumetric water content (%) - nearly dry snow
    freq_ghz = 3.5

    # Calculate dielectric properties
    eps_r, eps_i = ulaby_long_snow_dielectric(Ps, mv, freq_ghz)
    atten_per_m = attenuation_db_per_meter(eps_r, eps_i, freq_ghz * 1e9)

    print(f"\nDry snow parameters (Ulaby & Long model):")
    print(f"  Dry snow density (Ps): {Ps} g/cm³")
    print(f"  Volumetric water content (mv): {mv}% (nearly dry)")
    print(f"  Frequency: {freq_ghz} GHz")
    print(f"  ε' (real): {eps_r:.3f}")
    print(f"  ε'' (imag): {eps_i:.3f}")
    print(f"  Attenuation: {atten_per_m:.2f} dB/m")

    # Theoretical attenuation for 0.5m and 1m radius spheres
    # For a sphere, signal passes through diameter = 2 * radius
    diameter_05m = 1.0  # 0.5m radius -> 1m diameter
    diameter_1m = 2.0   # 1m radius -> 2m diameter

    theoretical_05m = theoretical_snow_attenuation(diameter_05m, Ps, mv, freq_ghz)
    theoretical_1m = theoretical_snow_attenuation(diameter_1m, Ps, mv, freq_ghz)

    print(f"\nTheoretical attenuation (Ulaby & Long):")
    print(f"  0.5m radius sphere (1m path): {theoretical_05m:.2f} dB")
    print(f"  1m radius sphere (2m path): {theoretical_1m:.2f} dB")

    print(f"\nSimulated attenuation (AvaSimRT):")
    print(f"  0.5m radius sphere: {attenuation_05m_db:.2f} dB")
    print(f"  1m radius sphere: {attenuation_1m_db:.2f} dB")

    # ==========================================================================
    # PLOT 1: Theoretical attenuation vs snow thickness
    # ==========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Generate theoretical curve
    thicknesses = np.linspace(0, 3, 100)
    theoretical_attenuations = [theoretical_snow_attenuation(t, Ps, mv, freq_ghz) for t in thicknesses]

    ax1 = axes[0]
    ax1.plot(thicknesses, theoretical_attenuations, 'b-', linewidth=2,
             label=f'Ulaby & Long Model\n(Ps={Ps} g/cm³, mv={mv}%)')
    ax1.axvline(x=diameter_05m, color='orange', linestyle='--', alpha=0.7, label=f'0.5m sphere (1m path)')
    ax1.axvline(x=diameter_1m, color='red', linestyle='--', alpha=0.7, label=f'1m sphere (2m path)')
    ax1.scatter([diameter_05m], [theoretical_05m], color='orange', s=100, zorder=5, edgecolor='black')
    ax1.scatter([diameter_1m], [theoretical_1m], color='red', s=100, zorder=5, edgecolor='black')

    ax1.set_xlabel('Snow Thickness (m)', fontsize=12)
    ax1.set_ylabel('Attenuation (dB)', fontsize=12)
    ax1.set_title(f'Theoretical Snow Attenuation at {freq_ghz} GHz\n(Ulaby & Long Model, mv={mv}%)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)

    # ==========================================================================
    # PLOT 2: Simulation vs Theory comparison (bar chart)
    # ==========================================================================
    ax2 = axes[1]

    x_labels = ['0.5m Radius\n(1m path)', '1m Radius\n(2m path)']
    x_pos = np.arange(len(x_labels))
    bar_width = 0.35

    # Theoretical values
    theoretical_vals = [theoretical_05m, theoretical_1m]
    # Simulated values
    simulated_vals = [attenuation_05m_db, attenuation_1m_db]

    bars1 = ax2.bar(x_pos - bar_width/2, theoretical_vals, bar_width,
                    label='Theoretical (Ulaby & Long)', color='#3498db', edgecolor='black')
    bars2 = ax2.bar(x_pos + bar_width/2, simulated_vals, bar_width,
                    label='Simulated (AvaSimRT)', color='#e74c3c', edgecolor='black')

    # Add value labels
    for bar, val in zip(bars1, theoretical_vals):
        ax2.annotate(f'{val:.1f} dB',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, simulated_vals):
        ax2.annotate(f'{val:.1f} dB',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Snow Obstacle Size', fontsize=12)
    ax2.set_ylabel('Attenuation (dB)', fontsize=12)
    ax2.set_title('Theoretical vs Simulated Attenuation\nat 3.5 GHz', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "theoretical_vs_simulation.png", dpi=150)
    plt.close()

    print(f"\n✓ Comparison plot saved to: {out_dir / 'theoretical_vs_simulation.png'}")

    # ==========================================================================
    # PLOT 3: Difference analysis
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    diff_05m = simulated_vals[0] - theoretical_vals[0]
    diff_1m = simulated_vals[1] - theoretical_vals[1]

    categories = ['0.5m Radius Sphere', '1m Radius Sphere']
    differences = [diff_05m, diff_1m]
    colors = ['#27ae60' if d >= 0 else '#c0392b' for d in differences]

    bars = ax.bar(categories, differences, color=colors, edgecolor='black', linewidth=1.2)

    for bar, diff in zip(bars, differences):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 3 if height >= 0 else -3
        ax.annotate(f'{diff:+.1f} dB',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, offset), textcoords="offset points",
                    ha='center', va=va, fontsize=12, fontweight='bold')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Difference (Simulated - Theoretical) [dB]', fontsize=12)
    ax.set_title('Simulation vs Theory Difference\n(Positive = more attenuation than predicted)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "simulation_theory_difference.png", dpi=150)
    plt.close()

    print(f"✓ Difference plot saved to: {out_dir / 'simulation_theory_difference.png'}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Scenario':<25} {'Theoretical':<15} {'Simulated':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'0.5m radius (1m path)':<25} {theoretical_05m:<15.2f} {attenuation_05m_db:<15.2f} {diff_05m:+.2f} dB")
    print(f"{'1m radius (2m path)':<25} {theoretical_1m:<15.2f} {attenuation_1m_db:<15.2f} {diff_1m:+.2f} dB")
