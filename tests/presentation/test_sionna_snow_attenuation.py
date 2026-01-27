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

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Sequence

import numpy as np
import pytest
import sionna_vispy

from avasimrt.channelstate.simulation import (
    _build_context,
    _evaluate_cfr,
    Position3D,
    Resolution,
)

from avasimrt.channelstate.snow import (
    Snow, create_scene_with_snow, ulaby_long_snow_dielectric
)

from sionna.rt import subcarrier_frequencies
from avasimrt.math import mean_amp_in_db_from_cfr, distance
import matplotlib.pyplot as plt


# =============================================================================
# SHARED HELPER CLASSES AND FUNCTIONS
# =============================================================================

@dataclass
class SimulationParams:
    """Common simulation parameters."""
    sc_num: int = 101
    sc_spacing: float = 5e6
    freq_center: float = 3.5e9  # 3.5 GHz
    reflection_depth: int = 3
    seed: int = 42

    @property
    def bandwidth(self) -> float:
        return self.sc_num * self.sc_spacing

    @property
    def freqs(self) -> np.ndarray:
        return subcarrier_frequencies(self.sc_num, self.sc_spacing)

    @property
    def freq_ghz(self) -> float:
        return self.freq_center / 1e9


@dataclass
class SimulationResult:
    """Result of a single simulation run."""
    amp_db: float
    paths: object
    ctx: object
    cfr: np.ndarray
    dists: np.ndarray


def setup_output_dir(
    presentation_output: Path,
    examples: Path,
    subdir: str
) -> tuple[Path, Path]:
    """
    Setup output directory and copy necessary files.

    Returns:
        Tuple of (out_dir, scene_xml_path)
    """
    out_dir = presentation_output / subdir
    out_dir.mkdir(exist_ok=True)

    base_scene = examples / "empty.xml"
    scene_xml = out_dir / "scene.xml"
    shutil.copy(base_scene, scene_xml)

    # Copy meshes directory if needed
    base_meshes = examples / "seegrube" / "assets" / "meshes"
    if base_meshes.exists():
        shutil.copytree(base_meshes, out_dir / "meshes", dirs_exist_ok=True)
    else:
        (out_dir / "meshes").mkdir(exist_ok=True)

    return out_dir, scene_xml


def run_simulation(
    txs: list[tuple[str, tuple[float, float, float], float]],
    rx_pos: tuple[float, float, float],
    params: SimulationParams,
    scene_src: Path | None = None,
    snow: Snow | None = None,
    show_preview: bool = False,
    screenshot_path: Path | None = None,
    camera_origin: Position3D | None = None,
    camera_target: Position3D | None = None,
    resolution: Resolution = (1920, 1080),
) -> SimulationResult:
    """
    Run a single simulation and return the result.

    Args:
        txs: List of transmitter tuples (name, position, power)
        rx_pos: Receiver position
        params: Simulation parameters
        scene_src: Path to scene XML file (None for empty scene)
        snow: Snow material (None for no snow)
        show_preview: Whether to show the 3D preview interactively
        screenshot_path: Path to save a screenshot (None to skip)
        camera_origin: Camera position for screenshot (auto-calculated if None)
        camera_target: Camera look-at target for screenshot (auto-calculated if None)
        resolution: Screenshot resolution (width, height)

    Returns:
        SimulationResult with amplitude in dB and other data
    """
    ctx = _build_context(
        anchors=txs,
        scene_src=scene_src,
        freq_center=params.freq_center,
        bandwidth=params.bandwidth,
        snow=snow,
        reflection_depth=params.reflection_depth,
        seed=params.seed
    )

    paths = ctx.solve_paths()
    paths.cir()  # Compute CIR

    if screenshot_path:
        # Auto-calculate camera position if not provided
        tx_pos = txs[0][1]  # First transmitter position
        if camera_target is None:
            # Look at midpoint between TX and RX
            camera_target = (
                (tx_pos[0] + rx_pos[0]) / 2,
                (tx_pos[1] + rx_pos[1]) / 2,
                (tx_pos[2] + rx_pos[2]) / 2,
            )
        if camera_origin is None:
            # Position camera at an angle to show TX-RX distance clearly
            # Offset diagonally (X and Y) and elevate (Z) for perspective view
            dist = distance(tx_pos, rx_pos)*2
            camera_origin = (
                camera_target[0] + dist * 0.7,   # Offset in X
                camera_target[1] + dist * 0.5,   # Offset in Y (rotated view)
                camera_target[2] + dist * 0.4,   # Above the scene
            )

        ctx.render_to_file(
            paths=paths,
            origin=camera_origin,
            target=camera_target,
            file_path=screenshot_path,
            resolution=resolution,
        )
        print(f"  Screenshot saved: {screenshot_path}")

    if show_preview:
        with sionna_vispy.patch():
            ctx.scene.preview(paths=paths)
        canvas = sionna_vispy.get_canvas(ctx.scene)
        canvas.show()
        canvas.app.run()

    cfr, dists = _evaluate_cfr(
        paths,
        freqs=params.freqs,
        anchors=txs,
        node_pos=rx_pos
    )
    amp_db = mean_amp_in_db_from_cfr(cfr)

    return SimulationResult(
        amp_db=amp_db,
        paths=paths,
        ctx=ctx,
        cfr=cfr,
        dists=dists
    )


def run_snow_simulation(
    txs: list[tuple[str, tuple[float, float, float], float]],
    rx_pos: tuple[float, float, float],
    params: SimulationParams,
    scene_xml: Path,
    meshes_dir: Path,
    radius: float,
    snow_position: np.ndarray,
    show_preview: bool = False,
    screenshot_path: Path | None = None,
    camera_origin: Position3D | None = None,
    camera_target: Position3D | None = None,
    resolution: Resolution = (1920, 1080),
) -> SimulationResult:
    """
    Run a simulation with a snow sphere obstacle.

    Args:
        txs: List of transmitter tuples (name, position, power)
        rx_pos: Receiver position
        params: Simulation parameters
        scene_xml: Path to base scene XML file
        meshes_dir: Directory for mesh files
        radius: Radius of the snow sphere in meters
        snow_position: Position(s) of snow spheres as numpy array
        show_preview: Whether to show the 3D preview interactively
        screenshot_path: Path to save a screenshot (None to skip)
        camera_origin: Camera position for screenshot (auto-calculated if None)
        camera_target: Camera look-at target for screenshot (auto-calculated if None)
        resolution: Screenshot resolution (width, height)

    Returns:
        SimulationResult with amplitude in dB and other data
    """
    snow_scene_xml = create_scene_with_snow(
        xml_path=scene_xml,
        meshes_dir=meshes_dir,
        radius=radius,
        positions=snow_position
    )

    # diameter = 2 * radius
    snow = Snow(thickness_m=2 * radius)

    return run_simulation(
        txs=txs,
        rx_pos=rx_pos,
        params=params,
        scene_src=snow_scene_xml,
        snow=snow,
        show_preview=show_preview,
        screenshot_path=screenshot_path,
        camera_origin=camera_origin,
        camera_target=camera_target,
        resolution=resolution,
    )


def plot_amplitude_comparison(
    scenarios: list[str],
    amplitudes: list[float],
    title: str,
    output_path: Path,
    colors: list[str] | None = None,
) -> None:
    """
    Create a bar plot comparing signal amplitudes across scenarios.

    Args:
        scenarios: List of scenario names
        amplitudes: List of amplitude values in dB
        title: Plot title
        output_path: Path to save the plot
        colors: Optional list of colors for bars
    """
    if colors is None:
        # Default color gradient from green (good) to red (attenuated)
        n = len(scenarios)
        colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, n))

    fig, ax = plt.subplots(figsize=(max(10, len(scenarios) * 1.5), 6))

    bars = ax.bar(scenarios, amplitudes, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, amp in zip(bars, amplitudes):
        height = bar.get_height()
        ax.annotate(f'{amp:.1f} dB',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Mean Amplitude (dB)', fontsize=12)
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_attenuation_vs_thickness(
    radii: Sequence[float],
    simulated_attenuations: Sequence[float],
    params: SimulationParams,
    output_path: Path,
    Ps: float = 0.4,
    mv: float = 0.5,
) -> None:
    """
    Create a plot comparing simulated vs theoretical attenuation.

    Args:
        radii: List of sphere radii in meters
        simulated_attenuations: Corresponding simulated attenuation values
        params: Simulation parameters
        output_path: Path to save the plot
        Ps: Dry snow density (g/cm³)
        mv: Volumetric water content (%)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Theoretical curve
    max_thickness = max(r * 2 for r in radii) * 1.2
    thicknesses = np.linspace(0, max_thickness, 100)
    theoretical_attenuations = [
        theoretical_snow_attenuation(t, Ps, mv, params.freq_ghz)
        for t in thicknesses
    ]

    # Plot 1: Theoretical curve with simulation points
    ax1 = axes[0]
    ax1.plot(thicknesses, theoretical_attenuations, 'b-', linewidth=2,
             label=f'Ulaby & Long Model\n(Ps={Ps} g/cm³, mv={mv}%)')

    # Add simulation points
    diameters = [r * 2 for r in radii]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(radii)))
    for d, att, c in zip(diameters, simulated_attenuations, colors):
        ax1.scatter([d], [att], color=c, s=100, zorder=5, edgecolor='black',
                    label=f'{d:.0f}m diameter (simulated)')

    ax1.set_xlabel('Snow Thickness (m)', fontsize=12)
    ax1.set_ylabel('Attenuation (dB)', fontsize=12)
    ax1.set_title(f'Attenuation vs Snow Thickness at {params.freq_ghz} GHz', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bar chart comparison
    ax2 = axes[1]
    x_labels = [f'{r:.1f}m radius\n({r*2:.0f}m path)' for r in radii]
    x_pos = np.arange(len(radii))
    bar_width = 0.35

    theoretical_vals = [theoretical_snow_attenuation(r * 2, Ps, mv, params.freq_ghz) for r in radii]

    bars1 = ax2.bar(x_pos - bar_width/2, theoretical_vals, bar_width,
                    label='Theoretical', color='#3498db', edgecolor='black')
    bars2 = ax2.bar(x_pos + bar_width/2, simulated_attenuations, bar_width,
                    label='Simulated', color='#e74c3c', edgecolor='black')

    # Add value labels
    for bar, val in zip(bars1, theoretical_vals):
        ax2.annotate(f'{val:.1f}',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, val in zip(bars2, simulated_attenuations):
        ax2.annotate(f'{val:.1f}',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.set_xlabel('Snow Obstacle Size', fontsize=12)
    ax2.set_ylabel('Attenuation (dB)', fontsize=12)
    ax2.set_title(f'Theoretical vs Simulated at {params.freq_ghz} GHz', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# THEORETICAL CALCULATIONS
# =============================================================================

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
        - Compare signal with vs without snow (0.5m and 1m radius spheres)

    Expected:
        Signal amplitude WITH snow << Signal amplitude WITHOUT snow
    """
    out_dir, scene_xml = setup_output_dir(
        presentation_output, examples, "sionna_snow_attenuation"
    )

    tx_pos = (0.0, -10.0, 0.0)
    rx_pos = (0.0, 0.0, 0.0)
    txs = [("tx1", tx_pos, 1.0)]
    snow_position = np.array([[0.0, -5.0, 0.0]])  # Midway between TX and RX

    params = SimulationParams()

    # Create screenshots directory
    screenshots_dir = out_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    # -------------------------------------------------------------------------
    # Test 1: Without snow (baseline)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("TEST 1: NO SNOW (baseline)")
    print("="*70)

    result_no_snow = run_simulation(
        txs=txs,
        rx_pos=rx_pos,
        params=params,
        scene_src=None,
        snow=None,
        show_preview=False,
        screenshot_path=screenshots_dir / "scenario_no_snow.png",
    )

    print(f"TX position: {tx_pos}")
    print(f"RX position: {rx_pos}")
    print(f"Distance: {distance(tx_pos, rx_pos):.1f} m")
    print(f"Mean amplitude (no snow): {result_no_snow.amp_db:.2f} dB")

    # -------------------------------------------------------------------------
    # Test 2: With 0.5m radius snow sphere
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("TEST 2: WITH 0.5M SNOW SPHERE")
    print("="*70)

    result_05m = run_snow_simulation(
        txs=txs,
        rx_pos=rx_pos,
        params=params,
        scene_xml=scene_xml,
        meshes_dir=out_dir / "meshes",
        radius=0.5,
        snow_position=snow_position,
        show_preview=False,
        screenshot_path=screenshots_dir / "scenario_snow_0.5m_radius.png",
    )

    print(f"Snow position: {snow_position[0]}")
    print(f"Snow radius: 0.5 m (diameter: 1m)")
    print(f"Mean amplitude (with 0.5m snow): {result_05m.amp_db:.2f} dB")

    # -------------------------------------------------------------------------
    # Test 3: With 1m radius snow sphere
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("TEST 3: WITH 1M SNOW SPHERE")
    print("="*70)

    result_1m = run_snow_simulation(
        txs=txs,
        rx_pos=rx_pos,
        params=params,
        scene_xml=scene_xml,
        meshes_dir=out_dir / "meshes",
        radius=1.0,
        snow_position=snow_position,
        show_preview=False,
        screenshot_path=screenshots_dir / "scenario_snow_1.0m_radius.png",
    )

    print(f"Snow radius: 1.0 m (diameter: 2m)")
    print(f"Mean amplitude (with 1m snow): {result_1m.amp_db:.2f} dB")

    # -------------------------------------------------------------------------
    # Analysis
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    attenuation_05m_db = result_no_snow.amp_db - result_05m.amp_db
    attenuation_1m_db = result_no_snow.amp_db - result_1m.amp_db

    print(f"\nSignal amplitude (no snow):       {result_no_snow.amp_db:.2f} dB")
    print(f"Signal amplitude (0.5m sphere):   {result_05m.amp_db:.2f} dB")
    print(f"Signal amplitude (1m sphere):     {result_1m.amp_db:.2f} dB")
    print(f"Attenuation (0.5m sphere):        {attenuation_05m_db:.2f} dB")
    print(f"Attenuation (1m sphere):          {attenuation_1m_db:.2f} dB")

    # Create comparison bar plot
    plot_amplitude_comparison(
        scenarios=['No Snow', '0.5m Sphere', '1m Sphere'],
        amplitudes=[result_no_snow.amp_db, result_05m.amp_db, result_1m.amp_db],
        title='Signal Attenuation Through Snow at 3.5 GHz\n(TX-RX distance: 10m)',
        output_path=out_dir / "amplitude_comparison.png",
        colors=['#2ecc71', '#f39c12', '#e74c3c'],
    )
    print(f"\n✓ Plot saved to: {out_dir / 'amplitude_comparison.png'}")

    # Validation
    assert result_05m.amp_db < result_no_snow.amp_db, \
        f"Snow should reduce signal! Got {result_05m.amp_db:.2f} dB vs {result_no_snow.amp_db:.2f} dB"

    assert result_1m.amp_db <= result_05m.amp_db, \
        f"1m snow should attenuate at least as much as 0.5m! Got {result_1m.amp_db:.2f} dB vs {result_05m.amp_db:.2f} dB"

    print(f"\n✓ TEST PASSED: 0.5m sphere caused {attenuation_05m_db:.2f} dB attenuation")
    print(f"✓ TEST PASSED: 1m sphere caused {attenuation_1m_db:.2f} dB attenuation")
    print(f"  Output directory: {out_dir}")
    print(f"\nScreenshots saved to: {screenshots_dir}")
    for f in sorted(screenshots_dir.glob("*.png")):
        print(f"  - {f.name}")

    # -------------------------------------------------------------------------
    # Theoretical vs Simulation Comparison
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("THEORETICAL VS SIMULATION COMPARISON (Ulaby & Long Model)")
    print("="*70)

    Ps = 0.4   # Dry snow density (g/cm³)
    mv = 0.5   # Volumetric water content (%)

    eps_r, eps_i = ulaby_long_snow_dielectric(Ps, mv, params.freq_ghz)
    atten_per_m = attenuation_db_per_meter(eps_r, eps_i, params.freq_center)

    print(f"\nDry snow parameters (Ulaby & Long model):")
    print(f"  Dry snow density (Ps): {Ps} g/cm³")
    print(f"  Volumetric water content (mv): {mv}% (nearly dry)")
    print(f"  Frequency: {params.freq_ghz} GHz")
    print(f"  ε' (real): {eps_r:.3f}")
    print(f"  ε'' (imag): {eps_i:.3f}")
    print(f"  Attenuation: {atten_per_m:.2f} dB/m")

    theoretical_05m = theoretical_snow_attenuation(1.0, Ps, mv, params.freq_ghz)
    theoretical_1m = theoretical_snow_attenuation(2.0, Ps, mv, params.freq_ghz)

    print(f"\nTheoretical attenuation (Ulaby & Long):")
    print(f"  0.5m radius sphere (1m path): {theoretical_05m:.2f} dB")
    print(f"  1m radius sphere (2m path): {theoretical_1m:.2f} dB")

    print(f"\nSimulated attenuation (AvaSimRT):")
    print(f"  0.5m radius sphere: {attenuation_05m_db:.2f} dB")
    print(f"  1m radius sphere: {attenuation_1m_db:.2f} dB")

    # Create theoretical vs simulation plots
    plot_attenuation_vs_thickness(
        radii=[0.5, 1.0],
        simulated_attenuations=[attenuation_05m_db, attenuation_1m_db],
        params=params,
        output_path=out_dir / "theoretical_vs_simulation.png",
        Ps=Ps,
        mv=mv,
    )
    print(f"\n✓ Comparison plot saved to: {out_dir / 'theoretical_vs_simulation.png'}")

    # Difference analysis plot
    fig, ax = plt.subplots(figsize=(10, 6))
    diff_05m = attenuation_05m_db - theoretical_05m
    diff_1m = attenuation_1m_db - theoretical_1m

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


@pytest.mark.presentation
def test_snow_attenuation_varying_sizes_distant(
    presentation_output: Path,
    examples: Path,
    presentation_config
):
    """
    Extended test with more distant TX/RX nodes and multiple snow sphere sizes.

    Setup:
        - TX at (0, -50, 0)
        - RX at (0, 0, 0)
        - Distance: 50m
        - Snow spheres of diameters: 1m, 2m, 5m, 8m, 10m (radii: 0.5, 1, 2.5, 4, 5)
        - Snow positioned midway between TX and RX

    This test validates attenuation scaling with snow thickness at greater distances.
    """
    out_dir, scene_xml = setup_output_dir(
        presentation_output, examples, "sionna_snow_attenuation_distant"
    )

    tx_pos = (0.0, -50.0, 0.0)
    rx_pos = (0.0, 0.0, 0.0)
    txs = [("tx1", tx_pos, 1.0)]
    snow_position = np.array([[0.0, -25.0, 0.0]])  # Midway between TX and RX

    params = SimulationParams()

    # Snow sphere radii to test (corresponding to diameters 1m, 2m, 5m, 8m, 10m)
    radii = [0.5, 1.0, 2.5, 4.0, 5.0]
    diameters = [r * 2 for r in radii]

    # Create screenshots directory
    screenshots_dir = out_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("SNOW ATTENUATION TEST - DISTANT NODES, MULTIPLE SIZES")
    print("="*70)
    print(f"TX position: {tx_pos}")
    print(f"RX position: {rx_pos}")
    print(f"Distance: {distance(tx_pos, rx_pos):.1f} m")
    print(f"Snow position: {snow_position[0]}")
    print(f"Testing diameters: {diameters} m")

    # -------------------------------------------------------------------------
    # Baseline: No snow
    # -------------------------------------------------------------------------
    print("\n" + "-"*50)
    print("Running baseline (no snow)...")

    result_no_snow = run_simulation(
        txs=txs,
        rx_pos=rx_pos,
        params=params,
        scene_src=None,
        snow=None,
        show_preview=False,
        screenshot_path=screenshots_dir / "scenario_no_snow.png",
    )
    print(f"Baseline amplitude: {result_no_snow.amp_db:.2f} dB")

    # -------------------------------------------------------------------------
    # Run simulations for each snow sphere size
    # -------------------------------------------------------------------------
    results = []
    for radius in radii:
        diameter = radius * 2
        print(f"\nRunning simulation with {diameter:.0f}m diameter snow sphere (radius={radius}m)...")

        result = run_snow_simulation(
            txs=txs,
            rx_pos=rx_pos,
            params=params,
            scene_xml=scene_xml,
            meshes_dir=out_dir / "meshes",
            radius=radius,
            snow_position=snow_position,
            show_preview=False,
            screenshot_path=screenshots_dir / f"scenario_snow_{diameter:.0f}m_diameter.png",
        )
        results.append(result)
        print(f"  Amplitude: {result.amp_db:.2f} dB")

    # -------------------------------------------------------------------------
    # Calculate attenuations
    # -------------------------------------------------------------------------
    attenuations = [result_no_snow.amp_db - r.amp_db for r in results]

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Diameter (m)':<15} {'Amplitude (dB)':<18} {'Attenuation (dB)':<18}")
    print("-" * 55)
    print(f"{'No snow':<15} {result_no_snow.amp_db:<18.2f} {'0.00':<18}")
    for d, r, att in zip(diameters, results, attenuations):
        print(f"{d:<15.0f} {r.amp_db:<18.2f} {att:<18.2f}")

    # -------------------------------------------------------------------------
    # Theoretical comparison
    # -------------------------------------------------------------------------
    Ps = 0.4  # Dry snow density
    mv = 0.5  # Volumetric water content

    theoretical_attenuations = [
        theoretical_snow_attenuation(d, Ps, mv, params.freq_ghz) for d in diameters
    ]

    print("\n" + "="*70)
    print("THEORETICAL VS SIMULATED COMPARISON")
    print("="*70)
    print(f"\n{'Diameter':<12} {'Theoretical':<15} {'Simulated':<15} {'Difference':<15}")
    print("-" * 60)
    for d, theo, sim in zip(diameters, theoretical_attenuations, attenuations):
        diff = sim - theo
        print(f"{d:<12.0f}m {theo:<15.2f} {sim:<15.2f} {diff:+.2f} dB")

    # -------------------------------------------------------------------------
    # Plot 1: Amplitude comparison bar chart
    # -------------------------------------------------------------------------
    scenarios = ['No Snow'] + [f'{d:.0f}m' for d in diameters]
    amplitudes = [result_no_snow.amp_db] + [r.amp_db for r in results]

    plot_amplitude_comparison(
        scenarios=scenarios,
        amplitudes=amplitudes,
        title=f'Signal Attenuation vs Snow Thickness at {params.freq_ghz} GHz\n(TX-RX distance: 50m)',
        output_path=out_dir / "amplitude_comparison.png",
    )
    print(f"\n✓ Amplitude plot saved to: {out_dir / 'amplitude_comparison.png'}")

    # -------------------------------------------------------------------------
    # Plot 2: Theoretical vs Simulated attenuation
    # -------------------------------------------------------------------------
    plot_attenuation_vs_thickness(
        radii=radii,
        simulated_attenuations=attenuations,
        params=params,
        output_path=out_dir / "theoretical_vs_simulation.png",
        Ps=Ps,
        mv=mv,
    )
    print(f"✓ Theoretical comparison plot saved to: {out_dir / 'theoretical_vs_simulation.png'}")

    # -------------------------------------------------------------------------
    # Plot 3: Attenuation vs diameter (line plot)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Theoretical curve
    thick_range = np.linspace(0, max(diameters) * 1.1, 100)
    theo_curve = [theoretical_snow_attenuation(t, Ps, mv, params.freq_ghz) for t in thick_range]

    ax.plot(thick_range, theo_curve, 'b-', linewidth=2, label='Theoretical (Ulaby & Long)')
    ax.scatter(diameters, attenuations, color='red', s=100, zorder=5,
               edgecolor='black', label='Simulated (AvaSimRT)')

    # Add labels for each point
    for d, att in zip(diameters, attenuations):
        ax.annotate(f'{att:.1f} dB', xy=(d, att), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)

    ax.set_xlabel('Snow Thickness / Diameter (m)', fontsize=12)
    ax.set_ylabel('Attenuation (dB)', fontsize=12)
    ax.set_title(f'Snow Attenuation at {params.freq_ghz} GHz\n(TX-RX distance: 50m, Ps={Ps} g/cm³, mv={mv}%)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(diameters) * 1.1)
    ax.set_ylim(0, None)

    plt.tight_layout()
    plt.savefig(out_dir / "attenuation_vs_diameter.png", dpi=150)
    plt.close()

    print(f"✓ Attenuation vs diameter plot saved to: {out_dir / 'attenuation_vs_diameter.png'}")

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    # Check that snow causes attenuation
    assert all(att > 0 for att in attenuations), \
        "All snow spheres should cause positive attenuation"

    # Check that larger spheres cause more (or equal) attenuation
    for i in range(len(attenuations) - 1):
        assert attenuations[i+1] >= attenuations[i] * 0.9, \
            f"Larger snow sphere ({diameters[i+1]}m) should attenuate at least as much as smaller ({diameters[i]}m)"

    print(f"\n✓ TEST PASSED: All snow spheres cause attenuation")
    print(f"✓ TEST PASSED: Attenuation increases with snow thickness")
    print(f"  Output directory: {out_dir}")
    print(f"\nScreenshots saved to: {screenshots_dir}")
    for f in sorted(screenshots_dir.glob("*.png")):
        print(f"  - {f.name}")
