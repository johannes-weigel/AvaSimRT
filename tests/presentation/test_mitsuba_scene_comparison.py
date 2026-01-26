"""
Test comparing signal attenuation across different Mitsuba scenes with Snow material.

Scenarios:
- empty: No geometry
- 1: Single cube
- 3_close: Three cubes, close configuration
- 3_distant: Three cubes, distant configuration

Setup:
- TX at (-5, 0, 0)
- RX at (+5, 0, 0)
- Snow material assigned to all scene objects
"""
from __future__ import annotations

from pathlib import Path
import shutil

import mitsuba as mi
import numpy as np
import pytest
import matplotlib.pyplot as plt

from sionna.rt import (
    ITURadioMaterial,
    PathSolver,
    PlanarArray,
    Receiver,
    Scene,
    Transmitter,
    load_scene,
    subcarrier_frequencies,
    Paths,
)
import sionna_vispy

from avasimrt.channelstate.snow import Snow
from avasimrt.math import mean_amp_in_db_from_cfr


def setup_scene_with_snow_material(scene_xml: Path | None) -> Scene:
    """Load scene and assign Snow material to ALL objects."""
    if scene_xml is not None:
        scene = load_scene(scene_xml.as_posix(), merge_shapes=False)
    else:
        scene = load_scene()

    # Create Snow material
    snow = Snow()
    snow_material = snow._material
    scene.add(snow_material)

    # Apply snow material to ALL objects in the scene
    for obj_name, obj in scene.objects.items():
        obj.radio_material = snow_material

    # Setup antenna arrays
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )

    return scene


def run_scenario(
    scene_xml: Path | None,
    scenario_name: str,
    out_dir: Path,
    resolution: tuple[int, int],
) -> tuple[float, Paths, Scene]:
    """Run a single scenario and return amplitude in dB, paths, and scene."""
    scene = setup_scene_with_snow_material(scene_xml)

    # Add TX at x=-5
    tx = Transmitter(
        name="tx",
        position=mi.Point3f(-5.0, 0.0, 0.0),
        display_radius=0.3,
        color=(0.180, 0.282, 0.388),
    )
    scene.add(tx)

    # Add RX at x=+5
    rx = Receiver(
        name="rx",
        position=mi.Point3f(5.0, 0.0, 0.0),
        display_radius=0.3,
        color=(1.0, 0.308, 0.1),
    )
    scene.add(rx)

    # Set frequency
    freq_center = 3.5e9
    sc_num = 101
    sc_spacing = 5e6
    scene.frequency = freq_center
    scene.bandwidth = sc_num * sc_spacing

    # Solve paths
    solver = PathSolver()
    paths = solver(
        scene=scene,
        max_depth=10,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        synthetic_array=False,
        seed=42,
    )

    # Compute CFR and amplitude
    freqs = subcarrier_frequencies(sc_num, sc_spacing)
    h_raw = paths.cfr(frequencies=freqs, out_type="numpy", normalize_delays=False)
    cfr = h_raw[0][:, :, 0, 0, :]
    cfr = np.transpose(cfr, (1, 0, 2))
    amp_db = mean_amp_in_db_from_cfr(cfr)

    # Save screenshot
    from sionna.rt import Camera

    screenshot_path = out_dir / f"{scenario_name}.png"
    cam = Camera(
        position=mi.Point3f(0, -15, 10),
        look_at=mi.Point3f(0, 0, 0),
    )
    scene.render_to_file(
        camera=cam,
        #paths=paths,
        filename=screenshot_path.as_posix(),
        resolution=resolution,
        show_orientations=False,
    )

    return amp_db, paths, scene


@pytest.mark.presentation
def test_mitsuba_scene_comparison(
    presentation_output: Path,
    examples: Path,
    presentation_config,
):
    """
    Compare signal attenuation across different scene configurations.

    Scenes:
    - empty: No obstacles
    - 1: Single cube
    - 3_close: Three cubes, close together
    - 3_distant: Three cubes, spread apart
    """
    out_dir = presentation_output / "mitsuba_scene_comparison"
    out_dir.mkdir(exist_ok=True)

    # Define scenarios
    scenarios = {
        "empty": None,  # No scene file = empty scene
        "1_cube": examples / "mitsuba" / "1" / "scene.xml",
        "3_close": examples / "mitsuba" / "3_close" / "scene.xml",
        "3_distant": examples / "mitsuba" / "3_distant" / "scene.xml",
    }

    results = {}
    scenes = {}
    all_paths = {}

    print("\n" + "=" * 70)
    print("MITSUBA SCENE COMPARISON TEST")
    print("TX at (-5, 0, 0), RX at (+5, 0, 0)")
    print("Snow material applied to all objects")
    print("=" * 70)

    for name, scene_xml in scenarios.items():
        print(f"\nProcessing scenario: {name}")
        if scene_xml is not None and not scene_xml.exists():
            print(f"  WARNING: Scene file not found: {scene_xml}")
            continue

        amp_db, paths, scene = run_scenario(
            scene_xml=scene_xml,
            scenario_name=name,
            out_dir=out_dir,
            resolution=presentation_config.resolution,
        )

        results[name] = amp_db
        scenes[name] = scene
        all_paths[name] = paths

        print(f"  Objects in scene: {list(scene.objects.keys())}")
        print(f"  Amplitude: {amp_db:.2f} dB")
        print(f"  Screenshot saved: {out_dir / f'{name}.png'}")

    # Interactive preview for each scenario
    for name, scene in scenes.items():
        paths = all_paths[name]
        print(f"\nShowing preview for: {name}")

        with sionna_vispy.patch():
            scene.preview(paths=paths)

        sionna_vispy.get_canvas(scene).show()
        sionna_vispy.get_canvas(scene).app.run()

    # Print results summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    baseline = results.get("empty", 0)
    print(f"\n{'Scenario':<20} {'Amplitude (dB)':<15} {'Attenuation vs Empty (dB)':<25}")
    print("-" * 60)
    for name, amp_db in results.items():
        attenuation = baseline - amp_db
        print(f"{name:<20} {amp_db:<15.2f} {attenuation:<25.2f}")

    # Create comparison bar plot - Amplitude
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Absolute amplitudes
    ax1 = axes[0]
    scenario_names = list(results.keys())
    amplitudes = list(results.values())
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

    bars1 = ax1.bar(scenario_names, amplitudes, color=colors[: len(scenario_names)], edgecolor="black", linewidth=1.2)

    for bar, amp in zip(bars1, amplitudes):
        height = bar.get_height()
        ax1.annotate(
            f"{amp:.1f} dB",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax1.set_ylabel("Mean Amplitude (dB)", fontsize=12)
    ax1.set_xlabel("Scenario", fontsize=12)
    ax1.set_title("Signal Amplitude by Scene Configuration\n(TX-RX distance: 10m)", fontsize=12)
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.grid(axis="y", alpha=0.3)
    ax1.tick_params(axis="x", rotation=15)

    # Plot 2: Attenuation relative to empty scene
    ax2 = axes[1]
    attenuations = [baseline - amp for amp in amplitudes]
    bar_colors = ["#27ae60" if a >= 0 else "#c0392b" for a in attenuations]

    bars2 = ax2.bar(scenario_names, attenuations, color=bar_colors, edgecolor="black", linewidth=1.2)

    for bar, atten in zip(bars2, attenuations):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 3 if height >= 0 else -3
        ax2.annotate(
            f"{atten:+.1f} dB",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
        )

    ax2.set_ylabel("Attenuation vs Empty Scene (dB)", fontsize=12)
    ax2.set_xlabel("Scenario", fontsize=12)
    ax2.set_title("Signal Attenuation Relative to Empty Scene\n(Positive = more attenuation)", fontsize=12)
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(out_dir / "amplitude_comparison.png", dpi=150)
    plt.close()

    print(f"\nPlot saved to: {out_dir / 'amplitude_comparison.png'}")

    # Create pairwise difference heatmap
    n_scenarios = len(results)
    diff_matrix = np.zeros((n_scenarios, n_scenarios))
    scenario_list = list(results.keys())

    for i, name_i in enumerate(scenario_list):
        for j, name_j in enumerate(scenario_list):
            diff_matrix[i, j] = results[name_i] - results[name_j]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(diff_matrix, cmap="RdYlGn", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Amplitude Difference (dB)", fontsize=11)

    # Add labels
    ax.set_xticks(range(n_scenarios))
    ax.set_yticks(range(n_scenarios))
    ax.set_xticklabels(scenario_list, rotation=45, ha="right")
    ax.set_yticklabels(scenario_list)
    ax.set_xlabel("Scenario B", fontsize=12)
    ax.set_ylabel("Scenario A", fontsize=12)
    ax.set_title("Pairwise Amplitude Difference (A - B) in dB", fontsize=12)

    # Add text annotations
    for i in range(n_scenarios):
        for j in range(n_scenarios):
            text = ax.text(
                j, i, f"{diff_matrix[i, j]:+.1f}",
                ha="center", va="center",
                color="black" if abs(diff_matrix[i, j]) < 3 else "white",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(out_dir / "pairwise_difference_heatmap.png", dpi=150)
    plt.close()

    print(f"Heatmap saved to: {out_dir / 'pairwise_difference_heatmap.png'}")

    # Assertions
    if "empty" in results and "1_cube" in results:
        # With an obstacle, signal should be attenuated (or at least not stronger)
        assert results["1_cube"] <= results["empty"] + 1.0, \
            f"Single cube should not significantly increase signal"

    print(f"\nAll outputs saved to: {out_dir}")
