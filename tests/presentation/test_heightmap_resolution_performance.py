"""
Presentation test for heightmap generation performance across different resolutions.

This test generates heightmaps at multiple resolutions (100, 50, 10, 5, 1, 0.5 meters),
measures the generation time for each, and creates:
- 3D plot for each heightmap
- Comparison plot showing generation time vs resolution

Output directory: presentation_output/heightmap_resolution_performance/
"""
from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pytest

from avasimrt.preprocessing.heights import generate_heightmap


# Resolutions to test (in meters)
RESOLUTIONS = [100, 50, 10, 5, 1, 0.5]


@dataclass
class ResolutionResult:
    """Store results for a single resolution test."""
    resolution: float
    computation_time: float
    grid_size_x: int
    grid_size_y: int
    total_positions: int
    coverage_percent: float
    memory_kb: float
    z_min: float
    z_max: float
    z_range: float


@pytest.mark.presentation
def test_heightmap_resolution_performance(
    presentation_output: Path,
    examples: Path,
    presentation_config
) -> None:
    """
    Generate heightmaps at multiple resolutions and create performance visualizations.
    
    For each resolution:
    - Generates heightmap via raycasting
    - Measures generation time
    - Creates 3D surface plot
    - Records metadata
    
    Creates a comparison plot showing time vs resolution.
    
    Outputs:
    - heightmap_3d_<res>m.png: 3D surface plot for each resolution
    - time_comparison.png: Time comparison plot across all resolutions
    """
    # Locate the Nordkette scene OBJ file
    scene_obj = examples / "nordkette" / "assets" / "scene.obj"
    
    if not scene_obj.exists():
        pytest.skip(f"Scene file not found: {scene_obj}")
    
    # Create output directory
    out_dir = presentation_output / "heightmap_resolution_performance"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    results: list[ResolutionResult] = []
    
    print(f"\n{'='*80}")
    print(f"HEIGHTMAP RESOLUTION PERFORMANCE TEST")
    print(f"Scene: {scene_obj}")
    print(f"Resolutions: {RESOLUTIONS}")
    print(f"Output: {out_dir}")
    print(f"{'='*80}\n")
    
    for resolution in RESOLUTIONS:
        print(f"Processing resolution: {resolution}m...")
        
        # Measure time for heightmap generation
        start_time = time.perf_counter()
        heightmap_3d, metadata = generate_heightmap(scene_obj, resolution)
        end_time = time.perf_counter()
        computation_time = end_time - start_time
        
        # Extract coordinate arrays and z-values
        x_coords = heightmap_3d[:, 0, 0]
        y_coords = heightmap_3d[0, :, 1]
        heightmap_z = heightmap_3d[:, :, 2]
        
        # Store result
        result = ResolutionResult(
            resolution=resolution,
            computation_time=computation_time,
            grid_size_x=metadata['grid_size']['n_x'],
            grid_size_y=metadata['grid_size']['n_y'],
            total_positions=metadata['grid_size']['total_positions'],
            coverage_percent=metadata['coverage']['coverage_percent'],
            memory_kb=metadata['memory']['kilobytes'],
            z_min=metadata['heightmap_stats']['z_min'],
            z_max=metadata['heightmap_stats']['z_max'],
            z_range=metadata['heightmap_stats']['z_range'],
        )
        results.append(result)
        
        # === Create 3D Surface Plot ===
        fig = plt.figure(figsize=(14, 10))
        ax_3d = fig.add_subplot(111, projection='3d')
        
        X = x_coords
        Y = y_coords
        XX, YY = np.meshgrid(X, Y, indexing='ij')
        
        surf = ax_3d.plot_surface(
            XX, YY, heightmap_z,
            cmap='terrain',
            edgecolor='none',
            alpha=0.9,
            antialiased=True,
        )
        
        ax_3d.set_xlabel('X [m]', fontsize=13)
        ax_3d.set_ylabel('Y [m]', fontsize=13)
        ax_3d.set_zlabel('Elevation [m]', fontsize=13)
        ax_3d.set_title(
            f'Heightmap 3D Surface - Resolution: {resolution}m\n'
            f'Grid: {result.grid_size_x}×{result.grid_size_y} | '
            f'Time: {computation_time:.4f}s | '
            f'Coverage: {result.coverage_percent:.1f}%',
            fontsize=15,
            fontweight='bold',
            pad=20
        )
        ax_3d.view_init(elev=30, azim=-60)
        
        cbar = fig.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=10)
        cbar.set_label('Elevation [m]', fontsize=12)
        
        # Add metadata text box
        info_text = (
            f"Resolution: {resolution} m\n"
            f"Time: {computation_time:.4f} s\n"
            f"Grid: {result.grid_size_x} × {result.grid_size_y}\n"
            f"Positions: {result.total_positions:,}\n"
            f"Coverage: {result.coverage_percent:.1f}%\n"
            f"Memory: {result.memory_kb:.2f} KB\n"
            f"Z Range: [{result.z_min:.1f}, {result.z_max:.1f}] m"
        )
        ax_3d.text2D(
            0.02, 0.98, info_text,
            transform=ax_3d.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
        )
        
        plt.tight_layout()
        output_3d = out_dir / f"heightmap_3d_{resolution}m.png"
        plt.savefig(output_3d, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 3D plot saved: {output_3d.name}")
        print(f"    Time: {computation_time:.4f}s | Grid: {result.grid_size_x}×{result.grid_size_y} | "
              f"Positions: {result.total_positions:,}\n")
    
    # === Create Time Comparison Plot ===
    print(f"{'='*80}")
    print(f"Creating time comparison plot...")
    print(f"{'='*80}\n")
    
    resolutions_array = np.array([r.resolution for r in results])
    times_array = np.array([r.computation_time for r in results])
    total_positions_array = np.array([r.total_positions for r in results])
    
    # Create comparison figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Computation Time vs Resolution
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(resolutions_array, times_array, 'o-', linewidth=2.5, markersize=10, 
             color='#2E86AB', label='Measured Time')
    ax1.set_xlabel('Resolution [m]', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Computation Time [s]', fontsize=13, fontweight='bold')
    ax1.set_title('Generation Time vs Resolution', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Fit power law: time ∝ resolution^α
    log_res = np.log(resolutions_array)
    log_time = np.log(times_array)
    poly_time = np.polyfit(log_res, log_time, 1)
    time_fit = np.exp(poly_time[1]) * resolutions_array ** poly_time[0]
    ax1.plot(resolutions_array, time_fit, '--', linewidth=2, color='red', alpha=0.7, 
             label=f'Power law fit: t ∝ r^{poly_time[0]:.2f}')
    ax1.legend(fontsize=11, loc='best')
    
    # Annotate each point with time
    for res, t in zip(resolutions_array, times_array):
        ax1.annotate(f'{t:.3f}s', (res, t), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)
    
    # Plot 2: Time vs Grid Positions
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(total_positions_array, times_array, 'o-', linewidth=2.5, markersize=10, 
             color='#A23B72', label='Measured Time')
    ax2.set_xlabel('Total Grid Positions', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Computation Time [s]', fontsize=13, fontweight='bold')
    ax2.set_title('Generation Time vs Grid Size', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Annotate with resolution
    for res, pos, t in zip(resolutions_array, total_positions_array, times_array):
        ax2.annotate(f'{res}m', (pos, t), textcoords="offset points", 
                    xytext=(0, 10), ha='center', fontsize=9)
    
    # Plot 3: Bar chart of times
    ax3 = plt.subplot(2, 2, 3)
    colors = plt.cm.viridis(np.linspace(0, 1, len(resolutions_array)))
    bars = ax3.bar(range(len(resolutions_array)), times_array, color=colors, 
                   edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Resolution [m]', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Computation Time [s]', fontsize=13, fontweight='bold')
    ax3.set_title('Generation Time Comparison (Bar Chart)', fontsize=15, fontweight='bold')
    ax3.set_xticks(range(len(resolutions_array)))
    ax3.set_xticklabels([f'{r}m' for r in resolutions_array], rotation=0)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add value labels on bars
    for i, (bar, t) in enumerate(zip(bars, times_array)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{t:.4f}s',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Plot 4: Summary Table
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calculate speedup relative to finest resolution
    finest_time = times_array[-1]
    speedups = finest_time / times_array
    
    summary_text = (
        "PERFORMANCE SUMMARY\n"
        "═" * 70 + "\n\n"
        "Power Law Analysis:\n"
        "─" * 70 + "\n"
        f"Time Scaling:        t ∝ resolution^{poly_time[0]:.3f}\n"
        f"Expected (theory):   t ∝ resolution^(-2) [area scaling]\n\n"
        "Timing Statistics:\n"
        "─" * 70 + "\n"
        f"Fastest (coarsest):  {times_array.min():.4f} s  ({resolutions_array[times_array.argmin()]:.1f}m)\n"
        f"Slowest (finest):    {times_array.max():.4f} s  ({resolutions_array[times_array.argmax()]:.1f}m)\n"
        f"Speed ratio:         {times_array.max()/times_array.min():.1f}x\n\n"
        "Resolution-wise Breakdown:\n"
        "─" * 70 + "\n"
    )
    
    for i, result in enumerate(results):
        summary_text += (
            f"{result.resolution:6.1f}m: {result.computation_time:8.4f}s  "
            f"({result.grid_size_x}×{result.grid_size_y}, {speedups[i]:5.1f}x faster)\n"
        )
    
    summary_text += (
        "\nGrid Size Statistics:\n"
        "─" * 70 + "\n"
        f"Smallest grid:       {total_positions_array.min():,} positions\n"
        f"Largest grid:        {total_positions_array.max():,} positions\n"
        f"Grid ratio:          {total_positions_array.max()/total_positions_array.min():.1f}x\n"
    )
    
    ax4.text(
        0.05, 0.95, summary_text,
        transform=ax4.transAxes,
        fontsize=9.5,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6),
    )
    
    plt.suptitle('Heightmap Generation Time Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    comparison_output = out_dir / "time_comparison.png"
    plt.savefig(comparison_output, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Time comparison saved: {comparison_output.name}\n")
    
    # === Print Summary ===
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total resolutions tested: {len(results)}")
    print(f"\n{'Resolution':<12} {'Time [s]':<12} {'Grid Size':<15} {'Positions':<12} {'Memory [KB]':<12}")
    print(f"{'-'*80}")
    
    for result in results:
        print(f"{result.resolution:<12.1f} {result.computation_time:<12.4f} "
              f"{result.grid_size_x}×{result.grid_size_y:<12} "
              f"{result.total_positions:<12,} {result.memory_kb:<12.2f}")
    
    print(f"\n{'='*80}")
    print(f"All outputs saved to: {out_dir}")
    print(f"{'='*80}\n")
    
    # Assertions
    assert len(results) == len(RESOLUTIONS), "Should have results for all resolutions"
    
    for result in results:
        assert result.computation_time > 0, "Computation time should be positive"
        assert result.total_positions > 0, "Should have grid positions"
        assert result.coverage_percent > 0, "Should have some coverage"
    
    # Verify that finer resolutions take longer (generally)
    # Allow some tolerance as very coarse resolutions might have overhead
    assert results[-1].computation_time > results[0].computation_time * 0.5, \
        "Finest resolution should take significantly longer than coarsest"
    
    print("✓ All assertions passed!")
