"""
Visual performance test for heightmap generation via raycasting.

This test generates heightmaps at multiple resolutions and creates
visualizations showing the quality, performance, and metadata for presentation purposes.

For each resolution, creates 3 separate files:
- heightmap_2d_<resolution>m.png: Top-down 2D heightmap view
- heightmap_3d_<resolution>m.png: 3D surface plot
- heightmap_profile_<resolution>m.png: Cross-section profile with metadata

Also creates comparison plots showing scaling behavior:
- comparison_performance.png: Resolution vs. Time/Memory/Coverage
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from avasimrt.preprocessing.heights import generate_heightmap


# EASY TO CHANGE: Define resolutions to test (in meters)
RESOLUTIONS = [200, 100, 50, 10, 1]


def test_heightmap_performance_visualization(tmp_path: Path) -> None:
    """
    Generate heightmaps at multiple resolutions and create performance visualizations.
    
    For each resolution:
    - Generates heightmap via raycasting
    - Creates 3 separate visualization files (2D, 3D, profile)
    - Records comprehensive metadata
    
    Creates comparison plots showing scaling behavior.
    
    Outputs:
    - heightmap_2d_<res>m.png: 2D top-down view
    - heightmap_3d_<res>m.png: 3D surface plot
    - heightmap_profile_<res>m.png: Cross-section with metadata
    - comparison_performance.png: Performance comparison plots
    - heightmap_performance_results.json: All metadata
    """
    # Locate the Nordkette scene OBJ file
    scene_obj = Path(__file__).parent.parent / "examples" / "nordkette" / "assets" / "scene.obj"
    
    if not scene_obj.exists():
        pytest.skip(f"Scene file not found: {scene_obj}")
    
    all_results = []
    
    print(f"\n{'='*80}")
    print(f"HEIGHTMAP PERFORMANCE TEST")
    print(f"Scene: {scene_obj}")
    print(f"Resolutions: {RESOLUTIONS}")
    print(f"{'='*80}\n")
    
    for resolution in RESOLUTIONS:
        print(f"Processing resolution: {resolution}m...")
        
        # Generate heightmap with complete metadata
        heightmap_3d, metadata = generate_heightmap(scene_obj, resolution)
        
        # Extract coordinate arrays and z-values
        x_coords = heightmap_3d[:, 0, 0]  # First column, all rows, x-coordinate
        y_coords = heightmap_3d[0, :, 1]  # First row, all columns, y-coordinate
        heightmap_z = heightmap_3d[:, :, 2]  # All z-values
        
        # Store result
        all_results.append(metadata.copy())
        
        # === 1. 2D Heightmap View ===
        fig, ax = plt.subplots(figsize=(12, 10))
        
        im = ax.imshow(
            heightmap_z.T,
            origin='lower',
            cmap='terrain',
            extent=[metadata['bounds']['x_min'], metadata['bounds']['x_max'], 
                   metadata['bounds']['y_min'], metadata['bounds']['y_max']],
            aspect='equal',
        )
        
        ax.set_xlabel('X [m]', fontsize=14)
        ax.set_ylabel('Y [m]', fontsize=14)
        ax.set_title(f'Heightmap (Top View) - Resolution: {resolution}m', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Elevation [m]', fontsize=13)
        
        plt.tight_layout()
        output_2d = tmp_path / f"heightmap_2d_{resolution}m.png"
        plt.savefig(output_2d, dpi=200, bbox_inches='tight')
        plt.close()
        
        # === 2. 3D Surface Plot ===
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
        
        ax_3d.set_xlabel('X [m]', fontsize=12)
        ax_3d.set_ylabel('Y [m]', fontsize=12)
        ax_3d.set_zlabel('Elevation [m]', fontsize=12)
        ax_3d.set_title(f'3D Surface View - Resolution: {resolution}m', fontsize=16, fontweight='bold')
        ax_3d.view_init(elev=25, azim=-45)
        
        fig.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=10, label='Elevation [m]')
        
        plt.tight_layout()
        output_3d = tmp_path / f"heightmap_3d_{resolution}m.png"
        plt.savefig(output_3d, dpi=200, bbox_inches='tight')
        plt.close()
        
        # === 3. Cross-Section Profile with Metadata ===
        fig = plt.figure(figsize=(16, 10))
        
        # Cross-section plot
        ax_profile = plt.subplot(2, 1, 1)
        mid_y_idx = metadata['grid_size']['n_y'] // 2
        z_values = heightmap_z[:, mid_y_idx]
        
        valid_mask = ~np.isnan(z_values)
        ax_profile.plot(
            x_coords[valid_mask],
            z_values[valid_mask],
            'o-',
            linewidth=2,
            markersize=5,
            color='darkgreen',
            label=f'Y = {y_coords[mid_y_idx]:.1f}m',
        )
        
        ax_profile.set_xlabel('X [m]', fontsize=13)
        ax_profile.set_ylabel('Elevation [m]', fontsize=13)
        ax_profile.set_title(f'Cross-Section Profile - Resolution: {resolution}m', fontsize=16, fontweight='bold')
        ax_profile.grid(True, alpha=0.3)
        ax_profile.legend(fontsize=12)
        
        # Metadata panel
        ax_meta = plt.subplot(2, 1, 2)
        ax_meta.axis('off')
        
        meta_text = (
            f"PERFORMANCE METRICS\n"
            f"{'═'*70}\n"
            f"Resolution:                {resolution} m\n"
            f"Computation Time:          {metadata['computation_time_s']:.4f} s\n"
            f"\n"
            f"GRID INFORMATION\n"
            f"{'═'*70}\n"
            f"Grid Size:                 {metadata['grid_size']['n_x']} × {metadata['grid_size']['n_y']}\n"
            f"Total Positions:           {metadata['grid_size']['total_positions']:,}\n"
            f"Valid Positions:           {metadata['coverage']['valid_positions']:,}\n"
            f"Coverage:                  {metadata['coverage']['coverage_percent']:.2f}%\n"
            f"\n"
            f"MEMORY USAGE\n"
            f"{'═'*70}\n"
            f"Array Size:                {metadata['memory']['kilobytes']:.2f} KB ({metadata['memory']['megabytes']:.4f} MB)\n"
            f"Total Bytes:               {metadata['memory']['bytes']:,}\n"
            f"\n"
            f"ELEVATION STATISTICS\n"
            f"{'═'*70}\n"
            f"Minimum Elevation:         {metadata['heightmap_stats']['z_min']:.2f} m\n"
            f"Maximum Elevation:         {metadata['heightmap_stats']['z_max']:.2f} m\n"
            f"Elevation Range:           {metadata['heightmap_stats']['z_range']:.2f} m\n"
            f"\n"
            f"SCENE BOUNDS\n"
            f"{'═'*70}\n"
            f"X Range:                   [{metadata['bounds']['x_min']:.2f}, {metadata['bounds']['x_max']:.2f}] m\n"
            f"Y Range:                   [{metadata['bounds']['y_min']:.2f}, {metadata['bounds']['y_max']:.2f}] m\n"
        )
        
        ax_meta.text(
            0.05, 0.95, meta_text,
            transform=ax_meta.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
        )
        
        plt.tight_layout()
        output_profile = tmp_path / f"heightmap_profile_{resolution}m.png"
        plt.savefig(output_profile, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 2D view:     {output_2d.name}")
        print(f"  ✓ 3D surface:  {output_3d.name}")
        print(f"  ✓ Profile:     {output_profile.name}")
        print(f"    Time: {metadata['computation_time_s']:.4f}s | Grid: {metadata['grid_size']['n_x']}×{metadata['grid_size']['n_y']} | Coverage: {metadata['coverage']['coverage_percent']:.1f}%\n")
    
    # === 4. Comparison/Scaling Analysis ===
    print(f"{'='*80}")
    print(f"Creating performance comparison plots...")
    print(f"{'='*80}\n")
    
    # Extract data for comparison
    resolutions = np.array([r['resolution_m'] for r in all_results])
    times = np.array([r['computation_time_s'] for r in all_results])
    total_positions = np.array([r['grid_size']['total_positions'] for r in all_results])
    memory_kb = np.array([r['memory']['kilobytes'] for r in all_results])
    coverage = np.array([r['coverage']['coverage_percent'] for r in all_results])
    
    # Create comprehensive comparison figure
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Computation Time vs Resolution
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(resolutions, times, 'o-', linewidth=2, markersize=8, color='#2E86AB', label='Measured')
    ax1.set_xlabel('Resolution [m]', fontsize=12)
    ax1.set_ylabel('Computation Time [s]', fontsize=12)
    ax1.set_title('Computation Time vs Resolution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Fit power law: time ∝ resolution^-α
    log_res = np.log(resolutions)
    log_time = np.log(times)
    poly_time = np.polyfit(log_res, log_time, 1)
    time_fit = np.exp(poly_time[1]) * resolutions ** poly_time[0]
    ax1.plot(resolutions, time_fit, '--', linewidth=2, color='red', alpha=0.7, 
             label=f'Fit: t ∝ r^{poly_time[0]:.2f}')
    ax1.legend(fontsize=10)
    
    # Plot 2: Memory Usage vs Resolution
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(resolutions, memory_kb, 'o-', linewidth=2, markersize=8, color='#A23B72', label='Measured')
    ax2.set_xlabel('Resolution [m]', fontsize=12)
    ax2.set_ylabel('Memory Usage [KB]', fontsize=12)
    ax2.set_title('Memory Usage vs Resolution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    
    # Fit power law: memory ∝ resolution^-β
    log_mem = np.log(memory_kb)
    poly_mem = np.polyfit(log_res, log_mem, 1)
    mem_fit = np.exp(poly_mem[1]) * resolutions ** poly_mem[0]
    ax2.plot(resolutions, mem_fit, '--', linewidth=2, color='red', alpha=0.7,
             label=f'Fit: mem ∝ r^{poly_mem[0]:.2f}')
    ax2.legend(fontsize=10)
    
    # Plot 3: Coverage vs Resolution
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(resolutions, coverage, 'o-', linewidth=2, markersize=8, color='#F18F01')
    ax3.set_xlabel('Resolution [m]', fontsize=12)
    ax3.set_ylabel('Coverage [%]', fontsize=12)
    ax3.set_title('Coverage vs Resolution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_ylim([0, 105])
    
    # Plot 4: Grid Positions vs Resolution
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(resolutions, total_positions, 'o-', linewidth=2, markersize=8, color='#6A994E', label='Measured')
    ax4.set_xlabel('Resolution [m]', fontsize=12)
    ax4.set_ylabel('Total Grid Positions', fontsize=12)
    ax4.set_title('Grid Size vs Resolution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    # Fit power law: positions ∝ resolution^-γ
    log_pos = np.log(total_positions)
    poly_pos = np.polyfit(log_res, log_pos, 1)
    pos_fit = np.exp(poly_pos[1]) * resolutions ** poly_pos[0]
    ax4.plot(resolutions, pos_fit, '--', linewidth=2, color='red', alpha=0.7,
             label=f'Fit: N ∝ r^{poly_pos[0]:.2f}')
    ax4.legend(fontsize=10)
    
    # Plot 5: Time per Position
    ax5 = plt.subplot(2, 3, 5)
    time_per_pos = (times / total_positions) * 1e6  # Convert to microseconds
    ax5.plot(resolutions, time_per_pos, 'o-', linewidth=2, markersize=8, color='#BC4B51')
    ax5.set_xlabel('Resolution [m]', fontsize=12)
    ax5.set_ylabel('Time per Position [μs]', fontsize=12)
    ax5.set_title('Efficiency: Time per Grid Position', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    
    # Plot 6: Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = (
        "SCALING ANALYSIS SUMMARY\n"
        "═" * 60 + "\n\n"
        "Power Law Fits (log-log regression):\n"
        "─" * 60 + "\n"
        f"Computation Time:    t ∝ resolution^{poly_time[0]:.3f}\n"
        f"Memory Usage:        mem ∝ resolution^{poly_mem[0]:.3f}\n"
        f"Grid Positions:      N ∝ resolution^{poly_pos[0]:.3f}\n\n"
        "Expected theoretical scaling:\n"
        "─" * 60 + "\n"
        f"Grid positions:      N ∝ r^(-2)  (area scaling)\n"
        f"Memory usage:        mem ∝ r^(-2)  (proportional to N)\n"
        f"Computation time:    t ∝ r^(-2) to r^(-2.5)\n\n"
        "Performance Overview:\n"
        "─" * 60 + "\n"
        f"Fastest run:         {times.min():.4f} s  ({resolutions[times.argmin()]:.0f}m)\n"
        f"Slowest run:         {times.max():.4f} s  ({resolutions[times.argmax()]:.0f}m)\n"
        f"Speed ratio:         {times.max()/times.min():.1f}x\n\n"
        f"Smallest grid:       {total_positions.min():,} positions\n"
        f"Largest grid:        {total_positions.max():,} positions\n"
        f"Grid ratio:          {total_positions.max()/total_positions.min():.1f}x\n\n"
        f"Best coverage:       {coverage.max():.2f}%\n"
        f"Worst coverage:      {coverage.min():.2f}%\n"
    )
    
    ax6.text(
        0.05, 0.95, summary_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4),
    )
    
    plt.suptitle('Heightmap Performance Scaling Analysis', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    comparison_output = tmp_path / "comparison_performance.png"
    plt.savefig(comparison_output, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Comparison plots: {comparison_output.name}\n")
    
    # Save combined metadata JSON
    combined_metadata = {
        'test_info': {
            'scene_file': str(scene_obj),
            'test_date': '2026-01-20',
            'resolutions_tested': RESOLUTIONS,
        },
        'results': all_results,
    }
    
    json_output_path = tmp_path / "heightmap_performance_results.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(combined_metadata, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total runs: {len(all_results)}")
    print(f"Resolutions tested: {', '.join(f'{r}m' for r in RESOLUTIONS)}")
    print(f"\nPerformance comparison:")
    print(f"{'Resolution':<12} {'Time':<10} {'Grid Size':<15} {'Positions':<12} {'Size':<10}")
    print(f"{'-'*80}")
    
    for result in all_results:
        res = result['resolution_m']
        time_s = result['computation_time_s']
        grid = f"{result['grid_size']['n_x']}×{result['grid_size']['n_y']}"
        positions = result['grid_size']['total_positions']
        size_kb = result['memory']['kilobytes']
        
        print(f"{res:<12} {time_s:<10.4f} {grid:<15} {positions:<12,} {size_kb:<10.2f} KB")
    
    print(f"\n{'='*80}")
    print(f"All visualizations saved to: {tmp_path}")
    print(f"Combined metadata saved to: {json_output_path}")
    print(f"{'='*80}\n")
    
    # Assertions
    assert len(all_results) == len(RESOLUTIONS), "Should have results for all resolutions"
    
    for result in all_results:
        assert result['computation_time_s'] > 0, "Computation time should be positive"
        assert result['grid_size']['total_positions'] > 0, "Should have grid positions"
        assert result['coverage']['coverage_percent'] > 0, "Should have some coverage"
        
    # Verify that finer resolutions have more positions
    for i in range(len(all_results) - 1):
        coarser = all_results[i]
        finer = all_results[i + 1]
        assert finer['grid_size']['total_positions'] > coarser['grid_size']['total_positions'], \
            f"Finer resolution should have more positions: {finer['resolution_m']}m vs {coarser['resolution_m']}m"
    
    print("✓ All assertions passed!")
