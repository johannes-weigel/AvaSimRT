"""
Visual tests for heightmap generation.
These tests generate PNG outputs that can be manually inspected to verify correctness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pytest

from avasimrt.preprocessing.heights import generate_heightmap


def test_heightmap_flat_plane(tmp_path: Path) -> None:
    """Test heightmap generation on a flat horizontal plane."""
    scene_obj = tmp_path / "flat.obj"
    scene_obj.write_text(
        "v -10 -10 2\n"
        "v 10 -10 2\n"
        "v 10 10 2\n"
        "v -10 10 2\n"
        "f 1 2 3\n"
        "f 1 3 4\n",
        encoding="utf-8"
    )
    
    heightmap, metadata = generate_heightmap(scene_obj, resolution=0.1)
    
    # All heights should be approximately 2.0
    valid_mask = ~np.isnan(heightmap)
    assert np.any(valid_mask), "Heightmap should have valid data"
    
    valid_heights = heightmap[valid_mask]
    assert np.allclose(valid_heights, 2.0, atol=0.01), "Flat plane should have uniform height"
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    x_coords = np.array(metadata['x_coords'])
    y_coords = np.array(metadata['y_coords'])
    im = ax.imshow(
        heightmap.T,
        origin='lower',
        extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
        cmap='terrain'
    )
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'Flat Plane at z=2.0 (resolution={metadata["resolution"]}m)')
    plt.colorbar(im, ax=ax, label='Height (m)')
    
    output = tmp_path / "flat_plane_heightmap.png"
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    
    assert output.exists()
    print(f"\n✓ Flat plane visualization: {output}")


def test_heightmap_pyramid(tmp_path: Path) -> None:
    """Test heightmap generation on a pyramid/cone shape."""
    scene_obj = tmp_path / "pyramid.obj"
    
    # Create a simple pyramid with apex at (0, 0, 5)
    vertices = [
        "v -5 -5 0",   # Base corner 1
        "v 5 -5 0",    # Base corner 2
        "v 5 5 0",     # Base corner 3
        "v -5 5 0",    # Base corner 4
        "v 0 0 5",     # Apex
    ]
    faces = [
        "f 1 2 5",     # Side 1
        "f 2 3 5",     # Side 2
        "f 3 4 5",     # Side 3
        "f 4 1 5",     # Side 4
        "f 1 4 3",     # Base 1
        "f 1 3 2",     # Base 2
    ]
    scene_obj.write_text("\n".join(vertices + faces) + "\n", encoding="utf-8")
    
    heightmap, metadata = generate_heightmap(scene_obj, resolution=0.1)
    
    # Peak should be around z=5
    assert metadata['z_max'] is not None
    assert metadata['z_max'] > 4.5, "Pyramid peak should be near z=5"
    
    # Create visualization
    fig = plt.figure(figsize=(14, 5))
    
    x_coords = np.array(metadata['x_coords'])
    y_coords = np.array(metadata['y_coords'])
    
    # 2D heatmap
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(
        heightmap.T,
        origin='lower',
        extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
        cmap='terrain'
    )
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Pyramid Heightmap (Top View)')
    plt.colorbar(im1, ax=ax1, label='Height (m)')
    
    # 3D surface
    ax2 = fig.add_subplot(132, projection='3d')
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    ax2.plot_surface(X, Y, heightmap, cmap='terrain', alpha=0.9)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Height (m)')
    ax2.set_title('Pyramid (3D View)')
    ax2.view_init(elev=30, azim=45)
    
    # Contour plot
    ax3 = fig.add_subplot(133)
    contour = ax3.contour(x_coords, y_coords, heightmap.T, levels=10, cmap='terrain')
    ax3.clabel(contour, inline=True, fontsize=8)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Pyramid Contour Lines')
    ax3.set_aspect('equal')
    
    output = tmp_path / "pyramid_heightmap.png"
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    
    assert output.exists()
    print(f"\n✓ Pyramid visualization: {output}")
    print(f"  Height range: [{metadata['z_min']:.2f}, {metadata['z_max']:.2f}]")


def test_heightmap_slope_verification(tmp_path: Path) -> None:
    """
    Test heightmap on a linearly sloped terrain and verify values match expected gradient.
    This is the most rigorous test for accuracy.
    """
    scene_obj = tmp_path / "slope.obj"
    
    # Create a plane that slopes linearly from z=0 to z=10 as x goes from -5 to 5
    # Slope = 1.0 (1m vertical per 1m horizontal)
    vertices = [
        "v -5 -5 0",
        "v 5 -5 10",
        "v 5 5 10",
        "v -5 5 0",
    ]
    faces = ["f 1 2 3", "f 1 3 4"]
    scene_obj.write_text("\n".join(vertices + faces) + "\n", encoding="utf-8")
    
    heightmap, metadata = generate_heightmap(scene_obj, resolution=0.25)
    
    x_coords = np.array(metadata['x_coords'])
    y_coords = np.array(metadata['y_coords'])
    
    # Verify slope
    valid_mask = ~np.isnan(heightmap)
    errors = []
    
    for i in range(heightmap.shape[0]):
        for j in range(heightmap.shape[1]):
            if valid_mask[i, j]:
                x_val = x_coords[i]
                z_val = heightmap[i, j]
                # Expected: z = (x + 5) * 1.0 (slope of 1)
                expected_z = (x_val + 5)
                error = abs(z_val - expected_z)
                errors.append(error)
    
    errors = np.array(errors)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    # Create visualization with error analysis
    fig = plt.figure(figsize=(16, 10))
    
    # Heightmap
    ax1 = fig.add_subplot(221)
    im1 = ax1.imshow(heightmap.T, origin='lower', cmap='terrain',
                     extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Computed Heightmap')
    plt.colorbar(im1, ax=ax1, label='Height (m)')
    
    # Expected heightmap
    ax2 = fig.add_subplot(222)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    expected = X + 5
    im2 = ax2.imshow(expected.T, origin='lower', cmap='terrain',
                     extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Expected Heightmap (Analytical)')
    plt.colorbar(im2, ax=ax2, label='Height (m)')
    
    # Error map
    ax3 = fig.add_subplot(223)
    error_map = np.abs(heightmap - expected)
    im3 = ax3.imshow(error_map.T, origin='lower', cmap='hot',
                     extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title(f'Absolute Error (mean={mean_error:.4f}m, max={max_error:.4f}m)')
    plt.colorbar(im3, ax=ax3, label='Error (m)')
    
    # Cross-section
    ax4 = fig.add_subplot(224)
    mid_y_idx = heightmap.shape[1] // 2
    ax4.plot(x_coords, heightmap[:, mid_y_idx], 'o-', label='Computed', markersize=4)
    ax4.plot(x_coords, x_coords + 5, '--', label='Expected (z = x + 5)', linewidth=2)
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Height (m)')
    ax4.set_title('Cross-section at mid-Y')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    output = tmp_path / "slope_verification.png"
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    
    assert output.exists()
    print(f"\n✓ Slope verification visualization: {output}")
    print(f"  Mean error: {mean_error:.4f}m")
    print(f"  Max error: {max_error:.4f}m")
    print(f"  RMS error: {np.sqrt(np.mean(errors**2)):.4f}m")
    
    # Assert accuracy
    assert mean_error < 0.1, f"Mean error too high: {mean_error:.4f}m"
    assert max_error < 0.5, f"Max error too high: {max_error:.4f}m"


def test_heightmap_multiple_resolutions(tmp_path: Path) -> None:
    """Test heightmap generation at different resolutions."""
    scene_obj = tmp_path / "terrain.obj"
    
    # Create a simple hilly terrain
    vertices = [
        "v -10 -10 0",
        "v 10 -10 3",
        "v 10 10 0",
        "v -10 10 3",
        "v 0 0 5",  # Peak in middle
    ]
    faces = [
        "f 1 2 5",
        "f 2 3 5",
        "f 3 4 5",
        "f 4 1 5",
    ]
    scene_obj.write_text("\n".join(vertices + faces) + "\n", encoding="utf-8")
    
    resolutions = [2.0, 1.0, 0.5]
    fig, axes = plt.subplots(1, len(resolutions), figsize=(15, 4))
    
    for idx, res in enumerate(resolutions):
        heightmap, metadata = generate_heightmap(scene_obj, resolution=res)
        
        x_coords = np.array(metadata['x_coords'])
        y_coords = np.array(metadata['y_coords'])
        
        ax = axes[idx]
        im = ax.imshow(heightmap.T, origin='lower', cmap='terrain',
                       extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Resolution: {res}m\n({heightmap.shape[0]}x{heightmap.shape[1]} grid)')
        plt.colorbar(im, ax=ax, label='Height (m)')
    
    output = tmp_path / "resolution_comparison.png"
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    plt.close()
    
    assert output.exists()
    print(f"\n✓ Resolution comparison: {output}")
