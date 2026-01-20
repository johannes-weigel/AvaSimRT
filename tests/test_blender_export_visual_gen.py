"""
Visual test to verify Blender export produces identical geometry.

This test exports a .blend file and compares the resulting mesh with the original
to verify that the export process preserves geometry correctly.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import trimesh

from avasimrt.preprocessing.blender import run_blender_export


def test_blender_export_produces_identical_geometry(tmp_path: Path) -> None:
    """
    Export scene.blend and verify the exported OBJ matches the original geometry.
    
    This test:
    1. Loads the original scene.obj as reference
    2. Exports scene.blend to temporary files
    3. Loads the exported scene.obj
    4. Visually compares both meshes
    5. Verifies geometric properties match (vertex count, face count, bounds)
    """
    # Locate the example scene files
    examples_dir = Path(__file__).parent.parent / "examples" / "nordkette" / "assets"
    blend_file = examples_dir / "scene.blend"
    original_obj = examples_dir / "scene.obj"
    original_xml = examples_dir / "scene.xml"
    
    if not blend_file.exists():
        pytest.skip("scene.blend not found - requires Blender test data")
    
    # Load original mesh
    original_loaded = trimesh.load(original_obj)
    # Handle both single mesh and scene
    if isinstance(original_loaded, trimesh.Scene):
        # Get the first mesh from the scene
        original_mesh = list(original_loaded.geometry.values())[0]
    else:
        original_mesh = original_loaded
    
    # Export from Blender
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir = Path(tmpdir)
        exported_obj = tmp_dir / "exported.obj"
        exported_xml = tmp_dir / "exported.xml"
        exported_ply = tmp_dir / "exported.ply"
        
        try:
            run_blender_export(
                blend_path=blend_file,
                obj_output=exported_obj,
                xml_output=exported_xml,
                ply_output=exported_ply,
            )
        except FileNotFoundError:
            pytest.skip("Blender not installed or not in PATH")
        except RuntimeError as e:
            if "Mitsuba" in str(e):
                # Mitsuba addon not installed - that's OK for testing OBJ/PLY export
                print(f"\nNote: Mitsuba addon not available, skipping XML export")
                # Check if at least OBJ and PLY were exported
                if not exported_obj.exists():
                    pytest.skip("Blender export failed - no output files generated")
            else:
                raise
        
        # Load exported mesh
        exported_loaded = trimesh.load(exported_obj)
        # Handle both single mesh and scene
        if isinstance(exported_loaded, trimesh.Scene):
            # Get the first mesh from the scene
            exported_mesh = list(exported_loaded.geometry.values())[0]
        else:
            exported_mesh = exported_loaded
        
        # Create visual comparison
        fig = plt.figure(figsize=(16, 8))
        
        # Plot 1: Original mesh (top view)
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        vertices_orig = original_mesh.vertices
        ax1.plot_trisurf(
            vertices_orig[:, 0],
            vertices_orig[:, 1],
            vertices_orig[:, 2],
            triangles=original_mesh.faces,
            alpha=0.7,
            cmap='terrain',
        )
        ax1.set_title('Original Mesh (from scene.obj)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Plot 2: Exported mesh (top view)
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        vertices_exp = exported_mesh.vertices
        ax2.plot_trisurf(
            vertices_exp[:, 0],
            vertices_exp[:, 1],
            vertices_exp[:, 2],
            triangles=exported_mesh.faces,
            alpha=0.7,
            cmap='terrain',
        )
        ax2.set_title('Exported Mesh (from scene.blend)')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Plot 3: Overlay (both meshes together)
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        ax3.plot_trisurf(
            vertices_orig[:, 0],
            vertices_orig[:, 1],
            vertices_orig[:, 2],
            triangles=original_mesh.faces,
            alpha=0.4,
            color='blue',
            label='Original',
        )
        ax3.plot_trisurf(
            vertices_exp[:, 0],
            vertices_exp[:, 1],
            vertices_exp[:, 2],
            triangles=exported_mesh.faces,
            alpha=0.4,
            color='red',
            label='Exported',
        )
        ax3.set_title('Overlay Comparison')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        
        # Plot 4: Vertex count comparison (bar chart)
        ax4 = fig.add_subplot(2, 3, 4)
        counts = [len(original_mesh.vertices), len(exported_mesh.vertices)]
        ax4.bar(['Original', 'Exported'], counts, color=['blue', 'red'], alpha=0.7)
        ax4.set_ylabel('Vertex Count')
        ax4.set_title('Vertex Count Comparison')
        ax4.grid(True, alpha=0.3)
        for i, count in enumerate(counts):
            ax4.text(i, count, str(count), ha='center', va='bottom')
        
        # Plot 5: Face count comparison (bar chart)
        ax5 = fig.add_subplot(2, 3, 5)
        face_counts = [len(original_mesh.faces), len(exported_mesh.faces)]
        ax5.bar(['Original', 'Exported'], face_counts, color=['blue', 'red'], alpha=0.7)
        ax5.set_ylabel('Face Count')
        ax5.set_title('Face Count Comparison')
        ax5.grid(True, alpha=0.3)
        for i, count in enumerate(face_counts):
            ax5.text(i, count, str(count), ha='center', va='bottom')
        
        # Plot 6: Bounds comparison (box plot)
        ax6 = fig.add_subplot(2, 3, 6)
        bounds_orig = original_mesh.bounds
        bounds_exp = exported_mesh.bounds
        
        ax6.text(0.1, 0.9, 'Bounding Box Comparison', fontweight='bold', transform=ax6.transAxes)
        ax6.text(0.1, 0.75, f'Original:', fontweight='bold', color='blue', transform=ax6.transAxes)
        ax6.text(0.1, 0.65, f'  Min: [{bounds_orig[0][0]:.2f}, {bounds_orig[0][1]:.2f}, {bounds_orig[0][2]:.2f}]', 
                 color='blue', transform=ax6.transAxes, fontfamily='monospace')
        ax6.text(0.1, 0.55, f'  Max: [{bounds_orig[1][0]:.2f}, {bounds_orig[1][1]:.2f}, {bounds_orig[1][2]:.2f}]', 
                 color='blue', transform=ax6.transAxes, fontfamily='monospace')
        
        ax6.text(0.1, 0.40, f'Exported:', fontweight='bold', color='red', transform=ax6.transAxes)
        ax6.text(0.1, 0.30, f'  Min: [{bounds_exp[0][0]:.2f}, {bounds_exp[0][1]:.2f}, {bounds_exp[0][2]:.2f}]', 
                 color='red', transform=ax6.transAxes, fontfamily='monospace')
        ax6.text(0.1, 0.20, f'  Max: [{bounds_exp[1][0]:.2f}, {bounds_exp[1][1]:.2f}, {bounds_exp[1][2]:.2f}]', 
                 color='red', transform=ax6.transAxes, fontfamily='monospace')
        
        # Calculate differences
        min_diff = np.abs(bounds_orig[0] - bounds_exp[0])
        max_diff = np.abs(bounds_orig[1] - bounds_exp[1])
        
        ax6.text(0.1, 0.05, f'Max difference: {max(min_diff.max(), max_diff.max()):.6f}', 
                 fontweight='bold', transform=ax6.transAxes)
        ax6.axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = tmp_path / "blender_export_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n{'='*70}")
        print(f"Visual comparison saved to: {output_path}")
        print(f"{'='*70}")
        print(f"Original mesh:  {len(original_mesh.vertices):>6} vertices, {len(original_mesh.faces):>6} faces")
        print(f"Exported mesh:  {len(exported_mesh.vertices):>6} vertices, {len(exported_mesh.faces):>6} faces")
        print(f"{'='*70}")
        print(f"Bounds comparison:")
        print(f"  Original min: {bounds_orig[0]}")
        print(f"  Exported min: {bounds_exp[0]}")
        print(f"  Difference:   {min_diff}")
        print(f"  Original max: {bounds_orig[1]}")
        print(f"  Exported max: {bounds_exp[1]}")
        print(f"  Difference:   {max_diff}")
        print(f"{'='*70}")
        
        # Verify geometric properties are similar
        # Allow for small differences due to floating point precision and export/import
        vertex_count_diff = abs(len(original_mesh.vertices) - len(exported_mesh.vertices))
        face_count_diff = abs(len(original_mesh.faces) - len(exported_mesh.faces))
        
        # Vertex/face counts should be identical or very close
        assert vertex_count_diff <= 10, \
            f"Vertex count differs by {vertex_count_diff} (original: {len(original_mesh.vertices)}, exported: {len(exported_mesh.vertices)})"
        assert face_count_diff <= 10, \
            f"Face count differs by {face_count_diff} (original: {len(original_mesh.faces)}, exported: {len(exported_mesh.faces)})"
        
        # Bounds should be very similar (within 1% or 0.1 units)
        bounds_tolerance = max(0.1, np.max(np.abs(bounds_orig)) * 0.01)
        assert np.allclose(bounds_orig[0], bounds_exp[0], atol=bounds_tolerance), \
            f"Minimum bounds differ: {bounds_orig[0]} vs {bounds_exp[0]}"
        assert np.allclose(bounds_orig[1], bounds_exp[1], atol=bounds_tolerance), \
            f"Maximum bounds differ: {bounds_orig[1]} vs {bounds_exp[1]}"
        
        # Verify exported files exist
        assert exported_obj.exists(), "Exported OBJ file should exist"
        assert exported_ply.exists(), "Exported PLY file should exist"
        # XML may not exist if Mitsuba addon is not installed
        if exported_xml.exists():
            print(f"✓ XML file also exported successfully")
        else:
            print(f"⚠ XML file not exported (Mitsuba addon may not be installed)")
        
        print(f"\n✓ All geometric properties match within tolerance!")
        print(f"✓ Blender export preserves scene geometry correctly\n")
