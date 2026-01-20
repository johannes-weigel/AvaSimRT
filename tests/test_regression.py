from __future__ import annotations

import subprocess
from pathlib import Path
import filecmp
import pytest


PROJECT_ROOT = Path(__file__).parent.parent
EXAMPLE_DIR = PROJECT_ROOT / "examples" / "nordkette"
ASSETS_DIR = EXAMPLE_DIR / "assets"


def run_avasimrt(config_path: Path) -> Path:
    """Run avasimrt and return the output directory path."""
    result = subprocess.run(
        ["avasimrt", "--config", str(config_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    output_dir = Path(result.stdout.strip())
    
    result_file = output_dir / "result.json"
    if not result_file.exists():
        raise FileNotFoundError(f"result.json not found in {output_dir}")
    
    return output_dir


def assert_file_unchanged(actual_path: Path, expected_path: Path, filename: str) -> None:
    """Assert that generated file matches the reference file."""

    actual_file = actual_path / filename
    if not actual_file.exists():
        pytest.fail(f"Actual file does not exist: {actual_file}")

    expected_file = expected_path / filename
    if not expected_file.exists():
        pytest.fail(f"Expected files does not exist: {expected_file}")
    
    if not filecmp.cmp(expected_file, actual_file, shallow=False):
        pytest.fail(
            f"Generated file differs from reference:\n"
            f"  Generated: {expected_file}\n"
            f"  Reference: {actual_file}\n"
            f"Run 'git diff --no-index {expected_file} {actual_file}' to see differences."
        )


class TestExampleRegression:
    """Regression tests for nordkette example configurations."""

    def test_01_preprocess_full(self) -> None:
        """
        Test 01: Full preprocessing from Blender file.
        
        Generates: scene.obj, scene.xml, heightmap.npy from .blend file
        """
        
        config = EXAMPLE_DIR / "01-preprocess-full.yml"
        output_dir = run_avasimrt(config)
        
        assert_file_unchanged(output_dir, ASSETS_DIR, "scene.obj")
        assert_file_unchanged(output_dir, ASSETS_DIR, "scene.xml")
        

    def test_02_preprocess_heightmap(self) -> None:
        """
        Test 02: Heightmap generation from existing .obj/.xml files.
        
        Generates: heightmap.npy, heightmap_meta.json
        """

        config = EXAMPLE_DIR / "02-preprocess-heightmap.yml"
        output_dir = run_avasimrt(config)
        
        assert_file_unchanged(output_dir, ASSETS_DIR, "heightmap.npy")
        
        import numpy as np
        generated_heightmap = np.load(output_dir / "heightmap.npy")
        reference_heightmap = np.load(ASSETS_DIR / "heightmap.npy")
        
        np.testing.assert_array_equal(
            generated_heightmap,
            reference_heightmap,
            err_msg="Generated heightmap differs from reference"
        )
        

    def test_03_preprocess_positions(self) -> None:
        """
        Test 03: Resolve positions for nodes and anchors.
        
        Generates: positions_resolved.json
        """

        config = EXAMPLE_DIR / "03-preprocess-positions.yml"
        output_dir = run_avasimrt(config)
        
        assert_file_unchanged(output_dir, ASSETS_DIR, "positions_resolved.json")
        
