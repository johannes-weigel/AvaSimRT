#!/bin/bash

# Run all seegrube test configurations and copy outputs to assets
# Usage: ./run_tests.sh [start_step]
#   start_step: 0 (default, all steps), 1 (skip blender), 2 (skip preprocessing), 3 (skip motion)

set -e

# Parse start step argument
START_STEP=${1:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TESTS_DIR="$SCRIPT_DIR/tests"
ASSETS_DIR="$SCRIPT_DIR/assets"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Running seegrube test configurations"
echo "Starting from step: $START_STEP"
echo "=========================================="
echo ""

# ===========================================
# Step 1: Run preprocessing-blender.yml
# ===========================================
if [ $START_STEP -le 0 ]; then
echo "=== PREPROCESSING: BLENDER ==="
echo ""

config="$TESTS_DIR/preprocessing-blender.yml"
if [ -f "$config" ]; then
    echo "Running: preprocessing-blender.yml"
    output=$(avasimrt --config "$config")
    echo "$output"

    # Extract output directory
    output_dir=$(echo "$output" | grep -oP 'output/[a-f0-9]+' | tail -1)

    if [ -n "$output_dir" ] && [ -d "$output_dir" ]; then
        echo ""
        echo "Copying obj, xml, meshes to assets..."

        # Copy scene files without suffix
        [ -f "$output_dir/scene.obj" ] && cp "$output_dir/scene.obj" "$ASSETS_DIR/scene.obj" && echo "  Copied: scene.obj"
        [ -f "$output_dir/scene.mtl" ] && cp "$output_dir/scene.mtl" "$ASSETS_DIR/scene.mtl" && echo "  Copied: scene.mtl"
        [ -f "$output_dir/scene.ply" ] && cp "$output_dir/scene.ply" "$ASSETS_DIR/scene.ply" && echo "  Copied: scene.ply"
        [ -f "$output_dir/scene.xml" ] && cp "$output_dir/scene.xml" "$ASSETS_DIR/scene.xml" && echo "  Copied: scene.xml"

        # Copy meshes directory without suffix
        if [ -d "$output_dir/meshes" ]; then
            mkdir -p "$ASSETS_DIR/meshes"
            cp -r "$output_dir/meshes/"* "$ASSETS_DIR/meshes/"
            echo "  Copied: meshes/"
        fi
    else
        echo "Warning: Could not find output directory for preprocessing-blender"
    fi
    echo ""
else
    echo "Warning: Config not found: $config"
fi
fi  # End Step 1

# ===========================================
# Step 2: Run heightmap preprocessing tests
# ===========================================
if [ $START_STEP -le 1 ]; then
echo "=== PREPROCESSING: HEIGHTMAPS ==="
echo ""

# Resolutions: low to high (coarse to fine)
PREPROCESSING_RESOLUTIONS="100 50 25 10 5 4 3 2 1 0.5"

for res in $PREPROCESSING_RESOLUTIONS; do
    config="$TESTS_DIR/preprocessing-${res}.yml"
    if [ -f "$config" ]; then
        echo "----------------------------------------"
        echo "Running: preprocessing-${res}.yml"
        echo "----------------------------------------"

        output=$(avasimrt --config "$config")
        echo "$output"

        # Extract output directory
        output_dir=$(echo "$output" | grep -oP 'output/[a-f0-9]+' | tail -1)

        if [ -n "$output_dir" ] && [ -d "$output_dir" ]; then
            echo ""
            echo "Copying heightmap files with suffix -${res}..."

            # Copy heightmap with suffix
            [ -f "$output_dir/heightmap.npy" ] && cp "$output_dir/heightmap.npy" "$ASSETS_DIR/heightmap-${res}.npy" && echo "  Copied: heightmap-${res}.npy"
            [ -f "$output_dir/heightmap_meta.json" ] && cp "$output_dir/heightmap_meta.json" "$ASSETS_DIR/heightmap_meta-${res}.json" && echo "  Copied: heightmap_meta-${res}.json"
        else
            echo "Warning: Could not find output directory for preprocessing-${res}"
        fi
        echo ""
    else
        echo "Warning: Config not found: $config"
    fi
done
fi  # End Step 2

# ===========================================
# Step 3: Run motion simulation tests
# ===========================================
if [ $START_STEP -le 2 ]; then
echo ""
echo "=== MOTION SIMULATIONS ==="
echo ""

# Motion resolutions to test
MOTION_RESOLUTIONS="0.2 0.5 1 2 5 10 20"

for res in $MOTION_RESOLUTIONS; do
    config="$TESTS_DIR/motion-${res}.yml"
    if [ -f "$config" ]; then
        echo "----------------------------------------"
        echo "Running: motion-${res}.yml"
        echo "----------------------------------------"

        output=$(avasimrt --config "$config")
        echo "$output"

        # Extract output directory
        output_dir=$(echo "$output" | grep -oP 'output/[a-f0-9]+' | tail -1)

        if [ -n "$output_dir" ] && [ -d "$output_dir" ]; then
            echo ""
            echo "Copying trajectories folder as trajectories-${res}..."

            # Copy trajectories folder with suffix
            if [ -d "$output_dir/trajectories" ]; then
                mkdir -p "$ASSETS_DIR/trajectories-${res}"
                cp -r "$output_dir/trajectories/"* "$ASSETS_DIR/trajectories-${res}/"
                echo "  Copied: trajectories-${res}/"
            else
                echo "  Warning: trajectories folder not found in $output_dir"
            fi
        else
            echo "Warning: Could not find output directory for motion-${res}"
        fi
        echo ""
    else
        echo "Warning: Config not found: $config"
    fi
done
fi  # End Step 3

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
