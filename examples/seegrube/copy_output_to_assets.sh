#!/bin/bash

# Copy output files to seegrube assets folder with optional suffix
# Usage: ./copy_output_to_assets.sh <output_dir> [suffix]
# Example: ./copy_output_to_assets.sh output/abc123 -0.5
#          ./copy_output_to_assets.sh output/abc123  # no suffix

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <output_dir> [suffix]"
    echo "Example: $0 output/abc123 -0.5"
    exit 1
fi

OUTPUT_DIR="$1"
SUFFIX="${2:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASSETS_DIR="$SCRIPT_DIR/assets"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory '$OUTPUT_DIR' does not exist"
    exit 1
fi

echo "Copying from: $OUTPUT_DIR"
echo "Copying to: $ASSETS_DIR"
echo "Suffix: ${SUFFIX:-<none>}"
echo ""

# Copy heightmap.npy
if [ -f "$OUTPUT_DIR/heightmap.npy" ]; then
    cp "$OUTPUT_DIR/heightmap.npy" "$ASSETS_DIR/heightmap${SUFFIX}.npy"
    echo "Copied: heightmap${SUFFIX}.npy"
fi

# Copy heightmap_meta.json
if [ -f "$OUTPUT_DIR/heightmap_meta.json" ]; then
    cp "$OUTPUT_DIR/heightmap_meta.json" "$ASSETS_DIR/heightmap_meta${SUFFIX}.json"
    echo "Copied: heightmap_meta${SUFFIX}.json"
fi

# Copy scene.obj
if [ -f "$OUTPUT_DIR/scene.obj" ]; then
    cp "$OUTPUT_DIR/scene.obj" "$ASSETS_DIR/scene${SUFFIX}.obj"
    echo "Copied: scene${SUFFIX}.obj"
fi

# Copy scene.mtl
if [ -f "$OUTPUT_DIR/scene.mtl" ]; then
    cp "$OUTPUT_DIR/scene.mtl" "$ASSETS_DIR/scene${SUFFIX}.mtl"
    echo "Copied: scene${SUFFIX}.mtl"
fi

# Copy scene.ply
if [ -f "$OUTPUT_DIR/scene.ply" ]; then
    cp "$OUTPUT_DIR/scene.ply" "$ASSETS_DIR/scene${SUFFIX}.ply"
    echo "Copied: scene${SUFFIX}.ply"
fi

# Copy scene.xml
if [ -f "$OUTPUT_DIR/scene.xml" ]; then
    cp "$OUTPUT_DIR/scene.xml" "$ASSETS_DIR/scene${SUFFIX}.xml"
    echo "Copied: scene${SUFFIX}.xml"
fi

# Copy positions_resolved.json
if [ -f "$OUTPUT_DIR/positions_resolved.json" ]; then
    cp "$OUTPUT_DIR/positions_resolved.json" "$ASSETS_DIR/positions_resolved${SUFFIX}.json"
    echo "Copied: positions_resolved${SUFFIX}.json"
fi

# Copy meshes directory
if [ -d "$OUTPUT_DIR/meshes" ]; then
    mkdir -p "$ASSETS_DIR/meshes${SUFFIX}"
    cp -r "$OUTPUT_DIR/meshes/"* "$ASSETS_DIR/meshes${SUFFIX}/"
    echo "Copied: meshes${SUFFIX}/"
fi

echo ""
echo "Done!"
