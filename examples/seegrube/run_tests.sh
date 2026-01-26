#!/bin/bash

# Run all seegrube test configurations and copy outputs to assets
# Preprocessing runs from low resolution (100m) to high resolution (0.5m)
# Usage: ./run_tests.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TESTS_DIR="$SCRIPT_DIR/tests"
COPY_SCRIPT="$SCRIPT_DIR/copy_output_to_assets.sh"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Running seegrube test configurations"
echo "=========================================="
echo ""

# Preprocessing resolutions: low to high (coarse to fine)
PREPROCESSING_RESOLUTIONS="100 50 25 10 5 4 3 2 1 0.5"

echo "=== PREPROCESSING TESTS ==="
echo ""

for res in $PREPROCESSING_RESOLUTIONS; do
    config="$TESTS_DIR/preprocessing-${res}.yml"
    if [ -f "$config" ]; then
        echo "----------------------------------------"
        echo "Running: preprocessing-${res}.yml"
        echo "----------------------------------------"

        output=$(avasimrt --config "$config")
        echo "$output"

        # Extract output directory from the output (last line typically contains path)
        output_dir=$(echo "$output" | grep -oP 'output/[a-f0-9]+' | tail -1)

        if [ -n "$output_dir" ] && [ -d "$output_dir" ]; then
            echo ""
            echo "Copying output to assets with suffix -${res}..."
            "$COPY_SCRIPT" "$output_dir" "-${res}"
        else
            echo "Warning: Could not find output directory for preprocessing-${res}"
        fi
        echo ""
    else
        echo "Warning: Config not found: $config"
    fi
done

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
