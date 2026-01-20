#!/bin/bash
set -e

# Blender + Mitsuba Setup Script for AvaSimRT
# This script downloads and sets up Blender 4.2.15 with Mitsuba plugin

BLENDER_VERSION="4.2.15"
BLENDER_DIR="${HOME}/.local/blender-${BLENDER_VERSION}"
MITSUBA_PLUGIN_URL="https://github.com/mitsuba-renderer/mitsuba-blender/archive/refs/heads/master.zip"

echo "=========================================="
echo "Setting up Blender ${BLENDER_VERSION} + Mitsuba"
echo "=========================================="

# Detect OS and architecture
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

if [ "$OS" = "linux" ] && [ "$ARCH" = "x86_64" ]; then
    BLENDER_PLATFORM="linux-x64"
    BLENDER_URL="https://download.blender.org/release/Blender4.2/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
elif [ "$OS" = "darwin" ] && [ "$ARCH" = "arm64" ]; then
    BLENDER_PLATFORM="macos-arm64"
    BLENDER_URL="https://download.blender.org/release/Blender4.2/blender-${BLENDER_VERSION}-macos-arm64.dmg"
elif [ "$OS" = "darwin" ] && [ "$ARCH" = "x86_64" ]; then
    BLENDER_PLATFORM="macos-x64"
    BLENDER_URL="https://download.blender.org/release/Blender4.2/blender-${BLENDER_VERSION}-macos-x64.dmg"
else
    echo "❌ Unsupported OS/Architecture: $OS/$ARCH"
    echo "Please install Blender manually from: https://www.blender.org/download/"
    exit 1
fi

# Check if Blender is already installed
if [ -f "${BLENDER_DIR}/blender" ]; then
    echo "✓ Blender ${BLENDER_VERSION} already installed at ${BLENDER_DIR}"
else
    echo "📦 Downloading Blender ${BLENDER_VERSION} for ${BLENDER_PLATFORM}..."
    
    # Create temp directory
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    
    cd "$TEMP_DIR"
    
    if [ "$OS" = "linux" ]; then
        # Download and extract for Linux
        curl -L -o blender.tar.xz "$BLENDER_URL"
        echo "📂 Extracting Blender..."
        tar -xf blender.tar.xz
        
        # Move to installation directory
        mkdir -p "$(dirname "$BLENDER_DIR")"
        mv blender-${BLENDER_VERSION}-linux-x64 "$BLENDER_DIR"
        
        echo "✓ Blender installed to ${BLENDER_DIR}"
    else
        echo "⚠️  macOS detected - please install Blender manually:"
        echo "   1. Download from: $BLENDER_URL"
        echo "   2. Install to Applications"
        echo "   3. Set AVASIMRT_BLENDER_CMD=/Applications/Blender.app/Contents/MacOS/Blender"
        exit 1
    fi
fi

# Install Mitsuba plugin
echo ""
echo "📦 Installing Mitsuba Blender plugin..."

EXTENSIONS_DIR="${HOME}/.config/blender/${BLENDER_VERSION%.*}/extensions/user_default"
MITSUBA_PLUGIN_DIR="${EXTENSIONS_DIR}/mitsuba-blender"

if [ -d "$MITSUBA_PLUGIN_DIR" ]; then
    echo "✓ Mitsuba plugin already installed"
else
    mkdir -p "$EXTENSIONS_DIR"
    
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    
    cd "$TEMP_DIR"
    
    echo "   Downloading Mitsuba plugin..."
    curl -L -o mitsuba.zip "$MITSUBA_PLUGIN_URL"
    
    echo "   Extracting..."
    unzip -q mitsuba.zip
    
    mv mitsuba-blender-master "$MITSUBA_PLUGIN_DIR"
    
    echo "✓ Mitsuba plugin installed to ${MITSUBA_PLUGIN_DIR}"
fi

# Create .env file if it doesn't exist
ENV_FILE="$(pwd)/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo ""
    echo "📝 Creating .env file..."
    echo "AVASIMRT_BLENDER_CMD=${BLENDER_DIR}/blender" > "$ENV_FILE"
    echo "✓ Created ${ENV_FILE}"
else
    # Check if AVASIMRT_BLENDER_CMD is already set
    if ! grep -q "AVASIMRT_BLENDER_CMD" "$ENV_FILE"; then
        echo ""
        echo "📝 Adding AVASIMRT_BLENDER_CMD to .env..."
        echo "AVASIMRT_BLENDER_CMD=${BLENDER_DIR}/blender" >> "$ENV_FILE"
        echo "✓ Updated ${ENV_FILE}"
    else
        echo "✓ AVASIMRT_BLENDER_CMD already configured in .env"
    fi
fi

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo "Blender: ${BLENDER_DIR}/blender"
echo "Mitsuba: ${MITSUBA_PLUGIN_DIR}"
echo ""
echo "To use:"
echo "  source .venv/bin/activate"
echo "  python -m avasimrt --scene-blender examples/nordkette/assets/scene.blend ..."
echo ""
echo "To test:"
echo "  make test"
echo "=========================================="
