# Blender Setup for AvaSimRT

This project requires Blender 4.2.17 with the Mitsuba plugin for scene export.

## Quick Setup

```bash
make setup
```

This will:

- Set up Python virtual environment
- Install project dependencies
- Download and install Blender 4.2.17
- Install Mitsuba Blender plugin
- Create/update `.env` with Blender path

## Manual Setup

If you prefer to install Blender manually:

1. Download Blender 4.2.17 from https://www.blender.org/download/lts/4-2/
2. Install Mitsuba plugin from https://github.com/mitsuba-renderer/mitsuba-blender
3. Create `.env` file with:
   ```
   AVASIMRT_BLENDER_CMD=/path/to/blender
   ```

## Individual Commands

```bash
# Install Blender + Mitsuba only
make setup-blender

# Clean Blender installation
make clean-blender

# Run tests (uses Blender from .env)
make test
```

## Verification

Test the setup:

```bash
source .venv/bin/activate
pytest tests/avasimrt/preprocessing/test_blender.py -v
```

## Supported Platforms

- ✅ Linux x86_64
- ⚠️ macOS (requires manual installation)
- ❌ Windows (requires manual installation)
