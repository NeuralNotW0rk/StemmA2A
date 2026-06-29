#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

# Check if the Diffracture directory is mounted and contains a Python package definition
if [ -d "/Diffracture" ] && { [ -f "/Diffracture/setup.py" ] || [ -f "/Diffracture/pyproject.toml" ]; }; then
    echo "Found Diffracture mount. Installing in editable mode..."
    pip install -e /Diffracture
fi

# Run CUDA and system diagnostics to check for common container bottleneck issues
python diagnose_gpu.py

# Execute the container's main command (CMD from Dockerfile or compose.yaml)
exec "$@"