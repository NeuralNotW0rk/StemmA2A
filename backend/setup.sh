#!/usr/bin/env bash
# macOS counterpart of setup.ps1. Not yet tested on real hardware.
# Usage: bash backend/setup.sh   (set PYTHON=/path/to/python3.10 to override detection)
#
# Written for the bash 3.2 that ships with macOS: no associative arrays, no mapfile.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

# 0. Platform check
if [ "$(uname -s)" != "Darwin" ]; then
    fail "This script targets macOS. On Windows, run 'npm.cmd run backend:setup'."
fi
if [ "$(uname -m)" != "arm64" ]; then
    # PyTorch stopped publishing Intel macOS wheels after 2.2, and setup pins 2.7.1.
    fail "Apple Silicon (arm64) is required. torch==2.7.1 has no Intel macOS build."
fi

# 1. Check for Python 3.10
echo "Step 1: Checking for Python 3.10..."
PYTHON="${PYTHON:-python3.10}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    fail "Python 3.10 not found. Install it (for example 'brew install python@3.10') and ensure 'python3.10' works, or set PYTHON=/path/to/python3.10."
fi
if ! "$PYTHON" -c "import sys; assert sys.version_info[:2] == (3, 10); print('Found compatible Python version: ' + sys.version)"; then
    fail "'$PYTHON' is not Python 3.10."
fi
echo "Python check passed."

# 2. Create Virtual Environment
VENV_PATH="$PROJECT_ROOT/backend/.venv"
if [ ! -f "$VENV_PATH/pyvenv.cfg" ]; then
    echo "Step 2: Creating Python virtual environment..."
    "$PYTHON" -m venv "$VENV_PATH" || fail "Failed to create virtual environment."
    echo "Virtual environment created at $VENV_PATH"
else
    echo "Step 2: Virtual environment already exists."
fi

# Define path to venv python executable
PYTHON_EXE="$VENV_PATH/bin/python"
if [ ! -x "$PYTHON_EXE" ]; then
    # A venv copied from Windows has Scripts/python.exe instead of bin/python.
    fail "$PYTHON_EXE not found. If backend/.venv was created on another OS, delete it and rerun."
fi

# 3. Upgrade pip/wheel
echo "Step 3: Upgrading pip and wheel..."
"$PYTHON_EXE" -m pip install --upgrade pip wheel || fail "Failed to upgrade pip and wheel."

# 4. No CUDA on macOS; the default PyPI wheels are the CPU/MPS build.
echo "Step 4: macOS detected, skipping CUDA detection."

# 5. Install PyTorch
echo "Step 5: Installing PyTorch..."
"$PYTHON_EXE" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --default-timeout=1000 --retries 20 \
    || fail "Failed to install PyTorch."
echo "PyTorch installed successfully."

# 6. Install other dependencies
echo "Step 6: Installing other dependencies from requirements.txt..."
"$PYTHON_EXE" -m pip install -r "$PROJECT_ROOT/backend/requirements.txt" \
    || fail "Failed to install dependencies from requirements.txt."

# 6b. flash-attn is CUDA-only, so it is skipped on macOS.
# torchcodec loads FFmpeg's shared libraries at runtime instead of bundling them.
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "WARNING: ffmpeg not found. torchcodec needs it for audio decoding: 'brew install ffmpeg'." >&2
fi

# 7-8. Install sibling repos (if available)
install_sibling() {
    local name="$1"
    local dir="$PROJECT_ROOT/../$2"
    echo "Checking for $name locally..."
    if [ ! -d "$dir" ]; then
        echo "$name not found at ../$2. Skipping."
        return
    fi
    dir="$(cd "$dir" && pwd)"
    if [ ! -f "$dir/setup.py" ] && [ ! -f "$dir/pyproject.toml" ]; then
        echo "$name directory found, but no setup.py or pyproject.toml present. Skipping."
        return
    fi
    echo "Found $name at $dir. Installing in editable mode..."
    "$PYTHON_EXE" -m pip install -e "$dir" --config-settings editable_mode=compat \
        || fail "Failed to install $name."
    echo "$name installed successfully."
}

echo "Step 7: Installing Diffracture..."
install_sibling "Diffracture" "diffracture"
echo "Step 8: Installing NeutralSelection..."
install_sibling "NeutralSelection" "neutral-selection"

echo "Backend setup completed successfully!"
