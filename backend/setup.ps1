[CmdletBinding()]
param (
    [string]$CudaVersion
)

$ErrorActionPreference = 'Stop'

# Get the directory of the script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

# 1. Check for Python 3.11
Write-Host "Step 1: Checking for Python 3.11..."
try {
    py -3.11 -c "import sys; print('Found compatible Python version: ' + sys.version)"
    Write-Host "Python check passed."
} catch {
    Write-Error "ERROR: Python 3.11 not found. Please install it and ensure 'py -3.11' command works."
    exit 1
}

# 2. Create Virtual Environment
$VenvPath = Join-Path $ProjectRoot "backend/.venv"
if (-not (Test-Path (Join-Path $VenvPath "pyvenv.cfg"))) {
    Write-Host "Step 2: Creating Python virtual environment..."
    try {
        py -3.11 -m venv $VenvPath
        Write-Host "Virtual environment created at $VenvPath"
    } catch {
        Write-Error "ERROR: Failed to create virtual environment."
        exit 1
    }
} else {
    Write-Host "Step 2: Virtual environment already exists."
}

# Define paths to venv executables
$PipExe = Join-Path $VenvPath "Scripts/pip.exe"
$PythonExe = Join-Path $VenvPath "Scripts/python.exe"

# 3. Upgrade pip/setuptools
Write-Host "Step 3: Upgrading pip, setuptools, and wheel..."
try {
    & $PythonExe -m pip install --upgrade pip setuptools wheel
} catch {
    Write-Error "ERROR: Failed to upgrade pip and other packaging tools."
    exit 1
}

# 4. Auto-detect CUDA version if not provided
if (-not $CudaVersion) {
    Write-Host "Step 4: CUDA version not specified, attempting to auto-detect..."
    try {
        $nvidiaSmiOutput = nvidia-smi
        $match = $nvidiaSmiOutput | Select-String -Pattern "CUDA Version: (\d+\.\d+)"
        if ($match) {
            $CudaVersion = $match.Matches[0].Groups[1].Value
            Write-Host "Auto-detected CUDA Version: $CudaVersion"
        } else {
            Write-Host "Could not determine CUDA version from nvidia-smi. Assuming CPU-only."
        }
    } catch {
        Write-Host "nvidia-smi command not found or failed to run. Assuming CPU-only."
    }
} else {
    Write-Host "Step 4: Using user-specified CUDA version: $CudaVersion"
}


# 5. Install PyTorch
Write-Host "Step 5: Installing PyTorch..."
$arguments = @("install", "torch", "torchvision", "torchaudio", "--default-timeout=1000", "--retries", "20", "--extra-index-url", "https://download.pytorch.org/whl/cpu")

if ($CudaVersion) {
    # Force cuda version 12.1 for now
    $CudaVersion = "12.1"
    $CudaVersionFormatted = "cu" + $CudaVersion.Replace(".", "")
    $IndexUrl = "https://download.pytorch.org/whl/$CudaVersionFormatted"
    # Overwrite arguments for CUDA
    $arguments = @("install", "torch", "torchvision", "torchaudio", "--default-timeout=1000", "--retries", "20", "--index-url", $IndexUrl)
    Write-Host "Preparing to install PyTorch for CUDA $CudaVersion..."
} else {
    Write-Host "Preparing to install CPU-optimized version of PyTorch..."
}

try {
    & $PythonExe -m pip $arguments
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed with exit code $LASTEXITCODE"
    }
    Write-Host "PyTorch installed successfully."
} catch {
    Write-Error "ERROR: Failed to install PyTorch. If using a CUDA version, please ensure the NVIDIA drivers and toolkit are correctly installed."
    Write-Error "PIP ERROR: $_"
    exit 1
}

# 6. Install other dependencies
Write-Host "Step 6: Installing other dependencies from requirements.txt..."
try {
    if ($CudaVersion) {
        & $PythonExe -m pip install -r (Join-Path $ProjectRoot "backend/requirements.txt") --extra-index-url $IndexUrl
    } else {
        & $PythonExe -m pip install -r (Join-Path $ProjectRoot "backend/requirements.txt") --extra-index-url "https://download.pytorch.org/whl/cpu"
    }
} catch {
    Write-Error "ERROR: Failed to install dependencies from requirements.txt."
    Write-Error "PIP ERROR: $_"
    exit 1
}


# 7. Install Diffracture (if available)
Write-Host "Step 7: Checking for Diffracture locally..."
$DiffracturePath = (Resolve-Path (Join-Path $ProjectRoot "../Diffracture") -ErrorAction SilentlyContinue).Path
if ($DiffracturePath) {
    $SetupPy = Join-Path $DiffracturePath "setup.py"
    $PyProject = Join-Path $DiffracturePath "pyproject.toml"
    if ((Test-Path $SetupPy) -or (Test-Path $PyProject)) {
        Write-Host "Found Diffracture at $DiffracturePath. Installing in editable mode..."
        try {
            if ($CudaVersion) {
                & $PythonExe -m pip install -e $DiffracturePath --extra-index-url $IndexUrl
            } else {
                & $PythonExe -m pip install -e $DiffracturePath --extra-index-url "https://download.pytorch.org/whl/cpu"
            }
            Write-Host "Diffracture installed successfully."
        } catch {
            Write-Error "ERROR: Failed to install Diffracture."
            Write-Error "PIP ERROR: $_"
            exit 1
        }
    } else {
        Write-Host "Diffracture directory found, but no setup.py or pyproject.toml present. Skipping."
    }
} else {
    Write-Host "Diffracture not found at ../Diffracture. Skipping."
}

Write-Host "Backend setup completed successfully!"
