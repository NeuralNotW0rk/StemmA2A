[CmdletBinding()]
param (
    [string]$ImageName = "execution-engine",
    [string]$Tag = "latest",
    [string]$OutputFile = "execution-engine.tar",
    [string]$CudaArchs = "7.5;8.0;8.6;8.9;9.0",
    [int]$MaxJobs = 2,
    [string]$Platform = "linux/amd64"
)

$ErrorActionPreference = 'Stop'

# Get the directory of the script
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$BackendDir = (Resolve-Path (Join-Path $ProjectRoot "backend")).Path
$DockerfilePath = Join-Path $BackendDir "Dockerfile"

# Ensure the output file path is absolute
if (-not [System.IO.Path]::IsPathRooted($OutputFile)) {
    $OutputFile = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputFile))
}

# 1. Check if Docker is running
Write-Host "Checking if Docker is running..."
$OldEAP = $ErrorActionPreference
$ErrorActionPreference = 'SilentlyContinue'
$dockerRunning = $false
try {
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        & docker info *>$null
        if ($LASTEXITCODE -eq 0) {
            $dockerRunning = $true
        }
    }
}
catch {
    # Keep $dockerRunning as false
}
$ErrorActionPreference = $OldEAP

if (-not $dockerRunning) {
    Write-Host ""
    Write-Host "[ERROR] Docker is not running or the Docker daemon is unreachable." -ForegroundColor Red
    Write-Host " -> Please launch Docker Desktop and make sure the engine has started successfully." -ForegroundColor Yellow
    Write-Host " -> Run 'docker info' in your terminal to manually verify the connection status." -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# Ensure BuildKit is enabled (critical for --mount=type=cache)
$env:DOCKER_BUILDKIT = 1

Write-Host "`n=== Step 1: Building Docker Image for GPU target on CPU host ==="
Write-Host "Image Name:  ${ImageName}:${Tag}"
Write-Host "Platform:    $Platform"
Write-Host "CUDA Archs:  $CudaArchs"
Write-Host "Max Jobs:    $MaxJobs"
Write-Host "Context:     $BackendDir"
Write-Host "Dockerfile:  $DockerfilePath"

$BuildArgs = @(
    "build",
    "--platform", $Platform,
    "-t", "${ImageName}:${Tag}",
    "-f", $DockerfilePath,
    "--build-arg", "TORCH_CUDA_ARCH_LIST=$CudaArchs",
    "--build-arg", "MAX_JOBS=$MaxJobs",
    $BackendDir
)

# Execute Docker build
Write-Host "`nRunning command: docker $([string]::Join(' ', $BuildArgs))`n"
& docker $BuildArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: Docker build command failed."
    exit 1
}

# 2. Export the image to a tar file
Write-Host "`n=== Step 2: Exporting Image to TAR file ==="
Write-Host "Target location: $OutputFile"

if (Test-Path $OutputFile) {
    Write-Host "Overwriting existing archive at $OutputFile..."
    Remove-Item $OutputFile -Force
}

& docker save -o "$OutputFile" "${ImageName}:${Tag}"

if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: docker save command failed."
    exit 1
}

# Get file size info
$FileSizeGb = ([System.IO.FileInfo]$OutputFile).Length / 1GB
$FileSizeFormatted = "{0:N2}" -f $FileSizeGb

Write-Host "`n======================================================================"
Write-Host " SUCCESS: Docker image successfully built and exported!"
Write-Host " Image:          ${ImageName}:${Tag}"
Write-Host " Export File:    $OutputFile"
Write-Host " Archive Size:   $FileSizeFormatted GB"
Write-Host "======================================================================"
Write-Host "`nNext steps to run this on your GPU-enabled machine:"
Write-Host "----------------------------------------------------------------------"
Write-Host "1. Copy the TAR archive to your GPU machine (replace user@gpu-host with your actual login):"
Write-Host "   scp `"$OutputFile`" user@gpu-host:/path/to/destination/"
Write-Host "`n2. SSH/log in to your GPU machine and import the image:"
Write-Host "   docker load -i /path/to/destination/$([System.IO.Path]::GetFileName($OutputFile))"
Write-Host "`n3. Launch the container stack from the StemmA2A project root directory:"
Write-Host "   docker compose up -d"
Write-Host "======================================================================"
