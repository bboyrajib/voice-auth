Write-Host ""
Write-Host "=================================================="
Write-Host "     AI Speech Detector - Local Launcher"
Write-Host "=================================================="
Write-Host ""

# --------------------------------------------------
# Config
# --------------------------------------------------
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectDir "venv"
$Requirements = Join-Path $ProjectDir "requirements.txt"
$AppFile = Join-Path $ProjectDir "app.py"
$FfmpegLocal = Join-Path $ProjectDir "ffmpeg\bin"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

# --------------------------------------------------
# Check Python
# --------------------------------------------------
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Python is not installed or not in PATH."
    Write-Host "Install Python 3.10+ and try again."
    exit 1
}

Write-Host "[INFO] Python detected."
Write-Host ""

# Create venv if missing
if (-not (Test-Path $VenvDir)) {
    Write-Host "[INFO] Creating virtual environment..."
    python -m venv $VenvDir
}

# Verify venv python exists
if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment setup failed."
    exit 1
}

Write-Host "[INFO] Using virtual environment Python."
Write-Host ""

# Install dependencies
Write-Host "[INFO] Installing dependencies..."
& $VenvPython -m pip install --upgrade pip | Out-Null
& $VenvPython -m pip install -r $Requirements

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Dependency installation failed."
    exit 1
}

# --------------------------------------------------
# Setup FFmpeg (optional)
# --------------------------------------------------
$FfmpegExe = Join-Path $FfmpegLocal "ffmpeg.exe"

if (Test-Path $FfmpegExe) {
    Write-Host "[INFO] Local FFmpeg detected."
    $env:PATH = "$FfmpegLocal;$env:PATH"
} else {
    Write-Host "[INFO] Using system FFmpeg (if installed)."
}

Write-Host ""
Write-Host "=================================================="
Write-Host "     Starting Streamlit Application"
Write-Host "=================================================="
Write-Host ""

streamlit run $AppFile
