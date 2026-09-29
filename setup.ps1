# setup.ps1 — one-command setup for the real target: Windows on Snapdragon
# (ARM64). Run from PowerShell in the project root:
#
#     .\setup.ps1
#
# This installs onnxruntime-qnn (the QNN Execution Provider that actually
# talks to the Hexagon NPU) instead of the plain onnxruntime that
# requirements.txt lists for dev machines -- that package only ships useful
# QNN builds for Windows ARM64, so it's handled as a platform-specific extra
# step here rather than baked into requirements.txt.
#
# This does NOT fetch the real NPU model asset (models\yamnet_npu.onnx) --
# that requires a Qualcomm AI Hub account and submits real cloud compile
# jobs, so it's a separate, explicit step. See models\README.md.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "== Acoustic Anomaly Detector setup (Windows on Snapdragon) =="

if (-not (Test-Path ".venv")) {
    Write-Host "-- Creating virtual environment (.venv)..."
    python -m venv .venv
}

$venvPip = ".venv\Scripts\pip.exe"
$venvPython = ".venv\Scripts\python.exe"

Write-Host "-- Installing Python dependencies..."
& $venvPip install --quiet --upgrade pip
& $venvPip install --quiet -r requirements.txt

Write-Host "-- Installing the QNN Execution Provider (Hexagon NPU support)..."
& $venvPip uninstall --quiet --yes onnxruntime 2>$null
& $venvPip install --quiet onnxruntime-qnn

if (-not (Test-Path "vendor\torch_audioset")) {
    Write-Host "-- Vendoring torch_audioset (mel-spectrogram front end; doesn't pip-install reliably)..."
    git clone --quiet --depth 1 https://github.com/w-hc/torch_audioset.git vendor\torch_audioset
} else {
    Write-Host "-- vendor\torch_audioset already present, skipping."
}

if (-not (Test-Path "models\yamnet_npu.onnx") -and -not (Test-Path "models\yamnet_public.onnx")) {
    Write-Host "-- No model found; generating the CPU-only development placeholder..."
    & $venvPython scripts\build_dev_embedder_onnx.py
}

New-Item -ItemType Directory -Force -Path "profiles", "logs" | Out-Null

Write-Host ""
Write-Host "Setup complete. Next steps:"
Write-Host "  1. (Real submission asset) Follow models\README.md to export the real"
Write-Host "     NPU-compiled yamnet_npu.onnx via Qualcomm AI Hub, if you haven't."
Write-Host "  2. Benchmark NPU vs CPU on this device: $venvPython scripts\benchmark.py"
Write-Host "  3. Run the smoke test:                  $venvPython scripts\smoke_test.py"
Write-Host "  4. Launch the app:                       $venvPython src\main.py"
