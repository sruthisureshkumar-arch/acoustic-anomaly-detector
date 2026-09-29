#!/usr/bin/env bash
# setup.sh — one-command setup for macOS/Linux dev/test machines.
#
# On the actual Snapdragon target (Windows on ARM), use setup.ps1 instead --
# it also installs onnxruntime-qnn, which isn't available on this platform.
#
# This does NOT fetch the real NPU model asset (models/yamnet_npu.onnx) --
# that requires a Qualcomm AI Hub account and submits real cloud compile
# jobs, so it's a separate, explicit step. See models/README.md. Without it,
# the app runs fine against the CPU-only placeholder model for development.
set -euo pipefail
cd "$(dirname "$0")"

echo "== Acoustic Anomaly Detector setup =="

if [ ! -d ".venv" ]; then
    echo "-- Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "-- Installing Python dependencies..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

if [ ! -d "vendor/torch_audioset" ]; then
    echo "-- Vendoring torch_audioset (mel-spectrogram front end; doesn't pip-install reliably)..."
    git clone --quiet --depth 1 https://github.com/w-hc/torch_audioset.git vendor/torch_audioset
else
    echo "-- vendor/torch_audioset already present, skipping."
fi

if [ ! -f "models/yamnet_npu.onnx" ] && [ ! -f "models/yamnet_public.onnx" ]; then
    echo "-- No model found; generating the CPU-only development placeholder..."
    .venv/bin/python scripts/build_dev_embedder_onnx.py
fi

mkdir -p profiles logs

echo ""
echo "Setup complete. Next steps:"
echo "  1. (Real submission asset) Follow models/README.md to export the real"
echo "     NPU-compiled yamnet_npu.onnx via Qualcomm AI Hub, if you haven't."
echo "  2. Run the smoke test:      .venv/bin/python scripts/smoke_test.py"
echo "  3. Launch the app:          .venv/bin/python src/main.py"
