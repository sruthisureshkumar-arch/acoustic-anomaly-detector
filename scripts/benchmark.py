"""
scripts/benchmark.py

Run this once on the actual Snapdragon hardware (with models/yamnet_npu.onnx
in place) to produce the real NPU-vs-CPU numbers this project's README and
GUI cite. Also safe to run on a non-Snapdragon dev machine or with the
placeholder model -- it'll just report there was only one provider to
compare, or CPU-only numbers.

Run from the project root:
    python scripts/benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import benchmark  # noqa: E402


def main() -> None:
    print("Benchmarking YamNet inference latency (loads the model on each available "
          "execution provider and times ~50 inference calls per provider; takes a "
          "few seconds)...\n")
    results = benchmark.run_benchmark()

    for key, val in results.items():
        if not isinstance(val, dict):
            continue
        print(f"{key}:")
        print(f"  mean latency:   {val['mean_ms']:.2f} ms/window")
        print(f"  median latency: {val['median_ms']:.2f} ms/window")
        print(f"  p95 latency:    {val['p95_ms']:.2f} ms/window")
        print(f"  throughput:     {val['throughput_hz']:.1f} windows/sec")
        print()

    if "npu_speedup_x" in results:
        print(f"NPU (QNN) is {results['npu_speedup_x']}x faster than CPU on this device, "
              f"on the same model ({results['model']}).")
    else:
        print("Only one execution provider was available on this machine, so there's no "
              "NPU-vs-CPU comparison to show -- run this on the actual Snapdragon "
              "hardware with onnxruntime-qnn installed to get that number.")

    path = benchmark.save_results(results)
    print(f"\nSaved to {path.relative_to(path.parent.parent)} -- the GUI reads this "
          f"file automatically and shows the numbers next to the inference-provider "
          f"label, and it's also what README.md's benchmark numbers should be updated "
          f"from before the final submission/demo.")


if __name__ == "__main__":
    main()
