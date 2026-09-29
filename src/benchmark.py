"""
benchmark.py

Times YamNet inference on whichever execution providers are actually
available on this machine, so the app -- and the hackathon presentation --
has a real, on-this-device number for "how much faster is the NPU" instead
of just a boolean "NPU active" label.

Why this exists: the judging criteria for the Snapdragon AI Lab Build &
Present Challenge lead with Technical Implementation, and the strongest
comparable submissions we looked at (see README.md's "How this maps to the
judging criteria" section) all show a measured NPU-vs-CPU comparison rather
than just asserting NPU usage. This module is that measurement, done
apples-to-apples: same model file, same input, two providers.

Usage:
  - `python scripts/benchmark.py` from the project root: prints a table and
    writes models/benchmark_results.json.
  - gui.py calls `run_benchmark()` directly (in a background QThread) so it
    can be re-run live during a demo, and calls `load_results()` on startup
    to show the most recent saved numbers without re-benchmarking every
    launch.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from embedding import WAVEFORM_SAMPLES_PER_PATCH, YamNetEmbedder

logger = logging.getLogger(__name__)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "models" / "benchmark_results.json"


@dataclass
class ProviderBenchmark:
    provider: str
    n_iters: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    throughput_hz: float


def _synthetic_window(rng: np.random.Generator) -> np.ndarray:
    # Real acoustic content doesn't matter for a latency benchmark -- only
    # the shape/dtype the model sees does -- so plain noise is fine here and
    # keeps this runnable with no microphone.
    return (0.1 * rng.standard_normal(WAVEFORM_SAMPLES_PER_PATCH)).astype(np.float32)


def _time_embedder(embedder: YamNetEmbedder, n_warmup: int, n_iters: int,
                    rng: np.random.Generator) -> ProviderBenchmark:
    windows = [_synthetic_window(rng) for _ in range(n_warmup + n_iters)]

    # Warm-up: the first call on a fresh session pays for lazy graph
    # optimization, provider kernel compilation, and (for the real model)
    # constructing the mel-spectrogram transform. Excluding these from the
    # timed loop is what makes the numbers reflect steady-state inference,
    # not one-time setup cost.
    for w in windows[:n_warmup]:
        embedder.embed(w)

    durations_ms = []
    for w in windows[n_warmup:]:
        start = time.perf_counter()
        embedder.embed(w)
        durations_ms.append((time.perf_counter() - start) * 1000.0)

    durations_ms.sort()
    mean_ms = statistics.mean(durations_ms)
    p95_idx = min(len(durations_ms) - 1, int(round(0.95 * (len(durations_ms) - 1))))
    return ProviderBenchmark(
        provider=embedder.provider_info.active,
        n_iters=n_iters,
        mean_ms=mean_ms,
        median_ms=statistics.median(durations_ms),
        p95_ms=durations_ms[p95_idx],
        throughput_hz=(1000.0 / mean_ms) if mean_ms > 0 else float("inf"),
    )


def run_benchmark(n_warmup: int = 5, n_iters: int = 50, seed: int = 0) -> dict:
    """Benchmarks the configured YamNet model (models/README.md) on the NPU
    (QNN Execution Provider) if available, and always on CPU for comparison
    -- using the SAME model file for both, so this measures the provider,
    not a different model. Returns a JSON-serializable dict; also see
    save_results()/load_results()."""
    rng = np.random.default_rng(seed)
    results: dict = {}

    npu_embedder = YamNetEmbedder(prefer_npu=True)
    primary = _time_embedder(npu_embedder, n_warmup, n_iters, rng)
    results[primary.provider] = asdict(primary)

    if primary.provider != "CPUExecutionProvider":
        # Force a second session on the same model file, CPU only, so we get
        # a genuine same-model comparison rather than comparing against a
        # different (e.g. placeholder) model that happens to be CPU-only.
        cpu_embedder = YamNetEmbedder(model_path=npu_embedder.model_path, prefer_npu=False)
        cpu_bench = _time_embedder(cpu_embedder, n_warmup, n_iters, rng)
        results[cpu_bench.provider] = asdict(cpu_bench)

    if "QNNExecutionProvider" in results and "CPUExecutionProvider" in results:
        results["npu_speedup_x"] = round(
            results["CPUExecutionProvider"]["mean_ms"] / results["QNNExecutionProvider"]["mean_ms"], 2
        )

    results["model"] = npu_embedder.model_path.name
    results["window_seconds"] = round(WAVEFORM_SAMPLES_PER_PATCH / 16_000, 3)
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return results


def save_results(results: dict) -> Path:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    return RESULTS_PATH


def load_results() -> Optional[dict]:
    if not RESULTS_PATH.exists():
        return None
    try:
        return json.loads(RESULTS_PATH.read_text())
    except Exception:  # noqa: BLE001 - a corrupt/partial results file is never fatal
        logger.warning("Could not parse %s; ignoring cached benchmark results.", RESULTS_PATH)
        return None


def summarize(results: dict) -> str:
    """One-line summary for the GUI's provider label."""
    if "npu_speedup_x" in results:
        npu = results.get("QNNExecutionProvider", {})
        return f"{npu.get('mean_ms', 0):.1f} ms/window  •  {results['npu_speedup_x']}x faster than CPU"
    for key, val in results.items():
        if isinstance(val, dict):
            return f"{val.get('mean_ms', 0):.1f} ms/window (only one provider available on this device)"
    return ""
