"""
smoke_test.py

Headless end-to-end test of the pipeline (embedding -> baseline -> anomaly
scoring -> save/load) using synthetic audio instead of a live microphone, so
it runs in environments with no audio hardware (e.g. this build sandbox).

Uses the placeholder dev ONNX model (models/yamnet_public.onnx) built by
build_dev_embedder_onnx.py, NOT real YamNet, so this only proves the
pipeline's plumbing and math are correct -- it says nothing about real-world
detection accuracy, which requires the actual YamNet NPU asset and a real
machine's audio (see DEMO.md).

Run: python scripts/smoke_test.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import anomaly  # noqa: E402
import baseline  # noqa: E402
from embedding import YamNetEmbedder  # noqa: E402

SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 15_360  # must match embedding.WAVEFORM_SAMPLES_PER_PATCH (0.96s patch)


def synth_window(freq_hz: float, noise_level: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
    tone = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    noise = noise_level * rng.standard_normal(WINDOW_SAMPLES)
    signal = (tone + noise).astype(np.float32)
    return np.clip(signal, -1.0, 1.0)


def run_case(label: str, n_baseline: int, metric: str) -> None:
    print(f"\n--- {label} (n_baseline={n_baseline}, metric={metric}) ---")
    rng = np.random.default_rng(42)
    embedder = YamNetEmbedder()
    print(f"Model: {embedder.model_path.name} | active provider: {embedder.provider_info.active} "
          f"| is_npu: {embedder.provider_info.is_npu}")

    # "Normal" machine: a steady ~200Hz hum with modest noise.
    baseline_embeddings = np.stack([
        embedder.embed(synth_window(200.0, 0.05, rng)) for _ in range(n_baseline)
    ])
    profile = baseline.build_and_calibrate_profile("smoke-test-machine", baseline_embeddings)
    print(f"Profile built: cov_mode={profile.cov_mode}, n_samples={profile.n_samples}, "
          f"embedding_dim={profile.embedding_dim}")
    print(f"Calibrated thresholds: mahalanobis={profile.threshold_mahalanobis:.3f}  "
          f"cosine={profile.threshold_cosine:.4f}")

    saved_dir = baseline.save_profile(profile)
    reloaded = baseline.load_profile(profile.name)
    assert reloaded.cov_mode == profile.cov_mode
    assert np.allclose(reloaded.mean, profile.mean)
    print(f"Save/load round-trip OK ({saved_dir})")

    scorer = anomaly.AnomalyScorer(reloaded, metric=metric)

    # Feed more windows from the SAME distribution: should stay in "normal".
    normal_alerts = 0
    for _ in range(10):
        emb = embedder.embed(synth_window(200.0, 0.05, rng))
        event = scorer.update(emb)
        if scorer.in_alert:
            normal_alerts += 1
    print(f"Normal-condition windows that triggered alert state: {normal_alerts}/10 "
          f"(want this low; some false positives are expected from a placeholder, "
          f"non-acoustic embedder)")

    # Now feed a clearly different signal: a much higher frequency, more noise,
    # simulating e.g. a bearing starting to squeal. Should eventually alert.
    scorer2 = anomaly.AnomalyScorer(reloaded, metric=metric)
    triggered_at = None
    for i in range(15):
        emb = embedder.embed(synth_window(1800.0, 0.4, rng))
        scorer2.update(emb)
        if scorer2.in_alert and triggered_at is None:
            triggered_at = i
    print(f"Anomalous-condition windows: alert triggered at window index "
          f"{triggered_at if triggered_at is not None else 'NEVER (check thresholds/metric)'}")

    false_before = reloaded.confirmed_false_alarms
    threshold_before = reloaded.get_threshold(metric)
    scorer2.mark_false_alarm()
    assert reloaded.confirmed_false_alarms == false_before + 1
    assert reloaded.get_threshold(metric) >= threshold_before
    print(f"False-alarm feedback OK: threshold now {reloaded.get_threshold(metric):.3f} "
          f"(was {threshold_before:.3f})")


def main() -> None:
    # Small n forces the diagonal-covariance fallback path; large n exercises
    # the full-covariance + shrinkage path. Both must work cleanly.
    run_case("Short baseline -> diagonal covariance path", n_baseline=20, metric="mahalanobis")
    run_case("Long baseline -> full covariance path", n_baseline=400, metric="mahalanobis")
    run_case("Cosine metric", n_baseline=60, metric="cosine")
    print("\nAll smoke test cases completed without unhandled errors.")


if __name__ == "__main__":
    main()
