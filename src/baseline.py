"""
baseline.py

Records a machine's "normal" sound and turns it into a persisted statistical
profile: a mean embedding, per-dimension standard deviation, and (when there's
enough data to estimate it reliably) a regularized covariance matrix for
Mahalanobis scoring. Profiles are named per machine (e.g. "Lathe #1") and
saved under profiles/<name>/ so a workshop can keep several machines'
baselines side by side.

Numerical note: a 1024-d YamNet embedding needs a lot of samples to estimate
a full covariance matrix reliably (rule of thumb: several times the
dimensionality). A 60-120s baseline recording with 50% overlap only yields on
the order of 100-250 embeddings, nowhere near enough for a well-conditioned
1024x1024 covariance. Rather than silently produce a matrix that will make
Mahalanobis distance meaningless (or blow up from a near-singular matrix), we:

  - always compute the diagonal (per-dimension variance) statistics, which are
    stable with far less data, and
  - only attempt a shrinkage-regularized full covariance when there are
    enough samples, falling back to the diagonal form otherwise, and
    recording which mode was used so anomaly.py can pick the right formula
    and the GUI can be honest with the user about it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).resolve().parent.parent / "profiles"

# Minimum ratio of (samples / embedding_dim) before we trust a full covariance
# estimate at all, even with shrinkage regularization.
MIN_SAMPLES_PER_DIM_FOR_FULL_COV = 0.25
SHRINKAGE = 0.1  # Ledoit-Wolf-style linear shrinkage toward the diagonal


@dataclass
class BaselineProfile:
    name: str
    created_at: float
    embedding_dim: int
    n_samples: int
    mean: np.ndarray                      # (D,)
    std: np.ndarray                       # (D,), floor-clamped to avoid /0
    cov_mode: Literal["diagonal", "full"]
    cov_inv: Optional[np.ndarray] = None  # (D, D), only when cov_mode == "full"
    # Anomaly-scoring thresholds, one per metric, since Mahalanobis and cosine
    # distance live on completely different scales (Mahalanobis distance over
    # a D-dim independent baseline naturally sits around sqrt(D), not O(1)).
    # Left at 0.0 until anomaly.calibrate_thresholds() sets them from the
    # baseline's own score distribution -- do NOT hand-pick a fixed constant
    # here, it will be wrong for whatever D the embedding model happens to have.
    threshold_mahalanobis: float = 0.0
    threshold_cosine: float = 0.0
    # The as-calibrated value, kept separately so adaptive widening from
    # false-alarm feedback (see anomaly.mark_false_alarm) can be bounded as a
    # multiple of where calibration originally put it, rather than drifting
    # without limit.
    threshold_mahalanobis_base: float = 0.0
    threshold_cosine_base: float = 0.0
    confirmed_false_alarms: int = 0
    confirmed_true_alarms: int = 0

    def profile_dir(self) -> Path:
        return PROFILES_DIR / _slugify(self.name)

    def get_threshold(self, metric: str) -> float:
        value = self.threshold_mahalanobis if metric == "mahalanobis" else self.threshold_cosine
        if value <= 0.0:
            raise ValueError(
                f"Profile '{self.name}' has no calibrated threshold for metric "
                f"'{metric}' yet -- call anomaly.calibrate_thresholds(profile, "
                f"baseline_embeddings) right after build_profile() and before use."
            )
        return value

    def set_threshold(self, metric: str, value: float, is_calibration: bool = False) -> None:
        if metric == "mahalanobis":
            self.threshold_mahalanobis = value
            if is_calibration:
                self.threshold_mahalanobis_base = value
        else:
            self.threshold_cosine = value
            if is_calibration:
                self.threshold_cosine_base = value

    def get_threshold_base(self, metric: str) -> float:
        return self.threshold_mahalanobis_base if metric == "mahalanobis" else self.threshold_cosine_base


def record_baseline(name: str, embedder, capture, seconds: float = 90.0) -> BaselineProfile:
    """Record `seconds` of audio via `capture` (an AudioCapture), embed every
    window with `embedder`, and build a BaselineProfile from the result."""
    logger.info("Recording %.0fs baseline for profile '%s'...", seconds, name)
    windows = capture.record_seconds(seconds)
    if len(windows) < 5:
        raise ValueError(
            f"Only captured {len(windows)} audio windows in {seconds}s — check the "
            f"microphone/device selection before recording a baseline."
        )

    embeddings = np.stack([embedder.embed(w.samples) for w in windows])
    return build_and_calibrate_profile(name, embeddings)


def build_profile(name: str, embeddings: np.ndarray) -> BaselineProfile:
    n_samples, dim = embeddings.shape
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)
    # Floor relative to the *typical* per-dimension spread, not just an absolute
    # tiny constant: with only tens-to-hundreds of baseline samples, a handful
    # of dimensions will have their std underestimated close to zero by pure
    # sampling noise. Dividing by an unfloored (or absolute-tiny-floored) std
    # lets exactly those dimensions dominate the distance metric out of all
    # proportion, which is what caused implausibly huge Mahalanobis scores in
    # testing. Flooring relative to the median std keeps any one dimension
    # from blowing up the total distance.
    robust_floor = max(float(np.median(std)) * 0.1, 1e-4)
    std = np.maximum(std, robust_floor)

    cov_mode: Literal["diagonal", "full"] = "diagonal"
    cov_inv = None
    if n_samples / dim >= MIN_SAMPLES_PER_DIM_FOR_FULL_COV:
        try:
            cov = np.cov(embeddings, rowvar=False)
            diag_mean = np.trace(cov) / dim
            shrunk = (1 - SHRINKAGE) * cov + SHRINKAGE * diag_mean * np.eye(dim)
            cov_inv = np.linalg.inv(shrunk)
            cov_mode = "full"
        except np.linalg.LinAlgError:
            logger.warning(
                "Covariance matrix for '%s' was singular even after shrinkage; "
                "falling back to diagonal (per-dimension variance) scoring.", name
            )
    else:
        logger.info(
            "Only %d samples for a %d-d embedding (%.2fx dimensionality) — using "
            "diagonal covariance rather than a full Mahalanobis matrix, which "
            "would be unreliable with this little data. Record a longer baseline "
            "to enable full covariance scoring.",
            n_samples, dim, n_samples / dim,
        )

    return BaselineProfile(
        name=name,
        created_at=time.time(),
        embedding_dim=dim,
        n_samples=n_samples,
        mean=mean,
        std=std,
        cov_mode=cov_mode,
        cov_inv=cov_inv,
    )


def build_and_calibrate_profile(name: str, embeddings: np.ndarray,
                                 holdout_fraction: float = 0.2,
                                 min_holdout_samples: int = 8) -> BaselineProfile:
    """Build a profile AND calibrate its thresholds, correctly: the
    thresholds are calibrated on embeddings held OUT of the mean/covariance
    fit, not scored against themselves.

    Why this matters: fitting a covariance matrix and then scoring the same
    points that were used to fit it is optimistic (the fitted distribution
    "expects" those exact points to look normal), especially with a
    ~1000-dim embedding and only a few hundred baseline samples. Testing
    this on synthetic audio surfaced exactly that -- the full-covariance
    path had a *higher* false-positive rate on fresh audio than the
    diagonal fallback, which is backwards from what better modeling should
    give you, and is the signature of in-sample calibration bias rather
    than a real quality difference between the two covariance modes.

    With enough samples, we hold out `holdout_fraction` of the baseline
    recording, fit mean/std/covariance on the rest, and calibrate
    thresholds against the held-out slice. Below `min_holdout_samples`
    worth of holdout data, holding anything out would make both the fit and
    the calibration unstable, so we fall back to in-sample calibration and
    the resulting profile is a little more false-positive-prone -- expected
    and acceptable for a short baseline recording, but real, so it's worth
    recording a longer baseline (90s+) for a machine you'll actually rely on."""
    from anomaly import calibrate_thresholds  # deferred: avoids circular import

    n = len(embeddings)
    holdout_n = int(n * holdout_fraction)
    if holdout_n >= min_holdout_samples and (n - holdout_n) >= min_holdout_samples:
        fit_embeddings, calib_embeddings = embeddings[:-holdout_n], embeddings[-holdout_n:]
        profile = build_profile(name, fit_embeddings)
        calibrate_thresholds(profile, calib_embeddings)
        logger.info("Calibrated '%s' on a held-out %d/%d samples (fit on the other %d).",
                     name, holdout_n, n, n - holdout_n)
    else:
        profile = build_profile(name, embeddings)
        calibrate_thresholds(profile, embeddings)
        logger.warning(
            "Baseline for '%s' has only %d samples -- too few to hold out a separate "
            "calibration set, so thresholds are calibrated in-sample and will likely "
            "be a bit tighter (more false-positive-prone) than a longer baseline would "
            "give. Consider recording 90s+ for machines you'll rely on.", name, n,
        )
    return profile


def save_profile(profile: BaselineProfile) -> Path:
    out_dir = profile.profile_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "name": profile.name,
        "created_at": profile.created_at,
        "embedding_dim": profile.embedding_dim,
        "n_samples": profile.n_samples,
        "cov_mode": profile.cov_mode,
        "threshold_mahalanobis": profile.threshold_mahalanobis,
        "threshold_cosine": profile.threshold_cosine,
        "threshold_mahalanobis_base": profile.threshold_mahalanobis_base,
        "threshold_cosine_base": profile.threshold_cosine_base,
        "confirmed_false_alarms": profile.confirmed_false_alarms,
        "confirmed_true_alarms": profile.confirmed_true_alarms,
    }
    (out_dir / "profile.json").write_text(json.dumps(meta, indent=2))

    arrays = {"mean": profile.mean, "std": profile.std}
    if profile.cov_inv is not None:
        arrays["cov_inv"] = profile.cov_inv
    np.savez_compressed(out_dir / "stats.npz", **arrays)

    logger.info("Saved profile '%s' to %s (%s covariance mode)",
                profile.name, out_dir, profile.cov_mode)
    return out_dir


def load_profile(name: str) -> BaselineProfile:
    out_dir = PROFILES_DIR / _slugify(name)
    meta = json.loads((out_dir / "profile.json").read_text())
    arrays = np.load(out_dir / "stats.npz")

    return BaselineProfile(
        name=meta["name"],
        created_at=meta["created_at"],
        embedding_dim=meta["embedding_dim"],
        n_samples=meta["n_samples"],
        mean=arrays["mean"],
        std=arrays["std"],
        cov_mode=meta["cov_mode"],
        cov_inv=arrays["cov_inv"] if "cov_inv" in arrays else None,
        threshold_mahalanobis=meta.get("threshold_mahalanobis", meta.get("threshold", 0.0)),
        threshold_cosine=meta.get("threshold_cosine", 0.0),
        threshold_mahalanobis_base=meta.get("threshold_mahalanobis_base", meta.get("threshold_mahalanobis", 0.0)),
        threshold_cosine_base=meta.get("threshold_cosine_base", meta.get("threshold_cosine", 0.0)),
        confirmed_false_alarms=meta.get("confirmed_false_alarms", 0),
        confirmed_true_alarms=meta.get("confirmed_true_alarms", 0),
    )


def list_profiles() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    names = []
    for child in sorted(PROFILES_DIR.iterdir()):
        if (child / "profile.json").exists():
            names.append(json.loads((child / "profile.json").read_text())["name"])
    return names


def _slugify(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip()).strip("_") or "profile"
