"""
anomaly.py

Turns a stream of per-window embeddings into a smoothed anomaly score and an
alert decision, scored against a BaselineProfile (see baseline.py).

Two distance metrics are supported and selectable at runtime:

  - "mahalanobis": distance in units of the baseline's own variance per
    dimension. Uses the full shrinkage-regularized covariance when the
    profile has one (cov_mode == "full"), otherwise automatically falls back
    to the per-dimension diagonal form ((x-mean)/std) — this is exactly a
    diagonal Mahalanobis distance, so the two cov_mode cases share one
    formula rather than needing separate code paths.
  - "cosine": 1 - cosine_similarity(embedding, baseline mean). Cheaper and
    less sensitive to which embedding dimensions happen to be noisy for a
    given machine; a reasonable default when a baseline recording was short.

A single anomalous window is not enough to raise an alert — a short bang, a
door slam, someone talking nearby will all spike a single window's score.
Instead we keep a rolling buffer of recent smoothed scores and only fire an
alert once the score has stayed above threshold for `sustain_windows`
consecutive windows (roughly `sustain_windows * HOP_SAMPLES / 16000` seconds
of sustained deviation), and only clear the alert after it drops back below
threshold for `release_windows` windows, to avoid flapping right at the
threshold boundary.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from baseline import BaselineProfile

logger = logging.getLogger(__name__)

ScoreMetric = Literal["mahalanobis", "cosine"]

DEFAULT_SUSTAIN_WINDOWS = 3     # consecutive over-threshold windows to trigger
DEFAULT_RELEASE_WINDOWS = 4     # consecutive under-threshold windows to clear
SMOOTHING_WINDOW = 3            # simple moving average length over raw scores

# How thresholds are calibrated: rather than a hand-picked constant (which is
# meaningless across metrics/dimensionalities -- Mahalanobis distance over a
# D-dim baseline naturally sits around sqrt(D), cosine distance sits near 0),
# we score the baseline's OWN embeddings against itself and set the threshold
# at `mean(self-scores) + CALIBRATION_K * std(self-scores)`. That's self-
# calibrating for any metric, any embedding model, any dimensionality.
CALIBRATION_K = 4.0
_THRESHOLD_FLOOR = {"mahalanobis": 0.5, "cosine": 0.01}

# Adaptive threshold bounds: false-alarm feedback can widen the threshold,
# but never past these multiples of the originally-calibrated value, so the
# detector can't be trained into uselessness.
THRESHOLD_WIDEN_MIN_MULTIPLE = 1.0
THRESHOLD_WIDEN_MAX_MULTIPLE = 4.0
FALSE_ALARM_WIDEN_FACTOR = 1.08


@dataclass
class AnomalyEvent:
    timestamp: float
    raw_score: float
    smoothed_score: float
    threshold: float
    metric: ScoreMetric


def raw_score(embedding: np.ndarray, profile: BaselineProfile, metric: ScoreMetric) -> float:
    if metric == "cosine":
        mean = profile.mean
        denom = (np.linalg.norm(embedding) * np.linalg.norm(mean)) or 1e-8
        cos_sim = float(np.dot(embedding, mean) / denom)
        return 1.0 - cos_sim

    if metric == "mahalanobis":
        delta = embedding - profile.mean
        if profile.cov_mode == "full" and profile.cov_inv is not None:
            return float(np.sqrt(max(delta @ profile.cov_inv @ delta, 0.0)))
        # Diagonal fallback: equivalent to Mahalanobis distance under an
        # independence assumption between embedding dimensions.
        return float(np.sqrt(np.sum((delta / profile.std) ** 2) / len(delta)) * np.sqrt(len(delta)))

    raise ValueError(f"Unknown metric: {metric}")


def calibrate_thresholds(profile: BaselineProfile, baseline_embeddings: np.ndarray,
                          k: float = CALIBRATION_K) -> None:
    """Set profile.threshold_mahalanobis / threshold_cosine from the spread of
    the baseline's own embeddings scored against itself. Must be called once,
    right after baseline.build_profile(), before the profile is used for
    monitoring or saved. Self-consistent by construction: a metric/embedding
    model with more natural spread gets a proportionally higher threshold,
    rather than everything being compared against one hand-picked constant
    that only happens to make sense for one particular dimensionality."""
    for metric in ("mahalanobis", "cosine"):
        scores = np.array([raw_score(e, profile, metric) for e in baseline_embeddings])
        mu, sigma = float(scores.mean()), float(scores.std())
        threshold = max(mu + k * max(sigma, 1e-6), _THRESHOLD_FLOOR[metric])
        profile.set_threshold(metric, threshold, is_calibration=True)
        logger.info(
            "Calibrated '%s' threshold for profile '%s': self-scores mean=%.4f "
            "std=%.4f -> threshold=%.4f", metric, profile.name, mu, sigma, threshold,
        )


class AnomalyScorer:
    """Stateful scorer: feed it embeddings one at a time via `update()`, get
    back whether the machine is currently in an alert state, plus enough
    detail for the GUI to show a live gauge and log events."""

    def __init__(self, profile: BaselineProfile, metric: ScoreMetric = "mahalanobis",
                 sustain_windows: int = DEFAULT_SUSTAIN_WINDOWS,
                 release_windows: int = DEFAULT_RELEASE_WINDOWS):
        if metric == "mahalanobis" and profile.cov_mode != "full":
            logger.info(
                "Profile '%s' has no full covariance (only %d samples were "
                "recorded); Mahalanobis scoring will use the diagonal "
                "(per-dimension) fallback automatically.", profile.name, profile.n_samples,
            )
        # Fail fast and clearly rather than silently scoring against threshold
        # 0.0 (which would alert on every single window).
        self.profile = profile
        self.metric = metric
        self.threshold = profile.get_threshold(metric)  # raises if uncalibrated
        self.sustain_windows = sustain_windows
        self.release_windows = release_windows

        self._raw_history: deque[float] = deque(maxlen=SMOOTHING_WINDOW)
        self._recent_flags: deque[bool] = deque(maxlen=max(sustain_windows, release_windows))
        self.in_alert = False
        self.last_event: Optional[AnomalyEvent] = None

    def update(self, embedding: np.ndarray) -> AnomalyEvent:
        # Re-read from the profile each time (not just at __init__) so that
        # mark_false_alarm() widening the threshold mid-monitoring takes
        # effect on the very next window, not just on a future session.
        self.threshold = self.profile.get_threshold(self.metric)

        score = raw_score(embedding, self.profile, self.metric)
        self._raw_history.append(score)
        smoothed = float(np.mean(self._raw_history))

        over_threshold = smoothed >= self.threshold
        self._recent_flags.append(over_threshold)

        if not self.in_alert:
            recent = list(self._recent_flags)[-self.sustain_windows:]
            if len(recent) == self.sustain_windows and all(recent):
                self.in_alert = True
                logger.warning(
                    "ANOMALY: '%s' smoothed score %.3f sustained over threshold "
                    "%.3f for %d windows.", self.profile.name, smoothed,
                    self.threshold, self.sustain_windows,
                )
        else:
            recent = list(self._recent_flags)[-self.release_windows:]
            if len(recent) == self.release_windows and not any(recent):
                self.in_alert = False
                logger.info("Alert cleared for '%s' (back under threshold).", self.profile.name)

        event = AnomalyEvent(
            timestamp=time.time(), raw_score=score, smoothed_score=smoothed,
            threshold=self.threshold, metric=self.metric,
        )
        self.last_event = event
        return event

    def gauge_value(self, event: Optional[AnomalyEvent] = None) -> float:
        """0-100 value for a GUI gauge/progress bar: 50 == right at threshold."""
        event = event or self.last_event
        if event is None or event.threshold <= 0:
            return 0.0
        return float(np.clip((event.smoothed_score / event.threshold) * 50.0, 0.0, 100.0))

    def mark_false_alarm(self) -> None:
        """User feedback: this alert wasn't a real issue. Widen the threshold
        modestly so near-identical noise doesn't retrigger immediately, bounded
        to a multiple of the originally-calibrated threshold so it can't drift
        into uselessness after repeated false alarms."""
        self.profile.confirmed_false_alarms += 1
        base = self.profile.get_threshold_base(self.metric)
        current = self.profile.get_threshold(self.metric)
        widened = current * FALSE_ALARM_WIDEN_FACTOR
        clamped = float(np.clip(
            widened, base * THRESHOLD_WIDEN_MIN_MULTIPLE, base * THRESHOLD_WIDEN_MAX_MULTIPLE
        ))
        self.profile.set_threshold(self.metric, clamped)
        self.threshold = clamped
        logger.info("False alarm confirmed for '%s'; threshold widened to %.3f (base was %.3f)",
                     self.profile.name, clamped, base)

    def mark_confirmed_issue(self) -> None:
        """User feedback: this alert was a real issue. We don't tighten the
        threshold automatically (that risks false positives spiraling), we
        just record it for the profile's history/stats."""
        self.profile.confirmed_true_alarms += 1
        logger.info("Confirmed issue recorded for '%s' (total: %d)",
                     self.profile.name, self.profile.confirmed_true_alarms)
