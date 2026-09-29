"""
audio_capture.py

Microphone streaming for the Acoustic Anomaly Detector.

Captures raw audio from the default (or a chosen) input device, resamples/
downmixes it to 16 kHz mono (the format YamNet expects), and slices it into
overlapping ~1-second windows. Used identically by both baseline recording
and live monitoring, so the two modes always see audio processed the same
way.

Uses `sounddevice` (PortAudio bindings) rather than PyAudio: it has no
compiled-extension headaches on Windows and exposes a simple callback-based
InputStream, which is what we want for a continuously-running monitor.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np

try:
    import sounddevice as sd
except OSError as exc:  # pragma: no cover - no audio backend on this machine
    sd = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16_000          # YamNet's expected input rate
# YamNet's mel-spectrogram patch is exactly 0.96s (PATCH_WINDOW_IN_SECONDS in
# torch_audioset's params.py) -- NOT a round 1.0s. This has to match
# embedding.WAVEFORM_SAMPLES_PER_PATCH exactly, or the front end either pads
# with silence for no reason or silently drops the tail of every window.
WINDOW_SECONDS = 0.96
WINDOW_SAMPLES = int(round(TARGET_SAMPLE_RATE * WINDOW_SECONDS))  # 15,360
OVERLAP = 0.5                        # 50% overlap between consecutive windows
HOP_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP))


@dataclass
class AudioWindow:
    """One ~1s mono 16kHz audio window, plus when it was captured."""

    samples: np.ndarray          # float32, shape (WINDOW_SAMPLES,), range [-1, 1]
    timestamp: float             # time.time() when this window finished capturing


def list_input_devices() -> list[dict]:
    """Return sounddevice's input-capable devices, for a device picker in the GUI."""
    if sd is None:
        raise RuntimeError(f"No audio backend available: {_IMPORT_ERROR}")
    devices = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            devices.append({"index": idx, "name": dev["name"],
                             "default_samplerate": dev["default_samplerate"]})
    return devices


def _downmix_to_mono(block: np.ndarray) -> np.ndarray:
    if block.ndim == 1:
        return block
    return block.mean(axis=1)


class AudioCapture:
    """
    Continuously reads mic audio in a background thread and hands back
    overlapping WINDOW_SAMPLES-length windows, already resampled to
    TARGET_SAMPLE_RATE mono float32.

    Two ways to consume it:
      - `windows()`: a blocking generator, used by both baseline recording
        (call `stop()` after N seconds) and live monitoring (run until the
        user clicks Stop).
      - `on_window` callback, if you'd rather not block a thread on a
        generator (used by the GUI's Qt event loop).
    """

    def __init__(self, device: Optional[int] = None,
                 on_window: Optional[Callable[[AudioWindow], None]] = None):
        if sd is None:
            raise RuntimeError(
                f"sounddevice could not open a PortAudio backend on this machine: "
                f"{_IMPORT_ERROR}. Live capture is unavailable here; you can still "
                f"exercise the rest of the pipeline with synthetic audio (see "
                f"scripts/smoke_test.py)."
            )
        self.device = device
        self._on_window = on_window
        self._native_rate = self._resolve_device_rate(device)
        self._resample_ratio = TARGET_SAMPLE_RATE / self._native_rate

        self._ring: np.ndarray = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._queue: "queue.Queue[AudioWindow]" = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._running = threading.Event()

    def _resolve_device_rate(self, device: Optional[int]) -> float:
        info = sd.query_devices(device, "input") if device is not None else sd.query_devices(
            kind="input"
        )
        rate = float(info["default_samplerate"]) if info else float(TARGET_SAMPLE_RATE)
        return rate or float(TARGET_SAMPLE_RATE)

    # -- internal: sounddevice calls this on its own audio thread ----------
    def _callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.warning("Audio input status: %s", status)

        mono = _downmix_to_mono(indata.astype(np.float32))

        if abs(self._resample_ratio - 1.0) > 1e-6:
            mono = _linear_resample(mono, self._resample_ratio)

        with self._lock:
            self._ring = np.concatenate([self._ring, mono])
            while len(self._ring) >= WINDOW_SAMPLES:
                window_samples = self._ring[:WINDOW_SAMPLES].copy()
                self._ring = self._ring[HOP_SAMPLES:]
                window = AudioWindow(samples=window_samples, timestamp=time.time())
                self._queue.put(window)
                if self._on_window is not None:
                    self._on_window(window)

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        blocksize = int(self._native_rate * 0.1)  # 100ms callback chunks
        self._stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self._native_rate,
            blocksize=blocksize,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "Audio capture started (device=%s, native_rate=%.0fHz -> %dHz, window=%.1fs)",
            self.device, self._native_rate, TARGET_SAMPLE_RATE, WINDOW_SECONDS,
        )

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        logger.info("Audio capture stopped.")

    def windows(self, timeout: Optional[float] = None) -> Iterator[AudioWindow]:
        """Blocking generator of AudioWindow objects. Stops yielding once `stop()`
        is called and the queue drains."""
        while self._running.is_set() or not self._queue.empty():
            try:
                yield self._queue.get(timeout=timeout or 0.5)
            except queue.Empty:
                continue

    def record_seconds(self, seconds: float) -> list[AudioWindow]:
        """Convenience for baseline capture: record for a fixed duration and
        return every window produced."""
        collected: list[AudioWindow] = []
        self.start()
        deadline = time.time() + seconds
        try:
            while time.time() < deadline:
                try:
                    collected.append(self._queue.get(timeout=0.5))
                except queue.Empty:
                    continue
        finally:
            self.stop()
        # drain anything left in the queue
        while not self._queue.empty():
            collected.append(self._queue.get_nowait())
        return collected


def _linear_resample(signal: np.ndarray, ratio: float) -> np.ndarray:
    """Cheap linear-interpolation resampler. Good enough for a classification/
    embedding model that isn't sensitive to phase-perfect audio; if you need
    higher fidelity, swap this for `scipy.signal.resample_poly`."""
    if len(signal) == 0:
        return signal
    n_out = max(1, int(round(len(signal) * ratio)))
    x_old = np.linspace(0, 1, num=len(signal), endpoint=False)
    x_new = np.linspace(0, 1, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, signal).astype(np.float32)
