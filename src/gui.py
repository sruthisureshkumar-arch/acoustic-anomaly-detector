"""
gui.py

PyQt6 desktop UI for the Acoustic Anomaly Detector. Wires together
audio_capture, embedding, baseline, and anomaly into something demoable in
under two minutes (see ../DEMO.md).

Threading model:
  - sounddevice's InputStream calls back on its own audio thread. That
    callback (via AudioCapture's `on_window`) only ever does one thing:
    push the finished window onto a plain thread-safe queue.Queue. It never
    touches Qt widgets directly -- Qt widgets may only be touched from the
    main/GUI thread.
  - A QTimer on the GUI thread polls that queue roughly 10x/second, and for
    each window: runs the (potentially somewhat slow, NPU/CPU-bound) YamNet
    embedding + anomaly scoring, then updates the widgets. This keeps model
    inference off the audio thread (which must stay fast to avoid dropped
    audio) without needing a second worker thread for inference.
  - Baseline recording blocks for tens of seconds, so it runs on a small
    QThread subclass instead of the GUI thread, to keep the window responsive
    and show a progress indicator.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen,
    QRadialGradient,
)
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

import anomaly
import baseline
import benchmark
import hardware
from audio_capture import AudioCapture, AudioWindow
from embedding import YamNetEmbedder

logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
EVENTS_LOG = LOGS_DIR / "events.jsonl"

# --------------------------------------------------------------------------
# Palette. One place to change the look; everything else (stylesheet, custom
# widgets) references these instead of hardcoding colors.
# --------------------------------------------------------------------------
COLORS = {
    "bg": "#0b0e13",
    "surface": "#131720",
    "surface_alt": "#1a2029",
    "border": "#252c38",
    "text": "#e8ebf0",
    "text_dim": "#8b93a3",
    "accent": "#4fd1c5",
    "accent_dim": "#2d7d74",
    "normal": "#38c172",
    "warning": "#e0a72e",
    "alert": "#ef4a5f",
    "idle": "#4a5261",
}

STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {COLORS['bg']};
    color: {COLORS['text']};
    font-family: -apple-system, 'Segoe UI', 'Helvetica Neue', sans-serif;
    font-size: 13px;
}}
QSplitter::handle {{
    background: {COLORS['bg']};
    width: 1px;
}}
QLabel {{
    color: {COLORS['text']};
    background: transparent;
}}
QLabel[role="heading"] {{
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: {COLORS['text_dim']};
    text-transform: uppercase;
    padding-bottom: 2px;
}}
QLabel[role="caption"] {{
    color: {COLORS['text_dim']};
    font-size: 12px;
}}
QListWidget {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    padding: 4px;
    outline: none;
}}
QListWidget::item {{
    padding: 8px 10px;
    border-radius: 6px;
    color: {COLORS['text']};
}}
QListWidget::item:selected {{
    background: {COLORS['accent_dim']};
    color: white;
}}
QListWidget::item:hover:!selected {{
    background: {COLORS['surface_alt']};
}}
QComboBox, QSpinBox, QLineEdit {{
    background: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 6px 8px;
    color: {COLORS['text']};
}}
QComboBox:hover, QSpinBox:hover, QLineEdit:hover {{
    border: 1px solid {COLORS['accent_dim']};
}}
QLineEdit:focus {{
    border: 1px solid {COLORS['accent']};
}}
QDialog {{
    background: {COLORS['bg']};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {COLORS['surface_alt']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent_dim']};
    color: {COLORS['text']};
    outline: none;
}}
QPushButton {{
    background: {COLORS['surface_alt']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px 14px;
    color: {COLORS['text']};
    font-weight: 600;
}}
QPushButton:hover {{
    border: 1px solid {COLORS['accent']};
    color: {COLORS['accent']};
}}
QPushButton:disabled {{
    color: {COLORS['text_dim']};
    border: 1px solid {COLORS['border']};
}}
QPushButton[role="primary"] {{
    background: {COLORS['accent_dim']};
    border: 1px solid {COLORS['accent']};
    color: white;
}}
QPushButton[role="primary"]:hover {{
    background: {COLORS['accent']};
    color: {COLORS['bg']};
}}
QMessageBox {{
    background: {COLORS['surface']};
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['border']};
    border-radius: 5px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
"""


def set_list_placeholder(list_widget: QListWidget, text: str) -> None:
    """Shows a single greyed-out, unselectable row in an otherwise-empty
    QListWidget instead of leaving it a blank box -- a blank card reads as
    broken on first launch, before any profile or event exists."""
    list_widget.clear()
    item = QListWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled)
    list_widget.addItem(item)


def card(inner_layout, title: str | None = None) -> QFrame:
    """Wraps a layout in a rounded, bordered 'card' panel — the visual unit
    the whole UI is built from, instead of loose widgets floating on the
    bare background."""
    frame = QFrame()
    frame.setStyleSheet(
        f"QFrame {{ background: {COLORS['surface']}; border: 1px solid {COLORS['border']}; "
        f"border-radius: 10px; }}"
    )
    outer = QVBoxLayout(frame)
    outer.setContentsMargins(14, 12, 14, 14)
    outer.setSpacing(8)
    if title:
        heading = QLabel(title)
        heading.setProperty("role", "heading")
        outer.addWidget(heading)
    outer.addLayout(inner_layout)
    return frame


# --------------------------------------------------------------------------
# Small custom widgets
# --------------------------------------------------------------------------

class WaveformWidget(QWidget):
    """Live waveform: a filled, gradient-glow line over a faint center grid,
    styled to match the rest of the UI instead of a bare debug plot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._samples = None

    def set_samples(self, samples) -> None:
        self._samples = samples
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming convention)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["surface_alt"]))
        painter.drawRoundedRect(rect, 8, 8)

        w, h = self.width(), self.height()
        mid = h / 2
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.drawLine(0, int(mid), w, int(mid))

        if self._samples is None or len(self._samples) < 2:
            painter.setPen(QColor(COLORS["text_dim"]))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "waiting for audio...")
            painter.end()
            return

        step = max(1, len(self._samples) // max(1, w))
        points = self._samples[::step]
        n = len(points)
        path = QPainterPath()
        fill_path = QPainterPath()
        margin = 6
        usable_h = mid - margin
        for i in range(n):
            x = i / max(1, n - 1) * w
            y = mid - float(points[i]) * usable_h
            if i == 0:
                path.moveTo(x, y)
                fill_path.moveTo(x, mid)
                fill_path.lineTo(x, y)
            else:
                path.lineTo(x, y)
                fill_path.lineTo(x, y)
        fill_path.lineTo(w, mid)
        fill_path.closeSubpath()

        gradient = QLinearGradient(0, 0, 0, h)
        fill_color = QColor(COLORS["accent"])
        fill_color.setAlpha(70)
        gradient.setColorAt(0, fill_color)
        transparent = QColor(COLORS["accent"])
        transparent.setAlpha(0)
        gradient.setColorAt(1, transparent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawPath(fill_path)

        painter.setPen(QPen(QColor(COLORS["accent"]), 1.6))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.end()


class StatusLight(QWidget):
    STATE_COLORS = {
        "idle": COLORS["idle"],
        "normal": COLORS["normal"],
        "warning": COLORS["warning"],
        "alert": COLORS["alert"],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(22, 22)
        self._state = "idle"

    def set_state(self, state: str) -> None:
        self._state = state if state in self.STATE_COLORS else "idle"
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(self.STATE_COLORS[self._state])

        # Soft glow halo behind the dot, subtler when idle, more prominent
        # for warning/alert so the eye is drawn to state changes.
        glow_alpha = 40 if self._state != "idle" else 15
        glow = QRadialGradient(11, 11, 11)
        glow_color = QColor(color)
        glow_color.setAlpha(glow_alpha)
        transparent = QColor(color)
        transparent.setAlpha(0)
        glow.setColorAt(0, glow_color)
        glow.setColorAt(1, transparent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(0, 0, 22, 22)

        painter.setBrush(color)
        painter.drawEllipse(6, 6, 10, 10)
        painter.end()


class ArcGauge(QWidget):
    """A semicircular anomaly-score gauge, replacing a plain progress bar.
    0 sits at the left, 100 (= at/over threshold) at the right; the arc's
    color shifts from accent (calm) through warning to alert as the score
    approaches and crosses threshold, so the state reads at a glance even
    from across a room during a demo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._value = 0.0  # 0-100

    def setValue(self, value: float) -> None:
        self._value = max(0.0, min(150.0, value))  # allow a bit past 100 to show overshoot
        self.update()

    def _color_for_value(self) -> QColor:
        if self._value >= 100:
            return QColor(COLORS["alert"])
        if self._value >= 75:
            return QColor(COLORS["warning"])
        return QColor(COLORS["accent"])

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        side = min(w, h * 2)
        cx, cy = w / 2, h - 14
        radius = side / 2 - 16
        rect = QRectF(cx - radius, cy - radius, radius * 2, radius * 2)

        track_pen = QPen(QColor(COLORS["border"]), 14, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawArc(rect, 0 * 16, 180 * 16)

        fraction = min(1.0, self._value / 100.0)
        span = int(180 * fraction * 16)
        arc_pen = QPen(self._color_for_value(), 14, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(arc_pen)
        painter.drawArc(rect, 180 * 16, -span)

        # Threshold tick at 100% (straight up the right edge of "safe").
        painter.setPen(QPen(QColor(COLORS["text_dim"]), 2))

        painter.setPen(QColor(COLORS["text"]))
        font = QFont(painter.font())
        font.setPointSize(20)
        font.setWeight(QFont.Weight.DemiBold)
        painter.setFont(font)
        label = f"{self._value:.0f}" if self._value < 150 else "150+"
        painter.drawText(QRectF(0, cy - radius - 6, w, radius), Qt.AlignmentFlag.AlignCenter, label)

        small_font = QFont(painter.font())
        small_font.setPointSize(9)
        small_font.setWeight(QFont.Weight.Normal)
        painter.setFont(small_font)
        painter.setPen(QColor(COLORS["text_dim"]))
        painter.drawText(QRectF(0, cy - 18, w, 16), Qt.AlignmentFlag.AlignCenter, "of threshold")
        painter.end()


class ScoreHistoryWidget(QWidget):
    """Rolling line chart of the smoothed anomaly score over the last few
    minutes of monitoring, with the current threshold drawn as a dashed
    reference line. The arc gauge only shows the instantaneous score; this
    turns "the score is 40 right now" into "watch it climb toward threshold
    over the last two minutes" -- a much stronger demo beat, and closer to
    how real predictive-maintenance tools present trend data rather than a
    single live number."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._scores: list[float] = []
        self._threshold: float = 0.0

    def set_history(self, scores: list[float], threshold: float) -> None:
        self._scores = scores
        self._threshold = threshold
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(COLORS["surface_alt"]))
        painter.drawRoundedRect(rect, 8, 8)

        w, h = self.width(), self.height()
        margin_top, margin_bottom = 10, 10

        if len(self._scores) < 2:
            painter.setPen(QColor(COLORS["text_dim"]))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "score history will appear here")
            painter.end()
            return

        top_value = max(self._threshold * 1.25, max(self._scores) * 1.05, 1e-6)
        usable_h = h - margin_top - margin_bottom

        def y_for(value: float) -> float:
            frac = min(1.0, value / top_value)
            return margin_top + usable_h - frac * usable_h

        if self._threshold > 0:
            ty = y_for(self._threshold)
            painter.setPen(QPen(QColor(COLORS["text_dim"]), 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(0, ty), QPointF(w, ty))

        n = len(self._scores)
        path = QPainterPath()
        fill_path = QPainterPath()
        for i, score in enumerate(self._scores):
            x = i / max(1, n - 1) * w
            y = y_for(score)
            if i == 0:
                path.moveTo(x, y)
                fill_path.moveTo(x, h - margin_bottom)
                fill_path.lineTo(x, y)
            else:
                path.lineTo(x, y)
                fill_path.lineTo(x, y)
        fill_path.lineTo(w, h - margin_bottom)
        fill_path.closeSubpath()

        last_score = self._scores[-1]
        line_color = QColor(COLORS["alert"] if last_score >= self._threshold > 0 else COLORS["accent"])
        fill_color = QColor(line_color)
        fill_color.setAlpha(45)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(fill_color))
        painter.drawPath(fill_path)

        painter.setPen(QPen(line_color, 1.8))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        painter.end()


# --------------------------------------------------------------------------
# Background worker for baseline recording (keeps the GUI responsive)
# --------------------------------------------------------------------------

class BaselineRecorderThread(QThread):
    progress = pyqtSignal(int)        # seconds elapsed
    finished_ok = pyqtSignal(object)  # baseline.BaselineProfile
    failed = pyqtSignal(str)

    def __init__(self, name: str, embedder: YamNetEmbedder, device, seconds: float, parent=None):
        super().__init__(parent)
        self.name = name
        self.embedder = embedder
        self.device = device
        self.seconds = seconds
        self._stop_requested = threading.Event()

    def request_stop(self) -> None:
        """Ends the recording early instead of running the full configured
        duration. Used by MainWindow.closeEvent so quitting the app (e.g.
        mid-baseline) doesn't leave the mic stream open and the terminal
        hanging for up to `seconds` while this thread keeps running."""
        self._stop_requested.set()

    def run(self) -> None:
        try:
            capture = AudioCapture(device=self.device)
            capture.start()
            windows: list[AudioWindow] = []
            start = time.time()
            while time.time() - start < self.seconds and not self._stop_requested.is_set():
                elapsed = int(time.time() - start)
                self.progress.emit(elapsed)
                time.sleep(0.2)
                try:
                    while True:
                        windows.append(capture._queue.get_nowait())  # internal, but we own the thread
                except queue.Empty:
                    pass
            capture.stop()
            while not capture._queue.empty():
                windows.append(capture._queue.get_nowait())

            if self._stop_requested.is_set():
                # Cut short deliberately (app closing) -- don't save a
                # partial/misleading profile, and don't bother emitting into
                # a window that may already be gone.
                return

            if len(windows) < 5:
                self.failed.emit(
                    f"Only captured {len(windows)} audio windows -- check the microphone "
                    f"and try again."
                )
                return

            import numpy as np
            embeddings = np.stack([self.embedder.embed(w.samples) for w in windows])
            profile = baseline.build_and_calibrate_profile(self.name, embeddings)
            baseline.save_profile(profile)
            self.finished_ok.emit(profile)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the GUI
            logger.exception("Baseline recording failed")
            self.failed.emit(str(exc))


class BenchmarkThread(QThread):
    """Runs the NPU-vs-CPU inference benchmark off the GUI thread -- it does
    ~100 real inference calls (across both providers), which is enough to
    visibly freeze the UI if run inline."""
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            results = benchmark.run_benchmark()
            benchmark.save_results(results)
            self.finished_ok.emit(results)
        except Exception as exc:  # noqa: BLE001 - surface any failure to the GUI
            logger.exception("Benchmark failed")
            self.failed.emit(str(exc))


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Acoustic Anomaly Detector — offline, on-device")
        self.resize(900, 600)

        self.embedder: YamNetEmbedder | None = None
        self.capture: AudioCapture | None = None
        self.scorer: anomaly.AnomalyScorer | None = None
        self.current_profile: baseline.BaselineProfile | None = None
        self._window_queue: "queue.Queue[AudioWindow]" = queue.Queue()
        self._was_in_alert = False  # tracks the previous window's alert state,
                                     # so the sound plays once per alert onset,
                                     # not once per window while it's sustained
        self._event_log_has_real_entries = False  # so the first real event
                                                    # clears the placeholder row
        # ~400 windows at roughly 2/sec (50%-overlapping 0.96s windows) is a
        # few minutes of trend -- enough to show a score climbing toward
        # threshold without the chart getting so dense it's unreadable.
        self._score_history: deque[float] = deque(maxlen=400)

        self._build_ui()
        self._show_hardware_name()
        self._load_model()
        self._refresh_profiles()
        self._load_alert_sound()

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(100)  # 10x/sec
        self.poll_timer.timeout.connect(self._poll_windows)

    # -- setup -----------------------------------------------------------
    def _load_alert_sound(self) -> None:
        self._alert_sound = None
        alert_path = Path(__file__).resolve().parent.parent / "assets" / "alert.wav"
        try:
            import soundfile as sf
            data, sr = sf.read(str(alert_path), dtype="float32")
            self._alert_sound = (data, sr)
        except Exception as exc:  # noqa: BLE001 - alert sound is a nice-to-have, never fatal
            logger.warning("Could not load alert sound (%s): %s", alert_path, exc)

    def _play_alert_sound(self) -> None:
        if self._alert_sound is None:
            return
        try:
            import sounddevice as sd
            data, sr = self._alert_sound
            sd.play(data, sr)
        except Exception as exc:  # noqa: BLE001 - never let a sound glitch break monitoring
            logger.warning("Could not play alert sound: %s", exc)

    def _show_hardware_name(self) -> None:
        # Independent of model loading (runs even if the model fails to
        # load) and never allowed to raise -- hardware.detect_hardware_name()
        # already catches its own failures, but this is one more belt to go
        # with that suspenders, since a "nice to have" label should never be
        # able to break app startup.
        try:
            name = hardware.detect_hardware_name()
            self.hardware_label.setText(f"Hardware: {name}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not display hardware name: %s", exc)
            self.hardware_label.setText("")

    def _load_model(self) -> None:
        try:
            self.embedder = YamNetEmbedder()
            provider = self.embedder.provider_info
            label = "NPU (QNN)" if provider.is_npu else f"CPU ({provider.active})"
            self.provider_label.setText(f"Inference: {label} — model: {self.embedder.model_path.name}")
            if not provider.is_npu:
                self.provider_label.setStyleSheet(f"color: {COLORS['warning']};")
            self._show_cached_benchmark()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load embedding model")
            self.provider_label.setText(f"Model load failed: {exc}")
            self.provider_label.setStyleSheet(f"color: {COLORS['alert']};")

    def _show_cached_benchmark(self) -> None:
        """Shows the most recently saved NPU-vs-CPU benchmark (if any) next
        to the provider label, without re-running it on every launch -- the
        "Benchmark NPU vs CPU" button re-runs it live, e.g. on stage."""
        results = benchmark.load_results()
        if results and self.embedder is not None and results.get("model") == self.embedder.model_path.name:
            self.benchmark_label.setText(benchmark.summarize(results))
        else:
            self.benchmark_label.setText("No benchmark yet — click \"Benchmark NPU vs CPU\".")

    def _build_ui(self) -> None:
        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(14, 14, 14, 14)
        central_layout.setSpacing(14)

        # -- left column: profiles + controls -------------------------------
        left_col = QVBoxLayout()
        left_col.setSpacing(14)

        profiles_layout = QVBoxLayout()
        self.profile_list = QListWidget()
        self.profile_list.currentTextChanged.connect(self._on_profile_selected)
        profiles_layout.addWidget(self.profile_list)
        left_col.addWidget(card(profiles_layout, title="Machine profiles"))

        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(6)
        mic_caption = QLabel("Microphone")
        mic_caption.setProperty("role", "caption")
        controls_layout.addWidget(mic_caption)
        self.device_combo = QComboBox()
        self._populate_devices()
        controls_layout.addWidget(self.device_combo)

        self.baseline_seconds = QSpinBox()
        self.baseline_seconds.setRange(20, 300)
        self.baseline_seconds.setValue(90)
        self.baseline_seconds.setSuffix(" s baseline")
        controls_layout.addWidget(self.baseline_seconds)

        new_profile_btn = QPushButton("Record new baseline...")
        new_profile_btn.setProperty("role", "primary")
        new_profile_btn.clicked.connect(self._start_new_baseline)
        controls_layout.addWidget(new_profile_btn)

        metric_caption = QLabel("Scoring metric")
        metric_caption.setProperty("role", "caption")
        controls_layout.addWidget(metric_caption)
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["mahalanobis", "cosine"])
        controls_layout.addWidget(self.metric_combo)
        left_col.addWidget(card(controls_layout, title="Setup"))

        left_col.addStretch(1)

        bench_layout = QVBoxLayout()
        bench_layout.setSpacing(6)
        self.hardware_label = QLabel("")
        self.hardware_label.setProperty("role", "caption")
        self.hardware_label.setWordWrap(True)
        bench_layout.addWidget(self.hardware_label)
        self.provider_label = QLabel("Inference: (loading model...)")
        self.provider_label.setProperty("role", "caption")
        self.provider_label.setWordWrap(True)
        bench_layout.addWidget(self.provider_label)
        self.benchmark_label = QLabel("")
        self.benchmark_label.setProperty("role", "caption")
        self.benchmark_label.setWordWrap(True)
        self.benchmark_label.setStyleSheet(f"color: {COLORS['accent']};")
        bench_layout.addWidget(self.benchmark_label)
        self.benchmark_btn = QPushButton("Benchmark NPU vs CPU")
        self.benchmark_btn.clicked.connect(self._run_benchmark)
        bench_layout.addWidget(self.benchmark_btn)
        left_col.addWidget(card(bench_layout, title="Performance proof"))

        left_widget = QWidget()
        left_widget.setLayout(left_col)
        left_widget.setMinimumWidth(260)
        left_widget.setMaximumWidth(320)
        central_layout.addWidget(left_widget)

        # -- right column: monitor ------------------------------------------
        right_col = QVBoxLayout()
        right_col.setSpacing(14)

        status_layout = QVBoxLayout()
        top_row = QHBoxLayout()
        self.status_light = StatusLight()
        top_row.addWidget(self.status_light)
        self.status_text = QLabel("No profile selected")
        self.status_text.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {COLORS['text']};")
        top_row.addWidget(self.status_text)
        top_row.addStretch(1)
        self.start_btn = QPushButton("Start monitoring")
        self.start_btn.setProperty("role", "primary")
        self.start_btn.clicked.connect(self._start_monitoring)
        self.start_btn.setEnabled(False)
        top_row.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop_monitoring)
        self.stop_btn.setEnabled(False)
        top_row.addWidget(self.stop_btn)
        status_layout.addLayout(top_row)

        self.waveform = WaveformWidget()
        status_layout.addWidget(self.waveform)
        right_col.addWidget(card(status_layout, title="Live monitor"))

        gauge_layout = QVBoxLayout()
        gauge_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.gauge = ArcGauge()
        gauge_layout.addWidget(self.gauge)
        self.score_history = ScoreHistoryWidget()
        gauge_layout.addWidget(self.score_history)
        right_col.addWidget(card(gauge_layout, title="Anomaly score"))

        log_layout = QVBoxLayout()
        self.event_log = QListWidget()
        set_list_placeholder(self.event_log, "No events yet — they'll appear here once monitoring starts.")
        log_layout.addWidget(self.event_log)

        feedback_row = QHBoxLayout()
        self.false_alarm_btn = QPushButton("Mark last alert: false alarm")
        self.false_alarm_btn.clicked.connect(self._mark_false_alarm)
        self.false_alarm_btn.setEnabled(False)
        feedback_row.addWidget(self.false_alarm_btn)
        self.confirmed_btn = QPushButton("Mark last alert: real issue")
        self.confirmed_btn.clicked.connect(self._mark_confirmed)
        self.confirmed_btn.setEnabled(False)
        feedback_row.addWidget(self.confirmed_btn)
        log_layout.addLayout(feedback_row)
        right_col.addWidget(card(log_layout, title="Event log"), stretch=1)

        right_widget = QWidget()
        right_widget.setLayout(right_col)
        central_layout.addWidget(right_widget, stretch=1)

        self.setCentralWidget(central)

    def _populate_devices(self) -> None:
        self.device_combo.addItem("System default", None)
        try:
            from audio_capture import list_input_devices
            for dev in list_input_devices():
                self.device_combo.addItem(dev["name"], dev["index"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not enumerate audio devices: %s", exc)

    # -- profiles ---------------------------------------------------------
    def _refresh_profiles(self) -> None:
        self.profile_list.clear()
        names = baseline.list_profiles()
        if not names:
            set_list_placeholder(
                self.profile_list, "No profiles yet — record a baseline below to add one."
            )
            return
        for name in names:
            self.profile_list.addItem(QListWidgetItem(name))

    def _on_profile_selected(self, name: str) -> None:
        if not name:
            return
        try:
            self.current_profile = baseline.load_profile(name)
            self.status_text.setText(f"Profile: {name} (ready)")
            self.status_light.set_state("normal")
            self.start_btn.setEnabled(self.embedder is not None)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Could not load profile", str(exc))

    def _prompt_profile_name(self) -> str | None:
        """A small dialog for the new-profile name, styled to match the rest
        of the app -- QInputDialog.getText() renders as an unstyled native
        macOS sheet that ignores our stylesheet and looks visually broken
        sitting on top of the dark theme, so we build our own instead."""
        dialog = QDialog(self)
        dialog.setWindowTitle("New machine profile")
        dialog.setFixedWidth(360)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        label = QLabel('Name this machine (e.g. "Lathe #1"):')
        label.setWordWrap(True)
        layout.addWidget(label)

        line_edit = QLineEdit()
        line_edit.setPlaceholderText("Lathe #1")
        layout.addWidget(line_edit)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_row.addWidget(cancel_btn)
        create_btn = QPushButton("Create")
        create_btn.setProperty("role", "primary")
        create_btn.clicked.connect(dialog.accept)
        create_btn.setDefault(True)
        button_row.addWidget(create_btn)
        layout.addLayout(button_row)

        line_edit.returnPressed.connect(dialog.accept)
        line_edit.setFocus()

        if dialog.exec() == QDialog.DialogCode.Accepted:
            return line_edit.text().strip()
        return None

    def _start_new_baseline(self) -> None:
        if self.embedder is None:
            QMessageBox.warning(self, "Model not loaded", "The embedding model failed to load; "
                                                            "see the status text on the left.")
            return
        name = self._prompt_profile_name()
        if not name:
            return

        seconds = self.baseline_seconds.value()
        device = self.device_combo.currentData()
        self._baseline_thread = BaselineRecorderThread(name.strip(), self.embedder, device, seconds)
        self._baseline_thread.progress.connect(
            lambda s: self.status_text.setText(f"Recording baseline for '{name}'... {s}/{seconds}s")
        )
        self._baseline_thread.finished_ok.connect(self._on_baseline_done)
        self._baseline_thread.failed.connect(
            lambda msg: QMessageBox.warning(self, "Baseline recording failed", msg)
        )
        self.status_light.set_state("warning")
        self._baseline_thread.start()

    def _on_baseline_done(self, profile: baseline.BaselineProfile) -> None:
        self._refresh_profiles()
        items = self.profile_list.findItems(profile.name, Qt.MatchFlag.MatchExactly)
        if items:
            self.profile_list.setCurrentItem(items[0])
        mode_note = "" if profile.cov_mode == "full" else " (diagonal — record a longer baseline for full covariance)"
        self.status_text.setText(f"Baseline '{profile.name}' saved{mode_note}")
        self.status_light.set_state("normal")

    # -- performance proof ---------------------------------------------------
    def _run_benchmark(self) -> None:
        self.benchmark_btn.setEnabled(False)
        self.benchmark_label.setText("Benchmarking... (times ~50 inference calls per provider)")
        self._benchmark_thread = BenchmarkThread()
        self._benchmark_thread.finished_ok.connect(self._on_benchmark_done)
        self._benchmark_thread.failed.connect(self._on_benchmark_failed)
        self._benchmark_thread.start()

    def _on_benchmark_done(self, results: dict) -> None:
        self.benchmark_btn.setEnabled(True)
        self.benchmark_label.setText(benchmark.summarize(results))
        if "npu_speedup_x" in results:
            npu = results["QNNExecutionProvider"]
            cpu = results["CPUExecutionProvider"]
            QMessageBox.information(
                self, "Benchmark complete",
                f"Model: {results['model']}\n\n"
                f"NPU (QNN):  {npu['mean_ms']:.2f} ms/window  ({npu['throughput_hz']:.1f} windows/sec)\n"
                f"CPU:        {cpu['mean_ms']:.2f} ms/window  ({cpu['throughput_hz']:.1f} windows/sec)\n\n"
                f"NPU is {results['npu_speedup_x']}x faster than CPU on this device.\n\n"
                f"Saved to models/benchmark_results.json.",
            )
        else:
            for key, val in results.items():
                if isinstance(val, dict):
                    QMessageBox.information(
                        self, "Benchmark complete",
                        f"Model: {results['model']}\n\n"
                        f"{key}: {val['mean_ms']:.2f} ms/window ({val['throughput_hz']:.1f} windows/sec)\n\n"
                        f"Only one execution provider was available on this machine, so there's "
                        f"no NPU-vs-CPU comparison here -- run this on the actual Snapdragon "
                        f"hardware to get that number.",
                    )
                    break

    def _on_benchmark_failed(self, msg: str) -> None:
        self.benchmark_btn.setEnabled(True)
        self.benchmark_label.setText("Benchmark failed.")
        QMessageBox.warning(self, "Benchmark failed", msg)

    # -- monitoring ---------------------------------------------------------
    def _start_monitoring(self) -> None:
        if self.current_profile is None or self.embedder is None:
            return
        device = self.device_combo.currentData()
        metric = self.metric_combo.currentText()
        self.scorer = anomaly.AnomalyScorer(self.current_profile, metric=metric)

        self.capture = AudioCapture(device=device, on_window=self._on_window_from_audio_thread)
        self.capture.start()
        self._was_in_alert = False
        self._score_history.clear()  # don't carry a previous machine/session's trend into this one
        self.score_history.set_history([], 0.0)
        self.poll_timer.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.false_alarm_btn.setEnabled(True)
        self.confirmed_btn.setEnabled(True)
        self.status_text.setText(f"Monitoring '{self.current_profile.name}'...")
        self.status_light.set_state("normal")

    def _stop_monitoring(self) -> None:
        self.poll_timer.stop()
        if self.capture is not None:
            self.capture.stop()
            self.capture = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        # Deliberately NOT disabling the mark-alert buttons here: this used to
        # force them off on every stop, which made it impossible to give
        # feedback on an alert from the session you just stopped -- exactly
        # the moment you're most likely to want to (you stopped to go look at
        # the machine, and now you know whether it was real). They stay
        # enabled as long as this session actually scored something; only
        # reset them if this session never got as far as producing a scorer.
        has_scored_session = self.scorer is not None and self.scorer.last_event is not None
        self.false_alarm_btn.setEnabled(has_scored_session)
        self.confirmed_btn.setEnabled(has_scored_session)
        self.status_text.setText(f"Stopped. Profile: {self.current_profile.name if self.current_profile else '-'}")
        self.status_light.set_state("idle")

    def _on_window_from_audio_thread(self, window: AudioWindow) -> None:
        # Runs on sounddevice's audio thread -- must NOT touch Qt widgets.
        self._window_queue.put(window)

    def _poll_windows(self) -> None:
        try:
            while True:
                window = self._window_queue.get_nowait()
                self._process_window(window)
        except queue.Empty:
            pass

    def _process_window(self, window: AudioWindow) -> None:
        if self.embedder is None or self.scorer is None:
            return
        self.waveform.set_samples(window.samples)

        embedding = self.embedder.embed(window.samples)
        event = self.scorer.update(embedding)
        self.gauge.setValue(self.scorer.gauge_value(event))
        self._score_history.append(event.smoothed_score)
        self.score_history.set_history(list(self._score_history), event.threshold)

        if self.scorer.in_alert:
            self.status_light.set_state("alert")
            self.status_text.setText(
                f"ANOMALY — '{self.current_profile.name}' score {event.smoothed_score:.2f} "
                f"(threshold {event.threshold:.2f})"
            )
            if not self._was_in_alert:
                # Rising edge only: play once per alert onset, not every
                # window for as long as the anomaly is sustained.
                self._play_alert_sound()
            self._log_event(event, alert=True)
        else:
            state = "warning" if event.smoothed_score >= event.threshold * 0.8 else "normal"
            self.status_light.set_state(state)
            self.status_text.setText(f"Monitoring '{self.current_profile.name}' — nominal")
        self._was_in_alert = self.scorer.in_alert

    def _log_event(self, event: anomaly.AnomalyEvent, alert: bool) -> None:
        line = (f"{time.strftime('%H:%M:%S', time.localtime(event.timestamp))}  "
                f"score={event.smoothed_score:.3f}  threshold={event.threshold:.3f}  "
                f"metric={event.metric}")
        item = QListWidgetItem(("⚠ " if alert else "") + line)
        if not self._event_log_has_real_entries:
            self.event_log.clear()  # drop the "no events yet" placeholder row
            self._event_log_has_real_entries = True
        self.event_log.insertItem(0, item)

        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        import json
        with EVENTS_LOG.open("a") as fh:
            fh.write(json.dumps({
                "profile": self.current_profile.name if self.current_profile else None,
                "timestamp": event.timestamp, "smoothed_score": event.smoothed_score,
                "threshold": event.threshold, "metric": event.metric, "alert": alert,
            }) + "\n")

    def _mark_false_alarm(self) -> None:
        # Deliberately not using QMessageBox here: on this machine, native
        # unstyled dialogs (see _prompt_profile_name's comment above) have
        # rendered invisibly/behind-window before, which made a working
        # button look broken. The status label is always on screen and
        # already styled, so it can't fail silently the same way.
        if self.scorer is None or self.current_profile is None:
            self.status_text.setText(
                "Nothing to mark -- start monitoring a profile first."
            )
            return
        self.scorer.mark_false_alarm()
        baseline.save_profile(self.current_profile)
        self.status_text.setText(
            f"Noted: false alarm for '{self.current_profile.name}' -- threshold widened slightly."
        )

    def _mark_confirmed(self) -> None:
        if self.scorer is None or self.current_profile is None:
            self.status_text.setText(
                "Nothing to mark -- start monitoring a profile first."
            )
            return
        self.scorer.mark_confirmed_issue()
        baseline.save_profile(self.current_profile)
        self.status_text.setText(
            f"Noted: confirmed issue recorded for '{self.current_profile.name}'."
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_monitoring()

        # A running baseline-recording or benchmark thread otherwise keeps
        # the mic stream (or model sessions) alive after the window closes,
        # which is what made the terminal hang / not return to the prompt on
        # quit -- ask them to stop and give them a moment before falling
        # back to a hard terminate() so the app always actually exits.
        baseline_thread = getattr(self, "_baseline_thread", None)
        if baseline_thread is not None and baseline_thread.isRunning():
            baseline_thread.request_stop()
            if not baseline_thread.wait(3000):
                logger.warning("Baseline recording thread didn't stop in time; terminating it.")
                baseline_thread.terminate()
                baseline_thread.wait()

        benchmark_thread = getattr(self, "_benchmark_thread", None)
        if benchmark_thread is not None and benchmark_thread.isRunning():
            if not benchmark_thread.wait(3000):
                logger.warning("Benchmark thread didn't stop in time; terminating it.")
                benchmark_thread.terminate()
                benchmark_thread.wait()

        event.accept()


def run() -> None:
    import sys
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
