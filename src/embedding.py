"""
embedding.py

Loads the YamNet ONNX model and turns a ~0.96s audio window into a
fixed-length feature vector, preferring the NPU (via ONNX Runtime's QNN
Execution Provider) and falling back to CPU automatically.

Two model configurations are supported, switched by config/model source:

  1. The real submission asset: a YamNet model compiled for a Snapdragon
     target through Qualcomm AI Hub (`models/yamnet_npu.onnx`). This is what
     actually exercises the Hexagon NPU on the HP Omnibook.
  2. A plain public YamNet ONNX export (`models/yamnet_public.onnx`), used
     for development/testing on machines without QNN support (e.g. this
     build environment, or a non-Snapdragon laptop). It only ever runs on
     CPUExecutionProvider, but exercises identical downstream logic.

IMPORTANT, and different from this file's original design: the actual
`qai-hub-models export yamnet --runtime onnx` asset does NOT take a raw
waveform and does NOT expose a separate embedding layer. Inspecting the real
exported graph (`models/README.md` has the exact commands) shows:

    input  "audio"        float32 [1, 1, 96, 64]   -- a log-mel spectrogram patch
    output "class_scores" float32 [1, 521]         -- AudioSet classifier output

So there are two real changes from the original plan:

  1. We have to compute a log-mel spectrogram from the raw waveform
     ourselves before the model ever sees it -- feeding raw samples (what an
     earlier version of this file did) silently produces garbage, or in this
     case loudly fails with a rank-mismatch error, which is how this got
     caught. We reuse `torch_audioset`'s own `WaveformToInput` transform for
     this rather than hand-deriving the STFT/mel-filterbank math, since it's
     the exact reference implementation Qualcomm's export was traced from --
     matching it by hand from memory risks a subtle numerical mismatch
     (window function, centering, htk-vs-slaney mel scale, normalization)
     that wouldn't crash, it would just quietly make every embedding a little
     wrong. `torch`/`torchaudio`/`torch_audioset` are therefore a real
     (heavier-than-ideal) runtime dependency of this file specifically for
     that preprocessing step -- see requirements.txt. This is a reasonable
     target for a post-hackathon optimization pass (port the mel-spectrogram
     math to plain numpy once validated against this reference), not
     something worth risking correctness over under deadline.
  2. There is no separate embedding layer in this export -- only the
     521-dim classifier output. We use THAT as our feature vector instead.
     It's a real change from the original "use the embedding, not the
     classifier" design principle: the 521-d class-score vector has been
     squeezed through a classification bottleneck trained on AudioSet's
     categories, which are not our categories, so it's a noisier signal
     than a true embedding layer would be. It still works for this app's
     actual job (notice when a specific machine's sound profile shifts,
     not identify what the sound is) because build_and_calibrate_profile's
     held-out calibration and baseline.py's dimension-agnostic statistics
     don't care what the feature vector "means" semantically, only that a
     given machine's baseline occupies a consistent region of it. Everything
     downstream of `embed()` (baseline.py, anomaly.py) is already written
     dimension-agnostically, so this needed no changes outside this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
NPU_MODEL_PATH = MODELS_DIR / "yamnet_npu.onnx"          # from qai-hub compile (see README)
PUBLIC_MODEL_PATH = MODELS_DIR / "yamnet_public.onnx"    # CPU-only dev/test fallback

# Must match the window length the real YamNet export's mel-spectrogram
# front end expects a full 96-frame patch from: CommonParams.SAMPLE_RATE *
# CommonParams.PATCH_WINDOW_IN_SECONDS in torch_audioset's params.py
# (16000 * 0.96 = 15360). audio_capture.py's WINDOW_SAMPLES must match this.
WAVEFORM_SAMPLES_PER_PATCH = 15_360

# Heuristics for picking the feature output out of a YamNet ONNX graph, since
# export tooling doesn't always use the same output names, and (see module
# docstring) the real export only exposes classifier scores anyway.
_EMBEDDING_NAME_HINTS = ("embedding", "penultimate", "feature")
_SCORES_NAME_HINTS = ("scores", "logits", "class", "prediction")


@dataclass
class ProviderInfo:
    requested: list[str]
    active: str            # the provider ONNX Runtime actually placed the graph on
    is_npu: bool


class YamNetEmbedder:
    def __init__(self, model_path: Optional[Path] = None, prefer_npu: bool = True):
        self.model_path = Path(model_path) if model_path else self._resolve_model_path()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"No YamNet ONNX model found at {self.model_path}. "
                f"See models/README.md for how to obtain either the real "
                f"qai-hub-compiled NPU asset or the public CPU-testing model."
            )

        providers = self._build_provider_list(prefer_npu)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=so, providers=providers
        )
        self.provider_info = self._describe_active_provider(providers)
        logger.info(
            "YamNet loaded from %s | requested providers=%s | active=%s | NPU=%s",
            self.model_path.name, self.provider_info.requested,
            self.provider_info.active, self.provider_info.is_npu,
        )

        self._input_meta = self.session.get_inputs()[0]
        self._input_rank = len(self._input_meta.shape)
        self._feature_output_name = self._pick_output_name()
        self._mel_transform = None  # lazily constructed; see _get_mel_transform

    # -- setup helpers -------------------------------------------------
    def _resolve_model_path(self) -> Path:
        if NPU_MODEL_PATH.exists():
            return NPU_MODEL_PATH
        if PUBLIC_MODEL_PATH.exists():
            logger.warning(
                "Using the public CPU-only YamNet model (%s). This is fine for "
                "development, but the hackathon submission should ship with the "
                "qai-hub-compiled asset at %s so the NPU claim is real. See "
                "models/README.md.", PUBLIC_MODEL_PATH.name, NPU_MODEL_PATH.name,
            )
            return PUBLIC_MODEL_PATH
        # Neither exists yet; default to the NPU path so the FileNotFoundError
        # message points at the right target.
        return NPU_MODEL_PATH

    def _build_provider_list(self, prefer_npu: bool) -> list[str]:
        available = ort.get_available_providers()
        providers = []
        if prefer_npu and "QNNExecutionProvider" in available:
            providers.append("QNNExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _describe_active_provider(self, requested: list[str]) -> ProviderInfo:
        active = self.session.get_providers()[0]
        return ProviderInfo(requested=requested, active=active,
                             is_npu=(active == "QNNExecutionProvider"))

    def _pick_output_name(self) -> str:
        outputs = self.session.get_outputs()
        if len(outputs) == 1:
            return outputs[0].name

        for out in outputs:
            name_lower = out.name.lower()
            if any(hint in name_lower for hint in _EMBEDDING_NAME_HINTS):
                return out.name

        # No obvious "embedding" name: fall back to whichever output is NOT the
        # classification scores, then to the widest (most feature-dimensions)
        # output as a last resort. For the real export this correctly falls
        # through to using "class_scores" itself, since it's the only output.
        non_score_outputs = [
            out for out in outputs
            if not any(hint in out.name.lower() for hint in _SCORES_NAME_HINTS)
        ]
        candidates = non_score_outputs or outputs
        widest = max(candidates, key=lambda o: _static_width(o.shape))
        logger.warning(
            "Could not identify a preferred feature output by name among %s; "
            "using '%s'.", [o.name for o in outputs], widest.name,
        )
        return widest.name

    def _get_mel_transform(self):
        """Lazily construct the log-mel spectrogram front end. Deferred
        (rather than imported at module load time) so that a rank-2 input
        model (e.g. the raw-waveform dev placeholder) never needs torch/
        torchaudio/torch_audioset installed at all -- only the real 4D-input
        export does.

        torch_audioset is vendored (git-cloned into vendor/torch_audioset/)
        rather than pip-installed: its setup.py uses an old-style distutils
        `install_layout` option that breaks under newer setuptools on some
        platforms. Its own code has no packaging dependencies once on
        sys.path, so cloning it and importing directly sidesteps the whole
        problem -- see models/README.md for the clone command."""
        if self._mel_transform is None:
            _ensure_vendored_torch_audioset_on_path()
            import torch
            from torch_audioset.data.torch_input_processing import WaveformToInput
            # Belt-and-suspenders alongside main.py's OMP/VECLIB/MKL thread-count
            # env vars: this embedding runs on a background QThread (baseline
            # recording), and multi-threaded PyTorch CPU math is not reliably
            # thread-safe when called off the main thread on macOS (Accelerate
            # framework), which caused a bus error crash after a full baseline
            # recording. Forcing single-threaded torch avoids that regardless
            # of whether the env vars were set before torch's own import.
            torch.set_num_threads(1)
            self._mel_transform = WaveformToInput()
        return self._mel_transform

    # -- inference -------------------------------------------------------
    def embed(self, window: np.ndarray) -> np.ndarray:
        """window: float32 mono waveform, ~0.96s at 16kHz (audio_capture.py's
        WINDOW_SAMPLES == WAVEFORM_SAMPLES_PER_PATCH). Returns a 1D feature
        vector (L2-normalized) for use by baseline/anomaly scoring."""
        feed = {self._input_meta.name: self._preprocess(window)}
        outputs = self.session.run([self._feature_output_name], feed)
        feat = np.asarray(outputs[0]).reshape(-1).astype(np.float32)

        if feat.ndim != 1 or feat.size == 0:
            raise RuntimeError(
                f"Unexpected feature shape {np.asarray(outputs[0]).shape} from "
                f"output '{self._feature_output_name}'."
            )

        norm = np.linalg.norm(feat)
        return feat / norm if norm > 1e-8 else feat

    def _preprocess(self, window: np.ndarray) -> np.ndarray:
        if self._input_rank == 4:
            return self._waveform_to_mel_patch(window)
        if self._input_rank == 2:
            # Older/alternate export that takes a raw waveform directly.
            return self._fit_to_expected_length_2d(window).reshape(1, -1).astype(np.float32)
        raise RuntimeError(
            f"Don't know how to feed a rank-{self._input_rank} input "
            f"(shape {self._input_meta.shape}) -- only rank-4 [1,1,96,64] "
            f"log-mel-patch and rank-2 [1,N] raw-waveform inputs are handled."
        )

    def _waveform_to_mel_patch(self, window: np.ndarray) -> np.ndarray:
        import torch

        wav = window
        if len(wav) != WAVEFORM_SAMPLES_PER_PATCH:
            # Should not normally happen if audio_capture.py's WINDOW_SAMPLES
            # is kept in sync with this constant, but pad/trim defensively
            # rather than let torch_audioset's internal chunking silently
            # produce zero or multiple chunks for an odd-length window.
            wav = _pad_or_trim(wav, WAVEFORM_SAMPLES_PER_PATCH)

        waveform_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)  # [1, N]
        mel_transform = self._get_mel_transform()
        patches = mel_transform(waveform_t, sample_rate=16_000)  # [num_chunks, 1, 96, 64]

        if patches.shape[0] != 1:
            # Our window is sized to produce exactly one non-overlapping
            # patch; if it ever doesn't (e.g. WINDOW_SAMPLES drifted out of
            # sync with WAVEFORM_SAMPLES_PER_PATCH), take the first patch
            # rather than silently averaging/misinterpreting multiple ones.
            logger.warning(
                "Expected exactly 1 mel patch from a %d-sample window, got %d; "
                "using the first. Check audio_capture.WINDOW_SAMPLES matches "
                "embedding.WAVEFORM_SAMPLES_PER_PATCH.", len(window), patches.shape[0],
            )

        return patches[0:1].detach().cpu().numpy().astype(np.float32)  # [1, 1, 96, 64]

    def _fit_to_expected_length_2d(self, window: np.ndarray) -> np.ndarray:
        expected_len = None
        for dim in self._input_meta.shape:
            if isinstance(dim, int) and dim > 1:
                expected_len = dim
                break
        if expected_len is None or len(window) == expected_len:
            return window
        return _pad_or_trim(window, expected_len)


_VENDOR_PATH_ADDED = False


def _ensure_vendored_torch_audioset_on_path() -> None:
    global _VENDOR_PATH_ADDED
    if _VENDOR_PATH_ADDED:
        return
    import sys
    vendor_dir = Path(__file__).resolve().parent.parent / "vendor" / "torch_audioset"
    if not vendor_dir.exists():
        raise ModuleNotFoundError(
            f"torch_audioset isn't vendored at {vendor_dir}. Run: "
            f"git clone https://github.com/w-hc/torch_audioset.git vendor/torch_audioset "
            f"(see models/README.md)."
        )
    sys.path.insert(0, str(vendor_dir))
    _VENDOR_PATH_ADDED = True


def _pad_or_trim(window: np.ndarray, target_len: int) -> np.ndarray:
    if len(window) > target_len:
        return window[:target_len]
    return np.pad(window, (0, target_len - len(window)), mode="constant")


def _static_width(shape) -> int:
    for dim in shape:
        if isinstance(dim, int) and dim > 1:
            return dim
    return 0
