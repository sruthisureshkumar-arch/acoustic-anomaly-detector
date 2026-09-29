# Models

This app needs a YamNet ONNX model here as either:

- `yamnet_npu.onnx` (+ its external weights file, `yamnet.data`, which MUST sit next to
  it) — the **real submission asset**: YamNet compiled for a Snapdragon target through
  Qualcomm AI Hub. `embedding.py` prefers this automatically when it's present, and it's
  the only one of the two that actually exercises the Hexagon NPU via the QNN Execution
  Provider. This is what should ship in the hackathon submission.
- `yamnet_public.onnx` — a **CPU-only development placeholder** (see below). Used
  automatically when `yamnet_npu.onnx` isn't present, purely so the rest of the app can
  be built and tested on a machine with no Snapdragon NPU and no AI Hub account yet.

## Getting the real NPU asset (`yamnet_npu.onnx`)

These are the commands that actually worked (confirmed by real use, not a sketch) — an
earlier version of this doc guessed at a raw `qai_hub.submit_compile_job` snippet, which
turned out not to match how this specific model is actually exported. Use the real
per-model export tooling instead:

```bash
# 1. Authenticate. Get an API token from https://aihub.qualcomm.com after creating a
#    Qualcomm ID at https://workbench.aihub.qualcomm.com/signup.
pip install qai-hub
qai-hub configure --api_token <YOUR_API_TOKEN>
qai-hub list-devices     # confirm you can see hosted Snapdragon devices, e.g.
                          # "Snapdragon X Elite CRD", "Snapdragon X2 Elite CRD"

# 2. Install the per-model export tooling and YamNet's extra dependencies.
pip install qai_hub_models
pip install resampy
pip install "qai-hub-models[yamnet]" git+https://github.com/w-hc/torch_audioset.git
#   ^ if that git+https line fails to build (older setuptools/distutils
#     incompatibility with torch_audioset's packaging), skip it -- you don't
#     need torch_audioset for the EXPORT step, only this app's own runtime
#     preprocessing does, and this project vendors it separately (see below).

# 3. Export. This submits real compile/profile/inference jobs to Qualcomm's cloud and
#    waits for them, so it takes a few minutes. Pick your actual target device from
#    `qai-hub list-devices`.
qai-hub-models export yamnet \
  --runtime onnx \
  --precision float \
  --device "Snapdragon X Elite CRD" \
  --output-dir yamnet_export

# 4. Copy the result into this project. The export produces a FOLDER, not a single
#    file -- the graph (yamnet.onnx) and its weights (yamnet.data) are separate, and
#    BOTH have to be copied. Renaming yamnet.onnx is fine; do NOT rename yamnet.data,
#    the graph references it by that exact filename.
cp yamnet_export/yamnet-onnx-float/yamnet.onnx models/yamnet_npu.onnx
cp yamnet_export/yamnet-onnx-float/yamnet.data models/yamnet.data
cp yamnet_export/yamnet-onnx-float/labels.txt models/labels.txt   # optional, for reference
```

**Real shape contract, confirmed by inspecting the export** (this matters — see
`embedding.py`'s module docstring for the full story of why the original assumption
here was wrong): input `"audio"`, shape `[1, 1, 96, 64]` — a log-mel spectrogram patch,
**not raw audio samples**. Output `"class_scores"`, shape `[1, 521]` — YamNet's AudioSet
classifier output; this export does not expose a separate embedding layer. `embedding.py`
handles the mel-spectrogram conversion automatically (see the vendoring note below) and
uses `class_scores` as the feature vector — no code changes needed once the files are in
place, just run the app normally.

`embedding.py` logs which execution provider actually got used, so you can confirm on
stage that inference is really running on the NPU (`QNNExecutionProvider`) rather than
silently falling back to CPU.

**Quantify it, don't just assert it.** Once `yamnet_npu.onnx` is in place, run
`python scripts/benchmark.py` (or click "Benchmark NPU vs CPU" in the app itself) to get
a real, measured NPU-vs-CPU latency comparison on the actual submission hardware. This
writes `models/benchmark_results.json`, which the GUI reads automatically and shows next
to the provider label. Re-run this on the real HP Snapdragon device before the final
demo/submission — a number benchmarked on a dev machine without QNN isn't the number you
want to present.

## Required companion dependency: vendored `torch_audioset`

The real model's mel-spectrogram preprocessing is done with `torch_audioset`'s own
front end (`WaveformToInput`) — the exact reference implementation Qualcomm's export was
traced from, reused rather than re-derived by hand to avoid a subtle numerical mismatch
in window function, mel-scale formula, or normalization that wouldn't crash, just quietly
degrade every score. `torch_audioset` doesn't reliably install via pip (its `setup.py`
uses an old-style distutils option that breaks under newer setuptools on some platforms),
so it's vendored instead of pip-installed:

```bash
git clone https://github.com/w-hc/torch_audioset.git vendor/torch_audioset
```

`embedding.py` adds `vendor/torch_audioset` to `sys.path` automatically the first time it
needs it — no further setup after cloning. This is only needed for the real model
(rank-4 input); the CPU-only placeholder below also takes a pre-computed mel patch as
input, so it needs this too once you're testing against the corrected shape contract.

**Sanity-checking the real model** (confirms the whole preprocessing → model chain is
numerically correct, not just shape-compatible): feed a pure tone, white noise, and
silence through it and check the top predicted `labels.txt` classes make sense. In
testing: a 1kHz tone scored highest on "Beep, bleep" then "Sine wave"; white noise on
"Water"/"White noise"/"Noise"; silence on "Silence" by a huge margin over anything else.
If you change anything about the preprocessing path, rerunning this check is a fast way
to catch a regression before it shows up as a confusing anomaly-detection problem instead.

## The development placeholder (`yamnet_public.onnx`)

`scripts/build_dev_embedder_onnx.py` generates a tiny ONNX graph — a fixed random linear
projection + tanh over the flattened mel patch — with the **same input/output shape
contract as the real export** (`[1,1,96,64]` in, `[1,521]` out), but **no real acoustic
knowledge**. It exists only so `audio_capture.py` → `embedding.py` → `baseline.py` →
`anomaly.py` → `gui.py` can be built, wired together, and tested end to end (including
on a dev sandbox with no NPU) before the real model asset is available. Its contract was
corrected once (see git history / `embedding.py`'s docstring) after the real export
turned out not to match the original raw-waveform-in / embedding-out guess — keeping it
in sync with the real contract is what makes it useful as a stand-in at all.

Regenerate it any time with:

```bash
python scripts/build_dev_embedder_onnx.py
```

**Do not demo or submit with this placeholder model.** It will run, and the app's logic
(baseline recording, scoring, alerting, UI) all work identically either way — that's the
point — but its outputs don't encode anything about real machine sounds, so anomaly
detection quality with it is meaningless. Swap in the real `yamnet_npu.onnx` (+
`yamnet.data`) before any real testing or the actual presentation.
