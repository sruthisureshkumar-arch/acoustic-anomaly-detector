"""
build_dev_embedder_onnx.py

Builds models/yamnet_public.onnx: a tiny placeholder ONNX graph with the same
input/output *shape contract* as the REAL exported YamNet asset -- a
[1, 1, 96, 64] log-mel spectrogram patch in, a [1, 521] class-score vector
out -- so the rest of the pipeline (audio capture -> mel preprocessing ->
embed -> baseline -> anomaly scoring -> GUI) can be built, wired, and
smoke-tested on a machine with no QNN/NPU support and no Qualcomm AI Hub
account yet.

This shape contract was NOT the original guess (an earlier version of this
script used a [1, 16000] raw-waveform input and a 1024-d "embedding" output,
which is what a generic from-scratch YamNet port might look like). It was
corrected after actually exporting the real model via
`qai-hub-models export yamnet --runtime onnx` and inspecting its real input/
output metadata -- see embedding.py's module docstring and models/README.md
for the full story. Keeping this placeholder's contract in sync with the
real one is what makes it useful as a stand-in at all.

THIS IS NOT YAMNET. It's a fixed random linear projection + tanh over the
flattened mel patch -- it has no real acoustic knowledge, so anomaly scores
from it are meaningless for an actual demo. It exists purely so
`embedding.py`'s provider-selection and output-picking logic, plus
everything downstream of it, can be exercised end-to-end without the real
model asset or its torch/torchaudio/torch_audioset preprocessing dependency
(this placeholder still takes a pre-computed mel patch as input -- it
doesn't need the mel front end itself, since the shape contract starts
*after* that step).

For the actual hackathon submission, replace models/yamnet_public.onnx usage
with the real Qualcomm AI Hub-compiled asset at models/yamnet_npu.onnx --
see models/README.md for the exact `qai-hub-models export yamnet` command.
embedding.py already prefers yamnet_npu.onnx automatically when it's present.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

MEL_FRAMES = 96          # matches the real export's PATCH_FRAMES
MEL_BANDS = 64            # matches the real export's MEL_BANDS
NUM_CLASSES = 521         # matches the real export's AudioSet class count
FLAT_LEN = MEL_FRAMES * MEL_BANDS
SEED = 1234

OUT_PATH = Path(__file__).resolve().parent.parent / "models" / "yamnet_public.onnx"


def build() -> None:
    rng = np.random.default_rng(SEED)
    weight = (rng.standard_normal((FLAT_LEN, NUM_CLASSES)) / np.sqrt(FLAT_LEN)).astype(np.float32)

    input_tensor = helper.make_tensor_value_info(
        "audio", TensorProto.FLOAT, [1, 1, MEL_FRAMES, MEL_BANDS]
    )
    output_tensor = helper.make_tensor_value_info("class_scores", TensorProto.FLOAT, [1, NUM_CLASSES])
    weight_initializer = helper.make_tensor(
        "proj_weight", TensorProto.FLOAT, weight.shape, weight.flatten().tolist()
    )
    flat_shape_initializer = helper.make_tensor("flat_shape", TensorProto.INT64, [2], [1, FLAT_LEN])

    reshape_node = helper.make_node("Reshape", ["audio", "flat_shape"], ["flattened"])
    matmul_node = helper.make_node("MatMul", ["flattened", "proj_weight"], ["projected"])
    tanh_node = helper.make_node("Tanh", ["projected"], ["class_scores"])

    graph = helper.make_graph(
        [reshape_node, matmul_node, tanh_node],
        "dev_placeholder_embedder",
        [input_tensor],
        [output_tensor],
        initializer=[weight_initializer, flat_shape_initializer],
    )
    model = helper.make_model(graph, producer_name="acoustic-anomaly-detector-dev-tools")
    model.opset_import[0].version = 17
    # Pin the IR version to one the pinned onnxruntime release actually supports;
    # newer `onnx` packages default to a newer IR version than onnxruntime accepts.
    model.ir_version = 10
    onnx.checker.check_model(model)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(OUT_PATH))
    print(f"Wrote placeholder dev embedder to {OUT_PATH} "
          f"(input='audio' [1,1,{MEL_FRAMES},{MEL_BANDS}], output='class_scores' [1,{NUM_CLASSES}])")


if __name__ == "__main__":
    build()
