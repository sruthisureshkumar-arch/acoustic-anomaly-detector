"""
main.py

Entry point. Sets up logging and required directories, then launches the
PyQt6 GUI. Run from the project root:

    python src/main.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Must happen before numpy/torch are imported anywhere in the process (they
# size their internal math thread pools once, at import time, by reading
# these) -- both get imported transitively once `gui` (below) pulls in
# embedding.py. Baseline recording runs the mel-spectrogram + YamNet
# inference loop on a background QThread, and on macOS, Apple's Accelerate
# framework (which PyTorch's CPU backend uses there) is not reliably safe to
# call multi-threaded from a non-main thread -- it can silently corrupt
# memory until enough calls accumulate, which crashed as a bus error right
# after a 90s baseline recording finished processing ~180 windows. Forcing
# every math library here to single-threaded mode avoids that thread-safety
# hazard entirely, at the cost of somewhat slower per-window inference
# (still fast enough -- this is a ~1s-per-window workload, not a
# performance-critical hot loop).
for _env_var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_env_var, "1")

ROOT = Path(__file__).resolve().parent.parent
for required_dir in ("models", "profiles", "logs", "assets"):
    (ROOT / required_dir).mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT / "logs" / "app.log"),
    ],
)

logger = logging.getLogger("main")


def main() -> None:
    logger.info("Starting Acoustic Anomaly Detector")
    try:
        import gui
    except ImportError as exc:
        logger.error(
            "Could not import the GUI (%s). If PyQt6 isn't installed, run: "
            "pip install -r requirements.txt", exc,
        )
        sys.exit(1)

    gui.run()


if __name__ == "__main__":
    main()
