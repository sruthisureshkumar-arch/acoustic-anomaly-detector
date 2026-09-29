"""
main.py

Entry point. Sets up logging and required directories, then launches the
PyQt6 GUI. Run from the project root:

    python src/main.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

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
