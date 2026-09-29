"""
hardware.py

Best-effort detection of the actual CPU/SoC name (e.g. "Snapdragon(R) X
Elite - X1E80100 @ 3.40 GHz" on a real Copilot+ PC), so the app can show
concretely what it's running on instead of just "NPU (QNN)" -- closing the
last bit of ambiguity in the Technical Implementation story for judges who
don't take the NPU claim on faith.

Deliberately isolated from everything else: every platform lookup here is
wrapped so a failure (missing registry key, sandboxed environment, unusual
Linux distro, etc.) can never raise into the GUI or block app startup --
worst case, this just falls back to Python's generic platform.processor().
"""

from __future__ import annotations

import logging
import platform
import subprocess

logger = logging.getLogger(__name__)


def detect_hardware_name() -> str:
    """Returns a human-readable CPU/SoC name for the current machine.
    Never raises -- always returns *something*, even if it's just
    platform.machine()."""
    system = platform.system()
    try:
        if system == "Windows":
            name = _detect_windows_cpu_name()
            if name:
                return name
        elif system == "Darwin":
            name = _detect_macos_cpu_name()
            if name:
                return name
        elif system == "Linux":
            name = _detect_linux_cpu_name()
            if name:
                return name
    except Exception as exc:  # noqa: BLE001 - this is a "nice to have" label, never fatal
        logger.warning("Hardware name detection failed (%s), falling back.", exc)

    return platform.processor() or platform.machine() or "Unknown CPU"


def _detect_windows_cpu_name() -> str | None:
    # The registry's ProcessorNameString is what actually contains the real
    # marketing name on Windows (including "Snapdragon(R) X Elite ..." on
    # Copilot+ PCs) -- platform.processor() on Windows ARM64 just returns a
    # generic "ARMv8 (64-bit) Family ..." string that doesn't say Snapdragon
    # at all.
    import winreg  # only exists on Windows

    key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
        name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
    return name.strip() if name else None


def _detect_macos_cpu_name() -> str | None:
    out = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True, text=True, timeout=2,
    ).stdout.strip()
    if out:
        return out
    # Apple Silicon doesn't always populate machdep.cpu.brand_string; the
    # Mac's model identifier is a reasonable fallback (e.g. "Mac14,7").
    out = subprocess.run(
        ["sysctl", "-n", "hw.model"], capture_output=True, text=True, timeout=2,
    ).stdout.strip()
    return out or None


def _detect_linux_cpu_name() -> str | None:
    with open("/proc/cpuinfo") as fh:
        for line in fh:
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return None
