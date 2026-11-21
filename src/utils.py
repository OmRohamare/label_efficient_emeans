"""
utils.py
========
General-purpose utilities: logging setup, config loading, device selection,
Timer, atomic file saves, and hardware detection helpers.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

import torch
import yaml


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Configure root logger with console (and optional file) handler.

    Parameters
    ----------
    level:
        Logging level (e.g. logging.DEBUG or "DEBUG").
    log_file:
        Optional path; if given, also writes to this file.
    fmt:
        Log format string.
    datefmt:
        Date format for timestamps.

    Returns
    -------
    Root ``logging.Logger``.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)
    return logging.getLogger()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file and return it as a plain dict.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg or {}


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """
    Deep-merge multiple config dicts left to right (later dicts override earlier).

    Parameters
    ----------
    *configs:
        Config dicts to merge in order.

    Returns
    -------
    Merged config dict.
    """
    result: dict[str, Any] = {}
    for cfg in configs:
        _deep_merge(result, cfg)
    return result


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(prefer: str = "auto") -> torch.device:
    """
    Select the best available compute device.

    Parameters
    ----------
    prefer:
        ``"auto"`` – CUDA if available, else MPS (Apple), else CPU.
        ``"cuda"`` / ``"mps"`` / ``"cpu"`` – explicit override.

    Returns
    -------
    ``torch.device``
    """
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(prefer)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def get_hardware_info() -> dict[str, Any]:
    """Return a dictionary with hardware capabilities."""
    info: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [],
        "total_gpu_memory_gb": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            info["gpu_names"].append(torch.cuda.get_device_name(i))
            mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            info["total_gpu_memory_gb"].append(round(mem, 2))
    return info


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """
    Context-manager and callable timer.

    Usage::

        with Timer() as t:
            do_work()
        print(t.elapsed)  # seconds (float)

        t = Timer()
        t.start()
        do_work()
        t.stop()
        print(t.elapsed)
    """

    def __init__(self) -> None:
        self._start: float | None = None
        self._stop: float | None = None

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        self._stop = None
        return self

    def stop(self) -> float:
        if self._start is None:
            raise RuntimeError("Timer.start() was not called.")
        self._stop = time.perf_counter()
        return self.elapsed

    @property
    def elapsed(self) -> float:
        """Elapsed seconds. If still running, returns time since start."""
        if self._start is None:
            return 0.0
        end = self._stop if self._stop is not None else time.perf_counter()
        return end - self._start

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"Timer(elapsed={self.elapsed:.4f}s)"


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Generator[Timer, None, None]:
    """Context manager that logs elapsed time."""
    t = Timer()
    t.start()
    yield t
    t.stop()
    msg = f"{label}: {t.elapsed:.3f}s"
    if logger is not None:
        logger.info(msg)
    else:
        logging.info(msg)


# ---------------------------------------------------------------------------
# Atomic file save
# ---------------------------------------------------------------------------

def save_atomic(src: Path, dst: Path) -> None:
    """
    Atomically move ``src`` to ``dst`` using an intermediate temp file.

    This avoids leaving a partially-written ``dst`` if the process is killed
    mid-write. ``src`` and ``dst`` must be on the same filesystem for the
    rename to be atomic on POSIX; on Windows a copy+delete is used as fallback.

    Parameters
    ----------
    src:
        Path to the source file (will be moved / deleted).
    dst:
        Destination path.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        src.rename(dst)
    except OSError:
        # Cross-device or Windows cross-volume: copy then delete
        shutil.copy2(src, dst)
        src.unlink(missing_ok=True)


def write_atomic(data: bytes, dst: Path) -> None:
    """
    Write ``data`` to ``dst`` atomically via a sibling temp file.

    Parameters
    ----------
    data:
        Raw bytes to write.
    dst:
        Destination path.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dst.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        save_atomic(Path(tmp_path), dst)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
