"""
src/logger.py — Dual stdout + file logger with timestamps.
Writes to BOTH terminal (stdout) and logs/run_<timestamp>.log simultaneously.

"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Lazy import config to avoid circular imports ──────────────────────────────
_LOGS_DIR: Path | None = None

def _get_logs_dir() -> Path:
    global _LOGS_DIR
    if _LOGS_DIR is None:
        try:
            from src.config import LOGS_DIR
            _LOGS_DIR = LOGS_DIR
        except ImportError:
            _LOGS_DIR = Path("logs")
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return _LOGS_DIR


# ── Shared log file for the current process run ───────────────────────────────
_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
_LOG_FILENAME: str | None = None
_ROOT_LOGGER_CONFIGURED = False


def _configure_root_logger():
    """Configure root logger once per process: stdout + rotating file."""
    global _ROOT_LOGGER_CONFIGURED, _LOG_FILENAME

    if _ROOT_LOGGER_CONFIGURED:
        return

    logs_dir = _get_logs_dir()
    _LOG_FILENAME = str(logs_dir / f"run_{_RUN_TIMESTAMP}.log")

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # ── Stdout handler ─────────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # ── File handler ───────────────────────────────────────────────────────────
    fh = logging.FileHandler(_LOG_FILENAME, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    root.info(f"Logger initialised — writing to {_LOG_FILENAME}")
    _ROOT_LOGGER_CONFIGURED = True


def get_logger(name: str = __name__) -> logging.Logger:
    """Return a named logger (dual stdout + file output)."""
    _configure_root_logger()
    return logging.getLogger(name)


def get_log_path() -> str | None:
    """Return the current log file path (useful for sanity-check assertions)."""
    return _LOG_FILENAME


# ── Convenience banner helpers ────────────────────────────────────────────────
def log_section(log: logging.Logger, title: str, width: int = 60):
    """Print a formatted section banner."""
    bar = "=" * width
    log.info(bar)
    log.info(f"  {title}")
    log.info(bar)


def log_dict(log: logging.Logger, d: dict, prefix: str = ""):
    """Log a dict line by line (for metrics, configs, etc.)."""
    for k, v in d.items():
        if isinstance(v, float):
            log.info(f"  {prefix}{k}: {v:.6f}")
        else:
            log.info(f"  {prefix}{k}: {v}")


if __name__ == "__main__":
    log = get_logger("test")
    log_section(log, "Logger Smoke Test")
    log.info("Info message")
    log.warning("Warning message")
    log.debug("Debug message (file only)")
    log.info(f"Log file path: {get_log_path()}")
