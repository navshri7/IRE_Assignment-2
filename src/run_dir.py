"""
src/run_dir.py — Timestamped run folders for experiment artifacts.

Every invocation of the experiment runner writes its reports into its own
directory, `results/runs/<YYYYmmdd_HHMMSS>/`, so a run is a single self-contained
artefact and no earlier run's output is ever overwritten.

A run directory contains:

    manifest.md              parameters, git SHA, versions, file index
    summary.md               master benchmark table (--mode val)
    two_stage.md             Stage-1 Recall@K + before/after re-ranking
    serving_ablation_*.md    Q9 serving-feature ablation
    ablation.md              Q3 baseline-vs-improvement + paired bootstrap CIs
    commands.log             shell transcript, if piped through tee

A one-line pointer at `results/runs/LATEST` records the most recent run
directory, so the latest run is discoverable without a glob and without
overwriting any report.
"""

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import FEATURE_NAMES, HYPERPARAMS, N_FEATURES, PATHS
from src.logger import get_logger

log = get_logger(__name__)

RUNS_ROOT_NAME = "runs"
LATEST_POINTER = "LATEST"


def make_run_dir(stamp: Optional[str] = None, root: Optional[Path] = None) -> Path:
    """
    Creates and returns a fresh timestamped run directory.

    If the directory already exists (two runs in the same second), a numeric
    suffix is appended rather than reusing it, so a run's artefacts are never
    mixed with another's.
    """
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(root) if root else (Path(PATHS["results"]) / RUNS_ROOT_NAME)
    run_dir = base / stamp
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    (base / LATEST_POINTER).write_text(run_dir.name + "\n", encoding="utf-8")
    log.info(f"Run directory: {run_dir}")
    return run_dir


def resolve_run_dir(run_dir: Optional[str] = None) -> Path:
    """
    Returns the directory for this run.

    `--run_dir NAME` reuses a named directory (e.g. to add the ablation to the
    same folder as an earlier benchmark). Otherwise a fresh timestamped one is
    created. An existing named directory is NOT wiped: files are overwritten
    only if their names collide, which is why the default is a new timestamp.
    """
    if run_dir:
        p = Path(run_dir)
        if not p.is_absolute():
            p = Path(PATHS["results"]) / RUNS_ROOT_NAME / run_dir
        p.mkdir(parents=True, exist_ok=True)
        return p
    return make_run_dir()


def latest_run_dir() -> Optional[Path]:
    """The most recent run directory recorded by the LATEST pointer."""
    base = Path(PATHS["results"]) / RUNS_ROOT_NAME
    pointer = base / LATEST_POINTER
    if not pointer.exists():
        return None
    name = pointer.read_text(encoding="utf-8").strip()
    candidate = base / name
    return candidate if candidate.is_dir() else None


# ── Manifest ─────────────────────────────────────────────────────────────────

def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def write_manifest(run_dir: Path, params: dict, files: Optional[List[str]] = None) -> Path:
    """
    Writes manifest.md: everything needed to interpret or reproduce the run.

    Recorded because the numbers in a report are only meaningful alongside the
    feature schema and dense backend that produced them -- a checkpoint trained
    under tfidf_svd and one under minilm both emit 28 columns.
    """
    run_dir = Path(run_dir)
    sha = _git("rev-parse", "--short", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_git("status", "--porcelain"))

    try:
        import lightgbm
        import numpy
        import polars
        import sklearn
        import torch

        versions = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "lightgbm": lightgbm.__version__,
            "numpy": numpy.__version__,
            "polars": polars.__version__,
            "scikit-learn": sklearn.__version__,
        }
    except Exception:
        versions = {"python": platform.python_version()}

    try:
        device = __import__("src.config", fromlist=["DEVICE"]).DEVICE
    except Exception:
        device = "unknown"

    lines = [
        f"# Run manifest — {run_dir.name}",
        "",
        f"- **Created**: {datetime.now().isoformat(timespec='seconds')}",
        f"- **Git**: `{sha or 'unknown'}` on `{branch or 'unknown'}`"
        f"{' (working tree DIRTY)' if dirty else ''}",
        f"- **Platform**: {platform.platform()}",
        f"- **Compute device**: `{device}`",
        f"- **Features**: {N_FEATURES} columns",
        f"- **Dense backend**: `{HYPERPARAMS.get('dense_backend')}` "
        f"(`{HYPERPARAMS.get('dense_model_name')}`)",
        "",
        "## Parameters",
        "",
        "| Key | Value |",
        "|:--|:--|",
    ]
    for k, v in sorted(params.items()):
        lines.append(f"| `{k}` | `{v}` |")

    lines += ["", "## Environment", "", "| Package | Version |", "|:--|:--|"]
    for k, v in sorted(versions.items()):
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Feature schema",
        "",
        "```",
        json.dumps(list(FEATURE_NAMES), indent=1),
        "```",
    ]

    if files:
        lines += ["", "## Artefacts in this run", ""]
        lines += [f"- `{f}`" for f in sorted(files)]

    lines += [
        "",
        "## Reproduce",
        "",
        "```zsh",
        "cd \"" + str(REPO_ROOT) + "\"",
        f".venv/bin/python3 run_all_experiments.py "
        + " ".join(f"--{k.replace('_', '-')} {v}" for k, v in sorted(params.items())
                   if k in {"dataset", "mode", "sample_size", "n_bootstrap", "top_k"}),
        "```",
        "",
    ]

    path = run_dir / "manifest.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Wrote manifest: {path}")
    return path


def index_run_dir(run_dir: Path) -> List[str]:
    """Names of the report files present in a run directory."""
    return [p.name for p in sorted(Path(run_dir).glob("*.md"))
            if p.name != "manifest.md"]
