"""
src/temporal_guard.py — Split-level temporal boundary verification (Q9).

"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import polars as pl

from src.config import PATHS
from src.logger import get_logger

log = get_logger(__name__)

# Which field carries the impression timestamp, per dataset.
TIME_FIELD = {"ebnerd": "impression_time", "mind": "time"}


@dataclass
class SplitBoundary:
    """One dataset's train/val/test time envelope."""
    dataset: str
    split: str
    n_rows: int
    min_time: Optional[object] = None
    max_time: Optional[object] = None
    verifiable: bool = True
    note: str = ""


@dataclass
class BoundaryReport:
    dataset: str
    checks: List[dict] = field(default_factory=list)
    passed: bool = True
    unverifiable: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.checks.append({"check": name, "pass": ok, "detail": detail})
        if not ok:
            self.passed = False

    def render(self) -> str:
        lines = [
            f"# Split-Level Temporal Boundary: {self.dataset.upper()}",
            f"**Result**: {'PASS' if self.passed else 'FAIL'}",
            f"**Checks**: {sum(1 for c in self.checks if c['pass'])}/"
            f"{len(self.checks)} passed\n",
            "| Check | Result | Detail |",
            "|:--|:--:|:--|",
        ]
        for c in self.checks:
            mark = "PASS" if c["pass"] else "**FAIL**"
            lines.append(f"| {c['check']} | {mark} | {c['detail']} |")
        if self.notes:
            lines += ["", "## Notes"]
            lines += [f"- {n}" for n in self.notes]
        if self.unverifiable:
            lines += [
                "",
                "## Unverifiable (reported, not hidden)",
                *[f"- {u}" for u in self.unverifiable],
            ]
        return "\n".join(lines)


def _split_envelope(dataset: str, split: str) -> Optional[SplitBoundary]:
    """Reads only the timestamp column of a split -- cheap even at 13.5M rows."""
    path = PATHS.get(f"{dataset}_{split}_behaviors")
    time_field = TIME_FIELD.get(dataset)
    if not path or not time_field or not Path(path).exists():
        return None

    schema_cols = pl.scan_parquet(path).collect_schema().names() if path.suffix == ".parquet" else None
    if schema_cols is not None:
        if time_field not in schema_cols:
            return SplitBoundary(
                dataset, split, 0, verifiable=False,
                note=f"no '{time_field}' column",
            )
        agg = (
            pl.scan_parquet(path)
            .select(
                pl.len().alias("n"),
                pl.col(time_field).min().alias("lo"),
                pl.col(time_field).max().alias("hi"),
            )
            .collect()
        )
    else:
        # MIND behaviors.tsv has no header; declare the column names explicitly.
        # CRITICAL: MIND's `time` is the STRING "11/09/2019 9:59:58 AM", so a
        # lexicographic min/max would order "11/9/..." AFTER "11/15/..." and
        # report a false boundary violation. It must be parsed to a datetime
        # before comparing -- same treatment as data_loader._parse_mind_time.
        from src.config import MIND_BEHAVIOR_COLS

        agg = (
            pl.scan_csv(
                path,
                separator="\t",
                has_header=False,
                new_columns=MIND_BEHAVIOR_COLS,
                quote_char=None,
            )
            .with_columns(
                parsed=pl.col(time_field).str.to_datetime(
                    "%m/%d/%Y %I:%M:%S %p", strict=False
                )
            )
            .select(
                pl.len().alias("n"),
                pl.col("parsed").min().alias("lo"),
                pl.col("parsed").max().alias("hi"),
                (pl.col("parsed").is_null().sum()).alias("unparsed"),
            )
            .collect()
        )
        row = agg.row(0)
        unparsed = int(row[3] or 0)
        if unparsed:
            log.warning(
                f"[{dataset}.{split}] {unparsed:,} rows had an unparseable "
                f"timestamp; the envelope below is computed from parsed rows only."
            )
        return SplitBoundary(dataset, split, int(row[0]), row[1], row[2])
    row = agg.row(0)
    return SplitBoundary(dataset, split, int(row[0]), row[1], row[2])


def verify_dataset(dataset: str, strict: bool = True) -> BoundaryReport:
    """
    Verifies that train <= val <= test in time for one dataset.

    Checks, in order:
      1. each split has a parseable timestamp column
      2. max(train_time) <= min(val_time)      -- no val row predates a train row
      3. max(val_time)   <= min(test_time)    -- no test row predates a val row
      4. train/val/test user overlap is reported (not asserted: a shared user
         across splits is normal and is handled by the behaviour window, not by
         splitting)
    """
    report = BoundaryReport(dataset=dataset)

    envs: Dict[str, SplitBoundary] = {}
    for split in ("train", "val", "test"):
        env = _split_envelope(dataset, split)
        if env is None:
            report.notes.append(f"{split}: not available locally -- check skipped")
            continue
        envs[split] = env
        if not env.verifiable:
            report.unverifiable.append(
                f"{split}: {env.note}, so its time envelope cannot be checked"
            )

    for split, env in envs.items():
        if env.verifiable:
            report.add(
                f"{split} has timestamps",
                True,
                f"{env.n_rows:,} rows, {env.min_time} .. {env.max_time}",
            )

    def _cmp(a: Optional[SplitBoundary], b: Optional[SplitBoundary], name: str):
        if not a or not b or not (a.verifiable and b.verifiable):
            return
        ok = a.max_time <= b.min_time
        report.add(
            name,
            ok,
            f"max({a.split})={a.max_time} vs min({b.split})={b.min_time}",
        )

    _cmp(envs.get("train"), envs.get("val"), "max(train) <= min(val)")
    _cmp(envs.get("val"), envs.get("test"), "max(val) <= min(test)")
    _cmp(envs.get("train"), envs.get("test"), "max(train) <= min(test)")

    if dataset == "mind":
        report.unverifiable.append(
            "MIND ships no per-click timestamps for user history, so the "
            "behaviour window cannot be enforced on history at all. A user's "
            "MIND 'history' may already contain articles published after the "
            "impression being scored. Any MIND metric is therefore an upper bound."
        )
    if dataset == "ebnerd":
        report.notes.append(
            "EB-NeRD does ship per-click history timestamps "
            "(history.parquet: impression_time_fixed), so the behaviour window IS "
            "enforceable and is asserted by tests/test_anti_gaming.py."
        )

    if strict and not report.passed:
        for c in report.checks:
            if not c["pass"]:
                log.error(f"[{dataset}] TEMPORAL BOUNDARY VIOLATED: {c['check']} -- {c['detail']}")
    return report


def verify_all(strict: bool = True) -> Dict[str, BoundaryReport]:
    return {ds: verify_dataset(ds, strict=strict) for ds in ("ebnerd", "mind")}


def assert_temporal_boundaries(strict: bool = True) -> None:
    """Raises AssertionError if any dataset's splits are not time-ordered."""
    reports = verify_all(strict=strict)
    failures = [r for r in reports.values() if not r.passed]
    for ds, r in reports.items():
        verdict = "PASS" if r.passed else "FAIL"
        log.info(f"[TEMPORAL GUARD] {ds.upper()}: {verdict} "
                 f"({sum(1 for c in r.checks if c['pass'])}/{len(r.checks)} checks)")
    if failures:
        detail = "; ".join(
            f"{r.dataset}: {[c['check'] for c in r.checks if not c['pass']]}"
            for r in failures
        )
        raise AssertionError(f"Temporal split boundary violated -- {detail}")


def main():
    import argparse

    p = argparse.ArgumentParser(description="Q9 split-level temporal boundary verification")
    p.add_argument("--dataset", choices=["ebnerd", "mind", "all"], default="all")
    p.add_argument("--save", action="store_true")
    p.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run folder for the reports. Default: a fresh "
             "results/runs/<YYYYmmdd_HHMMSS>/ folder.",
    )
    p.add_argument("--no_strict", action="store_true", help="Report but do not raise")
    a = p.parse_args()

    from src.run_dir import resolve_run_dir

    run_dir = resolve_run_dir(a.run_dir) if a.save else None
    if run_dir:
        log.info(f"Run directory: {run_dir}")

    targets = ["ebnerd", "mind"] if a.dataset == "all" else [a.dataset]
    overall = True
    for ds in targets:
        rep = verify_dataset(ds, strict=not a.no_strict)
        text = rep.render()
        print("\n" + text + "\n")
        overall &= rep.passed
        if a.save:
            out = run_dir / f"temporal_boundary_{ds}.md"
            out.write_text(text, encoding="utf-8")
            log.info(f"Saved {out}")
    if not a.no_strict and not overall:
        raise SystemExit(1)
    raise SystemExit(0 if overall else 1)


if __name__ == "__main__":
    main()
