from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


STAGE_NAMES = ["broadphase", "narrowphase", "coloring", "solver", "update", "total"]

STEP_COLUMNS = [
    "run_id",
    "task",
    "config",
    "step",
    "dt",
    "broadphase_ms",
    "narrowphase_ms",
    "coloring_ms",
    "solver_ms",
    "update_ms",
    "total_ms",
]


def summary_columns() -> list[str]:
    columns = ["run_id", "task", "config", "steps", "warmup_steps", "dt", "task_config"]
    # All means, then p90, p99, max, min for each metric
    for suffix in ["mean", "p90", "p99", "max", "min"]:
        columns.extend([f"{stage}_{suffix}_ms" for stage in STAGE_NAMES])
    return columns


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_rows(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    """Append rows to CSV; write header only if file is new or empty."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metadata_to_string(metadata: dict) -> str:
    parts = [f"{key}={metadata[key]}" for key in sorted(metadata.keys())]
    return ";".join(parts)

