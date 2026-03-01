from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


STAGE_NAMES = ["broadphase", "narrowphase", "coloring", "solver", "update", "other", "total"]

STEP_COLUMNS = [
    "run_id",
    "task",
    "difficulty",
    "step",
    "dt",
    "broadphase_ms",
    "narrowphase_ms",
    "coloring_ms",
    "solver_ms",
    "update_ms",
    "other_ms",
    "total_ms",
]


def summary_columns() -> list[str]:
    columns = ["run_id", "task", "difficulty", "steps", "warmup_steps", "dt", "task_config"]
    for stage in STAGE_NAMES:
        columns.extend(
            [f"{stage}_mean_ms", f"{stage}_p50_ms", f"{stage}_p95_ms", f"{stage}_max_ms"]
        )
    return columns


def write_rows(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def metadata_to_string(metadata: dict) -> str:
    parts = [f"{key}={metadata[key]}" for key in sorted(metadata.keys())]
    return ";".join(parts)

