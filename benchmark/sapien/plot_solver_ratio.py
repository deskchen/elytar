#!/usr/bin/env python3
"""
Plot solver/total percentage across num_envs.
Reads from results_history.csv (or --input).
Run from repo root: python3 benchmark/sapien/plot_solver_ratio.py
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def parse_num_envs(task_config: str) -> int | None:
    """Extract num_envs from task_config, e.g. 'config=4;...;num_envs=512'."""
    for part in task_config.split(";"):
        m = re.match(r"num_envs=(\d+)", part.strip())
        if m:
            return int(m.group(1))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot solver/total % across num_envs")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("benchmark/sapien/results/results_history.csv"),
        help="Input CSV (history file)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path (default: stdout or display)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Filter by task name (default: all)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: {args.input} not found", file=__import__("sys").stderr)
        return 1

    rows: list[dict] = []
    with args.input.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.task and row.get("task") != args.task:
                continue
            num_envs = parse_num_envs(row.get("task_config", ""))
            if num_envs is None:
                continue
            try:
                solver = float(row.get("solver_mean_ms", 0))
                total = float(row.get("total_mean_ms", 1))
            except (ValueError, TypeError):
                continue
            if total <= 0:
                continue
            rows.append({
                "num_envs": num_envs,
                "solver_ratio": 100.0 * solver / total,
                "solver_ms": solver,
                "total_ms": total,
            })

    if not rows:
        print("No rows with num_envs found; ensure task_config contains num_envs=N")
        return 1

    # Dedupe by num_envs: keep latest (last) per num_envs
    by_num = {}
    for r in rows:
        by_num[r["num_envs"]] = r
    rows = [by_num[k] for k in sorted(by_num.keys())]

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        for r in rows:
            print(f"  num_envs={r['num_envs']}: solver/total={r['solver_ratio']:.1f}%")
        return 0

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [r["num_envs"] for r in rows]
    ys = [r["solver_ratio"] for r in rows]
    ax.plot(xs, ys, "o-", linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_xlabel("num_envs")
    ax.set_ylabel("solver / total (%)")
    ax.set_title("Solver fraction of total step time")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"Saved {args.output}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

