#!/usr/bin/env python3
"""Run A/B benchmark comparing PhysX snippet binaries (vanilla vs capybara)."""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Default steps per run by snippet (from snippetMain #else paths)
DEFAULT_STEPS = {
    "Isosurface": 100,
    "SDF": 100,
    "PBF": 100,
    "PBDCloth": 100,
    "SplitSim": 100,
    "RBDirectGPUAPI": 100,
    "SplitFetchResults": 250,
    "Triggers": 250,
}


def workspace_root() -> Path:
    """Infer workspace root from script location: benchmark/physx_snippets/run.py -> repo root."""
    script_dir = Path(__file__).resolve().parent
    # benchmark/physx_snippets -> go up 2 levels
    return script_dir.parent.parent


def resolve_binary(workspace: Path, tree: str, snippet: str, profile: str = "profile") -> Optional[Path]:
    """Resolve headless binary path. Tries common PhysX output locations."""
    tree_path = workspace / tree
    binary_name = f"Snippet{snippet}Headless_64"
    candidates = [
        tree_path / "bin" / "linux.x86_64" / profile / binary_name,
        tree_path / "bin" / "linux.clang" / profile / binary_name,
        # Build dir (single-config cmake)
        tree_path / "compiler" / f"linux-clang-{profile}" / "sdk_snippets_bin" / binary_name,
        tree_path / "compiler" / f"linux-clang-{profile}" / "sdk_snippets_bin" / "Snippets" / binary_name,
    ]
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return p
    return None


def compute_stats(values: list[float]) -> dict:
    """Compute min, max, mean statistics."""
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    """Write summary rows to CSV, overwriting existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def append_summary_csv(path: Path, rows: list[dict]) -> None:
    """Append summary rows to CSV, creating if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B benchmark PhysX headless snippets (vanilla vs capybara)."
    )
    parser.add_argument("--snippet", required=True, help="Snippet name (e.g. Isosurface)")
    parser.add_argument("--reps", type=int, default=10, help="Repetitions per variant")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/physx_snippets/results"),
        help="Output directory for CSV files",
    )
    parser.add_argument("--run-id", help="Run identifier for CSV (default: timestamp)")
    parser.add_argument("--verbose", action="store_true", help="Print subprocess output")
    parser.add_argument("--timeout", type=float, help="Per-run timeout in seconds")
    parser.add_argument(
        "--steps",
        type=int,
        help="Steps per run for throughput (default: snippet-specific)",
    )
    parser.add_argument("--label-a", default="vanilla_cu", help="Label for variant A")
    parser.add_argument("--label-b", default="capybara_ptx", help="Label for variant B")
    parser.add_argument(
        "--delay-between-variants",
        type=float,
        default=1.0,
        help="Seconds to sleep between variant A and B (lets GPU release; default 1.0)",
    )
    args = parser.parse_args()

    snippet = args.snippet
    workspace = workspace_root()
    tree_a = "physx-5.6.1"
    tree_b = "physx-5.6.1-capybara"

    cmd_a_path = resolve_binary(workspace, tree_a, snippet)
    cmd_b_path = resolve_binary(workspace, tree_b, snippet)

    if cmd_a_path is None:
        print(f"ERROR: Binary not found for {snippet} (vanilla).", file=sys.stderr)
        print("Build both trees with snippets and headless targets:", file=sys.stderr)
        print(f"  PHYSX_DIR={workspace}/{tree_a} ELYTAR_BUILD_PHYSX_SNIPPETS=1 ./scripts/update_toolchain.sh", file=sys.stderr)
        print(f"  PHYSX_DIR={workspace}/{tree_b} ELYTAR_BUILD_PHYSX_SNIPPETS=1 ./scripts/update_toolchain.sh", file=sys.stderr)
        return 1
    if cmd_b_path is None:
        print(f"ERROR: Binary not found for {snippet} (capybara).", file=sys.stderr)
        print("Build the capybara tree with snippets:", file=sys.stderr)
        print(f"  PHYSX_DIR={workspace}/{tree_b} ELYTAR_BUILD_PHYSX_SNIPPETS=1 ./scripts/update_toolchain.sh", file=sys.stderr)
        return 1

    steps = args.steps if args.steps is not None else DEFAULT_STEPS.get(snippet, 100)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect results per variant
    results_a: dict[str, list[float]] = {"elapsed_s": [], "throughput_steps_per_s": []}
    results_b: dict[str, list[float]] = {"elapsed_s": [], "throughput_steps_per_s": []}

    def run_one(label: str, cmd: list[str], rep: int, results: dict[str, list[float]]) -> bool:
        start = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                capture_output=not args.verbose,
                timeout=args.timeout,
                cwd=str(workspace),
            )
        except subprocess.TimeoutExpired:
            elapsed = args.timeout or 0
            print(f"[{label}] rep={rep} TIMEOUT after {elapsed}s")
            return False
        except Exception as e:
            print(f"[{label}] rep={rep} ERROR: {e}", file=sys.stderr)
            return False

        elapsed = time.perf_counter() - start
        throughput = steps / elapsed if elapsed > 0 else 0.0
        results["elapsed_s"].append(elapsed)
        results["throughput_steps_per_s"].append(throughput)
        print(f"[{label}] rep={rep} elapsed={elapsed:.3f}s throughput={throughput:.2f} steps/s")

        if result.returncode != 0:
            print(f"  (exit code {result.returncode})", file=sys.stderr)
        return result.returncode == 0

    cmd_a = [str(cmd_a_path)]
    cmd_b = [str(cmd_b_path)]

    print(f"\n=== Running {args.reps} repetitions per variant ===\n")
    for i in range(1, args.reps + 1):
        run_one(args.label_a, cmd_a, i, results_a)
        if args.delay_between_variants > 0:
            time.sleep(args.delay_between_variants)
        run_one(args.label_b, cmd_b, i, results_b)

    # Compute and print summary
    print("\n=== Summary Statistics ===\n")
    stats_a = {k: compute_stats(v) for k, v in results_a.items()}
    stats_b = {k: compute_stats(v) for k, v in results_b.items()}

    for metric in ["elapsed_s", "throughput_steps_per_s"]:
        print(f"{metric}:")
        print(f"  {args.label_a:20} min={stats_a[metric]['min']:.4f}  max={stats_a[metric]['max']:.4f}  mean={stats_a[metric]['mean']:.4f}")
        print(f"  {args.label_b:20} min={stats_b[metric]['min']:.4f}  max={stats_b[metric]['max']:.4f}  mean={stats_b[metric]['mean']:.4f}")
        if stats_a[metric]["mean"] > 0:
            ratio = stats_b[metric]["mean"] / stats_a[metric]["mean"]
            print(f"  Ratio (B/A): {ratio:.2f}x")
        print()

    # Write summary CSVs
    summary_rows = [
        {
            "run_id": run_id,
            "snippet": snippet,
            "variant": args.label_a,
            "steps": steps,
            "elapsed_s_min": stats_a["elapsed_s"]["min"],
            "elapsed_s_max": stats_a["elapsed_s"]["max"],
            "elapsed_s_mean": stats_a["elapsed_s"]["mean"],
            "throughput_min": stats_a["throughput_steps_per_s"]["min"],
            "throughput_max": stats_a["throughput_steps_per_s"]["max"],
            "throughput_mean": stats_a["throughput_steps_per_s"]["mean"],
        },
        {
            "run_id": run_id,
            "snippet": snippet,
            "variant": args.label_b,
            "steps": steps,
            "elapsed_s_min": stats_b["elapsed_s"]["min"],
            "elapsed_s_max": stats_b["elapsed_s"]["max"],
            "elapsed_s_mean": stats_b["elapsed_s"]["mean"],
            "throughput_min": stats_b["throughput_steps_per_s"]["min"],
            "throughput_max": stats_b["throughput_steps_per_s"]["max"],
            "throughput_mean": stats_b["throughput_steps_per_s"]["mean"],
        },
    ]

    current_path = output_dir / f"{snippet}_current.csv"
    history_path = output_dir / f"{snippet}_history.csv"
    write_summary_csv(current_path, summary_rows)
    append_summary_csv(history_path, summary_rows)

    print(f"Wrote {current_path}")
    print(f"Appended to {history_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
