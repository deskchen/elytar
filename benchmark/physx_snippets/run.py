#!/usr/bin/env python3
"""Run A/B benchmark comparing PhysX snippet binaries (vanilla vs capybara)."""

import argparse
import csv
import os
import subprocess
import sys
import time
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B benchmark PhysX headless snippets (vanilla vs capybara)."
    )
    parser.add_argument("--snippet", required=True, help="Snippet name (e.g. Isosurface)")
    parser.add_argument("--reps", type=int, default=10, help="Repetitions per variant")
    parser.add_argument(
        "--output",
        help="Output CSV path (default: benchmark/physx_snippets/results/{snippet}.csv)",
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
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output = args.output or str(
        workspace / "benchmark" / "physx_snippets" / "results" / f"{snippet}.csv"
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_exists = output_path.exists() and output_path.stat().st_size > 0
    csv_file = open(output_path, "a", newline="")
    writer = csv.writer(csv_file)
    if not csv_exists:
        writer.writerow(
            ["run_id", "variant", "rep", "elapsed_s", "throughput_steps_per_s", "steps_per_run", "command"]
        )

    def run_one(label: str, cmd: list[str], rep: int) -> bool:
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
            writer.writerow([run_id, label, rep, "", "", steps, f"\"{' '.join(cmd)}\" (timeout)"])
            return False
        except Exception as e:
            print(f"[{label}] rep={rep} ERROR: {e}", file=sys.stderr)
            writer.writerow([run_id, label, rep, "", "", steps, f"\"{' '.join(cmd)}\" (error: {e})"])
            return False

        elapsed = time.perf_counter() - start
        throughput = steps / elapsed if elapsed > 0 else 0.0
        writer.writerow([run_id, label, rep, f"{elapsed:.6f}", f"{throughput:.2f}", steps, f"\"{' '.join(cmd)}\""])
        print(f"[{label}] rep={rep} elapsed={elapsed:.3f}s throughput={throughput:.2f} steps/s")

        if result.returncode != 0:
            print(f"  (exit code {result.returncode})", file=sys.stderr)
        return result.returncode == 0

    cmd_a = [str(cmd_a_path)]
    cmd_b = [str(cmd_b_path)]

    for i in range(1, args.reps + 1):
        run_one(args.label_a, cmd_a, i)
        if args.delay_between_variants > 0:
            time.sleep(args.delay_between_variants)
        run_one(args.label_b, cmd_b, i)

    csv_file.close()
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
