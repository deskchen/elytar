#!/usr/bin/env python3
from __future__ import annotations

import os

# Set before any other imports so the installed SAPIEN respects it when loaded.
# (The wheel must be built with the SAPIEN_SKIP_VULKAN check in _vulkan_tricks.py.)
os.environ["SAPIEN_SKIP_VULKAN"] = "1"

import sys
from pathlib import Path

# When run as script (python benchmark/run.py), ensure repo root is on path
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
import math
import os
from datetime import datetime
from pathlib import Path

# CUDA check runs before importing sapien so Vulkan/GPU setup doesn't cause "invalid device context".
_CUDA_SUCCESS = 0


def _check_cuda_alloc_64mb() -> tuple[bool, str]:
    """Try to allocate 64MB on GPU 0 via CUDA runtime API (context handled by runtime). Returns (ok, message)."""
    import ctypes

    # Prefer runtime API: it manages the primary context and often works when driver API yields "invalid device context".
    for libname in ("libcudart.so", "libcudart.so.12", "libcudart.so.11"):
        try:
            lib = ctypes.CDLL(libname)
            break
        except OSError:
            continue
    else:
        return False, "libcudart.so not loadable"

    cuda_set_device = lib.cudaSetDevice
    cuda_set_device.argtypes = [ctypes.c_int]
    cuda_set_device.restype = int
    cuda_malloc = lib.cudaMalloc
    cuda_malloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cuda_malloc.restype = int
    cuda_free = lib.cudaFree
    cuda_free.argtypes = [ctypes.c_void_p]
    cuda_free.restype = int
    cuda_get_error_string = lib.cudaGetErrorString
    cuda_get_error_string.argtypes = [ctypes.c_int]
    cuda_get_error_string.restype = ctypes.c_char_p

    result = cuda_set_device(0)
    if result != _CUDA_SUCCESS:
        msg = cuda_get_error_string(ctypes.c_int(result))
        return False, f"cudaSetDevice(0) failed: {(msg or b'').decode('utf-8', errors='replace')}"
    size = 64 * 1024 * 1024  # 64 MB
    dptr = ctypes.c_void_p()
    result = cuda_malloc(ctypes.byref(dptr), size)
    if result != _CUDA_SUCCESS:
        msg = cuda_get_error_string(ctypes.c_int(result))
        return False, f"cudaMalloc(64MB) failed: {(msg or b'').decode('utf-8', errors='replace')}"
    if dptr.value:
        cuda_free(dptr.value)
    return True, "cudaMalloc(64MB) OK"


def _establish_cuda_primary_context() -> None:
    """Establish CUDA primary context on device 0 (runtime API) so PhysX GPU sees a valid context."""
    import ctypes

    for libname in ("libcudart.so", "libcudart.so.12", "libcudart.so.11"):
        try:
            lib = ctypes.CDLL(libname)
            break
        except OSError:
            continue
    else:
        return
    cuda_set_device = lib.cudaSetDevice
    cuda_set_device.argtypes = [ctypes.c_int]
    cuda_set_device.restype = int
    cuda_free = lib.cudaFree
    cuda_free.argtypes = [ctypes.c_void_p]
    cuda_free.restype = int
    if cuda_set_device(0) == _CUDA_SUCCESS:
        cuda_free(ctypes.c_void_p(0))  # cudaFree(0) establishes context


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-only PhysX benchmark runner")
    parser.add_argument(
        "--tasks",
        type=str,
        default="cube_stack,pouring_balls",
        help="Comma-separated list of tasks",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default="easy",
        help="Shared difficulty preset for all tasks",
    )
    parser.add_argument("--steps", type=int, default=600, help="Measured simulation steps")
    parser.add_argument("--warmup-steps", type=int, default=120, help="Warmup steps")
    parser.add_argument("--dt", type=float, default=1.0 / 240.0, help="Simulation timestep")
    parser.add_argument("--device", type=str, default="cuda", help="PhysX GPU device string")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark/results"),
        help="Directory to write CSV outputs",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Run identifier written to CSV rows",
    )
    parser.add_argument("--list-tasks", action="store_true", help="List available task names")
    parser.add_argument(
        "--check-cuda",
        action="store_true",
        help="Test 64MB GPU allocation (same size as PhysX heap) then exit; use to debug allocator failures.",
    )

    # cube_stack
    parser.add_argument("--cube-count", type=int, default=None)
    parser.add_argument("--cube-half-size", type=float, default=0.04)
    parser.add_argument("--cube-spacing", type=float, default=0.0)

    # pouring_balls
    parser.add_argument("--ball-count", type=int, default=None)
    parser.add_argument("--ball-radius", type=float, default=0.02)
    parser.add_argument("--container-half-extent", type=float, default=0.6)
    parser.add_argument("--container-wall-height", type=float, default=0.45)
    parser.add_argument("--container-wall-thickness", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=0)

    # humanoid_from_urdf
    parser.add_argument("--humanoid-urdf", type=Path, default=None)
    parser.add_argument(
        "--humanoid-motion",
        type=str,
        choices=["walk", "run"],
        default="walk",
    )
    parser.add_argument("--humanoid-target-scale", type=float, default=0.25)
    parser.add_argument("--humanoid-root-height", type=float, default=1.0)
    parser.add_argument("--humanoid-joint-stiffness", type=float, default=80.0)
    parser.add_argument("--humanoid-joint-damping", type=float, default=8.0)
    parser.add_argument("--humanoid-joint-force-limit", type=float, default=400.0)
    return parser.parse_args()


# Parse args early so we can run --check-cuda before importing sapien (Vulkan setup can cause "invalid device context").
_ARGS = _parse_args()
if _ARGS.check_cuda:
    _ok, _msg = _check_cuda_alloc_64mb()
    print(_msg)
    sys.exit(0 if _ok else 1)

import sapien

from benchmark.output_csv import (
    STEP_COLUMNS,
    STAGE_NAMES,
    metadata_to_string,
    summary_columns,
    write_rows,
)
from benchmark.tasks import get_task_builder, list_tasks, resolve_task_name


def parse_args() -> argparse.Namespace:
    return _ARGS


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * p / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]
    alpha = index - lower
    return sorted_values[lower] * (1.0 - alpha) + sorted_values[upper] * alpha


def summarize_task_rows(
    rows: list[dict], *, run_id: str, task: str, difficulty: str, steps: int, warmup_steps: int, dt: float, task_config: str
) -> dict:
    summary = {
        "run_id": run_id,
        "task": task,
        "difficulty": difficulty,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "dt": dt,
        "task_config": task_config,
    }
    for stage in STAGE_NAMES:
        key = f"{stage}_ms"
        values = [float(row[key]) for row in rows]
        if values:
            summary[f"{stage}_mean_ms"] = sum(values) / len(values)
            summary[f"{stage}_p50_ms"] = percentile(values, 50.0)
            summary[f"{stage}_p95_ms"] = percentile(values, 95.0)
            summary[f"{stage}_max_ms"] = max(values)
        else:
            summary[f"{stage}_mean_ms"] = 0.0
            summary[f"{stage}_p50_ms"] = 0.0
            summary[f"{stage}_p95_ms"] = 0.0
            summary[f"{stage}_max_ms"] = 0.0
    return summary


def run_task(args: argparse.Namespace, task_name: str) -> tuple[list[dict], dict]:
    builder = get_task_builder(task_name)
    runtime = builder(args)

    if not isinstance(runtime.physx_system, sapien.physx.PhysxGpuSystem):
        raise RuntimeError(f"Task '{task_name}' did not create a PhysxGpuSystem")

    runtime.physx_system.gpu_init()

    before_step = runtime.before_step
    dt = float(args.dt)

    # Warmup
    for step_idx in range(args.warmup_steps):
        if before_step is not None:
            before_step(step_idx, step_idx * dt)
        runtime.physx_system.step()

    rows: list[dict] = []
    for step_idx in range(args.steps):
        if before_step is not None:
            before_step(step_idx, step_idx * dt)

        sapien.physx.stage_profiler_begin_frame()
        runtime.physx_system.step()
        sapien.physx.stage_profiler_end_frame()

        stage = sapien.physx.get_stage_profiler_last_frame_stage_ms()
        row = {
            "run_id": args.run_id,
            "task": runtime.name,
            "difficulty": args.difficulty,
            "step": step_idx,
            "dt": dt,
            "broadphase_ms": float(stage.get("broadphase_ms", 0.0)),
            "narrowphase_ms": float(stage.get("narrowphase_ms", 0.0)),
            "coloring_ms": float(stage.get("coloring_ms", 0.0)),
            "solver_ms": float(stage.get("solver_ms", 0.0)),
            "update_ms": float(stage.get("update_ms", 0.0)),
            "total_ms": float(stage.get("total_ms", 0.0)),
        }
        rows.append(row)

    task_config = metadata_to_string(runtime.metadata)
    summary = summarize_task_rows(
        rows,
        run_id=args.run_id,
        task=runtime.name,
        difficulty=args.difficulty,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        dt=dt,
        task_config=task_config,
    )

    runtime.scene.clear()
    return rows, summary


def main() -> int:
    args = parse_args()

    if args.list_tasks:
        print("\n".join(list_tasks()))
        return 0

    requested_tasks = [resolve_task_name(t) for t in args.tasks.split(",") if t.strip()]
    if not requested_tasks:
        raise ValueError("No tasks selected")

    # Require local SAPIEN + PhysX build (no prebuilt fallback).
    if not getattr(sapien, "__local_physx_version__", None):
        print(
            "ERROR: This benchmark requires a local SAPIEN wheel built with local PhysX. "
            "Run: scripts/update_toolchain.sh",
            file=sys.stderr,
        )
        return 1

    # Point to repo PhysX when running from repo root so GPU lib is loaded from local build.
    if not os.environ.get("SAPIEN_PHYSX5_DIR"):
        local_physx = _repo_root / "physx"
        if (local_physx / "bin").exists():
            os.environ["SAPIEN_PHYSX5_DIR"] = str(local_physx)
    if not os.environ.get("SAPIEN_PHYSX5_DIR"):
        print(
            "ERROR: SAPIEN_PHYSX5_DIR is not set. Run this script from the repo root after "
            "scripts/update_toolchain.sh, or set SAPIEN_PHYSX5_DIR to your PhysX source directory.",
            file=sys.stderr,
        )
        return 1

    # Establish CUDA primary context before loading PhysX GPU .so so allocator sees a valid context.
    _establish_cuda_primary_context()
    # Global GPU setup + stage profiler setup (uses local PhysX GPU lib; errors if missing).
    sapien.physx.enable_gpu()
    sapien.physx.set_stage_profiler_enabled(True)

    all_step_rows: list[dict] = []
    summary_rows: list[dict] = []

    for task_name in requested_tasks:
        step_rows, summary = run_task(args, task_name)
        all_step_rows.extend(step_rows)
        summary_rows.append(summary)

        print(
            f"[{task_name}] total_mean_ms={summary['total_mean_ms']:.4f}, "
            f"total_p95_ms={summary['total_p95_ms']:.4f}"
        )

    output_dir = Path(args.output_dir)
    steps_path = output_dir / "results_steps.csv"
    summary_path = output_dir / "results_summary.csv"

    write_rows(steps_path, STEP_COLUMNS, all_step_rows)
    write_rows(summary_path, summary_columns(), summary_rows)

    print(f"Wrote {steps_path}")
    print(f"Wrote {summary_path}")

    # Warn if all stage timings are zero: PhysX profile zones are compiled out in Release
    if summary_rows and all(s.get("total_mean_ms", 0) == 0 for s in summary_rows):
        print(
            "\nWARNING: All stage timings are zero. PhysX only emits profile zones when "
            "built with PHYSX_CONFIG=checked or profile (they are compiled out in release). "
            "Rebuild PhysX and SAPIEN with e.g. PHYSX_CONFIG=profile /workspace/scripts/update_toolchain.sh"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

