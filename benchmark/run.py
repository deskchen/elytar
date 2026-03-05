#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Run from repo root so "benchmark" is importable:  python3 -m benchmark.run [args]

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU-only PhysX benchmark runner")
    parser.add_argument(
        "--tasks",
        type=str,
        default="cube_stack,pouring_balls",
        help="Comma-separated list of tasks",
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
        "--prefix",
        type=str,
        default="results",
        help="Prefix for output files: {prefix}_current.csv and {prefix}_history.csv",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Run identifier written to CSV rows",
    )
    parser.add_argument("--list-tasks", action="store_true", help="List available task names")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable viewer and update render each step (requires Vulkan; do not set SAPIEN_SKIP_VULKAN).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel envs (vectorized). N>1 uses one PhysX GPU system and N scenes. Only supported by some tasks.",
    )

    # cube_stack
    parser.add_argument("--cube-count", type=int, default=4, help="Number of cubes")
    parser.add_argument("--cube-half-size", type=float, default=0.04)
    parser.add_argument("--cube-spacing", type=float, default=0.0)

    # pouring_balls
    parser.add_argument("--ball-count", type=int, default=4, help="Number of balls")
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


_ARGS = _parse_args()

# Disable Vulkan only when not rendering (viewer needs Vulkan).
if not _ARGS.render:
    os.environ["SAPIEN_SKIP_VULKAN"] = "1"

import math

import sapien

from benchmark.output_csv import (
    STAGE_NAMES,
    metadata_to_string,
    summary_columns,
    append_rows,
    write_rows,
)
from benchmark.tasks import get_task_builder, list_tasks, resolve_task_name


def parse_args() -> argparse.Namespace:
    return _ARGS


def _has_display() -> bool:
    """True if we likely have a display (e.g. X11 DISPLAY set). Avoids creating a window in headless/Docker."""
    return bool(os.environ.get("DISPLAY", "").strip())


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
    rows: list[dict], *, run_id: str, task: str, config: str, steps: int, warmup_steps: int, dt: float, task_config: str
) -> dict:
    summary = {
        "run_id": run_id,
        "task": task,
        "config": config,
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
            summary[f"{stage}_p90_ms"] = percentile(values, 90.0)
            summary[f"{stage}_p99_ms"] = percentile(values, 99.0)
            summary[f"{stage}_max_ms"] = max(values)
            summary[f"{stage}_min_ms"] = min(values)
        else:
            summary[f"{stage}_mean_ms"] = 0.0
            summary[f"{stage}_p90_ms"] = 0.0
            summary[f"{stage}_p99_ms"] = 0.0
            summary[f"{stage}_max_ms"] = 0.0
            summary[f"{stage}_min_ms"] = 0.0
    return summary


def run_task(args: argparse.Namespace, task_name: str) -> tuple[list[dict], dict]:
    builder = get_task_builder(task_name)
    # Use maximum practical GPU memory config for vectorized runs (avoids PhysX buffer overflow).
    # heap_capacity must be a power of two per PhysX. 2GB needed for large contact/patch buffers (1GB can cause memcpy 700 / segfault).
    # Sizes tuned for 2k–4k envs: contact/patch from PhysX overflow messages (e.g. 2048 envs → 152M contact, 38M patch).
    num_envs = getattr(args, "num_envs", 1)
    if num_envs > 1:
        sapien.physx.set_gpu_memory_config(
            temp_buffer_capacity=256 * 1024 * 1024,
            max_rigid_contact_count=152 * 1024 * 1024,
            max_rigid_patch_count=40 * 1024 * 1024,
            heap_capacity=2**31,
            found_lost_pairs_capacity=152 * 1024 * 1024,
            found_lost_aggregate_pairs_capacity=65536,
            total_aggregate_pairs_capacity=65536,
            collision_stack_size=16 * 1024 * 1024,
        )
    runtime = builder(args)

    if not isinstance(runtime.physx_system, sapien.physx.PhysxGpuSystem):
        raise RuntimeError(f"Task '{task_name}' did not create a PhysxGpuSystem")

    print(f"[{task_name}] Initializing GPU ...", flush=True)
    runtime.physx_system.gpu_init()

    # Optional viewer for --render (scene(s) must have been built with RenderSystem).
    viewer = None
    if args.render:
        if not _has_display():
            print("Error: --render requires a display (set DISPLAY). No display detected.", file=sys.stderr)
            sys.exit(1)
        from sapien.utils.viewer.viewer import Viewer
        viewer = Viewer()
        scenes = getattr(runtime, "scenes", None)
        if scenes:
            import numpy as np
            # Use PhysX scene offsets so viewer matches simulation layout
            offsets = np.array(
                [runtime.physx_system.get_scene_offset(s) for s in scenes],
                dtype=np.float32,
            )
            viewer.set_scenes(scenes, offsets)
            # SceneGroup needs lighting/cubemap set on the viewer's internal scene (like gpu_viewer)
            vs = viewer.window._internal_scene
            vs.set_ambient_light([0.5, 0.5, 0.5])
            cubemap = scenes[0].render_system.get_cubemap()
            if cubemap is not None:
                vs.set_cubemap(cubemap._internal_cubemap)
        else:
            viewer.set_scene(runtime.scene)

    before_step = runtime.before_step
    dt = float(args.dt)

    # Warmup (cap for large num_envs so we don't sit on CPU-bound setup forever before GPU runs)
    warmup_steps = args.warmup_steps
    if num_envs > 512:
        warmup_steps = min(warmup_steps, 30)
    print(f"[{task_name}] Warmup ({warmup_steps} steps) ...", flush=True)
    for step_idx in range(warmup_steps):
        if before_step is not None:
            before_step(step_idx, step_idx * dt)
        runtime.physx_system.step()

    print(f"[{task_name}] Running {args.steps} steps ...", flush=True)
    rows: list[dict] = []
    for step_idx in range(args.steps):
        if before_step is not None:
            before_step(step_idx, step_idx * dt)

        sapien.physx.stage_profiler_begin_frame()
        runtime.physx_system.step()
        sapien.physx.stage_profiler_end_frame()

        if viewer is not None:
            runtime.physx_system.sync_poses_gpu_to_cpu()
            viewer.window.update_render()
            viewer.render()

        stage = sapien.physx.get_stage_profiler_last_frame_stage_ms()
        config = runtime.metadata.get("config", "N/A")
        row = {
            "run_id": args.run_id,
            "task": runtime.name,
            "config": config,
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
    config = runtime.metadata.get("config", "N/A")
    summary = summarize_task_rows(
        rows,
        run_id=args.run_id,
        task=runtime.name,
        config=config,
        steps=args.steps,
        warmup_steps=warmup_steps,
        dt=dt,
        task_config=task_config,
    )

    scenes = getattr(runtime, "scenes", None)
    if scenes:
        for s in scenes:
            s.clear()
    else:
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
    if args.num_envs < 1:
        args.num_envs = 1

    # Require local SAPIEN + PhysX build (no prebuilt fallback).
    if not getattr(sapien, "__local_physx_version__", None):
        print(
            "ERROR: This benchmark requires a local SAPIEN wheel built with local PhysX. "
            "Run: scripts/update_toolchain.sh",
            file=sys.stderr,
        )
        return 1

    # If not set, use PhysX under current working directory (run from repo root).
    if not os.environ.get("SAPIEN_PHYSX5_DIR"):
        local_physx = Path.cwd() / "physx"
        if (local_physx / "bin").exists():
            os.environ["SAPIEN_PHYSX5_DIR"] = str(local_physx)
    if not os.environ.get("SAPIEN_PHYSX5_DIR"):
        print(
            "ERROR: SAPIEN_PHYSX5_DIR is not set. Run from repo root (so cwd/physx exists) after "
            "scripts/update_toolchain.sh, or set SAPIEN_PHYSX5_DIR to your PhysX source directory.",
            file=sys.stderr,
        )
        return 1

    # Global GPU setup + stage profiler setup (uses local PhysX GPU lib; errors if missing).
    sapien.physx.enable_gpu()
    sapien.physx.set_stage_profiler_enabled(True)

    summary_rows: list[dict] = []

    for task_name in requested_tasks:
        _, summary = run_task(args, task_name)
        summary_rows.append(summary)

        print(
            f"[{task_name}] total_mean_ms={summary['total_mean_ms']:.4f}, "
            f"total_p90_ms={summary['total_p90_ms']:.4f}"
        )

    output_dir = Path(args.output_dir)
    current_path = output_dir / f"{args.prefix}_current.csv"
    history_path = output_dir / f"{args.prefix}_history.csv"

    write_rows(current_path, summary_columns(), summary_rows)
    append_rows(history_path, summary_columns(), summary_rows)

    print(f"Wrote {current_path}")
    print(f"Appended to {history_path}")

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

