#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Run from repo root so "benchmark" and "envs" are importable:  python3 -m benchmark.run [args]
#
# GPU memory flow (to debug OOM / CUDA 700):
# 1. main() calls sapien.physx.enable_gpu() (once, before any PhysX)
# 2. run_task() calls set_gpu_memory_config() with our config (must be before builder)
# 3. builder() creates PhysxGpuSystem; its ctor reads PhysxDefault::getGpuMemoryConfig()
#    and creates PxScene with that config. One PxScene for all vectorized envs.
# 4. gpu_init() runs first simulate()+fetchResults(); PhysX allocates from config.
# PhysX: contact/patch = pinned host memory; heap = GPU; found_lost = GPU.
# If OOM: try --debug-gpu-config to see allocations; reduce num_envs or multipliers.

# Parse --render early so we can set SAPIEN_SKIP_VULKAN before envs (and sapien) are imported.
_early_parser = argparse.ArgumentParser()
_early_parser.add_argument("--render", action="store_true")
_early_args, _ = _early_parser.parse_known_args()
if not _early_args.render:
    os.environ["SAPIEN_SKIP_VULKAN"] = "1"


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
        default=None,
        help="Prefix for output files: {prefix}_current.csv and {prefix}_history.csv. If not set, only print metrics (no file write).",
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
    parser.add_argument(
        "--debug-gpu-config",
        action="store_true",
        help="Print GPU memory config and rough memory estimates before running.",
    )

    from envs import add_all_env_args
    add_all_env_args(parser)
    return parser.parse_args()


_ARGS = _parse_args()

import math

import sapien

from benchmark.config import GPUMemoryConfig
from benchmark.output_csv import (
    STAGE_NAMES,
    metadata_to_string,
    summary_columns,
    append_rows,
    write_rows,
)
from envs import get_task_builder, list_tasks, resolve_task_name


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


def _print_gpu_config_debug(gpu_config: dict, num_envs: int) -> None:
    """Print GPU config and rough memory estimates. PhysX: contact/patch in pinned host, heap in GPU."""
    print("\n=== GPU memory config (PhysX) ===")
    for k, v in gpu_config.items():
        if isinstance(v, int) and v >= 1024:
            print(f"  {k}: {v:,} ({v / 1024 / 1024:.1f} M)")
        else:
            print(f"  {k}: {v}")
    # Rough estimates: contact/patch double-buffered, ~64B/contact, ~32B/patch (PhysX struct sizes vary)
    contact = gpu_config.get("max_rigid_contact_count", 0)
    patch = gpu_config.get("max_rigid_patch_count", 0)
    found = gpu_config.get("found_lost_pairs_capacity", 0)
    heap = gpu_config.get("heap_capacity", 0)
    temp = gpu_config.get("temp_buffer_capacity", 0)
    pinned_mb = 2 * (contact * 64 + patch * 32) / 1024 / 1024
    gpu_mb = (heap + found * 8 + temp) / 1024 / 1024
    print(f"\n  Rough pinned host (contact+patch, 2x buffered): ~{pinned_mb:.0f} MB")
    print(f"  Rough GPU (heap+found_lost+temp): ~{gpu_mb:.0f} MB")
    print(f"  num_envs: {num_envs}\n")


def _set_physx_scene_config() -> None:
    """Match ManiSkill's PhysX config (scene_config, body_config, shape_config, default_material).
    Must be called before creating PhysxGpuSystem; PhysxSystem reads PhysxDefault at construction."""
    import numpy as np

    sapien.physx.set_shape_config(contact_offset=0.02, rest_offset=0.0)
    sapien.physx.set_body_config(
        solver_position_iterations=15,
        solver_velocity_iterations=1,
        sleep_threshold=0.005,
    )
    sapien.physx.set_scene_config(
        gravity=np.array([0.0, 0.0, -9.81], dtype=np.float32),
        bounce_threshold=2.0,
        enable_pcm=True,
        enable_tgs=True,
        enable_ccd=False,
        enable_enhanced_determinism=False,
        enable_friction_every_iteration=True,
        cpu_workers=0,
    )
    sapien.physx.set_default_material(
        static_friction=0.3,
        dynamic_friction=0.3,
        restitution=0.0,
    )


def run_task(args: argparse.Namespace, task_name: str) -> tuple[list[dict], dict]:
    builder = get_task_builder(task_name)
    num_envs = getattr(args, "num_envs", 1)
    gpu_config = GPUMemoryConfig().to_dict()
    if num_envs > 1:
        # ManiSkill-style: heap/temp stay at defaults (64 MB, 16 MB); PhysX grows heap if needed.
        # Only scale contact/patch per ManiSkill (rotate_single_object_in_hand, insert_flower).
        n = num_envs
        base = n * max(1024, n)
        # gpu_config["max_rigid_contact_count"] = base * 8
        gpu_config["max_rigid_contact_count"] = 67141632
        gpu_config["max_rigid_patch_count"] = 16785408
        # gpu_config["max_rigid_patch_count"] = base * 2
        gpu_config["found_lost_pairs_capacity"] = 2**26
        # Do NOT override heap_capacity or temp_buffer_capacity - use GPUMemoryConfig defaults
    if getattr(args, "debug_gpu_config", False):
        _print_gpu_config_debug(gpu_config, num_envs)

    try:
        sapien.physx.set_gpu_memory_config(**gpu_config)
    except TypeError:
        gpu_config.pop("collision_stack_size")
        sapien.physx.set_gpu_memory_config(**gpu_config)

    # Match ManiSkill: set scene/body/shape/material config before creating PhysxGpuSystem.
    _set_physx_scene_config()

    runtime = builder(args)

    if not isinstance(runtime.physx_system, sapien.physx.PhysxGpuSystem):
        raise RuntimeError(f"Task '{task_name}' did not create a PhysxGpuSystem")

    print(f"[{task_name}] Initializing GPU ...", flush=True)
    runtime.physx_system.gpu_init()
    print(f"[{task_name}] GPU ready", flush=True)

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
            # SceneGroup needs lighting set on the viewer's internal scene (like gpu_viewer).
            # Skip cubemap to avoid bright sky HDR washing out colors to white.
            vs = viewer.window._internal_scene
            vs.set_ambient_light([0.15, 0.15, 0.15])
        else:
            viewer.set_scene(runtime.scene)
        # Close-up camera for stacked cubes
        viewer.set_camera_pose(sapien.Pose(p=[-0.2, 0, 0.12]))
        viewer.window.set_camera_parameters(0.01, 1000, math.pi / 6)
        print("Viewer: scroll wheel=zoom, right-drag=rotate, middle-drag=pan, F=focus selected", flush=True)

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
    print(f"[{task_name}] Warmup done", flush=True)

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
            scenes = getattr(runtime, "scenes", None) or ([runtime.scene] if runtime.scene else [])
            for s in scenes:
                s.update_render()
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

    print(f"[{task_name}] Done ({args.steps} steps)", flush=True)
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
        local_physx = Path.cwd() / "physx-5.6.1"
        if (local_physx / "bin").exists():
            os.environ["SAPIEN_PHYSX5_DIR"] = str(local_physx)
    if not os.environ.get("SAPIEN_PHYSX5_DIR"):
        print(
            "ERROR: SAPIEN_PHYSX5_DIR is not set. Run from repo root (so cwd/physx-5.6.1 exists) after "
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

        if args.prefix:
            print(
                f"[{task_name}] total_mean_ms={summary['total_mean_ms']:.4f}, "
                f"total_p90_ms={summary['total_p90_ms']:.4f}"
            )
        else:
            parts = [f"{s}_mean={summary[f'{s}_mean_ms']:.4f}" for s in STAGE_NAMES]
            print(f"[{task_name}] " + ", ".join(parts))

    if args.prefix:
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
