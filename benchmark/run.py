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
        help='Comma-separated tasks. Sequential mode: "cube_stack,pouring_balls". Mixed mode: "cube_stack:4,franka:16".',
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
from envs import get_task_scene_builder, list_tasks, resolve_task_name
from envs.base import TaskRuntime

TaskSpec = tuple[str, int | None]


def parse_args() -> argparse.Namespace:
    return _ARGS


def parse_task_specs(tasks_arg: str) -> tuple[list[TaskSpec], bool]:
    specs: list[TaskSpec] = []
    has_explicit_counts = False
    for token in tasks_arg.split(","):
        item = token.strip()
        if not item:
            continue
        if ":" in item:
            task_name_raw, count_raw = item.split(":", 1)
            task_name = resolve_task_name(task_name_raw)
            count = int(count_raw)
            if count < 1:
                raise ValueError(f"Task '{task_name_raw}' has invalid env count '{count_raw}' (must be >= 1)")
            specs.append((task_name, count))
            has_explicit_counts = True
        else:
            specs.append((resolve_task_name(item), None))

    if not specs:
        raise ValueError("No tasks selected")
    if has_explicit_counts and any(count is None for _, count in specs):
        raise ValueError(
            "Mixed task syntax is ambiguous. When using task counts, provide all as task:count "
            '(e.g. --tasks "cube_stack:4,franka:16").'
        )
    return specs, has_explicit_counts


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


def _build_gpu_memory_config(num_envs: int) -> dict:
    gpu_config = GPUMemoryConfig().to_dict()
    if num_envs > 1:
        # ManiSkill-style: heap/temp stay at defaults (64 MB, 16 MB); PhysX grows heap if needed.
        # Only scale contact/patch per ManiSkill (rotate_single_object_in_hand, insert_flower).
        gpu_config["max_rigid_contact_count"] = 67141632
        gpu_config["max_rigid_patch_count"] = 16785408
        gpu_config["found_lost_pairs_capacity"] = 2**26
        # Do NOT override heap_capacity or temp_buffer_capacity - use GPUMemoryConfig defaults
    return gpu_config


def _apply_gpu_memory_config(gpu_config: dict) -> None:
    try:
        sapien.physx.set_gpu_memory_config(**gpu_config)
    except TypeError:
        gpu_config.pop("collision_stack_size")
        sapien.physx.set_gpu_memory_config(**gpu_config)


def _scene_offset(scene_idx: int, total_envs: int, env_spacing: float = 50.0) -> list[float]:
    scene_grid_length = int(math.ceil(math.sqrt(total_envs)))
    scene_x = scene_idx % scene_grid_length - scene_grid_length // 2
    scene_y = scene_idx // scene_grid_length - scene_grid_length // 2
    return [scene_x * env_spacing, scene_y * env_spacing, 0.0]


def _build_runtime_from_specs(
    args: argparse.Namespace,
    task_specs: list[tuple[str, int]],
    *,
    runtime_name: str,
) -> TaskRuntime:
    total_envs = sum(count for _, count in task_specs)
    if total_envs < 1:
        raise ValueError("No environments requested")

    px = sapien.physx.PhysxGpuSystem(device=args.device)
    render = bool(getattr(args, "render", False))
    scenes = []
    before_steps = []
    metadata: dict[str, object] = {"total_envs": total_envs}
    scene_idx = 0

    for task_name, count in task_specs:
        scene_builder = get_task_scene_builder(task_name)
        if scene_builder is None:
            raise RuntimeError(
                f"Task '{task_name}' does not expose build_scene_{task_name}(). "
                "Please add a single-scene builder for centralized vectorization."
            )
        metadata[f"{task_name}_num_envs"] = count
        for _ in range(count):
            systems = [px]
            if render:
                systems.append(sapien.render.RenderSystem())
            scene = sapien.Scene(systems)
            px.set_scene_offset(scene, _scene_offset(scene_idx, total_envs))
            result = scene_builder(scene, args)
            scenes.append(scene)
            if getattr(result, "before_step", None) is not None:
                before_steps.append(result.before_step)
            if getattr(result, "metadata", None):
                for key, value in result.metadata.items():
                    metadata[f"{task_name}_{key}"] = value
            scene_idx += 1

    def combined_before_step(step_idx: int, time_s: float) -> None:
        for hook in before_steps:
            hook(step_idx, time_s)

    return TaskRuntime(
        name=runtime_name,
        scene=scenes[0],
        physx_system=px,
        before_step=combined_before_step if before_steps else None,
        metadata=metadata,
        scenes=scenes,
    )


def _run_runtime(args: argparse.Namespace, task_label: str, num_envs: int, runtime: TaskRuntime) -> tuple[list[dict], dict]:
    if not isinstance(runtime.physx_system, sapien.physx.PhysxGpuSystem):
        raise RuntimeError(f"Task '{task_label}' did not create a PhysxGpuSystem")

    print(f"[{task_label}] Initializing GPU ...", flush=True)
    runtime.physx_system.gpu_init()
    print(f"[{task_label}] GPU ready", flush=True)

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
            if len(scenes) > 1:
                center_xy = offsets[:, :2].mean(axis=0)
                span_xy = offsets[:, :2].max(axis=0) - offsets[:, :2].min(axis=0)
                span = float(max(span_xy[0], span_xy[1], 1.0))
                # Overview camera for mixed/vectorized runs so env clusters are visible.
                viewer.set_camera_pose(
                    sapien.Pose(
                        p=[float(center_xy[0] - 0.35 * span), float(center_xy[1]), 0.18 * span + 0.6]
                    )
                )
                viewer.window.set_camera_parameters(0.05, 3000, math.pi / 5)
            else:
                # Close-up camera for single scene.
                viewer.set_camera_pose(sapien.Pose(p=[-0.2, 0, 0.12]))
                viewer.window.set_camera_parameters(0.01, 1000, math.pi / 6)
        else:
            viewer.set_scene(runtime.scene)
            viewer.set_camera_pose(sapien.Pose(p=[-0.2, 0, 0.12]))
            viewer.window.set_camera_parameters(0.01, 1000, math.pi / 6)
        print("Viewer: scroll wheel=zoom, right-drag=rotate, middle-drag=pan, F=focus selected", flush=True)

    before_step = runtime.before_step
    dt = float(args.dt)

    # Warmup (cap for large num_envs so we don't sit on CPU-bound setup forever before GPU runs)
    warmup_steps = int(args.warmup_steps)
    if num_envs > 512:
        warmup_steps = min(warmup_steps, 30)
    print(f"[{task_label}] Warmup ({warmup_steps} steps) ...", flush=True)
    for step_idx in range(warmup_steps):
        if before_step is not None:
            before_step(step_idx, step_idx * dt)
        runtime.physx_system.step()
    print(f"[{task_label}] Warmup done", flush=True)

    print(f"[{task_label}] Running {args.steps} steps ...", flush=True)
    rows: list[dict] = []
    for step_idx in range(args.steps):
        if before_step is not None:
            before_step(step_idx, step_idx * dt)

        sapien.physx.stage_profiler_begin_frame()
        runtime.physx_system.step()
        sapien.physx.stage_profiler_end_frame()

        if viewer is not None:
            # Keep articulation links (e.g., Franka) in sync for rendering in GPU mode.
            # Without this, mixed articulated scenes can appear overlapped or scrambled.
            if hasattr(runtime.physx_system, "gpu_update_articulation_kinematics"):
                runtime.physx_system.gpu_update_articulation_kinematics()
            if hasattr(runtime.physx_system, "gpu_fetch_articulation_link_pose"):
                runtime.physx_system.gpu_fetch_articulation_link_pose()
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

    print(f"[{task_label}] Done ({args.steps} steps)", flush=True)
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


def run_task(args: argparse.Namespace, task_name: str) -> tuple[list[dict], dict]:
    num_envs = max(1, int(getattr(args, "num_envs", 1)))
    gpu_config = _build_gpu_memory_config(num_envs)
    if getattr(args, "debug_gpu_config", False):
        _print_gpu_config_debug(gpu_config, num_envs)
    _apply_gpu_memory_config(gpu_config)

    # Match ManiSkill: set scene/body/shape/material config before creating PhysxGpuSystem.
    _set_physx_scene_config()
    runtime = _build_runtime_from_specs(args, [(task_name, num_envs)], runtime_name=task_name)
    return _run_runtime(args, task_name, num_envs, runtime)


def run_combined_task(args: argparse.Namespace, task_specs: list[tuple[str, int]]) -> tuple[list[dict], dict]:
    total_envs = sum(count for _, count in task_specs)
    gpu_config = _build_gpu_memory_config(total_envs)
    if getattr(args, "debug_gpu_config", False):
        _print_gpu_config_debug(gpu_config, total_envs)
    _apply_gpu_memory_config(gpu_config)
    _set_physx_scene_config()
    task_name = "+".join(f"{name}:{count}" for name, count in task_specs)
    runtime = _build_runtime_from_specs(args, task_specs, runtime_name=task_name)
    return _run_runtime(args, task_name, total_envs, runtime)


def main() -> int:
    args = parse_args()

    if args.list_tasks:
        print("\n".join(list_tasks()))
        return 0

    requested_task_specs, has_explicit_counts = parse_task_specs(args.tasks)
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
    if has_explicit_counts:
        combined_specs = [(task_name, int(count)) for task_name, count in requested_task_specs if count is not None]
        _, summary = run_combined_task(args, combined_specs)
        summary_rows.append(summary)
        task_label = summary["task"]
        if args.prefix:
            print(
                f"[{task_label}] total_mean_ms={summary['total_mean_ms']:.4f}, "
                f"total_p90_ms={summary['total_p90_ms']:.4f}"
            )
        else:
            parts = [f"{s}_mean={summary[f'{s}_mean_ms']:.4f}" for s in STAGE_NAMES]
            print(f"[{task_label}] " + ", ".join(parts))
    else:
        requested_tasks = [name for name, _ in requested_task_specs]
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
