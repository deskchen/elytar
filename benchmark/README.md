# GPU PhysX Benchmarks

This benchmark suite runs **GPU-only PhysX** tasks and writes results in CSV:

- `results_steps.csv`: one row per measured step
- `results_summary.csv`: one row per task/configuration

## Tasks

- `cube_stack`: procedural rigid-body cube stacks
- `pouring_balls`: container + many dynamic spheres
- `humanoid_from_urdf`: load a humanoid URDF and run simple open-loop joint targets

Use `--difficulty easy|medium|hard` to scale task complexity.

## Prerequisites

- Run inside the dev container with NVIDIA GPU access.
- **Local build only**: The benchmark uses GPU and **requires** a local SAPIEN + PhysX build. It does **not** fall back to prebuilt downloads; if the local build is missing or not found, the script exits with an error.
- Build the toolchain once from repo root:
  - `scripts/update_toolchain.sh` (builds PhysX and SAPIEN from source, installs the wheel).
- Run the benchmark from **repo root** so `SAPIEN_PHYSX5_DIR` is set automatically to the repo’s `physx` tree. Otherwise set `SAPIEN_PHYSX5_DIR` (and optionally `PHYSX_CONFIG`) yourself.

**Per-stage timings** (broadphase, narrowphase, solver, etc.) are only reported when the PhysX GPU lib is built with profiling (e.g. `PHYSX_CONFIG=checked` or `profile`); in `release`, `PX_PROFILE_ZONE` is compiled out and stage columns will be zero.

## Run

From repo root:

```bash
python benchmark/run.py --tasks cube_stack,pouring_balls --difficulty easy
```

Include humanoid URDF benchmark:

```bash
python benchmark/run.py \
  --tasks cube_stack,pouring_balls,humanoid_from_urdf \
  --difficulty medium \
  --humanoid-urdf /absolute/path/to/humanoid.urdf \
  --humanoid-motion walk
```

List available tasks:

```bash
python benchmark/run.py --list-tasks
```

## Output columns

`results_steps.csv`:

- `run_id, task, difficulty, step, dt`
- `broadphase_ms, narrowphase_ms, coloring_ms, solver_ms, update_ms, total_ms`

`results_summary.csv`:

- `run_id, task, difficulty, steps, warmup_steps, dt, task_config`
- for each stage in `broadphase|narrowphase|coloring|solver|update|total`:
  - `<stage>_mean_ms`
  - `<stage>_p50_ms`
  - `<stage>_p95_ms`
  - `<stage>_max_ms`

## Notes

- Stage latency comes from PhysX profiling zones (`PxProfilerCallback`), reported in milliseconds.
- This is timeline attribution, not direct GPU kernel-only timing.
- Rendering is intentionally omitted for benchmark consistency.

## Troubleshooting

- **`PxDeviceAllocatorCallback failed to allocate memory 67108864 bytes` then segfault**  
  PhysX GPU’s first allocation (64 MB heap) is failing. Steps:
  1. **Check 64 MB allocation in this process** – Run `python benchmark/run.py --check-cuda`. It runs a raw `cuMemAlloc(64MB)` in this process. If it fails (e.g. with a CUDA error name/string), **GPU device allocation is broken in this environment** (driver, container, or context). Fix the container/driver so `--check-cuda` passes before running the full benchmark.
  2. **Confirm which GPU lib is loaded** – Run with `SAPIEN_DEBUG_GPU_LIB=1`; the log should show the path to the local `libPhysXGpu_64.so`.
  3. **CUDA version match** – Build the toolchain **inside the same container** where you run (`scripts/update_toolchain.sh`).
  4. **Multiple GPUs** – Try `CUDA_VISIBLE_DEVICES=0 python benchmark/run.py ...`

- **Vulkan warnings and then `PxDeviceAllocatorCallback failed`**  
  The benchmark sets `SAPIEN_SKIP_VULKAN=1` so Vulkan is not initialized. If you still see Vulkan warnings and then the allocator failure, the **installed SAPIEN wheel was built before this fix**. Re-run **`scripts/update_toolchain.sh`** in the repo so the wheel is reinstalled with the updated `_vulkan_tricks.py`, then run the benchmark again.

