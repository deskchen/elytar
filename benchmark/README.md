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
python3 -m benchmark.run --tasks cube_stack,pouring_balls --difficulty easy
```

Include humanoid URDF benchmark:

```bash
python3 -m benchmark.run \
  --tasks cube_stack,pouring_balls,humanoid_from_urdf \
  --difficulty medium \
  --humanoid-urdf /absolute/path/to/humanoid.urdf \
  --humanoid-motion walk
```

List available tasks:

```bash
python3 -m benchmark.run --list-tasks
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

## Display setup for `--render`

For live viewer rendering, use the host NVIDIA X display (not Xvfb). `./scripts/run-dev.sh` runs `host_display_setup.sh` automatically on the host before starting/reattaching the container.

```bash
# On host: starts display setup, then enters container
./scripts/run-dev.sh

# In container
export DISPLAY=:0
python3 -m benchmark.run --tasks cube_stack --steps 20 --render
```

To run display setup manually (e.g. for debugging): `./scripts/host_display_setup.sh` on the host.

From your laptop: `ssh -L 5900:localhost:5900 <user>@<host>`, then connect VNC to `localhost:5900`.

## Rendering and vectorized envs

- **`--render`**: Enables a viewer and updates render each step. Requires Vulkan (do not set `SAPIEN_SKIP_VULKAN`). Only **cube_stack** adds a `RenderSystem` and visuals; other tasks run headless.
- **`--num-envs N`**: Runs N parallel scenes sharing one PhysX GPU system (vectorized). Only **cube_stack** supports `N > 1`; others use a single scene.

Example:

```bash
# 4 parallel cube_stack envs, headless
python3 -m benchmark.run --tasks cube_stack --num-envs 4 --steps 200

# cube_stack with viewer (from repo root, with display or virtual display)
python3 -m benchmark.run --tasks cube_stack --steps 300 --render

# 4 envs + rendering
python3 -m benchmark.run --tasks cube_stack --num-envs 4 --steps 200 --render
```

## Verification

To confirm the implementation matches SAPIEN’s intended usage:

1. **SAPIEN’s own “vec env” demos** (same pattern: one `PhysxGpuSystem`, multiple scenes, `set_scene_offset`):
   - **`sapien/manualtest/gpu_viewer.py`** – 16 scenes, viewer, `px.step()` then `sync_poses_gpu_to_cpu()` and `viewer.window.update_render()`.
   - **`sapien/manualtest/gpu.py`** – minimal: one `PhysxGpuSystem`, two scenes with `set_scene_offset`, same content built in each.

2. **Quick sanity checks** (from repo root, after `scripts/update_toolchain.sh`):
   - Headless vectorized:  
     `python3 -m benchmark.run --tasks cube_stack --num-envs 2 --steps 5`  
     Should finish and write CSV; no viewer.
   - Single env with rendering (requires display):  
     `python3 -m benchmark.run --tasks cube_stack --steps 20 --render`  
     Should open a window and show cubes; close the window to end.
   - Vectorized + render:  
     `python3 -m benchmark.run --tasks cube_stack --num-envs 4 --steps 20 --render`  
     Should show 4 cube stacks in a 2×2 layout.

3. **Compare timing**: Run the same task with `--num-envs 1` and `--num-envs 4` (e.g. 200 steps). Total step time should be similar (one PhysX step advances all envs); small differences can come from GPU occupancy and profiling overhead.

## Notes

- Stage latency comes from PhysX profiling zones (`PxProfilerCallback`), reported in milliseconds.
- This is timeline attribution, not direct GPU kernel-only timing.
- By default rendering is off for benchmark consistency; use `--render` when you need a viewer.

