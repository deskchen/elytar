# GPU PhysX Benchmarks

This benchmark suite runs **GPU-only PhysX** tasks and writes results in CSV:

- `results_current.csv`: current run's summary (overwritten each run)
- `results_history.csv`: all runs appended (history)

## Tasks

- `cube_stack`: ManiSkill StackCube (table + 2 cubes)
- `pouring_balls`: container + many dynamic spheres
- `humanoid_from_urdf`: load a humanoid URDF and run simple open-loop joint targets

Use `--cube-count N` or `--ball-count N` to set object counts.

## Prerequisites

- Run inside the dev container with NVIDIA GPU access.
- **Local build only**: The benchmark uses GPU and **requires** a local SAPIEN + PhysX build. It does **not** fall back to prebuilt downloads; if the local build is missing or not found, the script exits with an error.
- Build the toolchain once from repo root:
  - `scripts/update_toolchain.sh` (builds PhysX and SAPIEN from source, installs the wheel).
- Run the benchmark from **repo root** so `SAPIEN_PHYSX5_DIR` is set automatically to the repo’s `physx-5.3.1` tree. Otherwise set `SAPIEN_PHYSX5_DIR` (and optionally `PHYSX_CONFIG`) yourself.

**Per-stage timings** (broadphase, narrowphase, solver, etc.) are only reported when the PhysX GPU lib is built with profiling (e.g. `PHYSX_CONFIG=checked` or `profile`); in `release`, `PX_PROFILE_ZONE` is compiled out and stage columns will be zero.

## Run

From repo root:

```bash
python3 -m benchmark.run --tasks cube_stack,pouring_balls
```

Include humanoid URDF benchmark:

```bash
python3 -m benchmark.run \
  --tasks cube_stack,pouring_balls,humanoid_from_urdf \
  --humanoid-urdf /absolute/path/to/humanoid.urdf \
  --humanoid-motion walk
```

List available tasks:

```bash
python3 -m benchmark.run --list-tasks
```

## Output columns

- `{prefix}_current.csv`: Current run's summary (overwritten each run).
- `{prefix}_history.csv`: All runs appended (history).

Use `--prefix` to set the prefix (default: `results`). Both use the same columns:
- `run_id, task, config, steps, warmup_steps, dt, task_config`
- all means, then p90, p99, max, min for each stage: `broadphase_mean_ms`, ..., `total_mean_ms`, `broadphase_p90_ms`, ..., `total_p90_ms`, `broadphase_p99_ms`, ..., `total_p99_ms`, `broadphase_max_ms`, ..., `total_max_ms`, `broadphase_min_ms`, ..., `total_min_ms`

**Sweep script** (`benchmark/run_sweep.sh`): Runs num_envs 2, 4, 8, ..., 1024. Configurable via env vars:
```bash
TASK=cube_stack STEPS=2000 ./benchmark/run_sweep.sh
```

**Plot script** (`benchmark/plot_solver_ratio.py`): Plots solver/total % across num_envs from history:
```bash
python3 benchmark/plot_solver_ratio.py [--input benchmark/results/results_history.csv] [--output plot.png] [--task cube_stack]
```

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

From your laptop: `ssh -L 5900:localhost:5900 <user>@<host>`, then connect VNC to `localhost:5900`. If `bind [127.0.0.1]:5900: Permission denied` (e.g. Cursor IDE uses 5900 for remote forwarding), use `ssh -L 15900:localhost:5900 <user>@<host>` and connect VNC to `localhost:15900`.

**If the VNC connection keeps spinning:**

1. **On the host** run display setup and confirm VNC is listening:
   ```bash
   ./scripts/host_display_setup.sh
   ss -tlnp | grep 5900   # or: netstat -tlnp | grep 5900
   ```
   You should see something like `127.0.0.1:5900` (x11vnc with `-localhost`). If nothing listens, check `/tmp/elytar-x11vnc.log` on the host.

2. **Keep the SSH session open** — the tunnel lives in that session; closing it drops the forward.

3. **Password**: The VNC server is started with **no password** (`-nopw`). When the Mac prompts for a password, **leave it blank** and connect (or press Enter). Typing anything can make some clients spin or hang.

4. **On macOS** use the VNC URL explicitly:
   - Finder → Go → Connect to Server (⌘K), then enter: `vnc://localhost:5900`
   - Or in Terminal: `open vnc://localhost:5900`
   If the built-in Screen Sharing still spins, try a VNC client (e.g. [TigerVNC Viewer](https://tigervnc.org/) or RealVNC) and connect to `localhost:5900`.

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
- **Large `--num-envs` (e.g. 2048–4096)**: Requires a GPU with enough memory (e.g. 24GB+). Use `--debug-gpu-config` to print the config and rough memory estimates before running.
- **CUDA 700 / PxgCudaMemoryAllocator / OOM**: The flow is: `enable_gpu()` → `set_gpu_memory_config()` → `builder()` creates `PhysxGpuSystem` (reads config, creates one PxScene) → `gpu_init()` runs first simulate. PhysX: contact/patch = pinned host RAM; heap = GPU. If only 17/24 GB GPU used but still 700, it may be a PhysX/driver bug rather than OOM. Try: `--num-envs 1024`, `CUDA_VISIBLE_DEVICES=0`, or `compute-sanitizer --tool memcheck python3 -m benchmark.run ...`.

