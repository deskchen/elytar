# PhysX Snippet Benchmarks (PTX A/B)

This suite benchmarks PhysX C++ snippets for end-to-end latency/throughput while
comparing PTX sources:

- PTX nvcc (`generate_ptx.sh`)
- PTX Capybara (`compile_capybara_ptx.py`, materialized by `update_toolchain.sh`)

Primary workload: `SnippetIsosurface` (touches `utility` via
`interleaveGpuBuffers` -> `interleaveBuffers` kernel path).

## Build matrix

Use one Elytar tree for PTX comparison (`physx-5.6.1-capybara`) and optionally
one frozen vanilla baseline (`physx-5.6.1`).

### PTX nvcc vs PTX Capybara (strict PTX A/B)

```bash
# A: nvcc PTX
PX_PTX_REPLACE_LIST="utility" PX_PTX_SOURCE=nvcc ./scripts/update_toolchain.sh

# B: capybara PTX
python3 scripts/compile_capybara_ptx.py -v
PX_PTX_REPLACE_LIST="utility" PX_PTX_SOURCE=capybara ./scripts/update_toolchain.sh
```

## Build snippets

`update_toolchain.sh` builds SDK/SAPIEN with snippets disabled by design, so use
a dedicated snippets configure/build:

```bash
cmake -S physx-5.6.1-capybara/compiler/public \
  -B physx-5.6.1-capybara/compiler/linux-clang-profile-snippets \
  -DPX_BUILDSNIPPETS=ON \
  -DPX_BUILDPVDRUNTIME=FALSE \
  -DPX_GENERATE_GPU_PROJECTS=ON \
  -DPX_PTX_REPLACE_LIST="utility" \
  -DPX_PTX_ARCH=compute_86 \
  -DELYTAR_BUILD_HEADLESS_SNIPPET_BENCHES=ON
cmake --build physx-5.6.1-capybara/compiler/linux-clang-profile-snippets -- -j"$(nproc)"
```

With `ELYTAR_BUILD_HEADLESS_SNIPPET_BENCHES=ON`, an extra
`SnippetIsosurfaceHeadless` target is built on Linux.

## Run repeated A/B benchmark

Use `run_ptx_ab.sh`:

```bash
BENCH_CMD_A="/abs/path/SnippetIsosurfaceHeadless" \
BENCH_CMD_B="/abs/path/SnippetIsosurfaceHeadless" \
REPS=10 \
STEPS_PER_RUN=100 \
./benchmark/physx_snippets/run_ptx_ab.sh
```

Outputs CSV by default to:
`benchmark/physx_snippets/results/ptx_ab_results.csv`

Columns:
`run_id,variant,rep,elapsed_s,throughput_steps_per_s,steps_per_run,command`

## Notes

- This benchmark is **end-to-end wall-clock**; it measures total snippet runtime,
  not isolated single-kernel time.
- Keep `PX_PTX_ARCH`, GPU, clock settings, and PhysX config consistent between A/B.
- For kernel-only profiling, add Nsight/CUDA instrumentation in a later phase.

