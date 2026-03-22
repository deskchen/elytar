# PhysX Snippet Benchmarks (Two-tree A/B)

This suite compares PhysX snippet binaries built from two source trees:

- **A (baseline):** `physx-5.6.1` with pure `.cu` kernels (`PX_PTX_REPLACE_LIST=""`)
- **B (capybara):** `physx-5.6.1-capybara` with selected kernels from
  `*.capybara.ptx`

There is no special headless snippet target in this workflow. Benchmark standard
`Snippet*` executables (for example `SnippetIsosurface`) from each tree.

## Build both variants with `update_toolchain.sh`

`update_toolchain.sh` now supports snippets via `ELYTAR_BUILD_PHYSX_SNIPPETS=1`,
always enables GPU project generation, and consumes PTX suffixes directly
(no copying `.capybara.ptx -> .ptx`).

### A: baseline tree (`physx-5.6.1`, pure `.cu`)

```bash
PHYSX_DIR="/workspace/physx-5.6.1" \
PX_PTX_REPLACE_LIST="" \
ELYTAR_BUILD_PHYSX_SNIPPETS=1 \
./scripts/update_toolchain.sh
```

### B: capybara tree (`physx-5.6.1-capybara`, PTX replacement)

```bash
python3 scripts/compile_capybara_ptx.py -v

PHYSX_DIR="/workspace/physx-5.6.1-capybara" \
PX_PTX_REPLACE_LIST="utility" \
PX_PTX_SOURCE=capybara \
ELYTAR_BUILD_PHYSX_SNIPPETS=1 \
./scripts/update_toolchain.sh
```

Notes:
- `PX_PTX_SOURCE=capybara` selects `*.capybara.ptx`.
- Override suffix directly with `ELYTAR_PTX_INPUT_SUFFIX` if needed.
- Pick a replace list appropriate for your experiment (`utility`, custom list, or `all`).

## Run repeated A/B benchmark

Use `run_ptx_ab.sh` with executable paths from each tree:

```bash
BENCH_CMD_A="/workspace/physx-5.6.1/bin/linux.x86_64/profile/SnippetIsosurface_64" \
BENCH_CMD_B="/workspace/physx-5.6.1-capybara/bin/linux.x86_64/profile/SnippetIsosurface_64" \
REPS=10 \
STEPS_PER_RUN=100 \
./benchmark/physx_snippets/run_ptx_ab.sh
```

Outputs CSV by default to:
`benchmark/physx_snippets/results/ptx_ab_results.csv`

Columns:
`run_id,variant,rep,elapsed_s,throughput_steps_per_s,steps_per_run,command`

## Notes

- This benchmark is **end-to-end wall-clock**; it measures total process runtime
  of each snippet command.
- Keep GPU, clocks, PhysX config, and run conditions consistent between A/B.
- You can benchmark any snippet binary by changing `BENCH_CMD_A/B`.

