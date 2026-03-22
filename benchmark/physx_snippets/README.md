# PhysX Snippet Benchmarks (Two-tree A/B)

This suite compares PhysX snippet binaries built from two source trees:

- **A (vanilla):** `physx-5.6.1` with pure `.cu` kernels (`PX_PTX_REPLACE_LIST=""`)
- **B (capybara):** `physx-5.6.1-capybara` with selected kernels from `*.capybara.ptx`

Use **headless** snippet binaries for benchmarking: they run a fixed step count and exit, so you get valid latency and throughput. Interactive `Snippet*` executables open a GLUT window and hang.

## Build both variants with `update_toolchain.sh`

Build PhysX only (`ELYTAR_PHYSX_ONLY=1`): compiles and verifies PhysX libs + headless snippets, skips the SAPIEN wheel. Requires `ELYTAR_BUILD_PHYSX_SNIPPETS=1` for headless snippet binaries.

### A: baseline tree (`physx-5.6.1`, pure `.cu`)

```bash
ELYTAR_PHYSX_ONLY=1 \
PHYSX_DIR="/workspace/physx-5.6.1" \
PX_PTX_REPLACE_LIST="" \
ELYTAR_BUILD_PHYSX_SNIPPETS=1 \
./scripts/update_toolchain.sh
```

### B: capybara tree (`physx-5.6.1-capybara`, PTX replacement)

```bash
python3 scripts/compile_capybara_ptx.py -v

ELYTAR_PHYSX_ONLY=1 \
PHYSX_DIR="/workspace/physx-5.6.1-capybara" \
PX_PTX_REPLACE_LIST="utility" \
PX_PTX_SOURCE=capybara \
ELYTAR_BUILD_PHYSX_SNIPPETS=1 \
./scripts/update_toolchain.sh
```

Notes:
- `PX_PTX_SOURCE=capybara` selects `*.capybara.ptx`.
- Override suffix with `ELYTAR_PTX_INPUT_SUFFIX` if needed.
- Pick a replace list for your experiment (`utility`, custom list, or `all`).

## Run benchmark

Paths are fixed: `physx-5.6.1` (vanilla) and `physx-5.6.1-capybara` (capybara) under workspace root.

```bash
python3 benchmark/physx_snippets/run.py --snippet Isosurface --reps 10
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--snippet` | (required) | Snippet name, e.g. Isosurface |
| `--reps` | 10 | Repetitions per variant |
| `--output` | `benchmark/physx_snippets/results/{snippet}.csv` | Output CSV path |
| `--run-id` | timestamp | Run identifier for CSV |
| `--verbose` | False | Print subprocess output |
| `--timeout` | None | Per-run timeout (seconds) |
| `--steps` | snippet-specific | Steps per run for throughput |
| `--label-a` | vanilla_cu | Label for variant A |
| `--label-b` | capybara_ptx | Label for variant B |
| `--delay-between-variants` | 1.0 | Seconds to sleep between A and B (helps GPU release) |

### Headless snippets

Any snippet in the allowlist gets `SnippetFooHeadless_64`:
Isosurface, SDF, PBF, PBDCloth, SplitSim, RBDirectGPUAPI.

Output CSV columns: `run_id,variant,rep,elapsed_s,throughput_steps_per_s,steps_per_run,command`

## Notes

- This benchmark is **end-to-end wall-clock**; it measures total process runtime.
- Keep GPU, clocks, PhysX config, and run conditions consistent between A/B.
- Run from the repository root so workspace paths resolve correctly.
