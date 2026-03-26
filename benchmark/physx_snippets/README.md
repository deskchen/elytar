# PhysX Snippet Benchmarks (Two-tree A/B)

This suite compares PhysX snippet binaries built from two source trees:

- **A (vanilla):** `physx-5.6.1` with pure `.cu` kernels (`PX_PTX_REPLACE_LIST=""`)
- **B (capybara):** `physx-5.6.1-capybara` with selected kernels from `*.capybara.ptx`

Use **headless** snippet binaries for benchmarking: they run a fixed step count and exit, so you get valid latency and throughput. Interactive `Snippet*` executables open a GLUT window and hang.

## Ported kernels

| Source `.cu` | Module | Kernels | Status | Notes |
|---|---|---|---|---|
| `utility.cu` | gpucommon | `interleaveBuffers`, `zeroNormals`, `normalVectorsAreaWeighted`, `normalizeNormals`, `interpolateSkinnedClothVertices`, `interpolateSkinnedSoftBodyVertices` | Ported (6) | Skinning kernels use host adapter (`ELYTAR_CAPYBARA_SKINNING`) |
| `MemCopyBalanced.cu` | gpucommon | `clampMaxValue`, `clampMaxValues` | Ported (2) | `MemCopyBalanced` kernel deferred (shared mem + 2D warp copy) |
| `integration.cu` | gpusolver | `integrateCoreParallelLaunch` | Ported (1) | Rigid body integration + sleep/freeze. Host adapter (`ELYTAR_CAPYBARA_INTEGRATION`) unpacks `PxgSolverCoreDesc` into 20 flat args |

**Total: 9 kernels ported across 3 `.cu` files.**

### Capybara PTX compilation

```bash
conda run -n triton-dev python scripts/compile_capybara_ptx.py -v
# Expected: Compiled 3 module(s), 9 kernel entry block(s).
```

Output files:
- `source/gpucommon/src/PTX/utility.capybara.ptx`
- `source/gpucommon/src/PTX/MemCopyBalanced.capybara.ptx`
- `source/gpusolver/src/PTX/integration.capybara.ptx`

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
conda run -n triton-dev python3 scripts/compile_capybara_ptx.py -v

ELYTAR_PHYSX_ONLY=1 \
PHYSX_DIR="/workspace/physx-5.6.1-capybara" \
PX_PTX_REPLACE_LIST="utility;MemCopyBalanced;integration" \
PX_PTX_SOURCE=capybara \
ELYTAR_BUILD_PHYSX_SNIPPETS=1 \
./scripts/update_toolchain.sh
```

Notes:
- `PX_PTX_SOURCE=capybara` selects `*.capybara.ptx`.
- Override suffix with `ELYTAR_PTX_INPUT_SUFFIX` if needed.
- Pick a replace list for your experiment (individual stems, custom list, or `all`).
- `integration` automatically enables `ELYTAR_CAPYBARA_INTEGRATION` (flat-arg host adapter).
- `utility` automatically enables `ELYTAR_CAPYBARA_SKINNING` (skinning host adapter).

## Run benchmark

Paths are fixed: `physx-5.6.1` (vanilla) and `physx-5.6.1-capybara` (capybara) under workspace root.

```bash
python3 -m benchmark.physx_snippets.run --snippet Isosurface --reps 10
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--snippet` | (required) | Snippet name, e.g. Isosurface |
| `--reps` | 10 | Repetitions per variant |
| `--output-dir` | `benchmark/physx_snippets/results` | Output directory for CSV files |
| `--run-id` | timestamp | Run identifier for CSV |
| `--verbose` | False | Print subprocess output |
| `--timeout` | None | Per-run timeout (seconds) |
| `--steps` | snippet-specific | Steps per run for throughput |
| `--label-a` | vanilla_cu | Label for variant A |
| `--label-b` | capybara_ptx | Label for variant B |
| `--delay-between-variants` | 1.0 | Seconds to sleep between A and B (helps GPU release) |

### Output

Each run produces two CSV files per snippet:
- `{snippet}_current.csv` — latest run (overwritten)
- `{snippet}_history.csv` — appended history across runs

Summary statistics (min/max/mean) are printed after all reps complete.

### Headless snippets

Any snippet in the allowlist gets `SnippetFooHeadless_64`:
Isosurface, SDF, PBF, PBDCloth, SplitSim, RBDirectGPUAPI, SplitFetchResults, Triggers.

## Notes

- This benchmark is **end-to-end wall-clock**; it measures total process runtime.
- Keep GPU, clocks, PhysX config, and run conditions consistent between A/B.
- Run from the repository root so workspace paths resolve correctly.
