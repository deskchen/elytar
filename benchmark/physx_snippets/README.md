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
| `algorithms.cu` | gpusimulationcontroller | `reorderKernel`, `scanPerBlockKernel`, `scanPerBlockKernel4x4`, `addBlockSumsKernel`, `addBlockSumsKernel4x4`, `radixFourBitCountPerBlockKernel`, `radixFourBitReorderKernel` | Ported (7) | Scan/sort/reorder. NULL ptrs replaced by int flags. int4x4 as int32[N,16] flat tensors |
| `sparseGridStandalone.cu` | gpusimulationcontroller | `sg_SparseGridCalcSubgridHashes`, `sg_SparseGridMarkRequiredNeighbors`, `sg_SparseGridSortedArrayToDelta`, `sg_SparseGridGetUniqueValues`, `sg_SparseGridClearDensity`, `sg_SparseGridBuildSubgridNeighbors`, `sg_MarkSubgridEndIndices`, `sg_ReuseSubgrids`, `sg_AddReleasedSubgridsToUnusedStack`, `sg_AllocateNewSubgrids` | Ported (10) | PxSparseGridParams decomposed to 5 scalars. NULL ptr flags. searchSorted while loop. Local buffer[8] as 8 scalars. Atomics for stack ops |
| `diffuseParticles.cu` | gpusimulationcontroller | `ps_diffuseParticleCopy`, `ps_diffuseParticleSum`, `ps_updateUnsortedDiffuseArrayLaunch`, `ps_diffuseParticleOneWayCollision`, `ps_diffuseParticleCreate`, `ps_diffuseParticleUpdatePBF`, `ps_diffuseParticleCompact` | Ported (7) | PxgParticleSystem decomposed to per-kernel flat args. blockCopy eliminated. Warp ballot/popc/shfl for compaction. goto → done-flag |
| `anisotropy.cu` | gpusimulationcontroller | `smoothPositionsLaunch`, `calculateAnisotropyLaunch`, `anisotropyKernel`, `smoothPositionsKernel` | Ported (4) | PxMat33 eigen decomposition via Jacobi rotation (manually inlined — tuple-return @cp.inline in loops fails). PxGpuParticleSystem decomposed. Subgrid neighbor iteration. |
| `pairManagement.cu` | gpunarrowphase | `removeContactManagers_Stage1`–`Stage5`, `Stage5_CvxTri`, `initializeManifolds` | Ported (7) | SparseRemove scan inlined (same as algorithms.py). Struct copy via flat tensors. binarySearch for swap indices. |
| `radixSortImpl.cu` | gpucommon | `radixSortCopyHigh32Bits`, `radixSortDoubleCopyHigh32Bits`, `radixSortCopy`, `radixSortDoubleCopy`, `radixSortCopyBits2`, `radixSortCopy2`, `radixSortMultiBlockLaunch` (+2 variants), `radixSortMultiCalculateRanksLaunch` (+2 variants) | Ported (12) | RadixSort.cuh templates inlined. PxgRadixSortBlockDesc decomposed. Copy kernels trivial. Alias kernels require full body duplication (no kernel-to-kernel calls). |
| `updateBodiesAndShapes.cu` | gpusimulationcontroller | `updateBodiesLaunch`, `updateBodiesLaunchDirectAPI`, `updateShapesLaunch`, `newArticulationsLaunch`, `updateArticulationsLaunch`, `updateBodyExternalVelocitiesLaunch`, `updateJointsLaunch`, 5 getters, 4 setters, `copyUserData`, `getD6JointForces`, `getD6JointTorques` | Ported (20) | Massive descriptor decomposition. warpCopy → while loops. Quaternion math manually inlined (MLIR vectorizer issue). 2000+ lines. |
| `integrationTGS.cu` | gpusolver | `integrateCoreParallelLaunchTGS` | Ported (1) | TGS variant of integration. Uses deltaBody2World from TxIData. Reuses BS_*/SBD_*/TXI_* constants. |
| `compressOutputContacts.cu` | gpunarrowphase | `compressContactStage1`, `compressContactStage2`, `updateFrictionPatches` | Ported (3) | Multi-iteration ballot+scan reduction. Pointer arithmetic decomposed to index arithmetic. GPU pointer fields as int64 lo/hi pairs. |
| `updateTransformAndBoundArray.cu` | gpusimulationcontroller | `mergeTransformCacheAndBoundArrayChanges`, `updateTransformCacheAndBoundArrayLaunch`, `updateChangedAABBMgrHandlesLaunch`, `mergeChangedAABBMgrHandlesLaunch`, `computeFrozenAndUnfrozenHistogramLaunch`, `outputFrozenAndUnfrozenHistogram`, `createFrozenAndUnfrozenArray` | Ported (7) | Geometry-specific bounds (sphere/capsule/box/convex). Ballot compaction. Frozen/unfrozen histogram scan. binarySearch partitioning. |

| `cudaSphere.cu` | gpunarrowphase | `sphereNphase_Kernel` | Ported (1) | Pure geometric collision math (sphere-sphere/plane/capsule/box, plane-capsule, capsule-capsule). All 6 collision functions + contact output helpers inlined as scalar math. |
| `convexMeshOutput.cu` | gpunarrowphase | `convexTrimeshFinishContacts` | Ported (1) | Warp-level contact output. Material combining. Warp-cooperative reads replaced by per-thread reads. |
| `accumulateThresholdStream.cu` | gpusolver | 14 kernels (bodyInputAndRanks*, initialRanks*, reorganize*, compute/output/writeout/set/create*) | Ported (14) | Radix sort bodies (duplicated from radixSortImpl). Multi-iteration warp scan for force accumulation. Binary search for threshold mask. |

| `cudaBox.cu` | gpunarrowphase | `boxBoxNphase_Kernel` | Ported (1) | SAT box-box collision. 2300+ lines Capybara. No `elif` (MergeFlatIfToSwitchPass crash). 4-thread cooperative ops → per-thread. |
| `preIntegration.cu` | gpusolver | `preIntegrationLaunch`, `initStaticKinematics` | Ported (2) | Swizzled smem eliminated → direct flat tensor reads. Velocity integration + inertia tensor + gyroscopic forces. |
| `rigidDeltaAccum.cu` | gpusimulationcontroller | `accumulateDeltaVRigidFirstLaunch`, `accumulateDeltaVRigidSecondLaunch`, `clearDeltaVRigidSecondLaunchMulti`, Stage1, Stage2 | Ported (5) | Conditional warp reduction. Shuffles moved outside divergent ifs. 2D atomic_add flattened to 1D. |

**Total: 111 kernels ported across 19 `.cu` files.**

### Capybara PTX compilation

```bash
conda run -n triton-dev python scripts/compile_capybara_ptx.py -v
# Expected: Compiled 19 module(s), 111 kernel entry block(s).
```

Output files:
- `source/gpucommon/src/PTX/utility.capybara.ptx`
- `source/gpucommon/src/PTX/MemCopyBalanced.capybara.ptx`
- `source/gpucommon/src/PTX/radixSortImpl.capybara.ptx`
- `source/gpusolver/src/PTX/integration.capybara.ptx`
- `source/gpunarrowphase/src/PTX/pairManagement.capybara.ptx`
- `source/gpusimulationcontroller/src/PTX/algorithms.capybara.ptx`
- `source/gpusimulationcontroller/src/PTX/sparseGridStandalone.capybara.ptx`
- `source/gpusimulationcontroller/src/PTX/anisotropy.capybara.ptx`
- `source/gpusimulationcontroller/src/PTX/updateBodiesAndShapes.capybara.ptx`
- `source/gpusimulationcontroller/src/PTX/diffuseParticles.capybara.ptx`
- `source/gpusolver/src/PTX/integrationTGS.capybara.ptx`
- `source/gpunarrowphase/src/PTX/compressOutputContacts.capybara.ptx`
- `source/gpusimulationcontroller/src/PTX/updateTransformAndBoundArray.capybara.ptx`

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
PX_PTX_REPLACE_LIST="utility;MemCopyBalanced;integration;algorithms" \
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
