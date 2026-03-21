# PhysX DSL Candidate Kernels (Ranked)

This document lists PhysX kernels in `physx-5.6.1-capybara` that are likely to benefit from Capybara DSL by reducing lines of code and CUDA boilerplate.

Selection principles (aligned to `docs/DSL_GRAMMAR_PHYSX_PORT.md`):

- Favor kernels dominated by structured block/warp/thread decomposition
- Favor scan/reduction/compaction/sort/cooperative-load patterns
- Favor regular map/gather/grid-stride/stencil loops
- Exclude kernels dominated by hard blockers:
  - raw-address ABI (`u64 -> typed ptr deref`)
  - packed byte-pointer stream parsing
  - inline PTX requirements
  - texture fetches
  - deep irregular collision/helper stacks

---

## Tier 1 - Highest ROI (clear DSL LOC reduction)

These kernels are mostly "structured parallel primitives + index math". DSL should compress control/scaffolding the most.

| Rank | Kernel | File | Module | Why DSL should reduce LOC | Portability confidence |
|---|---|---|---|---|---|
| 1 | `reorderKernel` | `physx-5.6.1-capybara/source/gpusimulationcontroller/src/CUDA/algorithms.cu` | gpusimulationcontroller | Pure gather/reorder (`out[i] = in[map[i]]`), minimal branching | High |
| 2 | `scanPerBlockKernel` | same | gpusimulationcontroller | Block-scan boilerplate maps to block/collective APIs | High |
| 3 | `addBlockSumsKernel` | same | gpusimulationcontroller | Standard post-scan block-sum fixup; repetitive CUDA indexing disappears | High |
| 4 | `radixSortCopy` family (`radixSortCopy`, `radixSortCopy2`, `radixSortDoubleCopy`, `radixSortCopyHigh32Bits`, `radixSortDoubleCopyHigh32Bits`, `radixSortCopyBits2`) | `physx-5.6.1-capybara/source/gpucommon/src/CUDA/radixSortImpl.cu` | gpucommon | Repetitive grid-stride copy/scatter loops; good fit for structured loops | Medium-High |
| 5 | `interleaveBuffers` | `physx-5.6.1-capybara/source/gpucommon/src/CUDA/utility.cu` | gpucommon | Straightforward stream interleave pattern with simple loads/stores | High |

---

## Tier 2 - Good cooperative primitive candidates

These kernels are still structured, but include moderate compaction/reduction details.

| Rank | Kernel | File | Module | Why DSL should reduce LOC | Caveat |
|---|---|---|---|---|---|
| 6 | `sg_SparseGridSortedArrayToDelta` | `physx-5.6.1-capybara/source/gpusimulationcontroller/src/CUDA/sparseGridStandalone.cu` | gpusimulationcontroller | Boundary-marking on sorted keys; compaction-like shape | Validate edge-mask behavior |
| 7 | `sg_SparseGridGetUniqueValues` | same | gpusimulationcontroller | Unique-on-sorted pipeline with regular neighbor loops | Careful with sentinel semantics |
| 8 | `sg_MarkSubgridEndIndices` | same | gpusimulationcontroller | Run-end detection maps cleanly to structured per-thread logic | Low caveat |
| 9 | `compressContactStage1` | `physx-5.6.1-capybara/source/gpunarrowphase/src/CUDA/compressOutputContacts.cu` | gpunarrowphase | Ballot/reduction style stage; can leverage collectives | Keep Stage 1 only (Stage 2 is pointer-heavy) |
| 10 | `ps_diffuseParticleSum` | `physx-5.6.1-capybara/source/gpusimulationcontroller/src/CUDA/diffuseParticles.cu` | gpusimulationcontroller | Small warp reduction kernel; huge boilerplate cut | Confirm reduction order tolerance |

---

## Tier 3 - Regular map/stencil kernels

These are good DSL targets where logic is mostly per-element math over typed arrays.

| Rank | Kernel | File | Module | Why DSL should reduce LOC | Caveat |
|---|---|---|---|---|---|
| 11 | `bvhCalculateMortonCodes` | `physx-5.6.1-capybara/source/gpusimulationcontroller/src/CUDA/SDFConstruction.cu` | gpusimulationcontroller | Per-item map with straightforward arithmetic + writes | None major |
| 12 | `bvhCalculateKeyDeltas` | same | gpusimulationcontroller | Adjacent-key delta logic; simple guards and integer ops | None major |
| 13 | `bvhComputeTriangleBounds` | same | gpusimulationcontroller | Regular per-triangle AABB compute | None major |
| 14 | `bvhBuildLeaves` | same | gpusimulationcontroller | Structured leaf construction with deterministic indexing | Struct field mapping validation |
| 15 | `bvhComputeTotalBounds` | same | gpusimulationcontroller | Reduction-style aggregation with explicit block cooperation | Validate reduction semantics |
| 16 | `initializeManifolds` | `physx-5.6.1-capybara/source/gpunarrowphase/src/CUDA/pairManagement.cu` | gpunarrowphase | Shared staging + regular copy patterns | Keep scope narrow; avoid adjacent heavy paths |
| 17 | `iso_CountCellVerts` | `physx-5.6.1-capybara/source/gpusimulationcontroller/src/CUDA/isosurfaceExtraction.cu` | gpusimulationcontroller | Structured grid-cell counting, regular per-cell branching | Confirm constant-table strategy |
| 18 | `iso_GridFilterGauss` | same | gpusimulationcontroller | Fixed-neighbor stencil smoothing pattern | Validate memory-access ordering/perf |

---

## Tier 4 - Optional quick wins (tiny kernels)

These are low-risk, small-scope wins if you want fast visible progress.

| Kernel | File | Why useful |
|---|---|---|
| `clampMaxValue` | `physx-5.6.1-capybara/source/gpucommon/src/CUDA/MemCopyBalanced.cu` | Tiny scalar clamp utility, easy to verify |
| `clampMaxValues` | same | Same as above for 3 values |

Note: this does **not** imply `MemCopyBalanced` itself is a good candidate; only the tiny clamp kernels are.

---

## Poor candidates / Not recommended now

These kernels are currently dominated by hard blockers, so DSL work is unlikely to reduce code cleanly without major semantic/ABI decisions.

| Kernel | File | Primary blocker reason(s) |
|---|---|---|
| `MemCopyBalanced` | `physx-5.6.1-capybara/source/gpucommon/src/CUDA/MemCopyBalanced.cu` | Raw-address ABI (`desc.source/dest` pointer deref) |
| `convexConvexNphase_stage1Kernel` | `physx-5.6.1-capybara/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu` | Packed streams, pointer aliasing, deep helper graph |
| `convexConvexNphase_stage2Kernel` | same | Same as above, plus high irregularity |
| `midphaseGeneratePairs` | `physx-5.6.1-capybara/source/gpunarrowphase/src/CUDA/convexMeshMidphase.cu` | Pointer-heavy stacks + inline PTX barrier ecosystem |
| `triangleTriangleCollision` | same | Shared-memory reinterpret overlays + heavy helper stack |
| `performIncrementalSAP` | `physx-5.6.1-capybara/source/gpubroadphase/src/CUDA/broadphase.cu` | Descriptor pointer streams + reinterpret-heavy paths |
| `artiUpdateKinematic` | `physx-5.6.1-capybara/source/gpuarticulation/src/CUDA/articulationDirectGpuApi.cu` | Raw pointer fields in descriptors |
| `setupInternalConstraintLaunch1T` | `physx-5.6.1-capybara/source/gpuarticulation/src/CUDA/internalConstraints2.cu` | Template-heavy articulation internals |
| `computeUnconstrainedVelocities1TLaunch` | `physx-5.6.1-capybara/source/gpuarticulation/src/CUDA/forwardDynamic2.cu` | Template + deep math helper parity burden |
| `iso_CreateVerts` | `physx-5.6.1-capybara/source/gpusimulationcontroller/src/CUDA/isosurfaceExtraction.cu` | `__constant__` tables + atomic/PTX-adjacent paths |

---

## Recommended execution order

If the goal is "prove DSL reduces code quickly," start in this order:

1. `reorderKernel`
2. `scanPerBlockKernel`
3. `addBlockSumsKernel`
4. `interleaveBuffers`
5. `sg_SparseGridSortedArrayToDelta`
6. `bvhCalculateMortonCodes`
7. `bvhComputeTriangleBounds`

This gives a balanced sequence of map/scan/stencil styles with low blocker exposure and high readability gains.

