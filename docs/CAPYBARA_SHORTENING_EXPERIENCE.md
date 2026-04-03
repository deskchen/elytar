# Capybara Port: Shortening Experience

Lessons learned from reviewing the first three ported files (`utility.py`, `MemCopyBalanced.py`, `integration.py`) against their CUDA counterparts. The goal: Capybara kernel code should be no longer than the CUDA kernel code. Shared data-structure definitions (in a separate module) are not counted.

---

## Current state (after first round of shortening)

| File | Before | After | CUDA kernel lines | Ratio |
|------|--------|-------|-------------------|-------|
| utility.py | 449 | 236 | ~196 | 1.2x |
| MemCopyBalanced.py | 42 | 41 | ~14 | 2.9x (tiny; boilerplate-dominated) |
| integration.py | 668 | 657 | ~550 (cu + cuh) | 1.2x |
| physx_math.py (new shared) | — | 54 | N/A (in C++ headers) | — |

The dominant expansion factor was **manual scalar decomposition** of PxVec3/PxQuat/PxMat33 operations. CUDA gets these for free via header-defined operator overloading. The PxVec3 struct + inline methods approach (Technique 1) provided the biggest win in utility.py.

---

## Technique 1: @cp.struct with @cp.inline methods

**Impact: HIGH** — saves ~100 lines in utility.py, ~42 lines in integration.py.

Capybara supports `@cp.struct` types with `@cp.inline` methods (see `DSL_GRAMMAR_PHYSX_PORT.md` section 7). This is the single biggest shortening opportunity.

### Before (manual scalar decomposition)

```python
# Cross product: 3 lines
nx = e10y * e20z - e10z * e20y
ny = e10z * e20x - e10x * e20z
nz = e10x * e20y - e10y * e20x

# Dot product: 1 long line or 3 lines
scale1 = (qx - ax) * nAx + (qy - ay) * nAy + (qz - az) * nAz

# Matrix-vector multiply: 3 lines
body_ang_x = m00 * ang_vel_x + m01 * ang_vel_y + m02 * ang_vel_z
body_ang_y = m10 * ang_vel_x + m11 * ang_vel_y + m12 * ang_vel_z
body_ang_z = m20 * ang_vel_x + m21 * ang_vel_y + m22 * ang_vel_z

# Quaternion multiply: ~14 lines (cross, scalar products, combine)
cross_x = qv_y * b2w_qz - qv_z * b2w_qy
cross_y = qv_z * b2w_qx - qv_x * b2w_qz
cross_z = qv_x * b2w_qy - qv_y * b2w_qx
res_x = b2w_qw * qv_x + cross_x
# ... 10 more lines
```

### After (struct methods)

```python
n = e10.cross(e20)                              # 1 line
scale1 = q.sub(a).dot(nA)                       # 1 line
body_ang = sqrt_inv_inertia.multiply_vec(ang_vel)  # 1 line
result = quat_vel.qmul(b2w_q)                   # 1 line
```

### Recommended struct definitions

Create a shared module (e.g., `physx_math.py`) with:

```python
@cp.struct
class PxVec3:
    x: cp.float32
    y: cp.float32
    z: cp.float32

    @cp.inline
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    @cp.inline
    def cross(self, other):
        return PxVec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    @cp.inline
    def add(self, other):
        return PxVec3(self.x + other.x, self.y + other.y, self.z + other.z)

    @cp.inline
    def sub(self, other):
        return PxVec3(self.x - other.x, self.y - other.y, self.z - other.z)

    @cp.inline
    def scale(self, s):
        return PxVec3(self.x * s, self.y * s, self.z * s)

    @cp.inline
    def magnitude_sq(self):
        return self.dot(self)

    @cp.inline
    def normalize_safe(self, thread):
        mag2 = self.magnitude_sq()
        eps = cp.float32(1.0e-20)
        tiny = cp.float32(1.0e-30)
        mask = mag2 / (mag2 + eps)
        inv = thread.rsqrt(mag2 + tiny)
        mag = thread.sqrt(mag2 + tiny) * mask
        return PxVec3(self.x * mask * inv, self.y * mask * inv, self.z * mask * inv), mag
```

Similar definitions for PxQuat (rotate, rotateInv, qmul, normalize) and PxMat33 (multiply_vec).

### Per-kernel savings estimate (utility.py)

| Kernel | Current | After | Saved |
|--------|---------|-------|-------|
| `_evaluate_point_phong` | 48 | ~23 | 25 |
| `_normalize_safe` | 20 | 0 (moved to struct) | 20 |
| `normalVectorsAreaWeighted` | 46 | ~28 | 18 |
| `normalizeNormals` | 14 | ~8 | 6 |
| `interpolateSkinnedClothVertices` | 87 | ~52 | 35 |
| `interpolateSkinnedSoftBodyVertices` | 48 | ~28 | 20 |

### Per-kernel savings estimate (integration.py)

| Section | Saved | What |
|---------|-------|------|
| Matrix-vector multiply (2x) | 4 | `sqrtInvInertia.multiply_vec(ang_vel)` |
| Quaternion rotateInv | 8 | `b2w_q.rotate_inv(thread, am)` |
| Quaternion integration | 30 | `quat_vel.qmul(b2w_q)` + normalize |

---

## Technique 2: First-class struct tiles instead of flat offset constants

**Impact: MEDIUM** — saves ~49 lines in integration.py.

Currently integration.py defines ~84 lines of offset constants and struct layout comments:

```python
SBD_INIT_ANG_X, SBD_INIT_ANG_Y, SBD_INIT_ANG_Z, SBD_PEN_BIAS = 0, 1, 2, 3
SBD_INIT_LIN_X, SBD_INIT_LIN_Y, SBD_INIT_LIN_Z, SBD_INV_MASS = 4, 5, 6, 7
# ... 80 more lines of offsets
```

With `@cp.struct` types matching the C++ layout, these become self-documenting field definitions (~50 lines total for 5 struct types). Kernel access changes from:

```python
node_id_f = solver_body_data[a, SBD_NODE_ID]
node_id = thread.bitcast(node_id_f, cp.int32)
```

to:

```python
data = solver_body_data[a]
node_id = data.islandNodeIndex
```

This also eliminates ~8 `thread.bitcast` calls for int-in-float fields.

**Caveat**: Verify that Capybara struct tiles handle mixed int32/float32 fields correctly when the underlying tensor is float32. The current bitcast workaround exists for a reason — test before migrating.

---

## Technique 3: Common math module

**Impact: ORGANIZATIONAL** — no line savings in kernels, but prevents duplication across files.

Extract struct definitions (PxVec3, PxQuat, PxMat33) and shared helpers (_tanh_approx) into a single module importable by all kernel files:

```python
# physx_math.py — shared across all Capybara kernel files
from physx_math import PxVec3, PxQuat, PxMat33
```

This is analogous to CUDA including `foundation/PxVec3.h`. The ~90 lines of struct definitions are a one-time cost, not counted against individual kernel budgets.

---

## Technique 4: thread.load() instead of + cp.float32(0.0)

**Impact: READABILITY** — no line count change, but clearer intent.

~40 lines in integration.py use a load-forcing workaround:

```python
inv_mass = solver_body_data[a, SBD_INV_MASS] + cp.float32(0.0)
```

The proper API is `thread.load(ref)`:

```python
inv_mass = thread.load(solver_body_data[a, SBD_INV_MASS])
```

Same line count, but communicates intent (force a load from memory reference to value) instead of disguising it as arithmetic. Note: if struct tiles (Technique 2) are adopted, field access on loaded struct values are already values — this workaround disappears entirely.

---

## Unavoidable expansion (cannot be shortened)

### Variable pre-definition for structured control flow

Capybara compiles to MLIR structured control flow. Variables assigned in both `if`/`else` branches must be defined before the `if`. CUDA avoids this via SSA phi nodes.

```python
# Required by Capybara — ~14 lines in integration.py sleep state machine
sla_x = cp.float32(0.0)
sla_y = cp.float32(0.0)
sla_z = cp.float32(0.0)
# ...
```

**Cost**: ~14 lines per branching scope with many outputs. Cannot be eliminated without compiler changes.

### Kernel boilerplate

Each kernel needs:

```python
with cp.Kernel(grid, threads=BLOCK_SIZE) as (bx, block):
    for tid, thread in block.threads():
        idx = bx * BLOCK_SIZE + tid
```

This is ~3 lines per kernel, roughly matching CUDA's `threadIdx.x + blockIdx.x * blockDim.x`. Negligible for large kernels, noticeable for tiny ones like `clampMaxValue` (explains 3.0x ratio).

### No operator overloading on structs

`v1.add(v2)` instead of `v1 + v2`. Multi-operand expressions like `uvw.x * a + uvw.y * b + uvw.z * c` must be written with explicit method calls:

```python
# CUDA: 1 line
PxVec3 q = uvw.x * a + uvw.y * b + uvw.z * c;

# Capybara: still multiple calls (no operator overloading)
q = a.scale(uvw.x).add(b.scale(uvw.y)).add(c.scale(uvw.z))
```

Still shorter than 3 lines of manual scalar math, but not as terse as CUDA. This is the floor.

---

## Actual results after first round

| File | Before | After | CUDA | Ratio |
|------|--------|-------|------|-------|
| utility.py | 449 | 236 | ~196 | 1.2x |
| MemCopyBalanced.py | 42 | 41 | ~14 | 2.9x |
| integration.py | 668 | 657 | ~550 | 1.2x |
| physx_math.py (new) | — | 54 | N/A | — |

utility.py achieved 1.2x ratio (better than projected 1.6x) thanks to PxVec3 struct methods
compressing vertex loads, cross products, dot products, and Phong interpolation. integration.py
gains were modest (lock-flag helpers + quaternion helpers) because most expansion comes from
flat tensor offsets and load-forcing patterns that require struct kernel parameters to eliminate.

---

## algorithms.cu — all 7 kernels

All kernels ported from `gpusimulationcontroller/CUDA/algorithms.cu` (527 lines Capybara vs 438 lines CUDA including headers).

| Kernel | CUDA body lines | Capybara body lines | Ratio | Notes |
|--------|----------------|---------------------|-------|-------|
| `reorderKernel` | 5 | 9 | 1.8x | No float4 type → 4-component copy |
| `scanPerBlockKernel` | ~90 (kernel + 3 device fns) | 88 | 0.98x | Template boilerplate eliminated |
| `addBlockSumsKernel` | 13 | 10 | 0.77x | No template wrapper needed |
| `scanPerBlockKernel4x4` | ~44 (+ shared scan ~90) | ~160 | 1.2x | 4 int4 iterations × 3 phases; no int4 type |
| `addBlockSumsKernel4x4` | 25 | ~45 | 1.8x | 16-component decomposition + exclusiveSumInt16 phases |
| `radixFourBitCountPerBlockKernel` | ~55 (+ shared scan ~90) | ~120 | 0.83x | u64 scan inlined; packed bit ops same length |
| `radixFourBitReorderKernel` | 20 | 15 | 0.75x | Simpler without template wrapper |

### Why reorderKernel is longer (1.8x)

CUDA `float4` is a native type — `reordered[id] = data[map[id]]` copies 16 bytes in one statement. Capybara has no float4; the gather must copy 4 components explicitly (+4 lines). Kernel boilerplate (`with cp.Kernel` / `for tid, thread`) adds 3 lines vs CUDA's 1-line `threadIdx.x + blockIdx.x * blockDim.x`.

### Why scanPerBlockKernel achieves near-parity (0.98x)

CUDA pays a heavy tax for the multi-level scan architecture:
- **4 separate functions**: `warpScan<T>` (12 lines) + `scanPerBlock<T>` (72 lines) + `scanPerBlockKernelShared<T>` (28 lines) + kernel wrapper (10 lines)
- **Template boilerplate**: `template<typename T>`, `extern __shared__`, `reinterpret_cast<T*>(sumsMemory)`
- **Separate compilation units**: functions defined independently with their own signatures

Capybara inlines everything into one flat kernel with 3 barrier-separated thread phases. The barrier-phase pattern is actually cleaner than CUDA's "one thread body with `__syncthreads()` inside" because each phase's data flow is explicit via shared memory reads/writes.

**New patterns discovered:**
- `block.barrier()` must be between `block.threads()` regions, not inside — forces multi-phase structure
- `cp.disjoint(value)` needed for shared memory writes inside `block.threads()` to prove race-freedom
- `cp.assume_uniform(cond)` needed for shuffle operations inside conditionals that are warp-uniform but not provably so to the compiler
- Ternary `a if cond else b` avoids creating `cp.if` (which causes warp-convergence failures before `shfl_up`)

### Why addBlockSumsKernel is shorter (0.77x)

The CUDA version uses a template wrapper `addBlockSumsKernelShared<PxU32>` that adds function signature + bounds check overhead. Capybara inlines it directly — no wrapper needed.

### New Capybara limitations discovered

See `docs/CAPYBARA_STRUCT_LIMITATIONS.md` for full details. Additional findings from algorithms.cu:

5. **`block.barrier()` illegal inside `block.threads()`** — must structure kernel as multiple thread regions separated by barriers, using shared memory to carry values between phases
6. **`shfl_up` rejected under divergent conditions** — even if the value is produced by a conditional load (via `cp.if`), the warp convergence verifier flags it. Fix: use ternary (`arith.select`) instead of `if` for the conditional load
7. **No sub-warp shuffle width** — `thread.shfl_up(val, delta)` always uses full warp (width=32). CUDA's `__shfl_up_sync(mask, val, delta, width)` supports sub-warp widths. This means BLOCK_SIZE must be 1024 so NUM_WARPS=32 (fills one full warp for the second-level scan)

---

## Summary table (all files)

| File | Capybara lines | CUDA lines | Ratio | Key factor |
|------|---------------|------------|-------|------------|
| utility.py | 249 | ~234 | 1.1x | PxVec3 struct methods compress vector math |
| MemCopyBalanced.py | 41 | ~106 | 0.4x | Only 2 tiny kernels ported (rest deferred) |
| integration.py | 668 | ~550 | 1.2x | Flat tensor offsets + load-forcing expand |
| algorithms.py | 527 | ~438 | 1.2x | int4x4 decomposition + multi-phase barriers expand |
| physx_math.py (shared) | 54 | N/A | — | Shared struct definitions |

Key takeaway: Capybara is **shorter** than CUDA when templates and device function call graphs dominate (scanPerBlockKernel 0.98x, radix kernels 0.75-0.83x). It is **longer** when CUDA uses vector types (int4, float4, int4x4) that have no Capybara equivalent, requiring per-component decomposition.

---

## sparseGridStandalone.cu — all 10 kernels

All kernels ported from `gpusimulationcontroller/CUDA/sparseGridStandalone.cu` (620 lines Capybara vs 373 lines .cu + 318 lines .cuh = ~691 lines CUDA total).

| Kernel | CUDA body lines | Capybara body lines | Ratio | Notes |
|--------|----------------|---------------------|-------|-------|
| `sg_SparseGridClearDensity` | 5 | 6 | 1.2x | Trivial fill kernel |
| `sg_MarkSubgridEndIndices` | 10 | 11 | 1.1x | Trivial run-end detection |
| `sg_SparseGridSortedArrayToDelta` | 10 | 14 | 1.4x | NULL mask flag → explicit if branches |
| `sg_SparseGridCalcSubgridHashes` | 15 | 20 | 1.3x | PxVec3/PxVec4 → component access + NULL flag |
| `sg_SparseGridGetUniqueValues` | 20 | 25 | 1.25x | int4 → tuple returns; 3x3x3 loop via range() |
| `sg_SparseGridBuildSubgridNeighbors` | 25 | 28 | 1.12x | 3x3x3 loop + tryFindHashkey |
| `sg_ReuseSubgrids` | 22 | 22 | 1.0x | Clean mapping with tryFindHashkey |
| `sg_AddReleasedSubgridsToUnusedStack` | 6 | 8 | 1.33x | Atomic + host-resolved scalar grid |
| `sg_AllocateNewSubgrids` | 22 | 20 | 0.91x | No reinterpret_cast needed |
| `sg_SparseGridMarkRequiredNeighbors` | 55 | 105 | 1.9x | Local buffer[8] → 8 scalars + unrolled branches |

**Overall ratio**: 620 / 691 = **0.90x** (Capybara is shorter overall because all .cuh helper functions are inlined as @cp.inline, eliminating separate compilation units).

### Why sg_SparseGridMarkRequiredNeighbors is longer (1.9x)

The CUDA kernel uses `PxU32 buffer[8]` as a local array, incrementally appending values with `buffer[indexer++]`. Capybara has no local arrays — must use 8 scalar variables with extensive conditional assignment logic. The final `for (j = 0; j < indexer; ++j) applyMask(...)` loop also requires unrolled conditional calls. This is an inherent DSL limitation; local arrays would eliminate ~50 lines.

### Why overall ratio is favorable (0.90x)

The CUDA version splits code across .cu (kernels) and .cuh (helper functions with separate signatures). Capybara inlines all helpers as `@cp.inline` in one file, eliminating function signatures, include guards, and type declarations. The `searchSorted` binary search translated cleanly as a `while` loop.

### ABI changes

- `PxSparseGridParams` struct → 5 scalar args (maxNumSubgrids, gridSpacing, subgridSizeX/Y/Z); haloSize hardcoded to 0
- NULL pointer args (phases, activeIndices, mask) → int flag + dummy tensor
- `PxU32*` device pointers to single values → `int32[1]` tensors (or host-resolved scalars for grid dimensions)

---

## diffuseParticles.cu — all 7 kernels

All kernels ported from `gpusimulationcontroller/CUDA/diffuseParticles.cu` (548 lines Capybara vs 552 lines CUDA).

| Kernel | CUDA body lines | Capybara body lines | Ratio | Notes |
|--------|----------------|---------------------|-------|-------|
| `ps_diffuseParticleCopy` | 14 | 12 | 0.86x | No blockCopy overhead |
| `ps_diffuseParticleSum` | 18 | 20 | 1.11x | Inlined warp reduction (shfl_xor) |
| `ps_updateUnsortedDiffuseArrayLaunch` | 30 | 22 | 0.73x | Host pre-computes bufferOffset; no warpReduction needed |
| `ps_diffuseParticleOneWayCollision` | 35 | 32 | 0.91x | Flat contact tensor + unrolled loop |
| `ps_diffuseParticleCreate` | 80 | 75 | 0.94x | Flat args eliminate blockCopy + pointer indirection |
| `ps_diffuseParticleUpdatePBF` | 80 | 95 | 1.19x | goto → done-flag pattern adds overhead in nested loops |
| `ps_diffuseParticleCompact` | 75 | 90 | 1.20x | Warp-scope ballot/popc/shfl; explicit lane management |

**Overall ratio**: 548 / 552 = **0.99x** (near-parity).

### Why overall ratio is near-parity (0.99x)

The major win is **eliminating blockCopy**. The CUDA version copies the entire PxgParticleSystem struct (hundreds of fields) into shared memory via `blockCopy<uint2>`, then accesses fields through pointer chains. Capybara receives only the needed fields as flat tensor/scalar args, eliminating:
- `blockCopy<uint2>` call + `__syncthreads()` (5 lines per kernel × 5 kernels = 25 lines saved)
- `reinterpret_cast<float4*>()` pointer aliasing (2-4 lines per kernel)
- `PxgParticleSystem` shared memory declaration (2 lines per kernel)

### New patterns

- **goto → done-flag**: Replaced `goto weight_sum` with `done = cp.int32(0)` flag checked in loop conditions. Adds ~5 lines but preserves control flow correctness.
- **Warp-scope ballot/popc**: `thread.coll.ballot()` requires warp scope (`for lane, thread in warp.threads()`), not block scope.
- **shfl_xor for reduction**: Must be manually inlined (cannot pass `thread` to `@cp.inline` functions).
- **Host-resolved PxgParticleSystem**: Each kernel receives only its accessed fields. The host adapter must resolve all pointer-of-pointer indirection before launch.

---

## Updated summary table (all files)

| File | Capybara lines | CUDA lines | Ratio | Key factor |
|------|---------------|------------|-------|------------|
| utility.py | 249 | ~234 | 1.1x | PxVec3 struct methods compress vector math |
| MemCopyBalanced.py | 41 | ~106 | 0.4x | Only 2 tiny kernels ported (rest deferred) |
| integration.py | 668 | ~550 | 1.2x | Flat tensor offsets + load-forcing expand |
| algorithms.py | 527 | ~438 | 1.2x | int4x4 decomposition + multi-phase barriers expand |
| sparseGridStandalone.py | 620 | ~691 | 0.90x | .cuh inlined as @cp.inline; no separate compilation |
| diffuseParticles.py | 548 | ~552 | 0.99x | blockCopy elimination offsets DSL overhead |
| physx_math.py (shared) | 54 | N/A | — | Shared struct definitions |

**Total: 33 kernels ported across 6 `.cu` files.**
