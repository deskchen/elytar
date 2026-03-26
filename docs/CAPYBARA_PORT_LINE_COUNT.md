# Capybara Port: Per-Kernel Line Count Analysis

Capybara DSL kernels are consistently longer than their CUDA counterparts. This document breaks down each ported kernel with its function, line counts, and the reasons for the difference.

## Per-kernel comparison

### utility.cu (gpucommon) — Deformable skinning and vertex utilities

| Kernel | Function | CUDA | Capybara | Ratio |
|---|---|---|---|---|
| `interleaveBuffers` | Interleaves separate vertex and normal float4 arrays into a packed PxVec3 result buffer for rendering. | 11 | 27 | 2.5x |
| `zeroNormals` | Zeroes out the guide normal array before accumulation (grid-stride write of (0,0,0) per vertex). | 22 | 23 | 1.0x |
| `normalVectorsAreaWeighted` | Computes area-weighted face normals from guide triangles and atomically accumulates them into per-vertex normal buffers. | 13 | 63 | 4.8x |
| `normalizeNormals` | Normalizes each accumulated normal vector to unit length. | 13 | 27 | 2.1x |
| `interpolateSkinnedClothVertices` | Evaluates cloth skinning: interpolates skinned vertex positions from barycentric coordinates on guide triangles, offset along the surface normal. | 45 | 116 | 2.6x |
| `interpolateSkinnedSoftBodyVertices` | Evaluates soft-body (volume) skinning: interpolates skinned vertex positions from tetrahedral barycentric coordinates inside guide tetrahedra. | 25 | 69 | 2.8x |

### MemCopyBalanced.cu (gpucommon) — Memory utilities

| Kernel | Function | CUDA | Capybara | Ratio |
|---|---|---|---|---|
| `clampMaxValue` | Clamps a single device-side PxU32 counter to a maximum (launched as 1-thread kernel). | 5 | 15 | 3.0x |
| `clampMaxValues` | Clamps three device-side PxU32 counters to a maximum. | 9 | 13 | 1.4x |
| `MemCopyBalanced` | Warp-cooperative balanced memory copy across descriptors. | 33 | — | Deferred |

### integration.cu (gpusolver) — Rigid body integration

| Kernel | Function | CUDA (kernel + inlined device fns) | Capybara | Ratio |
|---|---|---|---|---|
| `integrateCoreParallelLaunch` | Per-body velocity integration and pose update: applies solver velocity deltas, transforms angular velocity through inertia tensor, integrates position (Euler) and rotation (closed-form quaternion), runs the full sleep/freeze state machine (energy threshold checks, stabilization damping, wake counter management, freeze/unfreeze flag transitions). | 73 + 337 = 410 | 537 | 1.3x |

The 73-line CUDA kernel calls three `__device__` functions from `integration.cuh`: `integrateCore` (88 lines), `sleepCheck` (10 lines), and `updateWakeCounter` (239 lines). The effective CUDA line count is 410.

## Why Capybara kernels are longer

### 1. Scalar expansion of vector/matrix operations

CUDA uses operator-overloaded types (`PxVec3`, `PxQuat`, `PxMat33`). Capybara operates on scalar components.

```cpp
// CUDA: 1 line
PxVec3 linearVelocity = initialLinVel + solverBodyLinVel;
```
```python
# Capybara: 3 lines
body_lin_x = init_lin_x + lin_vel_x
body_lin_y = init_lin_y + lin_vel_y
body_lin_z = init_lin_z + lin_vel_z
```

Impact: **3x expansion per PxVec3 op, 4x per PxQuat op, 9x per PxMat33 multiply.** This is the dominant factor for `normalVectorsAreaWeighted` (4.8x) which has 9 atomic_add calls on 3-component normals, and for `interpolateSkinnedClothVertices` (2.6x) which does barycentric interpolation on vec3/vec4 types.

### 2. Manual struct field access via flat tensor offsets

CUDA accesses struct fields by name through typed pointers. Capybara passes struct arrays as `float32[N, stride]` tensors and reads fields by column index.

```cpp
// CUDA: 1 line
PxU32 nodeIndex = data.islandNodeIndex.index();
```
```python
# Capybara: 2 lines (load + bitcast for int field in float tensor)
node_id_f = solver_body_data[a, SBD_NODE_ID]
node_id = thread.bitcast(node_id_f, cp.int32)
```

This also requires offset constant definitions at the top of the file (~30 lines per struct type). The integration kernel defines offsets for 5 struct types.

### 3. Explicit load-forcing for MLIR type safety

Capybara compiles to MLIR structured control flow. A tensor subscript `tensor[i, j]` produces a `cp.ref` (memory reference), not a loaded value. If a variable is later reassigned in a conditional branch, the `cp.if` yield types conflict (`cp.ref` vs `f32`). The workaround is `+ cp.float32(0.0)` on every tensor read that may be conditionally reassigned.

```python
# Capybara: load-forcing pattern (not needed in CUDA)
inv_mass = solver_body_data[a, SBD_INV_MASS] + cp.float32(0.0)
```

Impact: ~40 lines in `integration.py` are pure load-forcing with no CUDA equivalent.

### 4. Variable pre-definition for structured control flow

Variables assigned in both `if`/`else` branches must be defined before the `if` in Capybara.

```python
# Required by Capybara — 14 variables for sleep state machine
sla_x = cp.float32(0.0)
sla_y = cp.float32(0.0)
...
```

Impact: ~14 lines in `integration.py`.

### 5. Device function inlining

CUDA compiles `.cu` and `.cuh` separately and links at PTX level. Capybara has no separate device-function compilation — all called logic must be in the same `.py` file.

Impact: `integration.py` inlines 337 lines of `.cuh` code that CUDA keeps in a header.

### 6. Kernel boilerplate

Each Capybara kernel requires a `cp.Kernel` scope and `block.threads()` loop that has no CUDA equivalent (CUDA uses implicit `threadIdx.x` / `blockIdx.x`).

```python
# Capybara: 3 lines of boilerplate per kernel
with cp.Kernel(grid_x, threads=BLOCK_SIZE) as (bx, block):
    for tid, thread in block.threads():
        idx = bx * BLOCK_SIZE + tid
```

Impact: 3 lines per kernel (minor for large kernels, significant for trivial ones like `clampMaxValue` where it explains the 3.0x ratio).

## Summary

| Cause | Typical expansion | Dominant in |
|---|---|---|
| Scalar expansion of vector ops | 3-9x per operation | `normalVectorsAreaWeighted`, `interpolateSkinnedClothVertices` |
| Manual struct field access | +2 lines per field read | `integration` |
| Load-forcing `+ cp.float32(0.0)` | +1 line per conditional tensor read | `integration` |
| Variable pre-definition | +N lines per branching scope | `integration` (sleep state machine) |
| Device function inlining | 1:1 (not expansion, but moves lines into kernel file) | `integration` |
| Kernel boilerplate | +3 lines per kernel | `clampMaxValue`, `clampMaxValues` |
