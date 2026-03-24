# PhysX Port Practice — Lessons from utility.cu

This document captures concrete issues discovered during the first Capybara DSL port (`utility.cu`) and establishes rules for future kernel ports.

---

## 1. PTX Replacement ABI Contract

The `ElytarPtxReplace.cmake` system works by:

1. Reading `.entry` names from the Capybara-generated PTX
2. Building a fatbin from that PTX
3. Registering each kernel name via `PxGpuCudaRegisterFunction`
4. The PhysX runtime calls `cuLaunchKernel` with arguments packed in the **original CUDA order and layout**

**Rule**: The Capybara-generated PTX must have the **exact same parameter signature** as the CUDA-compiled PTX. This means:

- Same number of parameters
- Same parameter types (`.u32`, `.u64`, `.f32`, etc.)
- Same parameter order
- Same data layout assumptions (stride, alignment)

If even one parameter is mismatched, the kernel will read garbage from parameter space, which typically manifests as:

- **Infinite GPU loops** (garbage loop bounds or zero stride in while loops)
- **Silent memory corruption** (writes to wrong addresses)
- **GPU timeout / hang** (driver kills the kernel after watchdog timeout)

### How to verify ABI parity

Before committing any `.capybara.ptx`, diff the `.entry` signatures:

```bash
# Extract param signatures from both PTX files
grep -A 20 '\.entry.*kernelName' utility.ptx > /tmp/cuda_sig.txt
grep -A 20 '\.entry.*kernelName' utility.capybara.ptx > /tmp/capybara_sig.txt
diff /tmp/cuda_sig.txt /tmp/capybara_sig.txt
```

Check:
- Parameter count matches
- `.u32` vs `.u64` vs `.f32` types match
- Parameter order matches

---

## 2. Argument Order Must Match CUDA `extern "C"` Signature

**Bug found**: `interleaveBuffers` had arguments `(vertices, normals, interleaved_result, length)` in Capybara but `(vertices, normals, length, interleavedResultBuffer)` in CUDA.

The CUDA signature is:

```cpp
extern "C" __global__ void interleaveBuffers(
    const float4* PX_RESTRICT vertices,
    const float4* PX_RESTRICT normals,
    PxU32 length,
    PxVec3* interleavedResultBuffer)
```

PTX parameters are laid out sequentially. Swapping a `u32` and a `u64` means the runtime's `length` value (e.g., 1024) is read as a pointer, and the result pointer is truncated to a 32-bit "length". This causes writes to garbage addresses and potentially enormous loop trip counts.

**Rule**: Always declare Capybara kernel arguments in the exact order they appear in the CUDA `extern "C"` signature. Cross-check against the `.cu` file before generating PTX.

---

## 3. Data Layout Stride: `float4*` vs `float32[N,3]`

**Bug found**: CUDA `interleaveBuffers` reads `float4*` (16-byte stride per element, `.w` ignored), but the Capybara port used `float32[N,3]` (12-byte stride). After element 0, every access is misaligned.

CUDA PTX:

```ptx
mul.wide.s32 %rd5, %r1, 16;   // stride = sizeof(float4) = 16
ld.global.nc.v4.f32 {%f1, %f2, %f3, %f4}, [%rd6];   // vector load
```

Capybara PTX:

```ptx
mul.lo.s32 %r6, %r1, 12;      // stride = 3 * sizeof(float32) = 12 — WRONG
```

**Fix**: Declare inputs as `float32[N, 4]` in the JSON descriptor to match the 16-byte stride:

```json
{ "kind": "tensor", "shape": [1024, 4], "dtype": "float32", "device": "cuda" }
```

The kernel code reads only columns 0–2 and ignores column 3 (the `.w`).

**Rule**: When the CUDA kernel takes `float4*`, `uint4*`, or any packed vector pointer, the Capybara tensor must use the **full vector width** (4 elements) even if not all components are used. The stride is what matters for address computation.

---

## 4. Single-Struct-Pointer ABI (Cannot Port Without Adapter)

**Root cause of GPU hang**: Five of six utility kernels take a single `PxTrimeshSkinningGpuData*` or `PxTetmeshSkinningGpuData*` argument in CUDA:

```cpp
extern "C" __global__ void normalVectorsAreaWeighted(
    PxTrimeshSkinningGpuData* data)
```

This generates PTX with **1 parameter** (a `.u64` pointer):

```ptx
.visible .entry normalVectorsAreaWeighted(
    .param .u64 normalVectorsAreaWeighted_param_0
)
```

The kernel then dereferences fields from the struct (`data[blockIdx.y].guideVerticesD`, etc.) — these fields themselves contain raw pointers (`PxTypedBoundedData` wraps a `void*` with a count).

The Capybara port decomposes this into **14 separate parameters** (9 struct field tensors + 4 data tensors + 1 grid_x + 1 num_batches):

```ptx
.visible .entry normalVectorsAreaWeighted(
    .param .u64 normalVectorsAreaWeighted_param_0,
    .param .u64 normalVectorsAreaWeighted_param_1,
    ... (12 more params) ...
    .param .u32 normalVectorsAreaWeighted_param_13
)
```

When the runtime passes 1 argument, params 1–13 read uninitialized parameter memory. Since `grid_x` and loop bounds come from these garbage params, the `while idx < loop_end: ... idx += x_dim` loop runs with `x_dim = 0` or `loop_end = 2^31`, creating an **infinite GPU loop**.

### Why this cannot be fixed in Capybara DSL alone

The CUDA `PxTrimeshSkinningGpuData` struct contains fields like:

```cpp
PxTypedBoundedData<PxVec3> guideVerticesD;  // wraps { void* data; PxU32 count; }
PxU32* guideTrianglesD;                      // raw pointer
```

Capybara's memory model requires all buffer bindings as explicit typed kernel arguments. There is no mechanism to:

1. Accept a single struct pointer
2. Dereference pointer fields from that struct at runtime
3. Use those runtime pointers for typed load/store

This is **Blocker 1** (raw-address ABI) from the port blockers audit.

### Options for resolution

| Option | Description | Effort | Drop-in? |
|--------|-------------|--------|----------|
| **A. Host-side adapter** | Write C++ shim that unpacks `PxTrimeshSkinningGpuData*` into separate args, calls Capybara kernel with decomposed args | Medium | No — requires modifying the PhysX kernel dispatcher |
| **B. Index-offset ABI** | Change PhysX host code to pass flat arrays + offset struct separately (no raw pointers in struct) | High | No — upstream PhysX change |
| **C. `cp.intr` raw-address bridge** | Extend Capybara with `inttoptr`-style intrinsic to dereference runtime addresses | High | Yes — but requires compiler work |
| **D. Defer these kernels** | Only replace `interleaveBuffers` via PTX replacement; leave skinning kernels as CUDA | Low | Partial |

**Current recommendation**: Option D for near-term. Only `interleaveBuffers` has a compatible ABI. The five skinning kernels should remain as CUDA-compiled PTX until one of options A–C is implemented.

---

## 5. Semantic Bug: `normalizeSafe` Must Return Magnitude

**Bug found**: CUDA's `PxVec3::normalizeSafe()` normalizes in-place AND returns the original magnitude. The Capybara `_normalize_safe` only returned the unit vector, discarding the magnitude.

This caused `_evaluate_point_phong` to compute `tanh(alpha / h)` instead of `tanh(|dir| * alpha / h)` — the displacement magnitude was lost.

CUDA:

```cpp
PxVec3 dir = qStar - q;
PxReal offset = dir.normalizeSafe() * alpha;
// normalizeSafe: dir is now unit vector, returns original |dir|
// offset = |dir| * alpha
PxReal ratio = offset / halfSurfaceThickness;
ratio = tanhf(ratio);
```

**Fix**: `_normalize_safe` now returns a 4-tuple `(nx, ny, nz, magnitude)`. All call sites updated to either use or discard the magnitude.

**Rule**: When porting PhysX math methods that modify state AND return a value (like `normalizeSafe`, `normalize`, `magnitudeSquared`), verify that both the side effect and the return value are preserved. Decomposing C++ methods into scalar DSL code makes it easy to lose one of the two.

---

## 6. Checklist for Future Kernel Ports

Before generating PTX for a new `.cu` file:

- [ ] **Argument order**: matches `extern "C"` CUDA signature exactly
- [ ] **Argument types**: `float4*` → `[N, 4]`, `PxVec3*` → `[N, 3]`, `PxU32` → `int`, etc.
- [ ] **Struct pointer ABI**: if CUDA takes `SomeStruct*`, verify Capybara can match (usually cannot — defer or use adapter)
- [ ] **Grid dimensions**: `gridDim.x/y/z` usage matches `cp.Kernel(...)` dimensions
- [ ] **Loop stride**: `gridDim.x * blockDim.x` must not be computed from decomposed args if CUDA uses hardware registers
- [ ] **Method return values**: PhysX methods that modify-and-return (e.g., `normalizeSafe`) must have both behaviors preserved
- [ ] **PTX signature diff**: run parameter signature diff before committing
- [ ] **Data stride**: verify byte-level stride matches (16 for `float4`, 12 for `PxVec3`, 8 for `float2`, etc.)

---

## 7. Current Status of utility.cu Port

| Kernel | ABI Compatible | PTX Replacement Ready | Notes |
|--------|:-:|:-:|-------|
| `interleaveBuffers` | Yes | Yes | Argument order and float4 stride fixed |
| `normalVectorsAreaWeighted` | No | No | Requires raw-pointer struct ABI (Blocker 1) |
| `zeroNormals` | No | No | Same as above |
| `normalizeNormals` | No | No | Same as above |
| `interpolateSkinnedClothVertices` | No | No | Same as above |
| `interpolateSkinnedSoftBodyVertices` | No | No | Same as above |

**Action**: Only `interleaveBuffers` should be included in `PX_PTX_REPLACE_LIST` for utility.cu. The remaining five kernels must stay as CUDA-compiled PTX until a host-side adapter or `cp.intr` raw-address bridge is available.
