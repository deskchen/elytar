# PhysX Kernel Port Blockers Audit

This document catalogs every CUDA pattern found across PhysX 5.6.1 GPU source that cannot be directly ported to Capybara DSL in an apple-to-apple way. It was produced by scanning all `.cu` and `.cuh` files under `physx-5.6.1/source/` (55 `.cu` files, 62+ `.cuh` files, hundreds of `__global__` kernels across six GPU modules).

Reference documents:

- Capybara DSL grammar guide: [`docs/DSL_GRAMMAR_PHYSX_PORT.md`](DSL_GRAMMAR_PHYSX_PORT.md)
- Existing blocker note: [`physx-5.6.1/source/gpucommon/src/capybara/MemCopyBalanced_PORT_BLOCKER.md`](../physx-5.6.1/source/gpucommon/src/capybara/MemCopyBalanced_PORT_BLOCKER.md)

---

## Scope of Source Scanned

| Module | Path |
|--------|------|
| gpucommon | `physx-5.6.1/source/gpucommon/src/CUDA/` |
| gpubroadphase | `physx-5.6.1/source/gpubroadphase/src/CUDA/` |
| gpunarrowphase | `physx-5.6.1/source/gpunarrowphase/src/CUDA/` |
| gpuarticulation | `physx-5.6.1/source/gpuarticulation/src/CUDA/` |
| gpusolver | `physx-5.6.1/source/gpusolver/src/CUDA/` |
| gpusimulationcontroller | `physx-5.6.1/source/gpusimulationcontroller/src/CUDA/` |

Existing Capybara ports (both in `gpucommon/src/capybara/`):

- `utility.py` -- successfully ported with SoA layout compromise
- `MemCopyBalanced.py` -- ported with ABI difference; see `MemCopyBalanced_PORT_BLOCKER.md`

---

## Severity Key

| Label | Meaning |
|-------|---------|
| **CRITICAL** | No Capybara equivalent exists; requires language extension or project-approved workaround before any apple-to-apple port |
| **HIGH** | Capybara has partial support but the gap is real and affects many kernels; requires explicit decision per kernel |
| **MEDIUM** | Workaround exists or impact is narrow; must be documented and decided per kernel, not silently patched |

---

## Blocker 1 — Raw Pointer Dereference from u64 Address Fields

**Severity: CRITICAL**

### What CUDA does

PhysX passes structs whose fields hold raw memory addresses (`size_t` / `uintptr_t`) and then casts those addresses to typed pointers to load/store data at runtime:

```cpp
// MemCopyBalanced.cu:61–62
PxU32* srcPtr = reinterpret_cast<PxU32*>(copyDesc[warpIdxInBlock].source);
PxU32* dstPtr = reinterpret_cast<PxU32*>(copyDesc[warpIdxInBlock].dest);
PxU32 sourceVal = srcPtr[a];
dstPtr[a] = sourceVal;

// articulationDirectGpuApi.cu:232
PxReal* linkAccelData = reinterpret_cast<PxReal*>(articulation.motionAccelerations);
```

### Why it cannot be ported directly

Capybara's memory model expresses all loads and stores through typed kernel arguments (tensors/views). There is no mechanism to take an integer field from a struct, reinterpret it as a pointer, and dereference it. All buffer bindings must be known at kernel entry.

### Gap summary

| CUDA operation | Capybara equivalent | Status |
|----------------|---------------------|--------|
| `(T*)addr_field` + `ptr[i]` | — | No equivalent |
| Struct field carries raw buffer address | All buffers as explicit kernel args | ABI mismatch |

### Affected files

`gpucommon/src/CUDA/MemCopyBalanced.cu`, `gpuarticulation/src/CUDA/articulationDirectGpuApi.cu`, `gpuarticulation/src/CUDA/articulationImpulseResponse.cuh`, `gpubroadphase/src/CUDA/broadphase.cu`, `gpusimulationcontroller/src/CUDA/diffuseParticles.cu`, `gpusimulationcontroller/src/CUDA/softBodyGM.cu`

### What needs a decision

- Is exact CUDA kernel signature (raw-address ABI) required for PTX replacement, or is offset/tensor ABI acceptable?
- If raw-address ABI is required, what `cp.intr`-level path implements `u64 -> typed load/store`?

See `MemCopyBalanced_PORT_BLOCKER.md` for the detailed checklist.

---

## Blocker 2 — reinterpret_cast and Type Punning (Pointer-Level)

**Severity: CRITICAL**

### What CUDA does

Beyond address-to-pointer casts (Blocker 1), PhysX pervasively reinterprets pointers between incompatible aggregate types to reuse shared memory buffers, access SIMD lanes of packed data, and alias struct arrays:

```cpp
// articulationImpulseResponse.cuh:161–163
size_t ptr = reinterpret_cast<size_t>(&impulses[deltaIdx]);
float2* f2 = reinterpret_cast<float2*>(ptr);
float4* f4 = reinterpret_cast<float4*>(f2 + 1);

// gpucommon/src/CUDA/vector.cuh:46
const PxVec3& in = reinterpret_cast<const PxVec3&>(vec);

// diffuseParticles.cu:77–78
float4* unsortedPositions = reinterpret_cast<float4*>(particleSystem.mDiffusePosition_LifeTime);
const uint2* sParticleSystem = reinterpret_cast<const uint2*>(&particleSystem);
```

Scalar-level type punning with CUDA intrinsics is also pervasive (30+ uses):

```cpp
// atomic.cuh:109–116
while (val < __int_as_float(old))
    old = atomicCAS(..., __float_as_int(val));
return __int_as_float(old);
```

### Why it cannot be ported directly

Capybara does have `thread.bitcast()` for scalar reinterpretation between same-width types. However, it does not support:

- Reinterpreting a pointer to a struct as a pointer to a different aggregate (`float4*`, `uint2*`)
- Accessing a `PxVec3` reference through a `float4*` alias
- Shared-memory union aliasing across type boundaries

### Gap summary

| CUDA operation | Capybara equivalent | Status |
|----------------|---------------------|--------|
| `__float_as_int(x)` / `__int_as_float(x)` | `thread.bitcast(x, cp.int32)` / `thread.bitcast(x, cp.float32)` | Supported for same-width scalars |
| `reinterpret_cast<float4*>(struct_ptr)` | — | Not supported |
| `reinterpret_cast<T&>(val)` (aliasing) | — | Not supported |
| `__fdividef(a, b)` | `thread.div_approx(a, b)` | Supported (approx semantics) |

### Affected files

`gpucommon/src/CUDA/vector.cuh`, `gpucommon/src/CUDA/atomic.cuh`, `gpunarrowphase/src/CUDA/cudaGJKEPA.cu`, `gpunarrowphase/src/CUDA/articulationImpulseResponse.cuh`, `gpusimulationcontroller/src/CUDA/diffuseParticles.cu`, and dozens more (100+ total uses)

---

## Blocker 3 — C++ Templates in Device Functions

**Severity: CRITICAL**

### What CUDA does

PhysX uses C++ template parameterization throughout device code to:

- Select algorithm variants at compile time via `bool` template parameters:

```cpp
// internalConstraints2.cu:1553
template <typename IterativeData, const bool isTGS, const bool residualReportingEnabled>
static __device__ void artiSolveInternalConstraints1T(...)
```

- Select block/warp sizes as compile-time integer constants:

```cpp
// RadixSort.cuh:69
template <PxU32 WARP_PERBLOCK_SIZE>
static __device__ PxU32 scanRadixWarps(...)

// MemCopyBalanced.cu:40
template<PxU32 warpsPerBlock>
__device__ void copyBalanced(...)
```

- Select reduction ops through functor types:

```cpp
// reduction.cuh:187
template<typename OP, typename T, PxU32 log2threadGroupSize>
__device__ static inline T warpReduction(...)

// warpHelpers.cuh:100
template <typename T, class OP>
__device__ static inline void reduceWarp(volatile T* sdata)
```

- Compile-time size selection via type traits:

```cpp
// cudaGJKEPA.cu:768–777
template<int A, int B>
struct compileTimeSelectLargest {
    enum { value = compileTimeSelect<(A > B), A, B>::value };
};
struct EpaAndClipScratch {
    PxU8 mem[compileTimeSelectLargest<sizeof(squawk::EpaScratch), sizeof(TemporaryContacts)>::value];
};
```

### Why it cannot be ported directly

Capybara has `cp.constexpr` for compile-time integer parameters (block size, trip counts), but has no generic type parameterization. There is no equivalent to `typename T`, `class OP`, or `const bool isTGS` as template parameters. Each template instantiation must be written as a separate named function with all types resolved.

### Affected files

`gpucommon/src/CUDA/RadixSort.cuh`, `gpucommon/src/CUDA/reduction.cuh`, `gpunarrowphase/src/CUDA/cudaGJKEPA.cu`, `gpunarrowphase/src/CUDA/warpHelpers.cuh`, `gpunarrowphase/src/CUDA/contactReduction.cuh`, `gpunarrowphase/src/CUDA/softbodySoftbodyMidPhase.cu`, `gpuarticulation/src/CUDA/internalConstraints2.cu`, `gpuarticulation/src/CUDA/forwardDynamic2.cu`, `gpusimulationcontroller/src/CUDA/FEMClothUtil.cuh` (50+ template device functions total)

### What needs a decision

For each template device function, the port must decide whether to:

- Instantiate each variant explicitly as a separate `@cp.inline` (verbose but straightforward)
- Restrict to the single instantiation used by a target kernel (may miss future variants)
- Use Python-level metaprogramming to generate variants (non-standard Capybara pattern)

---

## Blocker 4 — PhysX Math Types as Device Structs with Methods

**Severity: HIGH**

### What CUDA does

`PxVec3`, `PxVec4`, `PxQuat`, `PxTransform`, `PxMat33`, and `PxMat34` are used throughout all six GPU modules. These are C++ classes with operator overloads and methods callable in device code:

```cpp
// FEMClothUtil.cuh:447
PxVec3 x0 = PxLoad3(xx0);
const PxVec3 axis0 = x01.getNormalized();
float d = axis0.dot(edge);

// contactConstraintBlockPrep.cuh:354
const PxVec3 bodyFrame0p(bodyFrame0.p.x, bodyFrame0.p.y, bodyFrame0.p.z);
const PxMat33 sqrtInvInertia(...);
PxVec3 result = sqrtInvInertia * raXn;

// cudaGJKEPA.cu:2319–2328
PxQuat q = PxQuat(shfl_x, shfl_y, shfl_z, shfl_w);
n = q.rotate(n);
```

### Why it cannot be ported directly

Capybara `@cp.struct` supports struct field access but does not support method calls on structs (`struct_instance.dot(...)`, `struct_instance.rotate(...)`, operator `*` between struct instances). All such operations must be decomposed into explicit scalar arithmetic expressed in the DSL.

Additionally, PhysX math types are AoS by design (e.g., a `PxVec3` is three consecutive floats), whereas Capybara memory access works through views; a direct field-by-field `@cp.struct` mirror is possible but method equivalents must all be manually written as `@cp.inline` helpers.

### Existing port precedent

`utility.py` successfully handled `PxVec3` operations by decomposing them into component-wise scalar operations. This is portable but requires significant manual decomposition for every use site.

### Affected files

All six GPU modules. Representative: `gpusolver/src/CUDA/contactConstraintBlockPrep.cuh`, `gpunarrowphase/src/CUDA/cudaGJKEPA.cu`, `gpusimulationcontroller/src/CUDA/FEMClothUtil.cuh`, `gpusimulationcontroller/src/CUDA/SDFConstruction.cu`, `gpucommon/src/CUDA/shuffle.cuh`

---

## Blocker 5 — Complex Pointer Arithmetic and `sizeof`/`alignof` in Device Code

**Severity: HIGH**

### What CUDA does

PhysX encodes packed binary streams (contact buffers, convex geometry data, manifold pools) using raw byte-pointer arithmetic with `sizeof` and alignment math:

```cpp
// convexTriangle.cuh:84–90
return (const float4*)(convexPtrA
    + sizeof(float4)          // skip header
    + sizeof(uint4)
    + sizeof(float4));

// triangleMesh.cuh:154
trimeshGeomPtr = (PxU8*)(((size_t)trimeshGeomPtr + 15) & ~15);  // 16-byte alignment

// copy.cuh:40–42
assert((size_t(dest) & (alignof(T)-1)) == 0);
assert(totalSize % sizeof(T) == 0);

// cudaGJKEPA.cu:2248
uint4* persistentContactManifoldMoreData =
    (uint4*)(((float4*)manifoldPtr) + 3 * PXG_MAX_PCM_CONTACTS);
```

### Why it cannot be ported directly

Capybara has no `sizeof` or `alignof` in the DSL. Memory layouts must be expressed as typed views with known element sizes baked in at kernel definition time. Arbitrary byte offsets derived at runtime from struct sizes cannot be computed.

Additionally, the packed stream encoding used in contact/convex data assumes a specific binary layout that must either be preserved exactly (requiring byte-level access) or re-encoded to match a view-compatible layout.

### Affected files

`gpunarrowphase/src/CUDA/convexTriangle.cuh`, `gpunarrowphase/src/CUDA/cudaGJKEPA.cu`, `gpunarrowphase/src/CUDA/midphaseAllocate.cuh`, `gpunarrowphase/src/CUDA/triangleMesh.cuh`, `gpusolver/src/CUDA/contactConstraintBlockPrep.cuh` (and `constraintBlockPrepTGS.cu`), `gpucommon/src/CUDA/copy.cuh`

### What needs a decision

- Can the binary stream layouts for contacts, convex data, and manifolds be redefined to a view-friendly layout, or must they match the existing C++ ABI exactly?
- If exact binary layout must be preserved, this is a fundamental blocker requiring a byte-buffer escape hatch.

---

## Blocker 6 — `__constant__` and Module-Level Device Variables

**Severity: HIGH**

### What CUDA does

Nine files declare lookup tables and configuration data in CUDA constant memory (`__constant__`) or global device memory (`__device__`). These are read-only tables initialized before kernel launch:

```cpp
// epa.cuh:54–58
__constant__ __device__ schlock::Vec3I8 sStartVertices4[4] = {...};
__constant__ __device__ schlock::Vec3I8 sStartAdjacencies4[4] = {...};
__constant__ __device__ schlock::Vec3I8 sStartVertices3[2] = {...};
__constant__ __device__ schlock::Vec3I8 sStartAdjacency3[2] = {...};

// gpusimulationcontroller/src/CUDA/marchingCubesTables.cuh
__constant__ int marchingCubeCorners[8][3];       // marching cubes tables
__constant__ int firstMarchingCubesId[257];
__constant__ int marchingCubesIds[2460];
__constant__ int marchingCubesEdgeLocations[12][4];

// gpusolver/src/CUDA/constant.cuh:35
__constant__ PxgSolverCoreDesc constraintSolverCoreDescC;

// softBodyGM.cu
__constant__ __device__ PxU32 tets6PerVoxel[24];
__constant__ __device__ PxU32 tets5PerVoxel[40];

// vector.cuh:514–521
__constant__ int ind[16];
__constant__ float sign[16];

// cudaGJKEPA.cu:1895
__constant__ __device__ PxReal patchConstants[];
__constant__ __device__ PxU32 ((FinishContactsWarpScratch::*patchOffsets[16])[32]) = {...};
```

### Why it cannot be ported directly

Capybara has no concept of `__constant__` memory space. There is no module-level storage declaration in the DSL. Options are:

- Pass lookup tables as additional `cp.constexpr` kernel parameters (only for small, known-size tables)
- Pass them as read-only tensor arguments (changes kernel signature)
- Embed them as Python-level constants and materialize per-invocation (for small tables)

The `constraintSolverCoreDescC` pattern (a struct descriptor set once and read by all blocks) has no direct analog.

### Affected files

`gpunarrowphase/src/CUDA/epa.cuh`, `gpunarrowphase/src/CUDA/cudaGJKEPA.cu`, `gpunarrowphase/src/CUDA/vector.cuh` (via `gpucommon`), `gpusimulationcontroller/src/CUDA/marchingCubesTables.cuh`, `gpusimulationcontroller/src/CUDA/isosurfaceExtraction.cu`, `gpusimulationcontroller/src/CUDA/softBodyGM.cu`, `gpusolver/src/CUDA/constant.cuh`

---

## Blocker 7 — Warp Intrinsics Without Capybara Equivalents

**Severity: HIGH**

### What CUDA does

PhysX uses the full CUDA warp intrinsic suite. Most are concentrated in narrowphase and articulation:

**`__shfl_sync` with arbitrary source lane:**
```cpp
// cudaGJKEPA.cu:181–182
int in0p = __shfl_sync(FULL_MASK, in0, tI+1 == nbVerts0 ? 0 : tI+1);
index = __shfl_sync(FULL_MASK, (int)index, bestProjLane);
```

**`__any_sync` / `__all_sync`:**
```cpp
// cudaGJKEPA.cu:155
if (!__any_sync(FULL_MASK, out1))

// softBodyGM.cu:3200
awake = __any_sync(FULL_MASK, awake);
```

**64-bit popcount (`__popcll`) and count-leading-zeros (`__clzll`):**
```cpp
// internalConstraints2.cu:851
PxU32 stackCount = __popcll(bitStack);

// internalConstraints2.cu:867
PxU32 child = 63 - __clzll(bitStack);
```

**Find-first-set (`__ffs`, `__ffsll`):**
```cpp
// cudaGJKEPA.cu:423
int index = __ffs(m);

// articulationImpulseResponse.cuh:49
return val == 0 ? 0 : __ffsll(val) - 1;
```

### Gap summary

| CUDA intrinsic | Capybara equivalent | Status |
|----------------|---------------------|--------|
| `__shfl_sync(mask, val, srcLane)` | `thread.shfl_up/down/xor` only | Arbitrary-lane `shfl` not supported |
| `__ballot_sync(mask, pred)` | `thread.coll.ballot(pred)` | Supported in warp scope |
| `__any_sync(mask, pred)` | `thread.coll.any(pred)` | Supported in warp scope only |
| `__all_sync(mask, pred)` | `thread.coll.all(pred)` | Supported in warp scope only |
| `__popc(mask)` | `thread.popcount(mask)` | Supported (32-bit) |
| `__popcll(mask64)` | — | No 64-bit variant |
| `__clz(x)` / `__clzll(x)` | — | Not supported |
| `__ffs(x)` / `__ffsll(x)` | — | Not supported |

### Affected files

`gpunarrowphase/src/CUDA/cudaGJKEPA.cu`, `gpunarrowphase/src/CUDA/reduction.cuh`, `gpunarrowphase/src/CUDA/warpHelpers.cuh`, `gpuarticulation/src/CUDA/internalConstraints2.cu`, `gpuarticulation/src/CUDA/articulationImpulseResponse.cuh`, `gpubroadphase/src/CUDA/broadphase.cu`, `gpubroadphase/src/CUDA/aggregate.cu`, `gpucommon/src/CUDA/RadixSort.cuh`

### What needs a decision

For missing intrinsics:

- `__shfl_sync(mask, val, srcLane)`: is `cp.intr`-level implementation acceptable, or must the algorithm be restructured?
- `__clzll` / `__popcll` / `__ffsll`: are 64-bit variants planned for Capybara, or should the algorithm be rewritten to use 32-bit operations?

---

## Blocker 8 — Inline PTX Assembly

**Severity: HIGH**

### What CUDA does

Four files use inline PTX for operations with no CUDA C++ equivalent:

**Named barriers (split barriers distinct from `__syncthreads`):**
```cpp
// convexMeshMidphase.cu:796, 851, 894, 915, 968
asm volatile ("bar.sync 1;");
asm volatile ("bar.sync 2;");
asm volatile ("bar.sync 3;");
```

**Warp scan with PTX shfl+predicated add (lower latency than intrinsic path):**
```cpp
// warpHelpers.cuh:212–220
asm(
    "{"
    "  .reg .u32 r0;"
    "  .reg .pred p;"
    "  shfl.up.b32 r0|p, %1, %2, %3;"
    "  @p add.u32 r0, r0, %4;"
    "  mov.u32 %0, r0;"
    "}"
    : "=r"(input) : "r"(input), "r"(1 << STEP), "r"(SHFL_MASK), "r"(input));
```

**Global reduction with `red.global.add.f32`:**
```cpp
// atomic.cuh:149
asm volatile ("red.global.add.f32 [%0], %1;" :: __STG_PTR(addr), "f"(val));
```

**Cached global load (`ld.global.cg`):**
```cpp
// solverMultiBlock.cu:107 (commented path)
asm("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" ...);
```

### Why it cannot be ported directly

Capybara has no inline assembly path at the high-level DSL layer. The DSL grammar guide explicitly states that `cp.intr` escape hatch requires user approval and cannot be used automatically.

### Affected files

`gpunarrowphase/src/CUDA/convexMeshMidphase.cu` (named barriers), `gpunarrowphase/src/CUDA/warpHelpers.cuh` (shfl scan), `gpucommon/src/CUDA/atomic.cuh` (global reduction), `gpusolver/src/CUDA/solverMultiBlock.cu` (cached load, commented)

### What needs a decision

For each use site:

- `bar.sync N` (named barriers): are separate `block.barrier()` calls per region sufficient, or is the numbered barrier semantics required?
- `shfl.up.b32` scan: can the `thread.coll.scan` collective replace the PTX path?
- `red.global.add.f32`: can `thread.atomic_add` replace the unordered global reduction?

---

## Blocker 9 — `volatile` Shared Memory Access Patterns

**Severity: MEDIUM**

### What CUDA does

PhysX uses `volatile` shared memory pointers as the classical warp-synchronous programming pattern to prevent compiler reordering without inserting explicit `__syncwarp()` barriers:

```cpp
// warpHelpers.cuh:101, 115, 135, 188
template <typename T>
__device__ static inline void reduceWarp(volatile T* sdata)

__device__ static inline void reduceHalfWarp(volatile T* sdata)
__device__ static inline void reduceWarpKeepIndex(volatile T* sdata, volatile TIndex* sindices, ...)
__device__ static inline void scanWarp(unsigned int scanIdx, volatile T* sdata)

// constraintBlockPrep.cu:64, 87
volatile __shared__ PxU8 bodyLoadData[...];
volatile PxU32* bodyData = reinterpret_cast<volatile PxU32*>(&bodyLoadData[...]);

// epa.cuh:265
(reinterpret_cast<volatile PxReal*>(S.vertices))[tI-tI/4] = input;
```

### Why it cannot be ported directly

Capybara `block.alloc()` allocates shared memory but has no `volatile` qualifier concept. The memory ordering guarantee that `volatile` provides (preventing dead-store elimination and ensuring visibility across warp lanes without a full barrier) has no DSL-level expression.

In practice, since Capybara compiles to PTX through MLIR and does not reorder stores across thread boundaries, many `volatile` uses may be safely dropped. However, this must be verified case-by-case. Any `volatile` that actually serves as a warp-synchronous coordination mechanism (not just a compiler hint) represents a real semantic gap.

### Affected files

`gpunarrowphase/src/CUDA/warpHelpers.cuh`, `gpunarrowphase/src/CUDA/epa.cuh`, `gpunarrowphase/src/CUDA/nputils.cuh`, `gpunarrowphase/src/CUDA/convexTriangle.cuh`, `gpunarrowphase/src/CUDA/convexHeightfield.cu`, `gpusolver/src/CUDA/constraintBlockPrep.cu`, `gpusolver/src/CUDA/constraintBlockPrepTGS.cu`, `gpusolver/src/CUDA/solverMultiBlockTGS.cu`, `gpusimulationcontroller/src/CUDA/bvh.cuh`

---

## Blocker 10 — `#pragma unroll` with Large or Variable Trip Counts

**Severity: MEDIUM**

### What CUDA does

PhysX has 60+ `#pragma unroll` and `#pragma unroll N` annotations. The CUDA compiler unrolls unconditionally:

```cpp
// articulationDynamic.cuh:49–50
#pragma unroll 3
for (PxU32 ind = 0; ind < 3; ++ind) { ... }

// forwardDynamic2.cu:1917
#pragma unroll (6)
for (int i = 0; i < 6; ++i) { ... }

// RadixSort.cuh:157–158
#pragma unroll
for (PxU32 bit = 0; bit < 16; bit += 2) { ... }

// cudaGJKEPA.cu:1644
#pragma unroll 4
for (PxU32 baseOffs = 0; baseOffs < WARP_SIZE; baseOffs += numElemsPerIteration)
```

### Capybara constraint

Capybara constexpr unroll requires:
- Trip count ≤ 16
- `body_nodes × trip_count ≤ 1024`

Examples that fit: loops with trip count ≤ 3 or ≤ 6 (articulation DOF loops) are safe. The WARP_SIZE=32 loop above (`baseOffs < 32`, step 4 → 8 iterations) may fit depending on body size, but 16-iteration loops like `for (bit = 0; bit < 16; ...)` are borderline.

### What needs a decision

Per kernel, confirm which unrolled loops can remain as normal `for ... in range(N)` and which require constexpr unroll. Loops exceeding the limit may need to be re-expressed as tile operations (e.g., `warp.sort_kv`) or explicitly restructured.

### Affected files

`gpuarticulation/src/CUDA/articulationDynamic.cuh`, `gpuarticulation/src/CUDA/forwardDynamic2.cu`, `gpucommon/src/CUDA/RadixSort.cuh`, `gpucommon/src/CUDA/reduction.cuh`, `gpunarrowphase/src/CUDA/cudaGJKEPA.cu`, `gpusimulationcontroller/src/CUDA/softBodyGM.cu`, `gpusimulationcontroller/src/CUDA/isosurfaceExtraction.cu` (60+ total)

---

## Blocker 11 — Union Types in Device Code

**Severity: MEDIUM**

### What CUDA does

PhysX uses unions to reinterpret the same shared memory storage as different types, primarily to read/write packed contact data from a 32-bit word:

```cpp
// cudaGJKEPA.cu:1582–1592
union ContactUnion {
    ContactParams params;
    PxU32 val;
};

union MaterialUnion {
    MaterialParams params;
    PxU32 val;
};

// cudaGJKEPA.cu (within FinishContactsWarpScratch):
union {
    ContactParams contactParams[WARP_SIZE];
    PxU32 contactUnion[WARP_SIZE];
};

// dataReadWriteHelper.cuh:62–68
union PxgVelocityPack {
    PxgVelocityPackTGS tgs;
    PxgVelocityPackPGS pgs;
};
```

### Why it cannot be ported directly

Capybara `@cp.struct` does not support union variants. There is no anonymous union syntax and no way to alias two struct types over the same storage.

The standard workaround is to use `thread.bitcast()` for scalar-sized cases (struct fits in 32 bits) and to use separate named views/allocations for cases where the union covers larger storage.

### Affected files

`gpunarrowphase/src/CUDA/cudaGJKEPA.cu`, `gpunarrowphase/src/CUDA/dataReadWriteHelper.cuh`

### What needs a decision

For each union: confirm whether `thread.bitcast()` covers the struct↔integer reinterpretation, or whether a different approach (e.g., packing/unpacking functions) is needed.

---

## Blocker 12 — Texture Memory (`tex3D`)

**Severity: MEDIUM**

### What CUDA does

`sdfCollision.cuh` uses 3D CUDA texture fetches for hardware-interpolated SDF sampling:

```cpp
// sdfCollision.cuh:118
return tex3D<float>(texture, f.x + 0.5f, f.y + 0.5f, f.z + 0.5f);

// sdfCollision.cuh:134, 147
return tex3D<float>(texture, f.x + 0.5f, f.y + 0.5f, f.z + 0.5f);
PxReal v = tex3D<float>(textureSubgrids, f.x + 0.5f, f.y + 0.5f, f.z + 0.5f);
```

The texture unit provides hardware-accelerated trilinear interpolation and clamped addressing. This is the hot path for all SDF collision kernels.

### Why it cannot be ported directly

Capybara has no texture object model. There is no `tex3D`, no texture unit binding, and no hardware interpolation path.

### Scope

This blocker affects only `gpunarrowphase/src/CUDA/sdfCollision.cuh` and the kernels that include it (SDF collision, SDF mesh contact generation, cloth/softbody SDF queries). All other modules are unaffected.

### What needs a decision

- Can the SDF query be approximated by a software trilinear interpolation over a normal view/tensor argument (semantic change: no hardware clamping/filtering)?
- Or must this remain as a `cp.intr`-level texture fetch?

---

## Blocker 13 — Deep Device-Function Call Graphs and Cross-File Include Chains

**Severity: MEDIUM**

### What CUDA does

The largest kernels delegate to chains of 4–7 `__device__` helper functions spread across multiple `.cuh` files:

**Articulation (depth 6–7):**
```
artiPropagateRigidImpulsesAndSolveSelfConstraints1T   (__global__)
  → getImpulseSelfResponseDofAligned()                (internalConstraints2.cu)
    → propagateImpulseDofAligned()
      → propagateAccelerationW()                      (articulationImpulseResponse.cuh)
        → computeSpatialJointDelta()
          → FeatherstoneArticulation::translateSpatialVector()
```

**Narrowphase GJK/EPA (depth 4+):**
```
convexConvexNphase_stage1Kernel   (__global__)
  → collide()                     (cudaGJKEPA.cu)
    → runGJK()
      → squawk::epa()             (epa.cuh)
        → newPoint(), generateConvexContacts(), polyClip()
```

**Include depth (representative):**
```
internalConstraints2.cu
  → articulationDynamic.cuh
  → articulationImpulseResponse.cuh → MemoryAllocator.cuh, articulationDynamic.cuh
  → solverBlock.cuh → solverResidual.cuh, constraintPrepShared.cuh
  → solverBlockTGS.cuh → solverResidual.cuh, solverBlockCommon.cuh
  → reduction.cuh
```

### Why this is a porting concern

All `__device__` helpers must become `@cp.inline` functions in Capybara. This is structurally feasible (Capybara inlines everything), but:

- Each helper must be manually translated and organized as a Python module
- Deep inlining of large function chains increases Capybara compile time significantly
- Cross-file helpers that call into C++ class methods (e.g., `FeatherstoneArticulation::translateSpatialVector()`) bring in CPU-side math code that must also be ported
- Include chains of 10+ `.cuh` files per `.cu` must be consolidated into Python module hierarchies

### Affected files (most complex)

`gpuarticulation/src/CUDA/internalConstraints2.cu` (~4010 lines, depth 6–7), `gpunarrowphase/src/CUDA/cudaGJKEPA.cu` (~3230 lines, depth 4+), `gpunarticulation/src/CUDA/forwardDynamic2.cu` (~4160 lines), `gpusimulationcontroller/src/CUDA/femClothPrimitives.cu` (~3900 lines), `gpusimulationcontroller/src/CUDA/particlesystem.cu` (~5200 lines)

---

## Additional Structural Observations

### AoS vs SoA Data Layout

PhysX uses predominantly AoS (Array-of-Structs) layout for physics objects (`PxgContactManagerInput`, `PxgShape`, `PxgArticulationCoreDesc`). Capybara views work best with SoA (Struct-of-Arrays). The existing `utility.py` port demonstrates the conversion: `PxTriangleMeshEmbeddingInfo` (AoS) was split into separate `skin_uv`, `skin_offset_along_normal`, `skin_tri_id` arrays. This transformation is systematic and applies to every kernel in the codebase.

### Dynamic Shared Memory (`extern __shared__`)

Only one file uses dynamic shared memory: `gpusimulationcontroller/src/CUDA/algorithms.cu:128`:

```cpp
extern __shared__ PxU32 sumsMemory[];
T* sums = reinterpret_cast<T*>(sumsMemory);
```

Capybara's `block.alloc()` is compile-time sized. This single use site requires the shared size to be a `cp.constexpr` parameter.

### Non-Issues (No Port Blocker)

- **Dynamic parallelism**: Not used anywhere.
- **Cooperative groups**: Not used anywhere.
- **cub / thrust**: Not used; all warp/block algorithms are hand-written.
- **`cudaMemcpy` / device-side allocation**: Not used in `.cu`/`.cuh` files.
- **Variable-length iteration**: BV32 tree traversal and variable contact loops are portable via `while` + `break` as demonstrated in `bv32_traversal.py`.
- **Large fixed shared memory arrays**: Portable via `block.alloc()` with compile-time sizes.

---

## Per-Module Impact Summary

| Module | Blocker 1<br>u64 ptr | Blocker 2<br>type pun | Blocker 3<br>templates | Blocker 4<br>Px math | Blocker 5<br>ptr arith | Blocker 6<br>__constant__ | Blocker 7<br>warp intr | Blocker 8<br>PTX asm | Blocker 9<br>volatile | Blocker 10<br>unroll | Blocker 11<br>union | Blocker 12<br>tex3D | Blocker 13<br>call depth | Overall |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **gpucommon** | ✗ | ✗ | ✓ | — | ✓ | ✓ | ✓ | ✗ | — | ✓ | — | — | ✓ | **HIGH** |
| **gpubroadphase** | ✗ | ✓ | ✓ | — | ✓ | — | ✓ | — | — | ✓ | — | — | ✓ | **HIGH** |
| **gpunarrowphase** | — | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | **CRITICAL** |
| **gpuarticulation** | ✗ | ✓ | ✗ | ✓ | ✓ | — | ✗ | — | — | ✗ | — | — | ✗ | **CRITICAL** |
| **gpusolver** | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | ✓ | **HIGH** |
| **gpusimulationcontroller** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ | ✓ | — | — | ✓ | **HIGH** |

Legend: `✗` = critical/heavy use, `✓` = present/moderate use, `—` = absent or minor

### Ranking by porting difficulty

1. **gpunarrowphase** — Every major blocker is present at heavy use; GJK/EPA alone combines type punning, templates, inline PTX, volatile, 64-bit warp intrinsics, and 4+ call-depth chains.
2. **gpuarticulation** — Deep call graphs, 64-bit popcount/clzll, strong templates, and u64 pointer ABI in the direct GPU API.
3. **gpusimulationcontroller** — Broadest module (particles, softbody, FEM cloth, SDF construction, isosurface); most blockers present but at moderate depth.
4. **gpusolver** — Volatile shared memory, constant memory descriptor, and `sizeof`-heavy constraint prep; structurally simpler per-kernel.
5. **gpubroadphase** — Focused SAP + aggregate; template and warp-intrinsic heavy but manageable scope.
6. **gpucommon** — Smallest module; two ports already exist; remaining `.cu` files (`radixSortImpl.cu`, `utility.cu`) are medium complexity.

---

## Recommended Next Steps

For each blocker, the following decisions are needed before any new port can begin:

| # | Blocker | Required decision |
|---|---------|-------------------|
| 1 | u64 pointer ABI | Is offset/tensor ABI acceptable, or is raw-address ABI required? If the latter, what `cp.intr` path? |
| 2 | reinterpret_cast (aggregate) | Can contact/convex binary stream layouts be redefined? If not, byte-buffer escape hatch needed. |
| 3 | C++ templates | Will each template instantiation be a separate `@cp.inline`, or is Python-level metaprogramming allowed? |
| 5 | `sizeof`/`alignof` | Can stream layouts be view-compatible, or must byte offsets be preserved? |
| 6 | `__constant__` | Lookup tables: inline as Python constants, pass as tensor args, or `cp.intr` constant-memory path? |
| 7 | Missing warp intrinsics | `__clzll`, `__ffsll`, `__popcll`, arbitrary-lane `shfl`: `cp.intr` path or algorithm restructure? |
| 8 | Inline PTX | Named barriers: restructure or `cp.intr`? Global reduction: `atomic_add` sufficient? |
| 12 | `tex3D` | Software trilinear interpolation acceptable, or `cp.intr` texture path? |
