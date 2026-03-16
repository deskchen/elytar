# PhysX PTX Hybrid Build

## Overview

This document describes the **Elytar PTX hybrid build system** for PhysX 5.6.1. It enables
per-kernel control over whether device code comes from the original `.cu` file compiled by
`nvcc`, or from a pre-generated `.ptx` file (e.g. from your Triton-like DSL).

| Mode | `PX_PTX_REPLACE_LIST` | Behavior |
|------|----------------------|----------|
| **All .cu** (default) | `""` (empty) | All 61 kernels compiled by `nvcc` from `.cu` |
| **All PTX** | `"all"` | All 61 kernels built from pre-generated `.ptx` |
| **Hybrid** | `"integration;solver"` | Listed stems from PTX, rest from `.cu` |

PTX files are **not tracked in git**. They live in `source/<module>/src/PTX/` and must be
generated with `generate_ptx.sh` before building. The sanity-check PTX is generated from the
same `.cu` files so the build produces functionally identical binaries.

---

## Quick Start

### Original mode (all .cu, default)

```bash
./scripts/update_toolchain.sh
```

### All-PTX mode

```bash
# Step 1: generate PTX from all .cu files
PX_PTX_ARCH=compute_86 ./scripts/generate_ptx.sh --all

# Step 2: build with all kernels replaced by PTX
PX_PTX_REPLACE_LIST=all PX_PTX_ARCH=compute_86 ./scripts/update_toolchain.sh
```

### Hybrid mode (selected kernels)

```bash
# Generate PTX only for the stems you want replaced
PX_PTX_ARCH=compute_86 ./scripts/generate_ptx.sh --list "integration;solver"

# Build — integration and solver from PTX, the other 59 from .cu
PX_PTX_REPLACE_LIST="integration;solver" PX_PTX_ARCH=compute_86 ./scripts/update_toolchain.sh
```

### Generate a single file

```bash
./scripts/generate_ptx.sh --cu physx-5.6.1/source/gpusolver/src/CUDA/integration.cu
```

#### GPU architecture values

| GPU | Arch flag |
|-----|-----------|
| RTX 3090 | `compute_86` (default) |
| H200 | `compute_90` |

---

## Architecture: How PhysX Launches GPU Kernels

### PhysX does NOT use `<<<...>>>` syntax

All GPU kernels in PhysX 5.6.1 are launched through the **CUDA Driver API**, not the CUDA
Runtime API. The call chain is:

```
PxgSolver.cpp
  └─ mCudaContext->launchKernel("integrateCoreParallelLaunch", ...)
       └─ CudaContextManager::launchKernel()
            └─ cuModuleGetFunction(module, "integrateCoreParallelLaunch")  ← string lookup
                 └─ cuLaunchKernel(fn, ...)
```

Kernels are resolved **by name at runtime** through the `KernelWrangler`. This means the build
system only needs to make the correct compiled kernel code available under the right name —
it doesn't matter whether that code came from a `.cu` file or a `.ptx` file.

### Static registration of fat binaries

When `nvcc` compiles a `.cu` file it:
1. Embeds a **fat binary** (contains both PTX and SASS for the target architecture) inside the
   `.o` object file.
2. Generates static initializer stubs that call PhysX's custom interception hooks:
   - `PxGpuCudaRegisterFatBinary(void* fatBin)` — registers the module image, returns a handle
   - `PxGpuCudaRegisterFunction(int moduleIndex, const char* name)` — registers each kernel name

These hooks are implemented in `PxgPhysXGpu.cpp` and `CudaContextManager.cpp`. They populate
a module table that `CudaContextManager::init()` later uses to call `cuModuleLoadDataEx()` —
loading each fat binary into a `CUmodule` on the device.

### PTX-mode kernels replicate this exactly

In PTX mode, for each replaced `.cu` file the auto-generated `<stem>_ptx_register.cpp`
performs the same two registrations at program startup via a C++ static initializer:

```cpp
namespace {
struct Elytar_integration_Registrar {
    Elytar_integration_Registrar() {
        void** handle = PxGpuCudaRegisterFatBinary(
            static_cast<void*>(integration_fatbin));   // our custom fatbin byte array
        int moduleIndex = static_cast<int>(
            reinterpret_cast<std::size_t>(handle));
        PxGpuCudaRegisterFunction(moduleIndex, "integrateCoreParallelLaunch");
        PxGpuCudaRegisterFunction(moduleIndex, "integrateKinematicParallelLaunch");
        // ... all kernels listed in the .ptx .entry directives
    }
};
static Elytar_integration_Registrar s_integration_registrar;
} // namespace
```

The fat binary is generated from the PTX by `nvcc -fatbin`, converted to a C byte array by
`bin2c`, and `#include`d as `integration_fatbin.h`. From the runtime's perspective this is
indistinguishable from the fat binary that `nvcc` would have embedded in the `.cu` object.

---

## Build Pipeline Comparison

### .cu path (original)

```
integration.cu
    │
    ▼  nvcc (full .cu compilation)
integration.o
    │  ┌─ embedded fatbin (PTX + SASS)
    │  ├─ __cudaRegisterFatBinary() stub (static init)
    │  └─ __cudaRegisterFunction() stubs for each __global__ kernel
    │
    ▼  linker
libPhysXSolverGpu_static_64.a
    │
    ▼  runtime  cuModuleLoadDataEx(embedded_fatbin)
CUmodule  ──►  cuModuleGetFunction("integrateCoreParallelLaunch")  ──►  cuLaunchKernel
```

### PTX path (hybrid)

```
integration.cu
    │
    ▼  nvcc -ptx  (scripts/generate_ptx.sh, run once before build)
integration.ptx                              ← NOT in git
    │
    ▼  nvcc -fatbin  (CMake add_custom_command, runs during cmake --build)
integration.fatbin
    │
    ▼  bin2c  (CMake add_custom_command, runs during cmake --build)
integration_fatbin.h                         ← C byte array
    │
    ┌──────────────────────────────────────────────────────────────┐
    │  integration_ptx_register.cpp  (generated at cmake configure) │
    │  #include "integration_fatbin.h"                              │
    │  namespace { struct Elytar_integration_Registrar { ... }; }  │
    └──────────────────────────────────────────────────────────────┘
    │
    ▼  g++ / clang++ (plain C++ compilation, no nvcc)
integration_ptx_register.o
    │  ┌─ integration_fatbin data symbol (type D)
    │  └─ _GLOBAL__sub_I_integration_ptx_register.cpp
    │
    ▼  linker
libPhysXSolverGpu_static_64.a
    │
    ▼  runtime  (IDENTICAL path from here)
CUmodule  ──►  cuModuleGetFunction("integrateCoreParallelLaunch")  ──►  cuLaunchKernel
```

The two paths converge at `cuModuleLoadDataEx`. Everything from that point on is identical.

---

## Original .cu vs PTX path: Full Diff

| Aspect | `.cu` path | PTX path |
|--------|-----------|----------|
| **What compiles the GPU code** | `nvcc` during build | `nvcc -ptx` before build |
| **Who provides the fatbin** | `nvcc` embeds it in `.cu` object | `bin2c` converts `.fatbin` to C array |
| **Who calls `PxGpuCudaRegisterFatBinary`** | `nvcc`-generated static init | Auto-generated `_ptx_register.cpp` |
| **Who calls `PxGpuCudaRegisterFunction`** | `nvcc`-generated (name from source) | Auto-generated (names from `.ptx .entry`) |
| **Host init stub** (`initXxxKernelsN`) | Defined in `.cu` | Auto-generated `_host_stub.cpp` (empty body) |
| **Linker input** | `.cu.o` from `nvcc` | `.cpp.o` from `g++/clang++` |
| **Symbol in `.a` for verification** | `__cuda_module_id` per TU | `<stem>_fatbin` data symbol (type D) |
| **Runtime module loading** | `cuModuleLoadDataEx(fatbin)` | `cuModuleLoadDataEx(fatbin)` — identical |
| **Runtime kernel lookup** | `cuModuleGetFunction(name)` | `cuModuleGetFunction(name)` — identical |
| **Runtime kernel dispatch** | `cuLaunchKernel(fn, ...)` | `cuLaunchKernel(fn, ...)` — identical |
| **Kernel execution overhead** | baseline | **zero additional overhead** |
| **DSL swap** | Requires editing `.cu` | Replace `.ptx` file, rebuild |
| **Git tracking** | `.cu` is in git | `.ptx` is NOT in git (generated) |

---

## Zero-Overhead Proof

**Claim:** PTX path introduces zero kernel-execution overhead compared to the `.cu` path.

**Proof:**

1. **Same SASS at runtime.** `nvcc -fatbin -arch=sm_XX` compiles the PTX to SASS for the
   target architecture. The SASS is generated from the same source with the same flags —
   identical machine code.

2. **Same module loading.** Both paths call `cuModuleLoadDataEx()` with equivalent fat binary
   images. Since we use `nvcc -fatbin -arch=sm_XX` it includes pre-compiled SASS — no JIT.

3. **Same kernel dispatch.** Both paths resolve kernels via `cuModuleGetFunction` by string
   name and call `cuLaunchKernel`. The function handle points to the same SASS instructions.

4. **No wrapper layer.** There are no extra function calls, indirections, or data copies in
   the hot path.

---

## Per-File Decision Logic

For each `.cu` file in every GPU sub-library, cmake evaluates:

```
PX_PTX_REPLACE_LIST = ""      →  use .cu for all
PX_PTX_REPLACE_LIST = "all"   →  use PTX for all
PX_PTX_REPLACE_LIST = "a;b"   →  use PTX for stems a and b, .cu for the rest
```

The check in each sub-library cmake file (`PhysXSolverGpu.cmake`, etc.):

```cmake
IF(PX_PTX_REPLACE_LIST STREQUAL "all")
    SET(_use_ptx TRUE)
ELSE()
    LIST(FIND PX_PTX_REPLACE_LIST "${_stem}" _idx)
    IF(_idx GREATER_EQUAL 0)
        SET(_use_ptx TRUE)
    ENDIF()
ENDIF()
```

If `_use_ptx` is TRUE and the corresponding `.ptx` file exists, `ELYTAR_REPLACE_CU_WITH_PTX()`
is called — otherwise the `.cu` stays in the kernel list and is compiled normally.

---

## Implementation Details

### `ElytarPtxReplace.cmake` — the core macro

`physx-5.6.1/source/compiler/cmakegpu/ElytarPtxReplace.cmake` provides the
`ELYTAR_REPLACE_CU_WITH_PTX()` macro. For each replaced `.cu` file it:

1. Validates that the corresponding `.ptx` file exists (FATAL_ERROR if missing).
2. Parses the `.cu` file for `extern "C" __host__ void <name>()` — generates
   `<stem>_host_stub.cpp` providing the empty body.
3. Reads kernel names from `.ptx .entry` directives — the authoritative source for the exact
   names `cuModuleGetFunction` / `PxGpuCudaRegisterFunction` must use, handling both
   `extern "C"` (plain) and C++ (mangled) kernels transparently.
4. Generates `<stem>_ptx_register.cpp` using `file(GENERATE ...)`.
5. Registers `nvcc -fatbin` and `bin2c` as CMake custom commands.
6. Sets `OBJECT_DEPENDS` to ensure `_ptx_register.cpp` is not compiled until its `_fatbin.h`
   header exists.
7. Removes the `.cu` from the `KERNELS_VAR` list and appends generated sources to
   `ELYTAR_PTX_EXTRA_SOURCES`.

### Kernel name extraction: why from PTX, not `.cu`

An early version parsed `extern "C" __global__ void <name>(` from the `.cu` source. This
failed for most PhysX kernels because PhysX uses plain C++ linkage for many kernels — their
names are mangled by the compiler. The PTX `.entry` directive always records the exact symbol
name regardless of linkage.

### Anonymous namespace and symbol visibility

The `Elytar_<stem>_Registrar` struct is in an anonymous `namespace {}` (internal linkage).
The compiler does not export the struct name; it inlines the constructor into a file-scope
static initializer. The correct PTX-mode fingerprints visible to `nm` are:
- `<stem>_fatbin` (type `D`) — the fatbin byte array from `bin2c`
- `_GLOBAL__sub_I_<stem>_ptx_register.cpp` (type `t`) — the static initializer

### PTX files are not in git

All `source/<module>/src/PTX/` directories are excluded from git tracking (`.gitignore`).
Run `generate_ptx.sh` to regenerate them before any PTX-mode build. This keeps the repo clean
of large binary/text generated files.

---

## Verification

After building, `update_toolchain.sh` calls `verify_ptx_mode()` when `PX_PTX_REPLACE_LIST`
is non-empty. The checks are **exact** — they verify that the count of PTX artifacts matches
the configured `PX_PTX_REPLACE_LIST`, not a generic "at least N" count.

### Check 1 — PTX files on disk

For each stem in `PX_PTX_REPLACE_LIST`, verifies `source/<module>/src/PTX/<stem>.ptx` exists.

### Check 2 — Build stubs in the build tree

Counts `*_ptx_register.cpp` and `*.fatbin` in `<build>/sdk_gpu_source_bin/elytar_ptx/`.
Both counts must equal `len(PX_PTX_REPLACE_LIST)` exactly.

### Check 3 — Symbols per library (`nm`)

For each GPU static library, counts the number of configured PTX stems belonging to it. Then
verifies that `nm --defined-only` shows exactly that many `<stem>_fatbin` data symbols (type D)
and exactly that many `_GLOBAL__sub_I_*_ptx_register.cpp` static-init entries.

- `expected == 0`: verifies there are **no** `*_fatbin` symbols (pure `.cu` library).
- `expected > 0`: verifies the exact count is correct (no extra, no missing).

---

## File Structure

```
physx-5.6.1/source/
├── compiler/cmakegpu/
│   ├── CMakeLists.txt                     ← PX_PTX_REPLACE_LIST, PX_PTX_ARCH options
│   ├── ElytarPtxReplace.cmake             ← ELYTAR_REPLACE_CU_WITH_PTX() macro
│   ├── PhysXSolverGpu.cmake               ← per-stem PTX/cu selection (12 kernels)
│   ├── PhysXBroadphaseGpu.cmake           ← per-stem PTX/cu selection (2 kernels)
│   ├── PhysXNarrowphaseGpu.cmake          ← per-stem PTX/cu selection (25 kernels)
│   ├── PhysXSimulationControllerGpu.cmake ← per-stem PTX/cu selection (15 kernels)
│   ├── PhysXArticulationGpu.cmake         ← per-stem PTX/cu selection (4 kernels)
│   └── PhysXCommonGpu.cmake               ← per-stem PTX/cu selection (3 kernels)
│
├── gpusolver/src/
│   ├── CUDA/integration.cu                ← original source (unmodified)
│   └── PTX/integration.ptx                ← NOT in git; generated by generate_ptx.sh
│   (same pattern for all 6 modules)
│
scripts/
├── update_toolchain.sh    ← passes PX_PTX_REPLACE_LIST/PX_PTX_ARCH to cmake + verify
└── generate_ptx.sh        ← generates PTX; supports --all, --cu, --list

docs/
└── ptx_dual_mode.md       ← this file
```

### Generated files (build directory)

When any PTX stems are configured, cmake generates into
`<build>/sdk_gpu_source_bin/elytar_ptx/` (shared `CMAKE_CURRENT_BINARY_DIR` for all
sub-libraries):

```
<build>/sdk_gpu_source_bin/elytar_ptx/
├── <stem>_host_stub.cpp        ← extern "C" void initXxxKernelsN() {}
├── <stem>_ptx_register.cpp     ← static initializer for fatbin registration
├── <stem>.fatbin               ← build-time: nvcc -fatbin <stem>.ptx
└── <stem>_fatbin.h             ← build-time: bin2c <stem>.fatbin
    (one set per PTX-replaced stem)
```

---

## CMake Options Reference

Defined in `physx-5.6.1/source/compiler/cmakegpu/CMakeLists.txt`:

| Option | Default | Description |
|--------|---------|-------------|
| `PX_PTX_REPLACE_LIST` | `""` (empty) | Semicolon-separated kernel stems to build from PTX, or `"all"` |
| `PX_PTX_ARCH` | `compute_86` | GPU architecture for PTX→fatbin compilation |

Pass via environment in `update_toolchain.sh`:
```bash
PX_PTX_REPLACE_LIST="integration;solver" PX_PTX_ARCH=compute_90 ./scripts/update_toolchain.sh
```

Or directly to cmake:
```bash
cmake -DPX_PTX_REPLACE_LIST="integration;solver" -DPX_PTX_ARCH=compute_90 ...
```

---

## How to Add DSL-Generated PTX

When your Triton-like DSL is ready to replace a kernel:

1. Compile your DSL program to PTX targeting the appropriate architecture.
2. Ensure the `.entry` names in the PTX match the original kernel names. Find them with:
   ```bash
   grep '\.entry ' physx-5.6.1/source/gpusolver/src/PTX/integration.ptx
   ```
3. Place the PTX file at:
   ```
   physx-5.6.1/source/<module>/src/PTX/<stem>.ptx
   ```
4. Run:
   ```bash
   PX_PTX_REPLACE_LIST="<stem>" ./scripts/update_toolchain.sh
   ```

The CMake pipeline picks up your file automatically; no cmake changes are needed.

> **Tip:** The kernel name strings that PhysX uses for wrangler lookups are in the kernel
> index headers, e.g., `PxgSolverKernelIndices.h`, `PxgNarrowphaseKernelIndices.h`.
> These must match the `.entry` names in your PTX exactly (case-sensitive).

---

## Scope: All 61 Kernel Files

| Sub-library | cmake file | `.cu` count | PTX dir |
|-------------|-----------|-------------|---------|
| `PhysXSolverGpu` | `PhysXSolverGpu.cmake` | 12 | `gpusolver/src/PTX/` |
| `PhysXBroadphaseGpu` | `PhysXBroadphaseGpu.cmake` | 2 | `gpubroadphase/src/PTX/` |
| `PhysXNarrowphaseGpu` | `PhysXNarrowphaseGpu.cmake` | 25 | `gpunarrowphase/src/PTX/` |
| `PhysXSimulationControllerGpu` | `PhysXSimulationControllerGpu.cmake` | 15 | `gpusimulationcontroller/src/PTX/` |
| `PhysXArticulationGpu` | `PhysXArticulationGpu.cmake` | 4 | `gpuarticulation/src/PTX/` |
| `PhysXCommonGpu` | `PhysXCommonGpu.cmake` | 3 | `gpucommon/src/PTX/` |
| **Total** | | **61** | |

---

## Troubleshooting

### "PTX file not found" during cmake configure

```
[Elytar] PTX file not found: .../integration.ptx
         Run: scripts/generate_ptx.sh --all
```

The `.ptx` file must exist before cmake can configure in PTX mode.

For a specific stem:
```bash
./scripts/generate_ptx.sh --cu physx-5.6.1/source/gpusolver/src/CUDA/integration.cu
```

For a list:
```bash
./scripts/generate_ptx.sh --list "integration;solver"
```

### "bin2c not found" during cmake configure

`bin2c` ships with the CUDA toolkit. Verify:
```bash
which bin2c || ls "$(dirname "$(which nvcc)")/bin2c"
```

### Verification check 2 fails: wrong stub/fatbin count

Means the number of generated stubs doesn't match `PX_PTX_REPLACE_LIST`. The build tree may
have stale files from a previous PTX run. Delete and rebuild:
```bash
sudo rm -rf "${PHYSX_BUILD_DIR}/sdk_gpu_source_bin/elytar_ptx"
cmake --build "${PHYSX_BUILD_DIR}"
```

### Verification check 3 fails: wrong *_fatbin symbol count

Means a library has more or fewer PTX-mode objects than configured. Stale archives from a
different `PX_PTX_REPLACE_LIST` run are the usual cause. Clean rebuild:
```bash
cmake --build "${PHYSX_BUILD_DIR}" --clean-first
```

### Kernel name mismatch (DSL PTX)

If the wrangler cannot find your DSL kernel at runtime:
1. Check `.entry` names: `grep '\.entry' your_kernel.ptx`
2. Check the kernel index header (e.g., `PxgSolverKernelIndices.h`)
3. Check the auto-generated `_ptx_register.cpp` in the build tree — it was parsed from
   the `.ptx` `.entry` directives; if your DSL emits a different name you need to match
   the original.

### Build order error (`<stem>_fatbin.h` not found)

The `OBJECT_DEPENDS` property ensures `_ptx_register.cpp` is not compiled until its
`_fatbin.h` exists. A clean build resolves ordering issues:
```bash
cmake --build "${PHYSX_BUILD_DIR}" --clean-first
```
