# DSL Grammar Guideline for PhysX Port

> **Capybara DSL source code**: `~/capybara-triton` — compiler, codegen, language runtime, and test examples.

This guideline is for AI assistants translating PhysX CUDA kernels to Capybara Python DSL in an apple-to-apple way (same algorithmic semantics, same control-flow intent, and PTX-targeted output).

Primary use case:
- Source workspace: `/home/zhuochen/elytar/physx-5.6.1/source/...`
- Capybara workspace for references: `/home/zhuochen/capybara-triton`

---

## Zero-Guessing Rule (Read First)

If you cannot prove a translation is apple-to-apple, **stop and ask the user**.

Do not silently:
- simplify control flow,
- weaken memory ordering,
- replace unsupported intrinsics with guessed behavior,
- alter numeric semantics,
- drop synchronization,
- or swap data layout without explicit confirmation.

### Mandatory stop-and-ask triggers

Stop and ask when any of these appear:
- CUDA intrinsic has no clearly equivalent DSL primitive in current source code.
- Translation would change divergent vs uniform behavior.
- Struct/bitfield/packing semantics are unclear.
- Shared-memory aliasing or race-freedom cannot be proven.
- A workaround changes algorithm behavior (not just syntax).
- You see a compiler limitation and there are multiple plausible workarounds.
- Kernel requires raw-address semantics (`u64 -> typed pointer -> dereference`) and equivalence is not proven.
- Port requires exact CUDA ABI replacement but translation changes kernel argument shape.

### Required question format

Use this exact pattern:
- **What is unclear**: one sentence.
- **Why apple-to-apple is uncertain**: one sentence.
- **2-3 concrete options**: with tradeoffs.
- **Ask for choice**: "Which option do you want?"

Example:
- "I cannot prove this `__shfl_sync` variant maps exactly to a supported `thread.shfl_*` path in current Capybara codegen."
- "If I guess wrong, lane communication order may differ from CUDA."
- "Options: (A) keep warp-uniform path with `thread.shfl_down`; (B) use shared-memory fallback + barrier; (C) defer and use cp.intr-level implementation."
- "Which option do you want?"

---

## Scope and Objective

You are not writing a toy port. You are translating production PhysX kernels (`.cu` / `.cuh`) into Capybara DSL so they compile through the Capybara pipeline to PTX.

### Apple-to-apple means

- Same algorithmic contract and edge-case behavior.
- Same memory ordering and synchronization intent.
- Equivalent control-flow convergence/divergence characteristics where required.
- Equivalent data representation (or an explicitly approved representation change).
- Equivalent failure/sentinel conventions.

---

## Ground Truth Sources (Use These First)

Do not rely on docs alone. The source of truth is codegen + working kernels/tests.

### Compiler/codegen source

- `/home/zhuochen/capybara-triton/python/capybara/__init__.py`
- `/home/zhuochen/capybara-triton/python/capybara/runtime/__init__.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/main.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/stmts.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/control_flow.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/exprs.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/dispatch.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/dispatch_block.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/regions.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/view_construction.py`
- `/home/zhuochen/capybara-triton/python/capybara/language/__init__.py`
- `/home/zhuochen/capybara-triton/include/capybara/Dialect/CapybaraIntr/IR/CapybaraIntrOps.td`

### Canonical examples for this port

- PhysX-style BV32 traversal:
  - `/home/zhuochen/capybara-triton/python/test/capybara/bench/capybara_kernels/bv32_traversal.py`
- Hash map + atomics:
  - `/home/zhuochen/capybara-triton/python/test/capybara/bench/capybara_kernels/hashmap_insert.py`
- While + break + assume_uniform:
  - `/home/zhuochen/capybara-triton/python/test/capybara/bench/capybara_kernels/tsne_repulsion.py`
- Composite irregular kernel:
  - `/home/zhuochen/capybara-triton/python/test/capybara/bench/capybara_kernels/cagra_search.py`
- Struct patterns:
  - `/home/zhuochen/capybara-triton/python/test/capybara/test_struct_e2e.py`
- First-class struct load/store + methods:
  - `/home/zhuochen/capybara-triton/python/test/capybara/test_first_class_struct.py`

### PhysX source anchors

- Root GPU source:
  - `/home/zhuochen/elytar/physx-5.6.1/source`
- Narrowphase CUDA:
  - `/home/zhuochen/elytar/physx-5.6.1/source/gpunarrowphase/src/CUDA`
- BV32 reference header:
  - `/home/zhuochen/elytar/physx-5.6.1/source/gpunarrowphase/src/CUDA/bv32Traversal.cuh`
- Memcpy/ABI blocker note:
  - `/home/zhuochen/elytar/physx-5.6.1/source/gpucommon/src/capybara/MemCopyBalanced_PORT_BLOCKER.md`

---

## DSL Grammar Coverage (Source-Verified)

This section describes what the frontend/codegen actually supports for kernel code.

## 1) Kernel and inline definitions

Supported:
- `@cp.kernel` for entry kernels.
- `@cp.inline` helpers called inside kernels.
- `@cp.kernel(...)` options such as `max_regs`, `verbose_ptxas`, `debug`,
  `num_stages`, `grid_swizzle`, `index_bitwidth`, `readonly=[...]`,
  `constant=['arg']` (places named args in CUDA constant memory, addrspace 4),
  `constant_strategy='global'` (Strategy B global constants), and
  validated compiler knobs.

Reference:
- `/home/zhuochen/capybara-triton/python/capybara/language/__init__.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/main.py`

Notes:
- Kernel functions must be in real `.py` files (source inspection requirement in runtime path).
- Inline helpers are lowered/inlined by compiler pipeline; do not assume arbitrary Python call semantics.
- Kernel body is compiled at block scope; there is no legacy implicit thread-mode wrapping.

## 2) Launch and unit scopes

### Kernel launch

Pattern:
- `with cp.Kernel(grid..., threads=BLOCK_SIZE) as (..., block):`
- `cp.Kernel` also supports `grid=` keyword form.

Persistent work distribution:
- `with cp.PersistentKernel(n_items, threads=..., mode="dynamic"|"static") as (item_id, block):`
  with a single grid dimension and one persistent region per kernel.

Codegen override region:
- `with cp.codegen(knob=value, ...):` for constexpr integer knob overrides in a scoped block.

### Unit-binding forms

Supported region constructs (from `regions.py` + selector lowering):
- `for warp_id, warp in block.warps():`
- `for team_id, team in block.teams(TEAM_SIZE):`
- `for tid, thread in block.threads():`
- `for i, j, thread in block.threads(M, N):` with `M*N == BLOCK_SIZE`
- `for lane, thread in warp.threads():`
- `for lane, thread in team.threads():`
- Single-unit selection:
  - `with block.warps()[0] as (warp_id, warp):`
  - `with warp.threads()[0] as (lane, thread):`

Strict checks:
- `block.threads(M, N, ...)` dimensions must be compile-time constants.
- Dimension product must equal `BLOCK_SIZE`.
- Scope mismatches are hard errors.

References:
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/regions.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/dispatch.py`

## 3) Expressions and operators

Supported expression categories include:
- arithmetic, comparisons, boolean ops, ternary (`a if cond else b`),
- names, constants, attributes, subscripts,
- selected calls recognized by codegen dispatch.

Reference entry point:
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/exprs.py` (`_gen_expr`)

Supported helper calls include (non-exhaustive):
- `cp.ceildiv`
- type casts like `cp.int32`, `cp.float32`, etc.
- `cp.coord_grid`
- `cp.debug_print`
- `cp.assume_uniform`
- `cp.static_assert`

Unsupported expressions fail fast with `NotImplementedError`.

### 3a) Basic scalar types and casts (source-verified)

For PhysX translation, treat these as the primary scalar cast surface in kernel code:

- Integers:
  - `cp.int32(expr)`
  - `cp.int64(expr)`
  - `cp.uint32(expr)` (unsigned interpretation on i32 storage)
  - `cp.int8(expr)`
  - `cp.uint8(expr)`
- Floating-point:
  - `cp.float16(expr)`
  - `cp.bfloat16(expr)`
  - `cp.float32(expr)`
  - `cp.float64(expr)`

Source references:
- DSL symbols/stubs:
  - `/home/zhuochen/capybara-triton/python/capybara/language/__init__.py`
- Codegen call handling:
  - `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/exprs.py`
- Cast lowering:
  - `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/dispatch_thread.py`

Useful constants and annotations:
- `cp.constexpr` (compile-time parameter annotation)
- `cp.float16_max`, `cp.bfloat16_max`, `cp.float32_max`, `cp.float64_max`

Important nuance (apple-to-apple safety):
- `cp.uint64` is exposed at the language level, but `exprs.py` direct `cp.*` cast
  lowering only covers `int32`, `int64`, `uint32`, `int8`, `uint8`, and float casts.
- `int16`/`uint16`/`uint64` semantics often require `thread.cast`/`thread.bitcast`
  or a different representation; do not assume direct `cp.<type>(x)` parity.
- If a PhysX kernel requires explicit `u64`, `i16`, or `u16` cast semantics and
  you cannot prove end-to-end support, stop and ask before choosing a workaround.

---

## 4) Control flow grammar

Supported control flow:
- `if` / `else`
- `for ... in range(...)`
- `while ...`
- `break`, `continue`, `return` (scope-constrained)

Key implementation points:
- `while` without break/continue is lowered via structured `scf.while` path.
- `while` with break/continue uses `cp.while_loop` lowering path.
- `while True:` is explicitly recognized; no synthetic condition-break is injected at top.
- `for t in cp.pipelined(range(...), stages=K)` exists for software pipelining (`stages >= 2`).

References:
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/control_flow.py`
- `/home/zhuochen/capybara-triton/include/capybara/Dialect/Capybara/IR/CapybaraOps.td`

### Divergence and collectives

Critical legality rule:
- Do not place collectives after potentially divergent exits where not all lanes can reach them.

Practical rule for ports:
- If break condition is intended uniform, make that explicit and prove it, often via:
  - `if cp.assume_uniform(cond): break`

References:
- `/home/zhuochen/capybara-triton/docs/capybara/design/04_DESIGN_PROGRAMMING_MODEL.md`
- `/home/zhuochen/capybara-triton/python/test/capybara/bench/capybara_kernels/bv32_traversal.py`
- `/home/zhuochen/capybara-triton/python/test/capybara/bench/capybara_kernels/tsne_repulsion.py`

---

## 5) Memory model and data access grammar

## Allocation

- Shared/register-like storage through unit allocs:
  - `block.alloc(...)`
  - `warp.alloc(...)`
  - `team.alloc(...)`
- Struct tiles and cooperative struct I/O:
  - `block.alloc_struct(Schema, N)`
  - `block.load_struct(global_struct, smem_struct, offset, count, n_elems=...)`
  - `block.store_struct(smem_struct, global_struct, offset, count, n_elems=...)`

## Loads/stores

- Explicit scalar load: `thread.load(ref)`
- Cooperative transfer: `block.load(src, dst)` / `block.store(src, dst)`
- Subscript assignment stores:
  - `out[idx] = val`
  - scalar ref store form: `ref[:] = val`

Important:
- Rebinding Python variable names does not store into memory.
- Keep lvalue/rvalue semantics explicit when porting pointer-heavy CUDA code.
- High-level DSL does not provide direct runtime raw-address dereference from integer fields.

References:
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/exprs.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/stmts.py`
- `/home/zhuochen/capybara-triton/docs/capybara/design/05_DESIGN_TYPES_MEMORY_MODEL.md`

### Shared-memory lifetime aliasing

The compiler automatically performs cross-type SMEM aliasing for allocations with
non-overlapping liveness ranges. For sequential-phase kernels (e.g., phase 1 uses
`float32[N]`, then a barrier, then phase 2 uses `int32[N]`), no user intervention
is needed — the compiler reuses the same physical SMEM.

For data-dependent control flow where the compiler cannot prove non-overlap,
use the manual override:

```python
buf_a = block.alloc(cp.float32, N)
buf_b = block.alloc(cp.int32, N)
cp.alias_smem(buf_a, buf_b)
```

This forces `buf_a` and `buf_b` to share physical SMEM. The programmer takes
responsibility for ensuring non-overlapping access.

---

## 6) Views, padding, layout grammar

Supported high-level view methods:
- `view(...)`
- `permute(...)`
- `unfold(dim, size, step)`
- `pad(dim=..., val=...).view(...)` pipeline

Important source-verified detail:
- `padded_view()` is deprecated in codegen; use `pad(...).view(...)`.

Reference:
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/view_construction.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/exprs.py`

Porting guidance:
- Prefer multidimensional views over manual flattening math when preserving semantics allows.
- If a PhysX kernel relies on explicit linearized pointer arithmetic for alias/ordering reasons, confirm before rewriting to views.

---

## 7) Struct grammar

Supported:
- `@cp.struct` schemas
- host-side `Schema.from_arrays(...)`
- struct field validation and flattening
- struct use inside kernels with inline helpers
- first-class struct tiles with `block.alloc_struct`/`block.load_struct`/`block.store_struct`
- struct helper methods callable on struct values (non-mutating patterns)
- `@cp.inline` method definitions on `@cp.struct` types (works on both kernel-constructed
  instances and values loaded from tiles)

**Operator overloading** (`+`, `*`, etc.) on struct types is NOT supported. Use explicit
method calls (`v.add(u)`, `v.dot(u)`) or free `@cp.inline` functions.

Example (PhysX math type with struct methods):

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
```

Reference:
- `/home/zhuochen/capybara-triton/python/capybara/language/__init__.py`
- `/home/zhuochen/capybara-triton/python/test/capybara/test_struct_e2e.py`
- `/home/zhuochen/capybara-triton/python/test/capybara/test_first_class_struct.py`

Stop-and-ask trigger:
- If CUDA struct has bitfields, packed unions, or alignment-sensitive overlays and equivalent behavior is uncertain, stop and ask before changing representation.
- If struct fields carry raw pointer addresses and the kernel dereferences them
  (`reinterpret_cast`-style semantics), stop and ask before translating.

---

## 8) Primitive families and where they are legal

## Thread primitives

Examples:
- arithmetic/comparison
- `thread.load`
- atomics (`thread.atomic_cas`, `thread.atomic_add`)
- warp shuffles (`thread.shfl_up`, `thread.shfl_down`, `thread.shfl_xor`, `thread.shfl_idx`)
- bit intrinsics (`thread.popcount`, `thread.popcount64`, `thread.clz`, `thread.clz64`, `thread.ctz`, `thread.ffs`, `thread.ffs64`, and `thread.fns(mask, n)` — find n-th set bit)
- memory fences (`thread.threadfence`, `thread.threadfence_block`)
- casts/bitcasts (`thread.cast`, `thread.bitcast`)

## Collective namespace

- `thread.coll.*` is only valid in `warp.threads()` or `team.threads()`.
- It is rejected in `block.threads()` because there is no implicit parent warp/team domain for that API.

Reference:
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/dispatch.py`

## Unit-level primitives

- warp-level methods: `warp.sort_kv`, `warp.merge_sorted_kv`, etc.
- block-level methods: `block.barrier`, `block.scan`, `block.sort_kv`, `block.compact`, `block.dot` (where applicable)
- block async copy methods: `block.async_load`, `block.async_commit`, `block.async_wait`
- team-level methods as implemented in dispatch table.

Stop-and-ask trigger:
- Method exists in docs but dispatch table or codegen path is missing/limited for your scope.

---

## 9) CUDA -> Capybara mapping checklist

Use this as baseline mapping; verify per-kernel semantics.

| CUDA concept | Capybara mapping |
|---|---|
| `blockIdx.x` | `bx` from `with cp.Kernel(...) as (bx, block)` |
| `threadIdx.x` (CTA index) | `tid` from `for tid, thread in block.threads():` |
| warp lane id | `lane` from `for lane, thread in warp.threads():` |
| `__shared__` | `block.alloc(...)` |
| `__syncthreads()` | `block.barrier()` (or selector `barrier=True` exit semantics) |
| `__threadfence()` | `thread.threadfence()` |
| `__threadfence_block()` | `thread.threadfence_block()` |
| atomic CAS | `thread.atomic_cas(...)` |
| ballot + popc | `thread.coll.ballot(pred)` + `thread.popcount(mask)` |
| warp shuffle | `thread.shfl_up/down/xor(...)`, `thread.shfl_idx(...)` |
| `__shfl_sync(FULL_MASK, val, lane)` | `thread.shfl_idx(val, lane)` (already supported; no workaround needed) |
| warp-level sort/merge idioms | `warp.sort_kv`, `warp.merge_sorted_kv` |
| `__constant__ float lut[N]` | `@cp.kernel(constant=['lut'])` (places arg in constant memory, addrspace 4) |
| `union { float f; int i; }` SMEM lifetime overlap | Automatic (compiler) for non-overlapping liveness; or `cp.alias_smem(a, b)` for manual override |
| named PTX barriers (`bar.sync`, `bar.arrive`) | `block.barrier()` |
| `asm("red.global.add.f32 ...")` (global reduction) | `thread.atomic_add(...)` |

---

## 10) PhysX port workflow (strict)

For each `.cu`/`.cuh` target:

1. Identify kernel boundaries and helper/device-function graph.
2. Identify scope structure:
   - CTA work split
   - warp specialization
   - lane-level collaboration
3. Extract memory contracts:
   - shared/global usage
   - alias assumptions
   - barriers/fences
4. Translate control flow preserving convergence semantics.
5. Translate intrinsics to source-verified DSL equivalents.
6. Preserve sentinels, masks, and edge behavior.
7. Only then consider ergonomic refactors (and only with approval if semantic risk exists).

If any step is ambiguous, stop and ask.

---

## 11) Known high-risk translation zones

These require extra care and usually user confirmation if non-trivial:

- Warp-synchronous algorithms with implicit convergence assumptions.
- Ballot-driven compaction and stack writes (e.g., BV32 DFS compaction).
- Pointer alias tricks and type punning.
- Raw-address pointer fields in descriptors (`size_t` / `uint64`) that are dereferenced in-kernel.
- Mixed precision math where operation ordering matters.
- Control flow where CUDA relied on undefined-but-observed behavior.
- Any replacement of `while(true)+break` patterns that may alter divergence.

---

## 11a) Raw-address memcpy / ABI blocker

For PhysX-style kernels like `MemCopyBalanced`, treat this as a hard blocker until confirmed:

- Current high-level DSL does not support direct `u64 address -> typed pointer -> load/store`.
- `cp_intr.base_ptr` and `cp_intr.ptr_add` exist, but they operate on refs/pointers
  already in the IR; there is no source-verified `inttoptr`-style bridge from a runtime integer address.
- Therefore, an offset-based rewrite (adding explicit `src_words`/`dst_words` args) is
  an ABI change, not an apple-to-apple translation.

Required behavior:
- If exact CUDA ABI replacement is required (e.g., kernel args must stay `(desc, count)`),
  stop and ask for one of:
  1. approved intrinsic extension path,
  2. approved caller/ABI rewrite,
  3. explicit defer/block decision.

References:
- `/home/zhuochen/elytar/physx-5.6.1/source/gpucommon/src/capybara/MemCopyBalanced_PORT_BLOCKER.md`
- `/home/zhuochen/capybara-triton/include/capybara/Dialect/CapybaraIntr/IR/CapybaraIntrOps.td`

---

## 12) cp.assume_uniform and cp.disjoint

### `cp.assume_uniform(cond)`

- Compiler hint that condition is uniform across relevant lanes.
- Use when you know convergence property but compiler proof may fail.
- Do not use as a blind suppressor; ensure semantic truth first.

Reference:
- `/home/zhuochen/capybara-triton/python/capybara/language/__init__.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/exprs.py`

### `cp.disjoint(val)`

- Assignment annotation for race-free target assumption when static proof is hard.
- Common for ballot/popcount compaction store sites.
- Use only when you can explain disjointness.

Reference:
- `/home/zhuochen/capybara-triton/python/capybara/language/__init__.py`
- `/home/zhuochen/capybara-triton/python/test/capybara/bench/capybara_kernels/bv32_traversal.py`

---

## 13) Rejected/unsupported patterns to call out in the guide

Build this list from codegen errors, not assumptions.

Common examples from source:
- Unsupported AST expression/statement forms -> `NotImplementedError`.
- Scope misuse (e.g., `warp.*` outside warp scope) -> `RuntimeError`.
- `thread.coll.*` in `block.threads()` -> `RuntimeError`.
- invalid `block.threads(M,N,...)` dimensions/product -> `RuntimeError`.
- unsupported `sum(generator)` forms (filtered generators, unsupported iterators) -> `NotImplementedError`.
- `padded_view()` path deprecated (use `pad().view()`) -> `RuntimeError`.
- Assuming direct `cp.uint64(...)` cast parity with CUDA pointer casts -> unsupported unless source-verified.
- Dereferencing raw address integers from struct fields in high-level DSL -> unsupported.
- Silent ABI rewrites (e.g., `(desc, count)` to `(desc, src_words, dst_words, count)`) without approval -> not allowed.

### Union types and SMEM overlays — nuanced

Union/`reinterpret_cast` patterns in CUDA fall into three categories with different DSL status:

1. **Scalar bitcast** (e.g., `union { float f; uint32_t u; }` for bit manipulation):
   Use `thread.bitcast(val, cp.float32)` / `thread.bitcast(val, cp.uint32)`.

2. **Sequential-phase SMEM aliasing** (e.g., phase 1 writes `float32[]`, barrier, phase 2
   reads `int32[]` from same physical SMEM): Handled automatically by the compiler's
   lifetime aliasing pass. Use `cp.alias_smem(a, b)` as manual override when the compiler
   cannot prove non-overlap.

3. **Aggregate pointer aliasing / type-punning across incompatible struct types via
   `reinterpret_cast<T*>(smem_base)`**: Still unsupported. These require ABI-level
   decisions — stop and ask.

Reference files:
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/exprs.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/stmts.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/dispatch.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/regions.py`
- `/home/zhuochen/capybara-triton/python/capybara/compiler/codegen/view_construction.py`

---

## 14) Minimal apple-to-apple acceptance checklist

Before declaring a kernel translation done:

- Control flow equivalence validated.
- Synchronization/barrier/fence points preserved.
- Atomic semantics and scope preserved.
- Shared-memory indexing and race assumptions preserved.
- Sentinel/mask behavior preserved.
- Uniformity assumptions documented where used.
- Any unresolved mapping questions were explicitly asked and answered.

If any checkbox is uncertain, stop and ask.

---

## 15) Brief note on `cp.intr` escape hatch

Capybara has a lower-level intrinsics dialect (`cp.intr`) in the pipeline.
For this guideline, keep mention brief:
- use high-level DSL first,
- if an operation is not expressible with proven equivalence, stop and ask whether to use intrinsics-level implementation.
- do not assume intrinsics can recover raw-address dereference automatically; verify available ops first.

Do not auto-switch to an intrinsics workaround without user approval.

---

## 16) Quick reference examples by feature

- Warp specialization:
  - `/home/zhuochen/capybara-triton/python/test/capybara/bench/capybara_kernels/cagra_search.py`
- Ballot + popcount + compaction:
  - `/home/zhuochen/capybara-triton/python/test/capybara/bench/capybara_kernels/bv32_traversal.py`
- Atomic hash insertion:
  - `/home/zhuochen/capybara-triton/python/test/capybara/bench/capybara_kernels/hashmap_insert.py`
- While loop with explicit convergence hints:
  - `/home/zhuochen/capybara-triton/python/test/capybara/bench/capybara_kernels/tsne_repulsion.py`
- Structs:
  - `/home/zhuochen/capybara-triton/python/test/capybara/test_struct_e2e.py`
  - `/home/zhuochen/capybara-triton/python/test/capybara/test_first_class_struct.py`
- Thread fences:
  - `/home/zhuochen/capybara-triton/python/test/capybara/bench/capybara_kernels/tsne_summarize.py`

---

## 17) Canonical design docs (secondary to source)

Use for conceptual context, then verify against source behavior:

- `/home/zhuochen/capybara-triton/docs/capybara/README.md`
- `/home/zhuochen/capybara-triton/docs/capybara/ONBOARDING.md`
- `/home/zhuochen/capybara-triton/docs/capybara/context/EXAMPLE_DSL.md`
- `/home/zhuochen/capybara-triton/docs/capybara/design/04_DESIGN_PROGRAMMING_MODEL.md`
- `/home/zhuochen/capybara-triton/docs/capybara/design/05_DESIGN_TYPES_MEMORY_MODEL.md`
- `/home/zhuochen/capybara-triton/docs/capybara/design/06_DESIGN_VIEWS_LAYOUT.md`
- `/home/zhuochen/capybara-triton/docs/capybara/design/09_DESIGN_FRONTEND_LOWERING.md`

Note:
- Some secondary docs/examples may still mention legacy `padded_view()` wording.
  For current behavior, treat source code (`exprs.py`/`view_construction.py`) as authoritative.

---

## 18) Final instruction for any AI using this file

If you are porting a PhysX kernel and you are not certain a translation step is apple-to-apple:

1. Stop immediately.
2. Explain uncertainty and risk.
3. Provide options.
4. Ask the user to choose.

Never guess silently.

