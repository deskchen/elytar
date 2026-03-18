# MemCopyBalanced Port Blocker (Detailed)

This note explains, in concrete terms, why the current Capybara port of
`MemCopyBalanced` does not yet have the same input/output ABI as the original
CUDA kernel, and what needs to be verified/fixed.

Relevant files:

- Original CUDA: `/home/zhuochen/elytar/physx-5.6.1/source/gpucommon/src/CUDA/MemCopyBalanced.cu`
- Current Capybara port: `/home/zhuochen/elytar/physx-5.6.1/source/gpucommon/src/capybara/MemCopyBalanced.py`
- Guideline: `/home/zhuochen/elytar/docs/DSL_GRAMMAR_PHYSX_PORT.md`

---

## 1) What "same input/output" means here

For this kernel, "same input/output" means:

1. Same launched kernel name: `MemCopyBalanced`
2. Same argument list shape:
   - CUDA: `MemCopyBalanced(PxgCopyManager::CopyDesc* desc, PxU32 count)`
3. Same meaning of `desc` fields:
   - `source` and `dest` are raw addresses (`size_t`)
4. Same behavior:
   - Reads words from memory at `source + a`
   - Writes words to memory at `dest + a`

In the current Python port, the signature was changed to:

- `MemCopyBalanced(desc, src_words, dst_words, count, ...)`

That is functionally useful, but ABI-different.

---

## 2) The exact technical gap

Inside CUDA (`MemCopyBalanced.cu`), the core lines are:

```cpp
PxU32* srcPtr = reinterpret_cast<PxU32*>(copyDesc[warpIdxInBlock].source);
PxU32* dstPtr = reinterpret_cast<PxU32*>(copyDesc[warpIdxInBlock].dest);
PxU32 sourceVal = srcPtr[a];
dstPtr[a] = sourceVal;
```

This requires **u64 address -> typed pointer dereference**.

In current high-level Capybara DSL usage, reads/writes are normally done through
tensors/views passed as kernel arguments, not by dereferencing arbitrary `u64`
addresses at runtime.

So the blocker is not `cp.struct` itself (that part exists), but the pointer
dereference semantics from struct `u64` fields.

---

## 3) Simplified examples

### Example A: CUDA behavior (raw pointer ABI)

Given:

- `desc[0].source = 0x1000` (address of src words)
- `desc[0].dest   = 0x2000` (address of dst words)
- `desc[0].bytes  = 64`

Kernel behavior:

- Treat `0x1000` as `uint32_t* srcPtr`
- Treat `0x2000` as `uint32_t* dstPtr`
- Copy 16 words (`64 / 4`) from `srcPtr[a]` to `dstPtr[a]`

No extra explicit `src_words` / `dst_words` kernel args are required.

### Example B: Current Python port behavior (offset model)

Given:

- `desc[0].source = 1024`
- `desc[0].dest = 2048`
- `src_words` and `dst_words` are explicit tensors

Kernel behavior:

- Uses `src_words[source + a]`
- Writes `dst_words[dest + a]`

This is easier to compile/test in DSL, but **not** the same ABI as CUDA.

---

## 4) What is already equivalent

These parts are already ported with matching control flow intent:

- 2D thread decomposition (`threadIdx.y`, `threadIdx.x`) modeled with
  `block.threads(COPY_KERNEL_WARPS_PER_BLOCK, 32)`
- One descriptor copied per warp by lane 0
- Warp sync point between descriptor load and copy loop
- Strided copy loop with `a += WARP_SIZE * warpsPerBlock`
- `clampMaxValue` and `clampMaxValues` logic

---

## 5) What still needs a decision/implementation

To make ABI truly match, we need one of these:

1. A proven Capybara path for:
   - `u64 address -> typed pointer -> load/store`
2. A project-approved custom low-level intrinsic route that implements this
3. An explicit decision that this file remains offset/tensor-based (known ABI
   difference)

Per the guideline ("Zero-Guessing Rule"), this cannot be guessed silently.

---

## 6) Checklist for your report

When reporting back, these are the key points to validate:

- [ ] Is exact CUDA kernel signature required for PTX replacement?
- [ ] Is there existing Capybara support for `u64` pointer dereference from
      struct fields?
- [ ] If yes, what API/pattern should be used?
- [ ] If no, is custom intrinsic path approved?
- [ ] If neither, is offset/tensor ABI acceptable for this stage?

---

## 7) Minimal "same ABI" target for this file

Final target should look like:

- Python kernel name: `MemCopyBalanced`
- Python args: `(desc, count, ...)` only (plus constexpr launch params)
- `desc` remains `@cp.struct` mirroring `CopyDesc`
- Internal copy uses raw-address semantics equivalent to CUDA

Until pointer dereference path is confirmed, this remains the open blocker.
