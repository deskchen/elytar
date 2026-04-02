"""Capybara DSL port of gpusimulationcontroller/CUDA/algorithms.cu — ALL kernels.

Ported kernels (matching CUDA names for PTX replacement):
  - reorderKernel                  — pure gather on float4 arrays
  - scanPerBlockKernel             — Hillis-Steele warp scan (PxU32)
  - scanPerBlockKernel4x4          — Hillis-Steele warp scan (int4x4 = 16 ints)
  - addBlockSumsKernel             — post-scan block-sum fixup (PxU32)
  - addBlockSumsKernel4x4          — post-scan block-sum fixup (int4x4)
  - radixFourBitCountPerBlockKernel — radix count per block (packed u64 scan)
  - radixFourBitReorderKernel       — radix reorder (scatter)

ABI differences from CUDA:
  - NULL pointer args replaced by integer flags + always-valid dummy tensors.
  - int4x4 passed as int32[N, 16] flat tensors.
  - int4 components accessed via offset math: int4_i component_j = [idx, 4*i+j].
  - Host must launch with the BLOCK_SIZE matching __launch_bounds__.

Capybara structural notes:
  - block.barrier() must be BETWEEN thread regions (not inside).
  - cp.disjoint() required for smem writes inside block.threads().
  - cp.assume_uniform() required for shfl_up inside warp-uniform conditionals.
  - Ternary (not if) for conditional loads before shfl_up to avoid divergence errors.
"""

import capybara as cp

WARP_SIZE = 32


# ===== reorderKernel =====
@cp.kernel
def reorderKernel(data, reordered, length, reorderedToOriginalMap,
                  BLOCK_SIZE: cp.constexpr = 1024):
    with cp.Kernel(cp.ceildiv(length, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gid = bx * BLOCK_SIZE + tid
            if gid < length:
                src = reorderedToOriginalMap[gid]
                reordered[gid, 0] = data[src, 0]
                reordered[gid, 1] = data[src, 1]
                reordered[gid, 2] = data[src, 2]
                reordered[gid, 3] = data[src, 3]


# ===== scanPerBlockKernel (PxU32) =====
@cp.kernel
def scanPerBlockKernel(data, result, partialSums, length, exclusiveScan,
                       totalSum, has_total_sum, has_result,
                       BLOCK_SIZE: cp.constexpr = 1024,
                       NUM_WARPS: cp.constexpr = 32):
    with cp.Kernel(cp.ceildiv(length, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        vals = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        sums = block.alloc((NUM_WARPS,), dtype=cp.int32)

        # Phase 1: Load + warp scan + store to smem
        for tid, thread in block.threads():
            gid = bx * BLOCK_SIZE + tid
            lane_id = tid % cp.int32(WARP_SIZE)
            warp_id = tid // cp.int32(WARP_SIZE)
            safe_gid = gid if gid < length else cp.int32(0)
            raw = data[safe_gid] + cp.int32(0)
            value = raw if gid < length else cp.int32(0)
            n = thread.shfl_up(value, 1)
            if lane_id >= cp.int32(1):
                value = value + n
            n = thread.shfl_up(value, 2)
            if lane_id >= cp.int32(2):
                value = value + n
            n = thread.shfl_up(value, 4)
            if lane_id >= cp.int32(4):
                value = value + n
            n = thread.shfl_up(value, 8)
            if lane_id >= cp.int32(8):
                value = value + n
            n = thread.shfl_up(value, 16)
            if lane_id >= cp.int32(16):
                value = value + n
            vals[tid] = cp.disjoint(value)
            if lane_id == cp.int32(WARP_SIZE - 1):
                sums[warp_id] = cp.disjoint(value)

        block.barrier()

        # Phase 2: Warp 0 scans the warp sums
        for tid, thread in block.threads():
            if cp.assume_uniform(tid < cp.int32(NUM_WARPS)):
                ws = sums[tid] + cp.int32(0)
                n = thread.shfl_up(ws, 1)
                if tid >= cp.int32(1):
                    ws = ws + n
                n = thread.shfl_up(ws, 2)
                if tid >= cp.int32(2):
                    ws = ws + n
                n = thread.shfl_up(ws, 4)
                if tid >= cp.int32(4):
                    ws = ws + n
                n = thread.shfl_up(ws, 8)
                if tid >= cp.int32(8):
                    ws = ws + n
                n = thread.shfl_up(ws, 16)
                if tid >= cp.int32(16):
                    ws = ws + n
                sums[tid] = cp.disjoint(ws)

        block.barrier()

        # Phase 3: Uniform add + write output
        for tid, thread in block.threads():
            gid = bx * BLOCK_SIZE + tid
            warp_id = tid // cp.int32(WARP_SIZE)
            value = vals[tid] + cp.int32(0)
            if warp_id > cp.int32(0):
                value = value + sums[warp_id - cp.int32(1)]
            if tid == cp.int32(BLOCK_SIZE - 1):
                partialSums[bx] = value
            if has_total_sum != cp.int32(0):
                if gid == length - cp.int32(1):
                    totalSum[0] = value
            if has_result != cp.int32(0):
                if gid < length:
                    if exclusiveScan == cp.int32(0):
                        result[gid] = value
                    else:
                        if tid + cp.int32(1) < cp.int32(BLOCK_SIZE):
                            if gid + cp.int32(1) < length:
                                result[gid + cp.int32(1)] = value
                        if tid == cp.int32(0):
                            result[gid] = cp.int32(0)


# ===== addBlockSumsKernel (PxU32) =====
@cp.kernel
def addBlockSumsKernel(partialSums, data, length, totalSum,
                       has_total_sum, has_data,
                       BLOCK_SIZE: cp.constexpr = 1024):
    with cp.Kernel(cp.ceildiv(length, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gid = bx * BLOCK_SIZE + tid
            if gid < length:
                ps = partialSums[bx] + cp.int32(0)
                if has_total_sum != cp.int32(0):
                    if gid == length - cp.int32(1):
                        totalSum[0] = totalSum[0] + ps
                if has_data != cp.int32(0):
                    data[gid] = data[gid] + ps


# ===== addBlockSumsKernel4x4 =====
# int4x4 passed as int32[N, 16]. 16 components = 4 int4 × 4 (x,y,z,w).
# Also calls exclusiveSumInt16 conditionally at end.
@cp.kernel
def addBlockSumsKernel4x4(partialSums, data, length, totalSum,
                           has_total_sum, has_data, is_last_block,
                           BLOCK_SIZE: cp.constexpr = 1024):
    with cp.Kernel(cp.ceildiv(length, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        # Phase 1: Element-wise add of 16 components
        for tid, thread in block.threads():
            gid = bx * BLOCK_SIZE + tid
            if has_total_sum != cp.int32(0):
                if gid == length - cp.int32(1):
                    for _c in range(16):
                        totalSum[_c] = totalSum[_c] + partialSums[bx, _c]
            if has_data != cp.int32(0):
                if gid < length:
                    for _c in range(16):
                        data[gid, _c] = data[gid, _c] + partialSums[bx, _c]

        # exclusiveSumInt16 on totalSum (only when is_last_block and has_total_sum)
        # Phase 2: Load 16 values from totalSum, warp scan
        es_vals = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        for tid, thread in block.threads():
            v = cp.int32(0)
            if is_last_block != cp.int32(0):
                if has_total_sum != cp.int32(0):
                    if tid < cp.int32(16):
                        v = totalSum[tid] + cp.int32(0)
            n = thread.shfl_up(v, 1)
            if tid >= cp.int32(1):
                v = v + n
            n = thread.shfl_up(v, 2)
            if tid >= cp.int32(2):
                v = v + n
            n = thread.shfl_up(v, 4)
            if tid >= cp.int32(4):
                v = v + n
            n = thread.shfl_up(v, 8)
            if tid >= cp.int32(8):
                v = v + n
            n = thread.shfl_up(v, 16)
            if tid >= cp.int32(16):
                v = v + n
            es_vals[tid] = cp.disjoint(v)

        block.barrier()

        # Phase 3: Write shifted results (exclusive sum)
        for tid, thread in block.threads():
            if is_last_block != cp.int32(0):
                if has_total_sum != cp.int32(0):
                    if tid < cp.int32(15):
                        totalSum[tid + cp.int32(1)] = es_vals[tid]
                    if tid == cp.int32(15):
                        totalSum[0] = cp.int32(0)


# ===== scanPerBlockKernel4x4 =====
# int4x4 = 16 ints = 4 int4 vectors. Each int4 is scanned separately (component-wise).
# CUDA does `for i in 0..3: scanPerBlock(data[id].data[i], ...)` — 4 passes.
# Each pass has 3 barrier-separated phases. Total: 12 phases + exclusiveSumInt16 (3 phases).
# int4x4 tensors: int32[N, 16] where offset 4*i+j = i-th int4, j-th component.
@cp.kernel
def scanPerBlockKernel4x4(data, result, partialSums, length, exclusiveScan,
                           totalSum, has_total_sum, has_result,
                           is_single_block,
                           BLOCK_SIZE: cp.constexpr = 512,
                           NUM_WARPS: cp.constexpr = 16):
    with cp.Kernel(cp.ceildiv(length, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        # Shared memory for int4 scan (4 components at a time)
        vals_0 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        vals_1 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        vals_2 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        vals_3 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        sums_0 = block.alloc((NUM_WARPS,), dtype=cp.int32)
        sums_1 = block.alloc((NUM_WARPS,), dtype=cp.int32)
        sums_2 = block.alloc((NUM_WARPS,), dtype=cp.int32)
        sums_3 = block.alloc((NUM_WARPS,), dtype=cp.int32)

        # First: handle exclusive scan output zero-fill (tid==0 of each block)
        for tid, thread in block.threads():
            gid = bx * BLOCK_SIZE + tid
            if has_result != cp.int32(0):
                if exclusiveScan != cp.int32(0):
                    if gid < length:
                        if tid == cp.int32(0):
                            for _c in range(16):
                                result[gid, _c] = cp.int32(0)

        block.barrier()

        # ---- Repeat scan for each of the 4 int4 vectors (i=0..3) ----
        # Unrolled: each iteration is 3 phases.
        for _int4_idx in range(4):
            # Phase A: Load 4 components + warp scan + save to smem
            for tid, thread in block.threads():
                gid = bx * BLOCK_SIZE + tid
                lane_id = tid % cp.int32(WARP_SIZE)
                warp_id = tid // cp.int32(WARP_SIZE)
                safe_gid = gid if gid < length else cp.int32(0)
                c0 = data[safe_gid, _int4_idx * 4 + 0] + cp.int32(0)
                c1 = data[safe_gid, _int4_idx * 4 + 1] + cp.int32(0)
                c2 = data[safe_gid, _int4_idx * 4 + 2] + cp.int32(0)
                c3 = data[safe_gid, _int4_idx * 4 + 3] + cp.int32(0)
                c0 = c0 if gid < length else cp.int32(0)
                c1 = c1 if gid < length else cp.int32(0)
                c2 = c2 if gid < length else cp.int32(0)
                c3 = c3 if gid < length else cp.int32(0)
                # Warp scan each component (5 rounds, manually unrolled)
                n0 = thread.shfl_up(c0, 1)
                n1 = thread.shfl_up(c1, 1)
                n2 = thread.shfl_up(c2, 1)
                n3 = thread.shfl_up(c3, 1)
                if lane_id >= cp.int32(1):
                    c0 = c0 + n0
                    c1 = c1 + n1
                    c2 = c2 + n2
                    c3 = c3 + n3
                n0 = thread.shfl_up(c0, 2)
                n1 = thread.shfl_up(c1, 2)
                n2 = thread.shfl_up(c2, 2)
                n3 = thread.shfl_up(c3, 2)
                if lane_id >= cp.int32(2):
                    c0 = c0 + n0
                    c1 = c1 + n1
                    c2 = c2 + n2
                    c3 = c3 + n3
                n0 = thread.shfl_up(c0, 4)
                n1 = thread.shfl_up(c1, 4)
                n2 = thread.shfl_up(c2, 4)
                n3 = thread.shfl_up(c3, 4)
                if lane_id >= cp.int32(4):
                    c0 = c0 + n0
                    c1 = c1 + n1
                    c2 = c2 + n2
                    c3 = c3 + n3
                n0 = thread.shfl_up(c0, 8)
                n1 = thread.shfl_up(c1, 8)
                n2 = thread.shfl_up(c2, 8)
                n3 = thread.shfl_up(c3, 8)
                if lane_id >= cp.int32(8):
                    c0 = c0 + n0
                    c1 = c1 + n1
                    c2 = c2 + n2
                    c3 = c3 + n3
                n0 = thread.shfl_up(c0, 16)
                n1 = thread.shfl_up(c1, 16)
                n2 = thread.shfl_up(c2, 16)
                n3 = thread.shfl_up(c3, 16)
                if lane_id >= cp.int32(16):
                    c0 = c0 + n0
                    c1 = c1 + n1
                    c2 = c2 + n2
                    c3 = c3 + n3
                vals_0[tid] = cp.disjoint(c0)
                vals_1[tid] = cp.disjoint(c1)
                vals_2[tid] = cp.disjoint(c2)
                vals_3[tid] = cp.disjoint(c3)
                if lane_id == cp.int32(WARP_SIZE - 1):
                    sums_0[warp_id] = cp.disjoint(c0)
                    sums_1[warp_id] = cp.disjoint(c1)
                    sums_2[warp_id] = cp.disjoint(c2)
                    sums_3[warp_id] = cp.disjoint(c3)

            block.barrier()

            # Phase B: Warp 0 scans warp sums (4 components)
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(NUM_WARPS)):
                    w0 = sums_0[tid] + cp.int32(0)
                    w1 = sums_1[tid] + cp.int32(0)
                    w2 = sums_2[tid] + cp.int32(0)
                    w3 = sums_3[tid] + cp.int32(0)
                    n0 = thread.shfl_up(w0, 1)
                    n1 = thread.shfl_up(w1, 1)
                    n2 = thread.shfl_up(w2, 1)
                    n3 = thread.shfl_up(w3, 1)
                    if tid >= cp.int32(1):
                        w0 = w0 + n0
                        w1 = w1 + n1
                        w2 = w2 + n2
                        w3 = w3 + n3
                    n0 = thread.shfl_up(w0, 2)
                    n1 = thread.shfl_up(w1, 2)
                    n2 = thread.shfl_up(w2, 2)
                    n3 = thread.shfl_up(w3, 2)
                    if tid >= cp.int32(2):
                        w0 = w0 + n0
                        w1 = w1 + n1
                        w2 = w2 + n2
                        w3 = w3 + n3
                    n0 = thread.shfl_up(w0, 4)
                    n1 = thread.shfl_up(w1, 4)
                    n2 = thread.shfl_up(w2, 4)
                    n3 = thread.shfl_up(w3, 4)
                    if tid >= cp.int32(4):
                        w0 = w0 + n0
                        w1 = w1 + n1
                        w2 = w2 + n2
                        w3 = w3 + n3
                    n0 = thread.shfl_up(w0, 8)
                    n1 = thread.shfl_up(w1, 8)
                    n2 = thread.shfl_up(w2, 8)
                    n3 = thread.shfl_up(w3, 8)
                    if tid >= cp.int32(8):
                        w0 = w0 + n0
                        w1 = w1 + n1
                        w2 = w2 + n2
                        w3 = w3 + n3
                    n0 = thread.shfl_up(w0, 16)
                    n1 = thread.shfl_up(w1, 16)
                    n2 = thread.shfl_up(w2, 16)
                    n3 = thread.shfl_up(w3, 16)
                    if tid >= cp.int32(16):
                        w0 = w0 + n0
                        w1 = w1 + n1
                        w2 = w2 + n2
                        w3 = w3 + n3
                    sums_0[tid] = cp.disjoint(w0)
                    sums_1[tid] = cp.disjoint(w1)
                    sums_2[tid] = cp.disjoint(w2)
                    sums_3[tid] = cp.disjoint(w3)

            block.barrier()

            # Phase C: Uniform add + write output
            for tid, thread in block.threads():
                gid = bx * BLOCK_SIZE + tid
                warp_id = tid // cp.int32(WARP_SIZE)
                c0 = vals_0[tid] + cp.int32(0)
                c1 = vals_1[tid] + cp.int32(0)
                c2 = vals_2[tid] + cp.int32(0)
                c3 = vals_3[tid] + cp.int32(0)
                if warp_id > cp.int32(0):
                    prev = warp_id - cp.int32(1)
                    c0 = c0 + sums_0[prev]
                    c1 = c1 + sums_1[prev]
                    c2 = c2 + sums_2[prev]
                    c3 = c3 + sums_3[prev]
                # Write partial sums for this block
                if tid == cp.int32(BLOCK_SIZE - 1):
                    partialSums[bx, _int4_idx * 4 + 0] = c0
                    partialSums[bx, _int4_idx * 4 + 1] = c1
                    partialSums[bx, _int4_idx * 4 + 2] = c2
                    partialSums[bx, _int4_idx * 4 + 3] = c3
                # Write totalSum
                if has_total_sum != cp.int32(0):
                    if gid == length - cp.int32(1):
                        totalSum[_int4_idx * 4 + 0] = c0
                        totalSum[_int4_idx * 4 + 1] = c1
                        totalSum[_int4_idx * 4 + 2] = c2
                        totalSum[_int4_idx * 4 + 3] = c3
                # Write result
                if has_result != cp.int32(0):
                    if gid < length:
                        off = _int4_idx * 4
                        if exclusiveScan == cp.int32(0):
                            result[gid, off + 0] = c0
                            result[gid, off + 1] = c1
                            result[gid, off + 2] = c2
                            result[gid, off + 3] = c3
                        else:
                            if tid + cp.int32(1) < cp.int32(BLOCK_SIZE):
                                if gid + cp.int32(1) < length:
                                    result[gid + cp.int32(1), off + 0] = c0
                                    result[gid + cp.int32(1), off + 1] = c1
                                    result[gid + cp.int32(1), off + 2] = c2
                                    result[gid + cp.int32(1), off + 3] = c3

            block.barrier()

        # exclusiveSumInt16 on totalSum (only when single block)
        es_vals = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        for tid, thread in block.threads():
            v = cp.int32(0)
            if is_single_block != cp.int32(0):
                if has_total_sum != cp.int32(0):
                    if tid < cp.int32(16):
                        v = totalSum[tid] + cp.int32(0)
            n = thread.shfl_up(v, 1)
            if tid >= cp.int32(1):
                v = v + n
            n = thread.shfl_up(v, 2)
            if tid >= cp.int32(2):
                v = v + n
            n = thread.shfl_up(v, 4)
            if tid >= cp.int32(4):
                v = v + n
            n = thread.shfl_up(v, 8)
            if tid >= cp.int32(8):
                v = v + n
            n = thread.shfl_up(v, 16)
            if tid >= cp.int32(16):
                v = v + n
            es_vals[tid] = cp.disjoint(v)

        block.barrier()

        for tid, thread in block.threads():
            if is_single_block != cp.int32(0):
                if has_total_sum != cp.int32(0):
                    if tid < cp.int32(15):
                        totalSum[tid + cp.int32(1)] = es_vals[tid]
                    if tid == cp.int32(15):
                        totalSum[0] = cp.int32(0)


# ===== radixFourBitCountPerBlockKernel =====
# Packs 6 histogram bins into a u64 (10 bits each), scans via scanPerBlock<u64>,
# then unpacks. 3 iterations for 16 bins total.
# partialSums/totalSum are int4x4 = int32[N,16], accessed flat.
@cp.kernel
def radixFourBitCountPerBlockKernel(data, offsetsPerWarp, passIndex,
                                     partialSums, length, totalSum,
                                     has_total_sum,
                                     BLOCK_SIZE: cp.constexpr = 512,
                                     NUM_WARPS: cp.constexpr = 16):
    with cp.Kernel(cp.ceildiv(length, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        vals64 = block.alloc((BLOCK_SIZE,), dtype=cp.int64)
        sums64 = block.alloc((NUM_WARPS,), dtype=cp.int64)

        for _iter in range(3):
            # Phase 1: Compute packed value + warp scan (u64) + save to smem
            for tid, thread in block.threads():
                gid = bx * BLOCK_SIZE + tid
                lane_id = tid % cp.int32(WARP_SIZE)
                warp_id = tid // cp.int32(WARP_SIZE)
                slot = cp.int32(0)
                if gid < length:
                    raw = data[gid] + cp.int32(0)
                    slot = (raw >> (passIndex * cp.int32(4))) & cp.int32(15)
                # Adjust slot for this iteration
                adj_slot = slot - cp.int32(_iter * 6)
                value = cp.int64(0)
                if gid < length:
                    if adj_slot < cp.int32(6):
                        if adj_slot >= cp.int32(0):
                            value = cp.int64(1) << (cp.int64(adj_slot) * cp.int64(10))
                # Warp scan on u64
                n = thread.shfl_up(value, 1)
                if lane_id >= cp.int32(1):
                    value = value + n
                n = thread.shfl_up(value, 2)
                if lane_id >= cp.int32(2):
                    value = value + n
                n = thread.shfl_up(value, 4)
                if lane_id >= cp.int32(4):
                    value = value + n
                n = thread.shfl_up(value, 8)
                if lane_id >= cp.int32(8):
                    value = value + n
                n = thread.shfl_up(value, 16)
                if lane_id >= cp.int32(16):
                    value = value + n
                vals64[tid] = cp.disjoint(value)
                if lane_id == cp.int32(WARP_SIZE - 1):
                    sums64[warp_id] = cp.disjoint(value)

            block.barrier()

            # Phase 2: Warp 0 scans warp sums (u64)
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(NUM_WARPS)):
                    ws = sums64[tid] + cp.int64(0)
                    n = thread.shfl_up(ws, 1)
                    if tid >= cp.int32(1):
                        ws = ws + n
                    n = thread.shfl_up(ws, 2)
                    if tid >= cp.int32(2):
                        ws = ws + n
                    n = thread.shfl_up(ws, 4)
                    if tid >= cp.int32(4):
                        ws = ws + n
                    n = thread.shfl_up(ws, 8)
                    if tid >= cp.int32(8):
                        ws = ws + n
                    n = thread.shfl_up(ws, 16)
                    if tid >= cp.int32(16):
                        ws = ws + n
                    sums64[tid] = cp.disjoint(ws)

            block.barrier()

            # Phase 3: Uniform add + unpack + write
            for tid, thread in block.threads():
                gid = bx * BLOCK_SIZE + tid
                warp_id = tid // cp.int32(WARP_SIZE)
                value = vals64[tid] + cp.int64(0)
                if warp_id > cp.int32(0):
                    value = value + sums64[warp_id - cp.int32(1)]
                # Last thread writes partialSums (unpacked 10-bit bins)
                if tid == cp.int32(BLOCK_SIZE - 1):
                    partial = value
                    partialSums[bx, _iter * 6 + 0] = cp.int32(partial & cp.int64(0x3FF))
                    partialSums[bx, _iter * 6 + 1] = cp.int32((partial >> cp.int64(10)) & cp.int64(0x3FF))
                    partialSums[bx, _iter * 6 + 2] = cp.int32((partial >> cp.int64(20)) & cp.int64(0x3FF))
                    partialSums[bx, _iter * 6 + 3] = cp.int32((partial >> cp.int64(30)) & cp.int64(0x3FF))
                    if _iter < 2:
                        partialSums[bx, _iter * 6 + 4] = cp.int32((partial >> cp.int64(40)) & cp.int64(0x3FF))
                        partialSums[bx, _iter * 6 + 5] = cp.int32((partial >> cp.int64(50)) & cp.int64(0x3FF))
                # Last element writes totalSum
                if has_total_sum != cp.int32(0):
                    if gid == length - cp.int32(1):
                        totalSum[_iter * 6 + 0] = cp.int32(value & cp.int64(0x3FF))
                        totalSum[_iter * 6 + 1] = cp.int32((value >> cp.int64(10)) & cp.int64(0x3FF))
                        totalSum[_iter * 6 + 2] = cp.int32((value >> cp.int64(20)) & cp.int64(0x3FF))
                        totalSum[_iter * 6 + 3] = cp.int32((value >> cp.int64(30)) & cp.int64(0x3FF))
                        if _iter < 2:
                            totalSum[_iter * 6 + 4] = cp.int32((value >> cp.int64(40)) & cp.int64(0x3FF))
                            totalSum[_iter * 6 + 5] = cp.int32((value >> cp.int64(50)) & cp.int64(0x3FF))
                # Write per-thread offset
                slot = cp.int32(0)
                if gid < length:
                    raw = data[gid] + cp.int32(0)
                    slot = (raw >> (passIndex * cp.int32(4))) & cp.int32(15)
                adj_slot = slot - cp.int32(_iter * 6)
                if gid < length:
                    if adj_slot < cp.int32(6):
                        if adj_slot >= cp.int32(0):
                            offset_val = cp.int32((value >> (cp.int64(adj_slot) * cp.int64(10))) & cp.int64(0x3FF))
                            offsetsPerWarp[gid] = offset_val - cp.int32(1)

            block.barrier()


# ===== radixFourBitReorderKernel =====
# Simple scatter using precomputed offsets + cumulative sums.
# partialSums/cumulativeSum are int4x4 = int32[N, 16], accessed flat.
@cp.kernel
def radixFourBitReorderKernel(data, offsetsPerWarp, reordered, passIndex,
                               partialSums, length, cumulativeSum,
                               dependentData, dependentDataReordered,
                               has_dependent,
                               BLOCK_SIZE: cp.constexpr = 1024):
    with cp.Kernel(cp.ceildiv(length, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gid = bx * BLOCK_SIZE + tid
            if gid < length:
                raw = data[gid] + cp.int32(0)
                slot = (raw >> (passIndex * cp.int32(4))) & cp.int32(15)
                cum = cumulativeSum[slot] + cp.int32(0)
                off = offsetsPerWarp[gid] + cp.int32(0)
                ps = partialSums[bx, slot] + cp.int32(0)
                new_idx = cum + off + ps
                if new_idx < length:
                    reordered[new_idx] = raw
                    if has_dependent != cp.int32(0):
                        dep = cp.int32(gid) if passIndex == cp.int32(0) else dependentData[gid] + cp.int32(0)
                        dependentDataReordered[new_idx] = dep
