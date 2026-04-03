"""Capybara DSL port of gpucommon/CUDA/radixSortImpl.cu — all 12 kernels.

Ported kernels (matching CUDA names for PTX replacement):
  Group A — Copy kernels (trivial grid-stride gather/scatter):
    - radixSortCopyHigh32Bits
    - radixSortDoubleCopyHigh32Bits
    - radixSortCopy
    - radixSortDoubleCopy
    - radixSortCopyBits2
    - radixSortCopy2

  Group B — radixSortSingleBlock wrappers (histogram + warp scan):
    - radixSortMultiBlockLaunch
    - radixSortMultiBlockLaunchWithoutCount
    - radixSortMultiBlockLaunchWithCount

  Group C — radixSortCalculateRanks wrappers (reorder pass):
    - radixSortMultiCalculateRanksLaunch
    - radixSortMultiCalculateRanksLaunchWithoutCount
    - radixSortMultiCalculateRanksLaunchWithCount

ABI differences from CUDA:
  - Descriptor decomposition: host resolves PxgRadixSortBlockDesc / PxgRadixSortDesc
    per blockIdx.y and passes flat tensors. Each kernel operates on a single descriptor.
  - numKeys always passed as scalar (host reads device ptr before launch).
  - PxU64 tensors -> int64[N]. PxU32 tensors -> int32[N].
  - uint4 vectorized loads -> int32[N, 4] tensor (4 keys per row).
  - radixCount -> int32[RADIX_SIZE * gridDim.x] flat tensor.
  - outputKeys / outputRanks -> int32[N].
  - All 6 Multi* kernels share the same body since numKeys is always scalar.

Constants from CUDA:
  - BLOCK_SIZE = 1024 (PxgRadixSortKernelBlockDim::RADIX_SORT)
  - NUM_WARPS = BLOCK_SIZE / 32 = 32
  - NB_BLOCKS = 32 (PxgRadixSortKernelGridDim::RADIX_SORT = gridDim.x)
  - RADIX_SIZE = 16
  - RADIX_ACCUM_SIZE = 8
"""

import capybara as cp

WARP_SIZE = 32


# ===========================================================================
# Group A: Copy kernels
# ===========================================================================

# ===== radixSortCopyHigh32Bits =====
# Copies high 32 bits of PxU64 values using rank indirection.
# inValue: int64[N], outValue: int32[N], rank: int32[N], numKeys: scalar
@cp.kernel
def radixSortCopyHigh32Bits(inValue, outValue, rank, numKeys,
                            BLOCK_SIZE: cp.constexpr = 256):
    with cp.Kernel(cp.ceildiv(numKeys, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            workIndex = bx * BLOCK_SIZE + tid
            if workIndex < numKeys:
                index = rank[workIndex] + cp.int32(0)
                val64 = inValue[index] + cp.int64(0)
                outValue[workIndex] = cp.int32(val64 >> cp.int64(32))


# ===== radixSortDoubleCopyHigh32Bits =====
# Two-stream version of CopyHigh32Bits.
@cp.kernel
def radixSortDoubleCopyHigh32Bits(inValue0, outValue0, rank0,
                                  inValue1, outValue1, rank1,
                                  numKeys,
                                  BLOCK_SIZE: cp.constexpr = 256):
    with cp.Kernel(cp.ceildiv(numKeys, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            workIndex = bx * BLOCK_SIZE + tid
            if workIndex < numKeys:
                index0 = rank0[workIndex] + cp.int32(0)
                val64_0 = inValue0[index0] + cp.int64(0)
                outValue0[workIndex] = cp.int32(val64_0 >> cp.int64(32))

                index1 = rank1[workIndex] + cp.int32(0)
                val64_1 = inValue1[index1] + cp.int64(0)
                outValue1[workIndex] = cp.int32(val64_1 >> cp.int64(32))


# ===== radixSortCopy =====
# Copies PxU64 values using rank indirection.
# inValue: int64[N], outValue: int64[N], rank: int32[N], numKeys: scalar
@cp.kernel
def radixSortCopy(inValue, outValue, rank, numKeys,
                  BLOCK_SIZE: cp.constexpr = 256):
    with cp.Kernel(cp.ceildiv(numKeys, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            workIndex = bx * BLOCK_SIZE + tid
            if workIndex < numKeys:
                index = rank[workIndex] + cp.int32(0)
                outValue[workIndex] = inValue[index] + cp.int64(0)


# ===== radixSortDoubleCopy =====
# Two-stream version of Copy.
@cp.kernel
def radixSortDoubleCopy(inValue0, outValue0, rank0,
                        inValue1, outValue1, rank1,
                        numKeys,
                        BLOCK_SIZE: cp.constexpr = 256):
    with cp.Kernel(cp.ceildiv(numKeys, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            workIndex = bx * BLOCK_SIZE + tid
            if workIndex < numKeys:
                index0 = rank0[workIndex] + cp.int32(0)
                outValue0[workIndex] = inValue0[index0] + cp.int64(0)

                index1 = rank1[workIndex] + cp.int32(0)
                outValue1[workIndex] = inValue1[index1] + cp.int64(0)


# ===== radixSortCopyBits2 =====
# Extracts hi or lo 32 bits of PxU64 with sentinel handling.
# inValue: int64[N], outValue: int32[N], rank: int32[N], numKeys: scalar, lowBit: scalar
@cp.kernel
def radixSortCopyBits2(inValue, outValue, rank, numKeys, lowBit,
                       BLOCK_SIZE: cp.constexpr = 256):
    with cp.Kernel(cp.ceildiv(numKeys, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gid = bx * BLOCK_SIZE + tid
            if gid < numKeys:
                index = rank[gid] + cp.int32(0)
                # sentinel check: 0xffffffff = -1 in int32
                if index == cp.int32(-1):
                    outValue[gid] = cp.int32(-1)
                else:
                    val64 = inValue[index] + cp.int64(0)
                    lo = cp.int32(val64 & cp.int64(0x00000000FFFFFFFF))
                    hi = cp.int32(val64 >> cp.int64(32))
                    result = lo if lowBit != cp.int32(0) else hi
                    outValue[gid] = result


# ===== radixSortCopy2 =====
# Copies PxU64 with aggregate sentinel (rank == 0xFFFFFFFF -> output all-ones).
# inValue: int64[N], outValue: int64[N], rank: int32[N], numKeys: scalar
@cp.kernel
def radixSortCopy2(inValue, outValue, rank, numKeys,
                   BLOCK_SIZE: cp.constexpr = 256):
    with cp.Kernel(cp.ceildiv(numKeys, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gid = bx * BLOCK_SIZE + tid
            if gid < numKeys:
                index = rank[gid] + cp.int32(0)
                # 0xffffffff = -1 in signed int32
                if index == cp.int32(-1):
                    outValue[gid] = cp.int64(-1)
                else:
                    outValue[gid] = inValue[index] + cp.int64(0)


# ===========================================================================
# Group B: radixSortSingleBlock — histogram + warp scan pass
# ===========================================================================
#
# This ports radixSortSingleBlock<NUM_WARPS> from RadixSort.cuh.
#
# The algorithm:
#   1. Each thread processes multiple uint4 keys (4 keys each) in a grid-stride loop.
#      For each key, extract 4-bit radix and accumulate into radixAccum[8] (packed
#      pairs of 16-bit counters, one per radix bucket pair).
#   2. Warp-level inclusive scan on each packed pair, then extract per-radix totals
#      and write to sRadixSum[RADIX_SIZE * NUM_WARPS].
#   3. Cross-warp exclusive scan (scanRadixWarps) and write results to gRadixCount.
#
# inputKeys: int32[N, 4] — uint4 vectorized keys (4 per row)
# inputRanks: int32[N, 4] — unused in this kernel but part of API
# numKeys: scalar — total number of individual keys
# startBit: scalar — which 4-bit nibble to sort on
# radixCount: int32[RADIX_SIZE * NB_BLOCKS] — output histogram
#
# NB_BLOCKS = 32 = gridDim.x (compile-time constant from PxgRadixSortKernelGridDim)

@cp.kernel
def radixSortMultiBlockLaunch(inputKeys, inputRanks, numKeys, startBit, radixCount,
                              BLOCK_SIZE: cp.constexpr = 1024,
                              NUM_WARPS: cp.constexpr = 32,
                              RADIX_SIZE: cp.constexpr = 16,
                              NB_BLOCKS: cp.constexpr = 32):
    with cp.Kernel(NB_BLOCKS, threads=BLOCK_SIZE) as (bx, block):
        # Shared memory: sRadixSum[RADIX_SIZE * NUM_WARPS]
        sRadixSum = block.alloc((RADIX_SIZE * NUM_WARPS,), dtype=cp.int32)

        # numKeys is total scalar keys. numKeysU4 = ceil(numKeys/4)
        numKeysU4 = (numKeys + cp.int32(3)) >> cp.int32(2)
        totalBlockRequired = (numKeysU4 + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterationPerBlock = (totalBlockRequired + cp.int32(NB_BLOCKS - 1)) // cp.int32(NB_BLOCKS)

        # Phase 1: Each thread accumulates radix histogram over its assigned keys.
        # radixAccum[8] packed pairs of 16-bit counters — stored in 8 shared arrays per thread.
        # Capybara limitation: cannot have per-thread register arrays of dynamic size.
        # We use 8 shared memory arrays of size BLOCK_SIZE as register file substitute.
        rAccum0 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum1 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum2 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum3 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum4 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum5 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum6 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum7 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)

        # Initialize accumulators to 0
        for tid, thread in block.threads():
            rAccum0[tid] = cp.disjoint(cp.int32(0))
            rAccum1[tid] = cp.disjoint(cp.int32(0))
            rAccum2[tid] = cp.disjoint(cp.int32(0))
            rAccum3[tid] = cp.disjoint(cp.int32(0))
            rAccum4[tid] = cp.disjoint(cp.int32(0))
            rAccum5[tid] = cp.disjoint(cp.int32(0))
            rAccum6[tid] = cp.disjoint(cp.int32(0))
            rAccum7[tid] = cp.disjoint(cp.int32(0))

        block.barrier()

        # Grid-stride loop over keys: radixSortWarp
        # inputKeyIndex = min(numIterationPerBlock * bx * BLOCK_SIZE, numKeysU4)
        # endIndex = min(inputKeyIndex + numIterationPerBlock * BLOCK_SIZE, numKeysU4)
        # count = endIndex - inputKeyIndex
        # Loop: for i = idx; i < count; i += WARP_SIZE * NUM_WARPS
        # Each iteration loads inputKeys[i + inputKeyIndex] as uint4.
        # stride = WARP_SIZE * NUM_WARPS = 32 * 32 = 1024 = BLOCK_SIZE
        # So iterations = count / BLOCK_SIZE = numIterationPerBlock (at most)
        for _iter in range(64):
            # Upper bound on numIterationPerBlock: ceil(ceil(N/4)/1024 / 32)
            # For very large N this could exceed 64, but that would be millions of keys.
            # Using 64 as a safe upper bound for typical PhysX usage.
            for tid, thread in block.threads():
                warpIndexInBlock = tid // cp.int32(WARP_SIZE)
                inputKeyStart = numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)
                inputKeyStart_clamped = inputKeyStart if inputKeyStart < numKeysU4 else numKeysU4
                endIndex_raw = inputKeyStart + numIterationPerBlock * cp.int32(BLOCK_SIZE)
                endIndex = endIndex_raw if endIndex_raw < numKeysU4 else numKeysU4
                count = endIndex - inputKeyStart_clamped

                loopIdx = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid
                if loopIdx < count:
                    gInputIdx = loopIdx + inputKeyStart_clamped
                    k0 = inputKeys[gInputIdx, 0] + cp.int32(0)
                    k1 = inputKeys[gInputIdx, 1] + cp.int32(0)
                    k2 = inputKeys[gInputIdx, 2] + cp.int32(0)
                    k3 = inputKeys[gInputIdx, 3] + cp.int32(0)

                    # sanitizeKeys: if gInputIdx*4 + {1,2,3} >= numKeys, set to 0xFFFFFFFF
                    baseScalarIdx = gInputIdx * cp.int32(4)
                    goodVals = numKeys - baseScalarIdx
                    # fallthrough switch: badVals = 4 - goodVals when goodVals < 4
                    if goodVals < cp.int32(4):
                        k3 = cp.int32(-1)
                    if goodVals < cp.int32(3):
                        k2 = cp.int32(-1)
                    if goodVals < cp.int32(2):
                        k1 = cp.int32(-1)

                    r0 = (k0 >> startBit) & cp.int32(0xF)
                    r1 = (k1 >> startBit) & cp.int32(0xF)
                    r2 = (k2 >> startBit) & cp.int32(0xF)
                    r3 = (k3 >> startBit) & cp.int32(0xF)

                    # Accumulate into packed pairs: radixAccum[bit/2] += (1 << ((radix - bit) << 4))
                    # bit=0: radixes 0,1 packed
                    acc = rAccum0[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(0)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(0)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(0)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(0)) << cp.int32(4)))
                    rAccum0[tid] = cp.disjoint(acc)

                    # bit=2
                    acc = rAccum1[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(2)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(2)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(2)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(2)) << cp.int32(4)))
                    rAccum1[tid] = cp.disjoint(acc)

                    # bit=4
                    acc = rAccum2[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(4)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(4)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(4)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(4)) << cp.int32(4)))
                    rAccum2[tid] = cp.disjoint(acc)

                    # bit=6
                    acc = rAccum3[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(6)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(6)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(6)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(6)) << cp.int32(4)))
                    rAccum3[tid] = cp.disjoint(acc)

                    # bit=8
                    acc = rAccum4[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(8)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(8)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(8)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(8)) << cp.int32(4)))
                    rAccum4[tid] = cp.disjoint(acc)

                    # bit=10
                    acc = rAccum5[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(10)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(10)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(10)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(10)) << cp.int32(4)))
                    rAccum5[tid] = cp.disjoint(acc)

                    # bit=12
                    acc = rAccum6[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(12)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(12)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(12)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(12)) << cp.int32(4)))
                    rAccum6[tid] = cp.disjoint(acc)

                    # bit=14
                    acc = rAccum7[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(14)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(14)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(14)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(14)) << cp.int32(4)))
                    rAccum7[tid] = cp.disjoint(acc)

            block.barrier()

        # Phase 2: Warp-level inclusive scan on each packed pair, then extract counts.
        # For each radix pair i (0,2,4,...14):
        #   accum = rAccum[i/2][tid]
        #   val = warpScanAdd<WARP_SIZE>(accum, accumValue=0) -> inclusive scan - accum + accum = exclusive + inclusive - original
        #   Actually warpScanAdd returns inclusive_scan - value argument. With value=0, returns inclusive scan.
        #   Wait: warpScanAdd(originalValue=accum, value=accumValue) returns inclusive_scan(originalValue) - value.
        #   With value=0 (the 'accumValue' parameter), it returns the inclusive scan of originalValue.
        #   val2 = shfl(val, WARP_SIZE-1) = total sum for this warp
        #   For threadIndexInWarp < 2: sRadixSum[(i+threadIndexInWarp)*NUM_WARPS + warpIndex] = (val2 >> (tiw*16)) & 0xFFFF
        # This writes per-warp totals for each radix bucket.

        # We process all 8 packed pairs (radixes 0..15) in unrolled fashion.
        # After this, sRadixSum[radix * NUM_WARPS + warp] = count of keys with this radix in this warp.

        # Temporary shared mem for warp scan results (reuse rAccum arrays as scratch)
        for _pair in range(8):
            for tid, thread in block.threads():
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                warpIndexInBlock = tid >> cp.int32(5)  # tid / 32

                # Load accumulator for this pair
                acc = cp.int32(0)
                if _pair == 0:
                    acc = rAccum0[tid] + cp.int32(0)
                elif _pair == 1:
                    acc = rAccum1[tid] + cp.int32(0)
                elif _pair == 2:
                    acc = rAccum2[tid] + cp.int32(0)
                elif _pair == 3:
                    acc = rAccum3[tid] + cp.int32(0)
                elif _pair == 4:
                    acc = rAccum4[tid] + cp.int32(0)
                elif _pair == 5:
                    acc = rAccum5[tid] + cp.int32(0)
                elif _pair == 6:
                    acc = rAccum6[tid] + cp.int32(0)
                elif _pair == 7:
                    acc = rAccum7[tid] + cp.int32(0)

                # Inclusive warp scan
                val = acc
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n

                # val is now inclusive prefix sum. warpScanAdd returns val - accumValue.
                # With accumValue=0, result = val (inclusive sum).
                # Get last lane value (total for warp)
                val2 = thread.shfl_idx(val, 31)

                # Extract lo and hi 16-bit counts: radix pair is (2*_pair, 2*_pair+1)
                # threadIndexInWarp < 2: sRadixSum[(i+tiw)*NUM_WARPS + warpIdx] = (val2 >> (tiw*16)) & 0xFFFF
                radixBase = cp.int32(_pair * 2)
                if threadIndexInWarp < cp.int32(2):
                    shifted = val2 >> (threadIndexInWarp << cp.int32(4))
                    count_val = shifted & cp.int32(0xFFFF)
                    sRadixSum[(radixBase + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint(count_val)

        block.barrier()

        # Phase 3: Cross-warp scan (scanRadixWarps<NUM_WARPS>) and write to gRadixCount.
        # Only warpIndexInBlock < NUM_WARPS/2 participates (first 16 warps = 512 threads).
        # sRadixSum has RADIX_SIZE * NUM_WARPS = 16 * 32 = 512 elements.
        # Each of the 512 threads handles one element.
        # scanRadixWarps: scan within groups of NUM_WARPS (32) consecutive elements.
        # radixIndex = threadIndexInWarp & (NUM_WARPS - 1) = threadIndexInWarp (since NUM_WARPS=32=WARP_SIZE)
        # This is an exclusive prefix sum within each group of 32 (= within a warp).
        # Then threads where (idx & (NUM_WARPS-1)) == NUM_WARPS-1 write to gRadixCount.

        for tid, thread in block.threads():
            warpIndexInBlock = tid >> cp.int32(5)
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
            if cp.assume_uniform(warpIndexInBlock < cp.int32(NUM_WARPS // 2)):
                # idx = tid (first 512 threads)
                originalValue = sRadixSum[tid] + cp.int32(0)

                # Inclusive scan within warp (group of 32)
                val = originalValue
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(1))
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(2))
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(4))
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(8))
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(16))
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n

                # scanRadixWarps returns val - value (exclusive scan), where value=0
                # So output = val - 0 = val (inclusive). But the original code passes value=0
                # and returns val - value = val - 0 = inclusive scan.
                # Wait, re-reading: scanRadixWarps returns val - value where value is the
                # *original* value parameter (not originalVal). value=0 in the call.
                # So it returns the inclusive scan result (since val = inclusive_scan(originalValue)).
                # Actually looking more carefully:
                #   int val = originalVal;
                #   for(...) val += shfl(val, idx-a) if radixIndex >= a
                #   return val - value;  // value is the 6th parameter = 0
                # So it returns inclusive_scan(originalVal). That's the exclusive sum for the NEXT element.
                # The write condition: (idx & (NUM_WARPS-1)) == (NUM_WARPS-1), i.e., last in group.
                # gRadixIndex = bx + idx/NUM_WARPS * gridDim.x = bx + radix * NB_BLOCKS
                # gRadixCount[gRadixIndex] = output = inclusive total for this radix in this block

                output = val  # inclusive scan result (= exclusive scan of value 0)

                if threadIndexInWarp == cp.int32(NUM_WARPS - 1):
                    # gRadixIndex = bx + (tid / NUM_WARPS) * NB_BLOCKS
                    radixIdx = tid >> cp.int32(5)  # tid / NUM_WARPS (within first 512 threads)
                    # Wait: warpIndexInBlock = tid/32 which IS the radix index since
                    # sRadixSum layout is [radix * NUM_WARPS + warpIdx] and the scan
                    # is within each warp (32 elements), so warp i in the scan corresponds
                    # to radix i.
                    gRadixIndex = bx + warpIndexInBlock * cp.int32(NB_BLOCKS)
                    radixCount[gRadixIndex] = output



# The 3 BlockLaunch variants have identical code since numKeys is always scalar.
# But they must be separate @cp.kernel functions for distinct PTX entry names.
# Capybara limitation: cannot call another @cp.kernel, so we duplicate the body.
@cp.kernel
def radixSortMultiBlockLaunchWithoutCount(inputKeys, inputRanks, numKeys, startBit, radixCount,
                                          BLOCK_SIZE: cp.constexpr = 1024,
                                          NUM_WARPS: cp.constexpr = 32,
                                          RADIX_SIZE: cp.constexpr = 16,
                                          NB_BLOCKS: cp.constexpr = 32):
    with cp.Kernel(NB_BLOCKS, threads=BLOCK_SIZE) as (bx, block):
        # Shared memory: sRadixSum[RADIX_SIZE * NUM_WARPS]
        sRadixSum = block.alloc((RADIX_SIZE * NUM_WARPS,), dtype=cp.int32)

        # numKeys is total scalar keys. numKeysU4 = ceil(numKeys/4)
        numKeysU4 = (numKeys + cp.int32(3)) >> cp.int32(2)
        totalBlockRequired = (numKeysU4 + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterationPerBlock = (totalBlockRequired + cp.int32(NB_BLOCKS - 1)) // cp.int32(NB_BLOCKS)

        # Phase 1: Each thread accumulates radix histogram over its assigned keys.
        # radixAccum[8] packed pairs of 16-bit counters — stored in 8 shared arrays per thread.
        # Capybara limitation: cannot have per-thread register arrays of dynamic size.
        # We use 8 shared memory arrays of size BLOCK_SIZE as register file substitute.
        rAccum0 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum1 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum2 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum3 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum4 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum5 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum6 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum7 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)

        # Initialize accumulators to 0
        for tid, thread in block.threads():
            rAccum0[tid] = cp.disjoint(cp.int32(0))
            rAccum1[tid] = cp.disjoint(cp.int32(0))
            rAccum2[tid] = cp.disjoint(cp.int32(0))
            rAccum3[tid] = cp.disjoint(cp.int32(0))
            rAccum4[tid] = cp.disjoint(cp.int32(0))
            rAccum5[tid] = cp.disjoint(cp.int32(0))
            rAccum6[tid] = cp.disjoint(cp.int32(0))
            rAccum7[tid] = cp.disjoint(cp.int32(0))

        block.barrier()

        # Grid-stride loop over keys: radixSortWarp
        # inputKeyIndex = min(numIterationPerBlock * bx * BLOCK_SIZE, numKeysU4)
        # endIndex = min(inputKeyIndex + numIterationPerBlock * BLOCK_SIZE, numKeysU4)
        # count = endIndex - inputKeyIndex
        # Loop: for i = idx; i < count; i += WARP_SIZE * NUM_WARPS
        # Each iteration loads inputKeys[i + inputKeyIndex] as uint4.
        # stride = WARP_SIZE * NUM_WARPS = 32 * 32 = 1024 = BLOCK_SIZE
        # So iterations = count / BLOCK_SIZE = numIterationPerBlock (at most)
        for _iter in range(64):
            # Upper bound on numIterationPerBlock: ceil(ceil(N/4)/1024 / 32)
            # For very large N this could exceed 64, but that would be millions of keys.
            # Using 64 as a safe upper bound for typical PhysX usage.
            for tid, thread in block.threads():
                warpIndexInBlock = tid // cp.int32(WARP_SIZE)
                inputKeyStart = numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)
                inputKeyStart_clamped = inputKeyStart if inputKeyStart < numKeysU4 else numKeysU4
                endIndex_raw = inputKeyStart + numIterationPerBlock * cp.int32(BLOCK_SIZE)
                endIndex = endIndex_raw if endIndex_raw < numKeysU4 else numKeysU4
                count = endIndex - inputKeyStart_clamped

                loopIdx = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid
                if loopIdx < count:
                    gInputIdx = loopIdx + inputKeyStart_clamped
                    k0 = inputKeys[gInputIdx, 0] + cp.int32(0)
                    k1 = inputKeys[gInputIdx, 1] + cp.int32(0)
                    k2 = inputKeys[gInputIdx, 2] + cp.int32(0)
                    k3 = inputKeys[gInputIdx, 3] + cp.int32(0)

                    # sanitizeKeys: if gInputIdx*4 + {1,2,3} >= numKeys, set to 0xFFFFFFFF
                    baseScalarIdx = gInputIdx * cp.int32(4)
                    goodVals = numKeys - baseScalarIdx
                    # fallthrough switch: badVals = 4 - goodVals when goodVals < 4
                    if goodVals < cp.int32(4):
                        k3 = cp.int32(-1)
                    if goodVals < cp.int32(3):
                        k2 = cp.int32(-1)
                    if goodVals < cp.int32(2):
                        k1 = cp.int32(-1)

                    r0 = (k0 >> startBit) & cp.int32(0xF)
                    r1 = (k1 >> startBit) & cp.int32(0xF)
                    r2 = (k2 >> startBit) & cp.int32(0xF)
                    r3 = (k3 >> startBit) & cp.int32(0xF)

                    # Accumulate into packed pairs: radixAccum[bit/2] += (1 << ((radix - bit) << 4))
                    # bit=0: radixes 0,1 packed
                    acc = rAccum0[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(0)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(0)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(0)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(0)) << cp.int32(4)))
                    rAccum0[tid] = cp.disjoint(acc)

                    # bit=2
                    acc = rAccum1[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(2)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(2)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(2)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(2)) << cp.int32(4)))
                    rAccum1[tid] = cp.disjoint(acc)

                    # bit=4
                    acc = rAccum2[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(4)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(4)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(4)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(4)) << cp.int32(4)))
                    rAccum2[tid] = cp.disjoint(acc)

                    # bit=6
                    acc = rAccum3[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(6)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(6)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(6)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(6)) << cp.int32(4)))
                    rAccum3[tid] = cp.disjoint(acc)

                    # bit=8
                    acc = rAccum4[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(8)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(8)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(8)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(8)) << cp.int32(4)))
                    rAccum4[tid] = cp.disjoint(acc)

                    # bit=10
                    acc = rAccum5[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(10)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(10)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(10)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(10)) << cp.int32(4)))
                    rAccum5[tid] = cp.disjoint(acc)

                    # bit=12
                    acc = rAccum6[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(12)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(12)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(12)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(12)) << cp.int32(4)))
                    rAccum6[tid] = cp.disjoint(acc)

                    # bit=14
                    acc = rAccum7[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(14)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(14)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(14)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(14)) << cp.int32(4)))
                    rAccum7[tid] = cp.disjoint(acc)

            block.barrier()

        # Phase 2: Warp-level inclusive scan on each packed pair, then extract counts.
        # For each radix pair i (0,2,4,...14):
        #   accum = rAccum[i/2][tid]
        #   val = warpScanAdd<WARP_SIZE>(accum, accumValue=0) -> inclusive scan - accum + accum = exclusive + inclusive - original
        #   Actually warpScanAdd returns inclusive_scan - value argument. With value=0, returns inclusive scan.
        #   Wait: warpScanAdd(originalValue=accum, value=accumValue) returns inclusive_scan(originalValue) - value.
        #   With value=0 (the 'accumValue' parameter), it returns the inclusive scan of originalValue.
        #   val2 = shfl(val, WARP_SIZE-1) = total sum for this warp
        #   For threadIndexInWarp < 2: sRadixSum[(i+threadIndexInWarp)*NUM_WARPS + warpIndex] = (val2 >> (tiw*16)) & 0xFFFF
        # This writes per-warp totals for each radix bucket.

        # We process all 8 packed pairs (radixes 0..15) in unrolled fashion.
        # After this, sRadixSum[radix * NUM_WARPS + warp] = count of keys with this radix in this warp.

        # Temporary shared mem for warp scan results (reuse rAccum arrays as scratch)
        for _pair in range(8):
            for tid, thread in block.threads():
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                warpIndexInBlock = tid >> cp.int32(5)  # tid / 32

                # Load accumulator for this pair
                acc = cp.int32(0)
                if _pair == 0:
                    acc = rAccum0[tid] + cp.int32(0)
                elif _pair == 1:
                    acc = rAccum1[tid] + cp.int32(0)
                elif _pair == 2:
                    acc = rAccum2[tid] + cp.int32(0)
                elif _pair == 3:
                    acc = rAccum3[tid] + cp.int32(0)
                elif _pair == 4:
                    acc = rAccum4[tid] + cp.int32(0)
                elif _pair == 5:
                    acc = rAccum5[tid] + cp.int32(0)
                elif _pair == 6:
                    acc = rAccum6[tid] + cp.int32(0)
                elif _pair == 7:
                    acc = rAccum7[tid] + cp.int32(0)

                # Inclusive warp scan
                val = acc
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n

                # val is now inclusive prefix sum. warpScanAdd returns val - accumValue.
                # With accumValue=0, result = val (inclusive sum).
                # Get last lane value (total for warp)
                val2 = thread.shfl_idx(val, 31)

                # Extract lo and hi 16-bit counts: radix pair is (2*_pair, 2*_pair+1)
                # threadIndexInWarp < 2: sRadixSum[(i+tiw)*NUM_WARPS + warpIdx] = (val2 >> (tiw*16)) & 0xFFFF
                radixBase = cp.int32(_pair * 2)
                if threadIndexInWarp < cp.int32(2):
                    shifted = val2 >> (threadIndexInWarp << cp.int32(4))
                    count_val = shifted & cp.int32(0xFFFF)
                    sRadixSum[(radixBase + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint(count_val)

        block.barrier()

        # Phase 3: Cross-warp scan (scanRadixWarps<NUM_WARPS>) and write to gRadixCount.
        # Only warpIndexInBlock < NUM_WARPS/2 participates (first 16 warps = 512 threads).
        # sRadixSum has RADIX_SIZE * NUM_WARPS = 16 * 32 = 512 elements.
        # Each of the 512 threads handles one element.
        # scanRadixWarps: scan within groups of NUM_WARPS (32) consecutive elements.
        # radixIndex = threadIndexInWarp & (NUM_WARPS - 1) = threadIndexInWarp (since NUM_WARPS=32=WARP_SIZE)
        # This is an exclusive prefix sum within each group of 32 (= within a warp).
        # Then threads where (idx & (NUM_WARPS-1)) == NUM_WARPS-1 write to gRadixCount.

        for tid, thread in block.threads():
            warpIndexInBlock = tid >> cp.int32(5)
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
            if cp.assume_uniform(warpIndexInBlock < cp.int32(NUM_WARPS // 2)):
                # idx = tid (first 512 threads)
                originalValue = sRadixSum[tid] + cp.int32(0)

                # Inclusive scan within warp (group of 32)
                val = originalValue
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(1))
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(2))
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(4))
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(8))
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(16))
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n

                # scanRadixWarps returns val - value (exclusive scan), where value=0
                # So output = val - 0 = val (inclusive). But the original code passes value=0
                # and returns val - value = val - 0 = inclusive scan.
                # Wait, re-reading: scanRadixWarps returns val - value where value is the
                # *original* value parameter (not originalVal). value=0 in the call.
                # So it returns the inclusive scan result (since val = inclusive_scan(originalValue)).
                # Actually looking more carefully:
                #   int val = originalVal;
                #   for(...) val += shfl(val, idx-a) if radixIndex >= a
                #   return val - value;  // value is the 6th parameter = 0
                # So it returns inclusive_scan(originalVal). That's the exclusive sum for the NEXT element.
                # The write condition: (idx & (NUM_WARPS-1)) == (NUM_WARPS-1), i.e., last in group.
                # gRadixIndex = bx + idx/NUM_WARPS * gridDim.x = bx + radix * NB_BLOCKS
                # gRadixCount[gRadixIndex] = output = inclusive total for this radix in this block

                output = val  # inclusive scan result (= exclusive scan of value 0)

                if threadIndexInWarp == cp.int32(NUM_WARPS - 1):
                    # gRadixIndex = bx + (tid / NUM_WARPS) * NB_BLOCKS
                    radixIdx = tid >> cp.int32(5)  # tid / NUM_WARPS (within first 512 threads)
                    # Wait: warpIndexInBlock = tid/32 which IS the radix index since
                    # sRadixSum layout is [radix * NUM_WARPS + warpIdx] and the scan
                    # is within each warp (32 elements), so warp i in the scan corresponds
                    # to radix i.
                    gRadixIndex = bx + warpIndexInBlock * cp.int32(NB_BLOCKS)
                    radixCount[gRadixIndex] = output


@cp.kernel
def radixSortMultiBlockLaunchWithCount(inputKeys, inputRanks, numKeys, startBit, radixCount,
                                       BLOCK_SIZE: cp.constexpr = 1024,
                                       NUM_WARPS: cp.constexpr = 32,
                                       RADIX_SIZE: cp.constexpr = 16,
                                       NB_BLOCKS: cp.constexpr = 32):
    with cp.Kernel(NB_BLOCKS, threads=BLOCK_SIZE) as (bx, block):
        # Shared memory: sRadixSum[RADIX_SIZE * NUM_WARPS]
        sRadixSum = block.alloc((RADIX_SIZE * NUM_WARPS,), dtype=cp.int32)

        # numKeys is total scalar keys. numKeysU4 = ceil(numKeys/4)
        numKeysU4 = (numKeys + cp.int32(3)) >> cp.int32(2)
        totalBlockRequired = (numKeysU4 + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterationPerBlock = (totalBlockRequired + cp.int32(NB_BLOCKS - 1)) // cp.int32(NB_BLOCKS)

        # Phase 1: Each thread accumulates radix histogram over its assigned keys.
        # radixAccum[8] packed pairs of 16-bit counters — stored in 8 shared arrays per thread.
        # Capybara limitation: cannot have per-thread register arrays of dynamic size.
        # We use 8 shared memory arrays of size BLOCK_SIZE as register file substitute.
        rAccum0 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum1 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum2 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum3 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum4 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum5 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum6 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum7 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)

        # Initialize accumulators to 0
        for tid, thread in block.threads():
            rAccum0[tid] = cp.disjoint(cp.int32(0))
            rAccum1[tid] = cp.disjoint(cp.int32(0))
            rAccum2[tid] = cp.disjoint(cp.int32(0))
            rAccum3[tid] = cp.disjoint(cp.int32(0))
            rAccum4[tid] = cp.disjoint(cp.int32(0))
            rAccum5[tid] = cp.disjoint(cp.int32(0))
            rAccum6[tid] = cp.disjoint(cp.int32(0))
            rAccum7[tid] = cp.disjoint(cp.int32(0))

        block.barrier()

        # Grid-stride loop over keys: radixSortWarp
        # inputKeyIndex = min(numIterationPerBlock * bx * BLOCK_SIZE, numKeysU4)
        # endIndex = min(inputKeyIndex + numIterationPerBlock * BLOCK_SIZE, numKeysU4)
        # count = endIndex - inputKeyIndex
        # Loop: for i = idx; i < count; i += WARP_SIZE * NUM_WARPS
        # Each iteration loads inputKeys[i + inputKeyIndex] as uint4.
        # stride = WARP_SIZE * NUM_WARPS = 32 * 32 = 1024 = BLOCK_SIZE
        # So iterations = count / BLOCK_SIZE = numIterationPerBlock (at most)
        for _iter in range(64):
            # Upper bound on numIterationPerBlock: ceil(ceil(N/4)/1024 / 32)
            # For very large N this could exceed 64, but that would be millions of keys.
            # Using 64 as a safe upper bound for typical PhysX usage.
            for tid, thread in block.threads():
                warpIndexInBlock = tid // cp.int32(WARP_SIZE)
                inputKeyStart = numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)
                inputKeyStart_clamped = inputKeyStart if inputKeyStart < numKeysU4 else numKeysU4
                endIndex_raw = inputKeyStart + numIterationPerBlock * cp.int32(BLOCK_SIZE)
                endIndex = endIndex_raw if endIndex_raw < numKeysU4 else numKeysU4
                count = endIndex - inputKeyStart_clamped

                loopIdx = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid
                if loopIdx < count:
                    gInputIdx = loopIdx + inputKeyStart_clamped
                    k0 = inputKeys[gInputIdx, 0] + cp.int32(0)
                    k1 = inputKeys[gInputIdx, 1] + cp.int32(0)
                    k2 = inputKeys[gInputIdx, 2] + cp.int32(0)
                    k3 = inputKeys[gInputIdx, 3] + cp.int32(0)

                    # sanitizeKeys: if gInputIdx*4 + {1,2,3} >= numKeys, set to 0xFFFFFFFF
                    baseScalarIdx = gInputIdx * cp.int32(4)
                    goodVals = numKeys - baseScalarIdx
                    # fallthrough switch: badVals = 4 - goodVals when goodVals < 4
                    if goodVals < cp.int32(4):
                        k3 = cp.int32(-1)
                    if goodVals < cp.int32(3):
                        k2 = cp.int32(-1)
                    if goodVals < cp.int32(2):
                        k1 = cp.int32(-1)

                    r0 = (k0 >> startBit) & cp.int32(0xF)
                    r1 = (k1 >> startBit) & cp.int32(0xF)
                    r2 = (k2 >> startBit) & cp.int32(0xF)
                    r3 = (k3 >> startBit) & cp.int32(0xF)

                    # Accumulate into packed pairs: radixAccum[bit/2] += (1 << ((radix - bit) << 4))
                    # bit=0: radixes 0,1 packed
                    acc = rAccum0[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(0)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(0)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(0)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(0)) << cp.int32(4)))
                    rAccum0[tid] = cp.disjoint(acc)

                    # bit=2
                    acc = rAccum1[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(2)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(2)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(2)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(2)) << cp.int32(4)))
                    rAccum1[tid] = cp.disjoint(acc)

                    # bit=4
                    acc = rAccum2[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(4)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(4)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(4)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(4)) << cp.int32(4)))
                    rAccum2[tid] = cp.disjoint(acc)

                    # bit=6
                    acc = rAccum3[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(6)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(6)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(6)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(6)) << cp.int32(4)))
                    rAccum3[tid] = cp.disjoint(acc)

                    # bit=8
                    acc = rAccum4[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(8)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(8)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(8)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(8)) << cp.int32(4)))
                    rAccum4[tid] = cp.disjoint(acc)

                    # bit=10
                    acc = rAccum5[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(10)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(10)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(10)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(10)) << cp.int32(4)))
                    rAccum5[tid] = cp.disjoint(acc)

                    # bit=12
                    acc = rAccum6[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(12)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(12)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(12)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(12)) << cp.int32(4)))
                    rAccum6[tid] = cp.disjoint(acc)

                    # bit=14
                    acc = rAccum7[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(14)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(14)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(14)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(14)) << cp.int32(4)))
                    rAccum7[tid] = cp.disjoint(acc)

            block.barrier()

        # Phase 2: Warp-level inclusive scan on each packed pair, then extract counts.
        # For each radix pair i (0,2,4,...14):
        #   accum = rAccum[i/2][tid]
        #   val = warpScanAdd<WARP_SIZE>(accum, accumValue=0) -> inclusive scan - accum + accum = exclusive + inclusive - original
        #   Actually warpScanAdd returns inclusive_scan - value argument. With value=0, returns inclusive scan.
        #   Wait: warpScanAdd(originalValue=accum, value=accumValue) returns inclusive_scan(originalValue) - value.
        #   With value=0 (the 'accumValue' parameter), it returns the inclusive scan of originalValue.
        #   val2 = shfl(val, WARP_SIZE-1) = total sum for this warp
        #   For threadIndexInWarp < 2: sRadixSum[(i+threadIndexInWarp)*NUM_WARPS + warpIndex] = (val2 >> (tiw*16)) & 0xFFFF
        # This writes per-warp totals for each radix bucket.

        # We process all 8 packed pairs (radixes 0..15) in unrolled fashion.
        # After this, sRadixSum[radix * NUM_WARPS + warp] = count of keys with this radix in this warp.

        # Temporary shared mem for warp scan results (reuse rAccum arrays as scratch)
        for _pair in range(8):
            for tid, thread in block.threads():
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                warpIndexInBlock = tid >> cp.int32(5)  # tid / 32

                # Load accumulator for this pair
                acc = cp.int32(0)
                if _pair == 0:
                    acc = rAccum0[tid] + cp.int32(0)
                elif _pair == 1:
                    acc = rAccum1[tid] + cp.int32(0)
                elif _pair == 2:
                    acc = rAccum2[tid] + cp.int32(0)
                elif _pair == 3:
                    acc = rAccum3[tid] + cp.int32(0)
                elif _pair == 4:
                    acc = rAccum4[tid] + cp.int32(0)
                elif _pair == 5:
                    acc = rAccum5[tid] + cp.int32(0)
                elif _pair == 6:
                    acc = rAccum6[tid] + cp.int32(0)
                elif _pair == 7:
                    acc = rAccum7[tid] + cp.int32(0)

                # Inclusive warp scan
                val = acc
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n

                # val is now inclusive prefix sum. warpScanAdd returns val - accumValue.
                # With accumValue=0, result = val (inclusive sum).
                # Get last lane value (total for warp)
                val2 = thread.shfl_idx(val, 31)

                # Extract lo and hi 16-bit counts: radix pair is (2*_pair, 2*_pair+1)
                # threadIndexInWarp < 2: sRadixSum[(i+tiw)*NUM_WARPS + warpIdx] = (val2 >> (tiw*16)) & 0xFFFF
                radixBase = cp.int32(_pair * 2)
                if threadIndexInWarp < cp.int32(2):
                    shifted = val2 >> (threadIndexInWarp << cp.int32(4))
                    count_val = shifted & cp.int32(0xFFFF)
                    sRadixSum[(radixBase + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint(count_val)

        block.barrier()

        # Phase 3: Cross-warp scan (scanRadixWarps<NUM_WARPS>) and write to gRadixCount.
        # Only warpIndexInBlock < NUM_WARPS/2 participates (first 16 warps = 512 threads).
        # sRadixSum has RADIX_SIZE * NUM_WARPS = 16 * 32 = 512 elements.
        # Each of the 512 threads handles one element.
        # scanRadixWarps: scan within groups of NUM_WARPS (32) consecutive elements.
        # radixIndex = threadIndexInWarp & (NUM_WARPS - 1) = threadIndexInWarp (since NUM_WARPS=32=WARP_SIZE)
        # This is an exclusive prefix sum within each group of 32 (= within a warp).
        # Then threads where (idx & (NUM_WARPS-1)) == NUM_WARPS-1 write to gRadixCount.

        for tid, thread in block.threads():
            warpIndexInBlock = tid >> cp.int32(5)
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
            if cp.assume_uniform(warpIndexInBlock < cp.int32(NUM_WARPS // 2)):
                # idx = tid (first 512 threads)
                originalValue = sRadixSum[tid] + cp.int32(0)

                # Inclusive scan within warp (group of 32)
                val = originalValue
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(1))
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(2))
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(4))
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(8))
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(16))
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n

                # scanRadixWarps returns val - value (exclusive scan), where value=0
                # So output = val - 0 = val (inclusive). But the original code passes value=0
                # and returns val - value = val - 0 = inclusive scan.
                # Wait, re-reading: scanRadixWarps returns val - value where value is the
                # *original* value parameter (not originalVal). value=0 in the call.
                # So it returns the inclusive scan result (since val = inclusive_scan(originalValue)).
                # Actually looking more carefully:
                #   int val = originalVal;
                #   for(...) val += shfl(val, idx-a) if radixIndex >= a
                #   return val - value;  // value is the 6th parameter = 0
                # So it returns inclusive_scan(originalVal). That's the exclusive sum for the NEXT element.
                # The write condition: (idx & (NUM_WARPS-1)) == (NUM_WARPS-1), i.e., last in group.
                # gRadixIndex = bx + idx/NUM_WARPS * gridDim.x = bx + radix * NB_BLOCKS
                # gRadixCount[gRadixIndex] = output = inclusive total for this radix in this block

                output = val  # inclusive scan result (= exclusive scan of value 0)

                if threadIndexInWarp == cp.int32(NUM_WARPS - 1):
                    # gRadixIndex = bx + (tid / NUM_WARPS) * NB_BLOCKS
                    radixIdx = tid >> cp.int32(5)  # tid / NUM_WARPS (within first 512 threads)
                    # Wait: warpIndexInBlock = tid/32 which IS the radix index since
                    # sRadixSum layout is [radix * NUM_WARPS + warpIdx] and the scan
                    # is within each warp (32 elements), so warp i in the scan corresponds
                    # to radix i.
                    gRadixIndex = bx + warpIndexInBlock * cp.int32(NB_BLOCKS)
                    radixCount[gRadixIndex] = output


# ===========================================================================
# Group C: radixSortCalculateRanks — reorder pass
# ===========================================================================
#
# This ports radixSortCalculateRanks<NUM_WARPS> from RadixSort.cuh.
#
# The algorithm has two major parts:
#   Part 1: scanRadixes — scan gRadixCount across blocks to compute global prefix sums.
#   Part 2: For each chunk of keys, compute local radix ranks within the block,
#           then scatter keys and ranks to their globally correct positions.
#
# inputKeys: int32[N, 4] — uint4 vectorized keys
# inputRanks: int32[N, 4] — uint4 vectorized ranks (indices)
# numKeys: scalar — total number of individual keys
# startBit: scalar — which 4-bit nibble
# radixCount: int32[RADIX_SIZE * NB_BLOCKS] — histogram from SingleBlock pass
# outputKeys: int32[M] — output sorted keys
# outputRanks: int32[M] — output sorted ranks

@cp.kernel
def radixSortMultiCalculateRanksLaunch(inputKeys, inputRanks, numKeys, startBit,
                                       radixCount, outputKeys, outputRanks,
                                       BLOCK_SIZE: cp.constexpr = 1024,
                                       NUM_WARPS: cp.constexpr = 32,
                                       RADIX_SIZE: cp.constexpr = 16,
                                       NB_BLOCKS: cp.constexpr = 32):
    with cp.Kernel(NB_BLOCKS, threads=BLOCK_SIZE) as (bx, block):
        # Shared memory allocations matching CUDA:
        # sRadixSumBetweenBlocks[RADIX_SIZE] — global prefix sums per radix
        # sBuckets[NUM_WARPS * WARP_SIZE] — scratch for warp scans
        # sRadixSum[RADIX_SIZE * NUM_WARPS] — per-warp radix counts within block
        # sRadixSumSum[RADIX_SIZE] — prefix sum of radix totals
        # sRadixCount[RADIX_SIZE] — radix totals for current iteration
        # sKeys[WARP_SIZE * NUM_WARPS * 4] — scattered keys in shared mem
        # sRanks[WARP_SIZE * NUM_WARPS * 4] — scattered ranks in shared mem
        sRadixSumBetweenBlocks = block.alloc((RADIX_SIZE,), dtype=cp.int32)
        sBuckets = block.alloc((NUM_WARPS * WARP_SIZE,), dtype=cp.int32)
        sRadixSum = block.alloc((RADIX_SIZE * NUM_WARPS,), dtype=cp.int32)
        sRadixSumSum = block.alloc((RADIX_SIZE,), dtype=cp.int32)
        sRadixCount = block.alloc((RADIX_SIZE,), dtype=cp.int32)
        sKeys = block.alloc((WARP_SIZE * NUM_WARPS * 4,), dtype=cp.int32)
        sRanks = block.alloc((WARP_SIZE * NUM_WARPS * 4,), dtype=cp.int32)

        numKeysU4 = (numKeys + cp.int32(3)) >> cp.int32(2)
        totalBlockRequired = (numKeysU4 + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterationPerBlock = (totalBlockRequired + cp.int32(NB_BLOCKS - 1)) // cp.int32(NB_BLOCKS)

        # ---------------------------------------------------------------
        # Part 1: scanRadixes — scan radixCount across blocks per radix
        # ---------------------------------------------------------------
        # scanRadixes<NUM_WARPS>(warpIndexInBlock, threadIndexInWarp, gRadixCount,
        #                        sRadixCountBetweenBlocks=sBuckets, sRadixSumBetweenBlocks)
        #
        # For each radix i (i = warpIndexInBlock; i < RADIX_SIZE; i += NUM_WARPS):
        #   value = gRadixCount[i * gridDim.x + threadIndexInWarp]
        #   output = warpScanAddWriteToSharedMem<WARP_SIZE>(index=radixSumIndex, tiw, sData=sBuckets, value, value)
        #   if tiw == WARP_SIZE-1: sRadixSumBetweenBlocks[i] = output + value
        #
        # Since NUM_WARPS=32 and RADIX_SIZE=16, each warp handles at most 1 radix
        # (i = warpIndexInBlock, step NUM_WARPS=32 > 16, so only warpIndexInBlock < 16 runs).
        # threadIndexInWarp indexes across NB_BLOCKS=32 blocks.
        # warpScanAddWriteToSharedMem<WARP_SIZE> does inclusive scan and writes
        # exclusive result (val - value) to sData[index], returns exclusive result.
        # So sBuckets[i*gridDim.x + tiw] = exclusive prefix sum of radixCount for radix i.
        # sRadixSumBetweenBlocks[i] = total count for radix i across all blocks.

        for tid, thread in block.threads():
            warpIndexInBlock = tid >> cp.int32(5)
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
            if cp.assume_uniform(warpIndexInBlock < cp.int32(RADIX_SIZE)):
                radixSumIndex = warpIndexInBlock * cp.int32(NB_BLOCKS) + threadIndexInWarp
                value = radixCount[radixSumIndex] + cp.int32(0)

                # warpScanAddWriteToSharedMem<WARP_SIZE>: inclusive scan, write exclusive
                val = value
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n

                # Exclusive result = inclusive - original
                exclusive = val - value
                sBuckets[radixSumIndex] = cp.disjoint(exclusive)
                if threadIndexInWarp == cp.int32(WARP_SIZE - 1):
                    # Total = exclusive + value = val
                    sRadixSumBetweenBlocks[warpIndexInBlock] = cp.disjoint(val)

        block.barrier()

        # Part 1b: Exclusive scan of sRadixSumBetweenBlocks[RADIX_SIZE=16]
        # if idx < RADIX_SIZE:
        #   value = sRadixSumBetweenBlocks[idx]
        #   output = warpScanAdd<RADIX_SIZE>(value, value) -> inclusive - value = exclusive
        #   sRadixSumBetweenBlocks[idx] = output + sBuckets[idx * NB_BLOCKS + bx]
        # This gives each block's global starting offset for each radix.
        for tid, thread in block.threads():
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
            if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                value = sRadixSumBetweenBlocks[tid] + cp.int32(0)

                # warpScanAdd<RADIX_SIZE>: inclusive scan among first 16 threads
                val = value
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n

                # warpScanAdd returns val - value (exclusive prefix sum)
                output = val - value
                # Add this block's offset from the cross-block scan
                sRadixSumBetweenBlocks[tid] = cp.disjoint(output + sBuckets[tid * cp.int32(NB_BLOCKS) + bx])

        block.barrier()

        # ---------------------------------------------------------------
        # Part 2: Per-iteration reorder loop
        # ---------------------------------------------------------------
        for _iter in range(64):
            # Phase 2a: Load keys, compute radix, warp-level rank computation
            for tid, thread in block.threads():
                warpIndexInBlock = tid >> cp.int32(5)
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)

                inputKeyIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                # Load keys and ranks (or pad with 0xFF radix if out of bounds)
                k0 = cp.int32(0)
                k1 = cp.int32(0)
                k2 = cp.int32(0)
                k3 = cp.int32(0)
                ki0 = cp.int32(0)
                ki1 = cp.int32(0)
                ki2 = cp.int32(0)
                ki3 = cp.int32(0)
                r0 = cp.int32(0xFF)
                r1 = cp.int32(0xFF)
                r2 = cp.int32(0xFF)
                r3 = cp.int32(0xFF)

                if inputKeyIndex < numKeysU4:
                    ki0 = inputRanks[inputKeyIndex, 0] + cp.int32(0)
                    ki1 = inputRanks[inputKeyIndex, 1] + cp.int32(0)
                    ki2 = inputRanks[inputKeyIndex, 2] + cp.int32(0)
                    ki3 = inputRanks[inputKeyIndex, 3] + cp.int32(0)

                    k0 = inputKeys[inputKeyIndex, 0] + cp.int32(0)
                    k1 = inputKeys[inputKeyIndex, 1] + cp.int32(0)
                    k2 = inputKeys[inputKeyIndex, 2] + cp.int32(0)
                    k3 = inputKeys[inputKeyIndex, 3] + cp.int32(0)

                    # sanitizeKeys
                    baseScalarIdx = inputKeyIndex * cp.int32(4)
                    goodVals = numKeys - baseScalarIdx
                    if goodVals < cp.int32(4):
                        k3 = cp.int32(-1)
                    if goodVals < cp.int32(3):
                        k2 = cp.int32(-1)
                    if goodVals < cp.int32(2):
                        k1 = cp.int32(-1)

                    r0 = (k0 >> startBit) & cp.int32(0xF)
                    r1 = (k1 >> startBit) & cp.int32(0xF)
                    r2 = (k2 >> startBit) & cp.int32(0xF)
                    r3 = (k3 >> startBit) & cp.int32(0xF)

                # Compute radix offsets within warp using packed 8-bit counters.
                # For each group of 4 radixes (i=0,4,8,12):
                #   accum = sum of (1 << ((radix_j - i) << 3)) for j in {x,y,z,w}
                #   val = warpScanAdd<WARP_SIZE>(accum, 0) -> inclusive scan
                #   val2 = shfl(val, 31) -> total
                #   For tiw < 4: sRadixSum[(i+tiw)*NUM_WARPS + warpIdx] = (val2 >> (8*tiw)) & 0xFF
                #   val -= accum -> exclusive scan
                #   Then extract per-element offsets from packed val

                # radixOffset for each of the 4 keys
                ro0 = cp.int32(0)
                ro1 = cp.int32(0)
                ro2 = cp.int32(0)
                ro3 = cp.int32(0)

                # Group i=0 (radixes 0-3)
                accum = (cp.int32(1) << ((r0 - cp.int32(0)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r1 - cp.int32(0)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r2 - cp.int32(0)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r3 - cp.int32(0)) << cp.int32(3)))
                # Inclusive warp scan
                val = accum
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n
                val2 = thread.shfl_idx(val, 31)
                if threadIndexInWarp < cp.int32(4):
                    sRadixSum[(cp.int32(0) + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint((val2 >> (cp.int32(8) * threadIndexInWarp)) & cp.int32(0xFF))
                val = val - accum  # exclusive
                # Extract per-key offsets
                sb0 = (r0 - cp.int32(0)) << cp.int32(3)
                ro0 = ro0 | ((val >> sb0) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb0)
                sb1 = (r1 - cp.int32(0)) << cp.int32(3)
                ro1 = ro1 | ((val >> sb1) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb1)
                sb2 = (r2 - cp.int32(0)) << cp.int32(3)
                ro2 = ro2 | ((val >> sb2) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb2)
                sb3 = (r3 - cp.int32(0)) << cp.int32(3)
                ro3 = ro3 | ((val >> sb3) & cp.int32(0xFF))

                # Group i=4 (radixes 4-7)
                accum = (cp.int32(1) << ((r0 - cp.int32(4)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r1 - cp.int32(4)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r2 - cp.int32(4)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r3 - cp.int32(4)) << cp.int32(3)))
                val = accum
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n
                val2 = thread.shfl_idx(val, 31)
                if threadIndexInWarp < cp.int32(4):
                    sRadixSum[(cp.int32(4) + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint((val2 >> (cp.int32(8) * threadIndexInWarp)) & cp.int32(0xFF))
                val = val - accum
                sb0 = (r0 - cp.int32(4)) << cp.int32(3)
                ro0 = ro0 | ((val >> sb0) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb0)
                sb1 = (r1 - cp.int32(4)) << cp.int32(3)
                ro1 = ro1 | ((val >> sb1) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb1)
                sb2 = (r2 - cp.int32(4)) << cp.int32(3)
                ro2 = ro2 | ((val >> sb2) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb2)
                sb3 = (r3 - cp.int32(4)) << cp.int32(3)
                ro3 = ro3 | ((val >> sb3) & cp.int32(0xFF))

                # Group i=8 (radixes 8-11)
                accum = (cp.int32(1) << ((r0 - cp.int32(8)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r1 - cp.int32(8)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r2 - cp.int32(8)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r3 - cp.int32(8)) << cp.int32(3)))
                val = accum
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n
                val2 = thread.shfl_idx(val, 31)
                if threadIndexInWarp < cp.int32(4):
                    sRadixSum[(cp.int32(8) + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint((val2 >> (cp.int32(8) * threadIndexInWarp)) & cp.int32(0xFF))
                val = val - accum
                sb0 = (r0 - cp.int32(8)) << cp.int32(3)
                ro0 = ro0 | ((val >> sb0) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb0)
                sb1 = (r1 - cp.int32(8)) << cp.int32(3)
                ro1 = ro1 | ((val >> sb1) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb1)
                sb2 = (r2 - cp.int32(8)) << cp.int32(3)
                ro2 = ro2 | ((val >> sb2) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb2)
                sb3 = (r3 - cp.int32(8)) << cp.int32(3)
                ro3 = ro3 | ((val >> sb3) & cp.int32(0xFF))

                # Group i=12 (radixes 12-15)
                accum = (cp.int32(1) << ((r0 - cp.int32(12)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r1 - cp.int32(12)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r2 - cp.int32(12)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r3 - cp.int32(12)) << cp.int32(3)))
                val = accum
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n
                val2 = thread.shfl_idx(val, 31)
                if threadIndexInWarp < cp.int32(4):
                    sRadixSum[(cp.int32(12) + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint((val2 >> (cp.int32(8) * threadIndexInWarp)) & cp.int32(0xFF))
                val = val - accum
                sb0 = (r0 - cp.int32(12)) << cp.int32(3)
                ro0 = ro0 | ((val >> sb0) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb0)
                sb1 = (r1 - cp.int32(12)) << cp.int32(3)
                ro1 = ro1 | ((val >> sb1) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb1)
                sb2 = (r2 - cp.int32(12)) << cp.int32(3)
                ro2 = ro2 | ((val >> sb2) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb2)
                sb3 = (r3 - cp.int32(12)) << cp.int32(3)
                ro3 = ro3 | ((val >> sb3) & cp.int32(0xFF))

                # Store radixOffset and key/rank data to per-thread shared mem slots
                # for use after the cross-warp scan. We reuse sBuckets as scratch.
                sBuckets[tid * cp.int32(4) + cp.int32(0)] = cp.disjoint(ro0)
                sBuckets[tid * cp.int32(4) + cp.int32(1)] = cp.disjoint(ro1)
                sBuckets[tid * cp.int32(4) + cp.int32(2)] = cp.disjoint(ro2)
                sBuckets[tid * cp.int32(4) + cp.int32(3)] = cp.disjoint(ro3)

            block.barrier()

            # Phase 2b: Save lastRadixSum before cross-warp scan overwrites sRadixSum
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    lastRS = sRadixSum[tid * cp.int32(NUM_WARPS) + cp.int32(NUM_WARPS - 1)] + cp.int32(0)
                    sRadixCount[tid] = cp.disjoint(lastRS)

            block.barrier()

            # Phase 2c: Cross-warp exclusive scan of sRadixSum (scanRadixWarps<NUM_WARPS>)
            # Only first NUM_WARPS/2 = 16 warps (512 threads).
            # sRadixSum[radix * NUM_WARPS + warp] -> exclusive prefix sum within each group of NUM_WARPS.
            for tid, thread in block.threads():
                warpIndexInBlock = tid >> cp.int32(5)
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                if cp.assume_uniform(warpIndexInBlock < cp.int32(NUM_WARPS // 2)):
                    tempVal = sRadixSum[tid] + cp.int32(0)

                    # Inclusive scan within warp (NUM_WARPS=32=WARP_SIZE elements per group)
                    val = tempVal
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(1))
                    if threadIndexInWarp >= cp.int32(1):
                        val = val + n
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(2))
                    if threadIndexInWarp >= cp.int32(2):
                        val = val + n
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(4))
                    if threadIndexInWarp >= cp.int32(4):
                        val = val + n
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(8))
                    if threadIndexInWarp >= cp.int32(8):
                        val = val + n
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(16))
                    if threadIndexInWarp >= cp.int32(16):
                        val = val + n

                    # scanRadixWarps returns val - tempVal (exclusive)
                    sRadixSum[tid] = cp.disjoint(val - tempVal)

            block.barrier()

            # Phase 2d: Compute sRadixSumSum (cross-radix prefix sum of per-radix totals)
            for tid, thread in block.threads():
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    # Total for radix tid = last warp's exclusive + last warp's count
                    value = sRadixSum[tid * cp.int32(NUM_WARPS) + cp.int32(NUM_WARPS - 1)] + cp.int32(0)
                    value = value + sRadixCount[tid]
                    sRadixCount[tid] = cp.disjoint(value)
                    sRadixSumSum[tid] = cp.disjoint(value)

            block.barrier()

            # warpScanAddWriteToSharedMem<RADIX_SIZE> on sRadixSumSum
            for tid, thread in block.threads():
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    value = sRadixSumSum[tid] + cp.int32(0)
                    val = value
                    n = thread.shfl_up(val, 1)
                    if threadIndexInWarp >= cp.int32(1):
                        val = val + n
                    n = thread.shfl_up(val, 2)
                    if threadIndexInWarp >= cp.int32(2):
                        val = val + n
                    n = thread.shfl_up(val, 4)
                    if threadIndexInWarp >= cp.int32(4):
                        val = val + n
                    n = thread.shfl_up(val, 8)
                    if threadIndexInWarp >= cp.int32(8):
                        val = val + n
                    # Exclusive = inclusive - original
                    sRadixSumSum[tid] = cp.disjoint(val - value)

            block.barrier()

            # Phase 2e: Add sRadixSumSum to sRadixSum
            # if idx < NUM_WARPS * RADIX_SIZE: sRadixSum[idx] += sRadixSumSum[idx / NUM_WARPS]
            for tid, thread in block.threads():
                if tid < cp.int32(NUM_WARPS * RADIX_SIZE):
                    radixIdx = tid // cp.int32(NUM_WARPS)
                    addVal = sRadixSumSum[radixIdx] + cp.int32(0)
                    curVal = sRadixSum[tid] + cp.int32(0)
                    sRadixSum[tid] = cp.disjoint(curVal + addVal)

            block.barrier()

            # Phase 2f: Compute global offsets
            # sRadixSumSum[idx] = sRadixSumBetweenBlocks[idx] - sRadixSumSum[idx]
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    betweenBlocks = sRadixSumBetweenBlocks[tid] + cp.int32(0)
                    sumSum = sRadixSumSum[tid] + cp.int32(0)
                    sRadixSumSum[tid] = cp.disjoint(betweenBlocks - sumSum)

            block.barrier()

            # Phase 2g: Scatter keys and ranks to shared memory
            for tid, thread in block.threads():
                inputKeyIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                k0 = cp.int32(0)
                k1 = cp.int32(0)
                k2 = cp.int32(0)
                k3 = cp.int32(0)
                ki0 = cp.int32(0)
                ki1 = cp.int32(0)
                ki2 = cp.int32(0)
                ki3 = cp.int32(0)
                r0 = cp.int32(0xFF)
                r1 = cp.int32(0xFF)
                r2 = cp.int32(0xFF)
                r3 = cp.int32(0xFF)

                if inputKeyIndex < numKeysU4:
                    ki0 = inputRanks[inputKeyIndex, 0] + cp.int32(0)
                    ki1 = inputRanks[inputKeyIndex, 1] + cp.int32(0)
                    ki2 = inputRanks[inputKeyIndex, 2] + cp.int32(0)
                    ki3 = inputRanks[inputKeyIndex, 3] + cp.int32(0)

                    k0 = inputKeys[inputKeyIndex, 0] + cp.int32(0)
                    k1 = inputKeys[inputKeyIndex, 1] + cp.int32(0)
                    k2 = inputKeys[inputKeyIndex, 2] + cp.int32(0)
                    k3 = inputKeys[inputKeyIndex, 3] + cp.int32(0)

                    baseScalarIdx = inputKeyIndex * cp.int32(4)
                    goodVals = numKeys - baseScalarIdx
                    if goodVals < cp.int32(4):
                        k3 = cp.int32(-1)
                    if goodVals < cp.int32(3):
                        k2 = cp.int32(-1)
                    if goodVals < cp.int32(2):
                        k1 = cp.int32(-1)

                    r0 = (k0 >> startBit) & cp.int32(0xF)
                    r1 = (k1 >> startBit) & cp.int32(0xF)
                    r2 = (k2 >> startBit) & cp.int32(0xF)
                    r3 = (k3 >> startBit) & cp.int32(0xF)

                    warpIndexInBlock = tid >> cp.int32(5)

                    # Reload radix offsets from sBuckets scratch
                    ro0 = sBuckets[tid * cp.int32(4) + cp.int32(0)] + cp.int32(0)
                    ro1 = sBuckets[tid * cp.int32(4) + cp.int32(1)] + cp.int32(0)
                    ro2 = sBuckets[tid * cp.int32(4) + cp.int32(2)] + cp.int32(0)
                    ro3 = sBuckets[tid * cp.int32(4) + cp.int32(3)] + cp.int32(0)

                    # Add cross-warp offset: sRadixSum[radix * NUM_WARPS + warpIndexInBlock]
                    ro0 = ro0 + sRadixSum[r0 * cp.int32(NUM_WARPS) + warpIndexInBlock]
                    ro1 = ro1 + sRadixSum[r1 * cp.int32(NUM_WARPS) + warpIndexInBlock]
                    ro2 = ro2 + sRadixSum[r2 * cp.int32(NUM_WARPS) + warpIndexInBlock]
                    ro3 = ro3 + sRadixSum[r3 * cp.int32(NUM_WARPS) + warpIndexInBlock]

                    sKeys[ro0] = cp.disjoint(k0)
                    sKeys[ro1] = cp.disjoint(k1)
                    sKeys[ro2] = cp.disjoint(k2)
                    sKeys[ro3] = cp.disjoint(k3)

                    sRanks[ro0] = cp.disjoint(ki0)
                    sRanks[ro1] = cp.disjoint(ki1)
                    sRanks[ro2] = cp.disjoint(ki2)
                    sRanks[ro3] = cp.disjoint(ki3)

            block.barrier()

            # Phase 2h: Write from shared memory to global output
            for tid, thread in block.threads():
                inputKeyIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)
                baseInputKeyIndex = inputKeyIndex - tid

                if baseInputKeyIndex < numKeysU4:
                    keysToProcess_raw = numKeys - baseInputKeyIndex * cp.int32(4)
                    keysToProcess_max = cp.int32(WARP_SIZE * NUM_WARPS * 4)
                    keysToProcess = keysToProcess_raw if keysToProcess_raw < keysToProcess_max else keysToProcess_max

                    # Grid-stride write: for a = tid; a < keysToProcess; a += BLOCK_SIZE
                    # Since keysToProcess <= BLOCK_SIZE * 4 and stride = BLOCK_SIZE,
                    # each thread writes up to 4 elements.
                    a0 = tid
                    if a0 < keysToProcess:
                        key = sKeys[a0] + cp.int32(0)
                        radix = (key >> startBit) & cp.int32(0xF)
                        writeIndex = a0 + sRadixSumSum[radix]
                        outputKeys[writeIndex] = key
                        outputRanks[writeIndex] = sRanks[a0] + cp.int32(0)

                    a1 = tid + cp.int32(BLOCK_SIZE)
                    if a1 < keysToProcess:
                        key = sKeys[a1] + cp.int32(0)
                        radix = (key >> startBit) & cp.int32(0xF)
                        writeIndex = a1 + sRadixSumSum[radix]
                        outputKeys[writeIndex] = key
                        outputRanks[writeIndex] = sRanks[a1] + cp.int32(0)

                    a2 = tid + cp.int32(BLOCK_SIZE * 2)
                    if a2 < keysToProcess:
                        key = sKeys[a2] + cp.int32(0)
                        radix = (key >> startBit) & cp.int32(0xF)
                        writeIndex = a2 + sRadixSumSum[radix]
                        outputKeys[writeIndex] = key
                        outputRanks[writeIndex] = sRanks[a2] + cp.int32(0)

                    a3 = tid + cp.int32(BLOCK_SIZE * 3)
                    if a3 < keysToProcess:
                        key = sKeys[a3] + cp.int32(0)
                        radix = (key >> startBit) & cp.int32(0xF)
                        writeIndex = a3 + sRadixSumSum[radix]
                        outputKeys[writeIndex] = key
                        outputRanks[writeIndex] = sRanks[a3] + cp.int32(0)

            block.barrier()

            # Phase 2i: Update sRadixSumBetweenBlocks for next iteration
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    betweenBlocks = sRadixSumBetweenBlocks[tid] + cp.int32(0)
                    radixCnt = sRadixCount[tid] + cp.int32(0)
                    sRadixSumBetweenBlocks[tid] = cp.disjoint(betweenBlocks + radixCnt)

            block.barrier()



# Aliases: all 3 CalculateRanksLaunch variants share the same body.
# Capybara limitation: cannot call another @cp.kernel, so we duplicate the body.
@cp.kernel
def radixSortMultiCalculateRanksLaunchWithoutCount(inputKeys, inputRanks, numKeys, startBit,
                                                    radixCount, outputKeys, outputRanks,
                                                    BLOCK_SIZE: cp.constexpr = 1024,
                                                    NUM_WARPS: cp.constexpr = 32,
                                                    RADIX_SIZE: cp.constexpr = 16,
                                                    NB_BLOCKS: cp.constexpr = 32):
    with cp.Kernel(NB_BLOCKS, threads=BLOCK_SIZE) as (bx, block):
        # Shared memory allocations matching CUDA:
        # sRadixSumBetweenBlocks[RADIX_SIZE] — global prefix sums per radix
        # sBuckets[NUM_WARPS * WARP_SIZE] — scratch for warp scans
        # sRadixSum[RADIX_SIZE * NUM_WARPS] — per-warp radix counts within block
        # sRadixSumSum[RADIX_SIZE] — prefix sum of radix totals
        # sRadixCount[RADIX_SIZE] — radix totals for current iteration
        # sKeys[WARP_SIZE * NUM_WARPS * 4] — scattered keys in shared mem
        # sRanks[WARP_SIZE * NUM_WARPS * 4] — scattered ranks in shared mem
        sRadixSumBetweenBlocks = block.alloc((RADIX_SIZE,), dtype=cp.int32)
        sBuckets = block.alloc((NUM_WARPS * WARP_SIZE,), dtype=cp.int32)
        sRadixSum = block.alloc((RADIX_SIZE * NUM_WARPS,), dtype=cp.int32)
        sRadixSumSum = block.alloc((RADIX_SIZE,), dtype=cp.int32)
        sRadixCount = block.alloc((RADIX_SIZE,), dtype=cp.int32)
        sKeys = block.alloc((WARP_SIZE * NUM_WARPS * 4,), dtype=cp.int32)
        sRanks = block.alloc((WARP_SIZE * NUM_WARPS * 4,), dtype=cp.int32)

        numKeysU4 = (numKeys + cp.int32(3)) >> cp.int32(2)
        totalBlockRequired = (numKeysU4 + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterationPerBlock = (totalBlockRequired + cp.int32(NB_BLOCKS - 1)) // cp.int32(NB_BLOCKS)

        # ---------------------------------------------------------------
        # Part 1: scanRadixes — scan radixCount across blocks per radix
        # ---------------------------------------------------------------
        # scanRadixes<NUM_WARPS>(warpIndexInBlock, threadIndexInWarp, gRadixCount,
        #                        sRadixCountBetweenBlocks=sBuckets, sRadixSumBetweenBlocks)
        #
        # For each radix i (i = warpIndexInBlock; i < RADIX_SIZE; i += NUM_WARPS):
        #   value = gRadixCount[i * gridDim.x + threadIndexInWarp]
        #   output = warpScanAddWriteToSharedMem<WARP_SIZE>(index=radixSumIndex, tiw, sData=sBuckets, value, value)
        #   if tiw == WARP_SIZE-1: sRadixSumBetweenBlocks[i] = output + value
        #
        # Since NUM_WARPS=32 and RADIX_SIZE=16, each warp handles at most 1 radix
        # (i = warpIndexInBlock, step NUM_WARPS=32 > 16, so only warpIndexInBlock < 16 runs).
        # threadIndexInWarp indexes across NB_BLOCKS=32 blocks.
        # warpScanAddWriteToSharedMem<WARP_SIZE> does inclusive scan and writes
        # exclusive result (val - value) to sData[index], returns exclusive result.
        # So sBuckets[i*gridDim.x + tiw] = exclusive prefix sum of radixCount for radix i.
        # sRadixSumBetweenBlocks[i] = total count for radix i across all blocks.

        for tid, thread in block.threads():
            warpIndexInBlock = tid >> cp.int32(5)
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
            if cp.assume_uniform(warpIndexInBlock < cp.int32(RADIX_SIZE)):
                radixSumIndex = warpIndexInBlock * cp.int32(NB_BLOCKS) + threadIndexInWarp
                value = radixCount[radixSumIndex] + cp.int32(0)

                # warpScanAddWriteToSharedMem<WARP_SIZE>: inclusive scan, write exclusive
                val = value
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n

                # Exclusive result = inclusive - original
                exclusive = val - value
                sBuckets[radixSumIndex] = cp.disjoint(exclusive)
                if threadIndexInWarp == cp.int32(WARP_SIZE - 1):
                    # Total = exclusive + value = val
                    sRadixSumBetweenBlocks[warpIndexInBlock] = cp.disjoint(val)

        block.barrier()

        # Part 1b: Exclusive scan of sRadixSumBetweenBlocks[RADIX_SIZE=16]
        # if idx < RADIX_SIZE:
        #   value = sRadixSumBetweenBlocks[idx]
        #   output = warpScanAdd<RADIX_SIZE>(value, value) -> inclusive - value = exclusive
        #   sRadixSumBetweenBlocks[idx] = output + sBuckets[idx * NB_BLOCKS + bx]
        # This gives each block's global starting offset for each radix.
        for tid, thread in block.threads():
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
            if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                value = sRadixSumBetweenBlocks[tid] + cp.int32(0)

                # warpScanAdd<RADIX_SIZE>: inclusive scan among first 16 threads
                val = value
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n

                # warpScanAdd returns val - value (exclusive prefix sum)
                output = val - value
                # Add this block's offset from the cross-block scan
                sRadixSumBetweenBlocks[tid] = cp.disjoint(output + sBuckets[tid * cp.int32(NB_BLOCKS) + bx])

        block.barrier()

        # ---------------------------------------------------------------
        # Part 2: Per-iteration reorder loop
        # ---------------------------------------------------------------
        for _iter in range(64):
            # Phase 2a: Load keys, compute radix, warp-level rank computation
            for tid, thread in block.threads():
                warpIndexInBlock = tid >> cp.int32(5)
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)

                inputKeyIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                # Load keys and ranks (or pad with 0xFF radix if out of bounds)
                k0 = cp.int32(0)
                k1 = cp.int32(0)
                k2 = cp.int32(0)
                k3 = cp.int32(0)
                ki0 = cp.int32(0)
                ki1 = cp.int32(0)
                ki2 = cp.int32(0)
                ki3 = cp.int32(0)
                r0 = cp.int32(0xFF)
                r1 = cp.int32(0xFF)
                r2 = cp.int32(0xFF)
                r3 = cp.int32(0xFF)

                if inputKeyIndex < numKeysU4:
                    ki0 = inputRanks[inputKeyIndex, 0] + cp.int32(0)
                    ki1 = inputRanks[inputKeyIndex, 1] + cp.int32(0)
                    ki2 = inputRanks[inputKeyIndex, 2] + cp.int32(0)
                    ki3 = inputRanks[inputKeyIndex, 3] + cp.int32(0)

                    k0 = inputKeys[inputKeyIndex, 0] + cp.int32(0)
                    k1 = inputKeys[inputKeyIndex, 1] + cp.int32(0)
                    k2 = inputKeys[inputKeyIndex, 2] + cp.int32(0)
                    k3 = inputKeys[inputKeyIndex, 3] + cp.int32(0)

                    # sanitizeKeys
                    baseScalarIdx = inputKeyIndex * cp.int32(4)
                    goodVals = numKeys - baseScalarIdx
                    if goodVals < cp.int32(4):
                        k3 = cp.int32(-1)
                    if goodVals < cp.int32(3):
                        k2 = cp.int32(-1)
                    if goodVals < cp.int32(2):
                        k1 = cp.int32(-1)

                    r0 = (k0 >> startBit) & cp.int32(0xF)
                    r1 = (k1 >> startBit) & cp.int32(0xF)
                    r2 = (k2 >> startBit) & cp.int32(0xF)
                    r3 = (k3 >> startBit) & cp.int32(0xF)

                # Compute radix offsets within warp using packed 8-bit counters.
                # For each group of 4 radixes (i=0,4,8,12):
                #   accum = sum of (1 << ((radix_j - i) << 3)) for j in {x,y,z,w}
                #   val = warpScanAdd<WARP_SIZE>(accum, 0) -> inclusive scan
                #   val2 = shfl(val, 31) -> total
                #   For tiw < 4: sRadixSum[(i+tiw)*NUM_WARPS + warpIdx] = (val2 >> (8*tiw)) & 0xFF
                #   val -= accum -> exclusive scan
                #   Then extract per-element offsets from packed val

                # radixOffset for each of the 4 keys
                ro0 = cp.int32(0)
                ro1 = cp.int32(0)
                ro2 = cp.int32(0)
                ro3 = cp.int32(0)

                # Group i=0 (radixes 0-3)
                accum = (cp.int32(1) << ((r0 - cp.int32(0)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r1 - cp.int32(0)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r2 - cp.int32(0)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r3 - cp.int32(0)) << cp.int32(3)))
                # Inclusive warp scan
                val = accum
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n
                val2 = thread.shfl_idx(val, 31)
                if threadIndexInWarp < cp.int32(4):
                    sRadixSum[(cp.int32(0) + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint((val2 >> (cp.int32(8) * threadIndexInWarp)) & cp.int32(0xFF))
                val = val - accum  # exclusive
                # Extract per-key offsets
                sb0 = (r0 - cp.int32(0)) << cp.int32(3)
                ro0 = ro0 | ((val >> sb0) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb0)
                sb1 = (r1 - cp.int32(0)) << cp.int32(3)
                ro1 = ro1 | ((val >> sb1) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb1)
                sb2 = (r2 - cp.int32(0)) << cp.int32(3)
                ro2 = ro2 | ((val >> sb2) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb2)
                sb3 = (r3 - cp.int32(0)) << cp.int32(3)
                ro3 = ro3 | ((val >> sb3) & cp.int32(0xFF))

                # Group i=4 (radixes 4-7)
                accum = (cp.int32(1) << ((r0 - cp.int32(4)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r1 - cp.int32(4)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r2 - cp.int32(4)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r3 - cp.int32(4)) << cp.int32(3)))
                val = accum
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n
                val2 = thread.shfl_idx(val, 31)
                if threadIndexInWarp < cp.int32(4):
                    sRadixSum[(cp.int32(4) + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint((val2 >> (cp.int32(8) * threadIndexInWarp)) & cp.int32(0xFF))
                val = val - accum
                sb0 = (r0 - cp.int32(4)) << cp.int32(3)
                ro0 = ro0 | ((val >> sb0) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb0)
                sb1 = (r1 - cp.int32(4)) << cp.int32(3)
                ro1 = ro1 | ((val >> sb1) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb1)
                sb2 = (r2 - cp.int32(4)) << cp.int32(3)
                ro2 = ro2 | ((val >> sb2) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb2)
                sb3 = (r3 - cp.int32(4)) << cp.int32(3)
                ro3 = ro3 | ((val >> sb3) & cp.int32(0xFF))

                # Group i=8 (radixes 8-11)
                accum = (cp.int32(1) << ((r0 - cp.int32(8)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r1 - cp.int32(8)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r2 - cp.int32(8)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r3 - cp.int32(8)) << cp.int32(3)))
                val = accum
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n
                val2 = thread.shfl_idx(val, 31)
                if threadIndexInWarp < cp.int32(4):
                    sRadixSum[(cp.int32(8) + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint((val2 >> (cp.int32(8) * threadIndexInWarp)) & cp.int32(0xFF))
                val = val - accum
                sb0 = (r0 - cp.int32(8)) << cp.int32(3)
                ro0 = ro0 | ((val >> sb0) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb0)
                sb1 = (r1 - cp.int32(8)) << cp.int32(3)
                ro1 = ro1 | ((val >> sb1) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb1)
                sb2 = (r2 - cp.int32(8)) << cp.int32(3)
                ro2 = ro2 | ((val >> sb2) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb2)
                sb3 = (r3 - cp.int32(8)) << cp.int32(3)
                ro3 = ro3 | ((val >> sb3) & cp.int32(0xFF))

                # Group i=12 (radixes 12-15)
                accum = (cp.int32(1) << ((r0 - cp.int32(12)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r1 - cp.int32(12)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r2 - cp.int32(12)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r3 - cp.int32(12)) << cp.int32(3)))
                val = accum
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n
                val2 = thread.shfl_idx(val, 31)
                if threadIndexInWarp < cp.int32(4):
                    sRadixSum[(cp.int32(12) + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint((val2 >> (cp.int32(8) * threadIndexInWarp)) & cp.int32(0xFF))
                val = val - accum
                sb0 = (r0 - cp.int32(12)) << cp.int32(3)
                ro0 = ro0 | ((val >> sb0) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb0)
                sb1 = (r1 - cp.int32(12)) << cp.int32(3)
                ro1 = ro1 | ((val >> sb1) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb1)
                sb2 = (r2 - cp.int32(12)) << cp.int32(3)
                ro2 = ro2 | ((val >> sb2) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb2)
                sb3 = (r3 - cp.int32(12)) << cp.int32(3)
                ro3 = ro3 | ((val >> sb3) & cp.int32(0xFF))

                # Store radixOffset and key/rank data to per-thread shared mem slots
                # for use after the cross-warp scan. We reuse sBuckets as scratch.
                sBuckets[tid * cp.int32(4) + cp.int32(0)] = cp.disjoint(ro0)
                sBuckets[tid * cp.int32(4) + cp.int32(1)] = cp.disjoint(ro1)
                sBuckets[tid * cp.int32(4) + cp.int32(2)] = cp.disjoint(ro2)
                sBuckets[tid * cp.int32(4) + cp.int32(3)] = cp.disjoint(ro3)

            block.barrier()

            # Phase 2b: Save lastRadixSum before cross-warp scan overwrites sRadixSum
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    lastRS = sRadixSum[tid * cp.int32(NUM_WARPS) + cp.int32(NUM_WARPS - 1)] + cp.int32(0)
                    sRadixCount[tid] = cp.disjoint(lastRS)

            block.barrier()

            # Phase 2c: Cross-warp exclusive scan of sRadixSum (scanRadixWarps<NUM_WARPS>)
            # Only first NUM_WARPS/2 = 16 warps (512 threads).
            # sRadixSum[radix * NUM_WARPS + warp] -> exclusive prefix sum within each group of NUM_WARPS.
            for tid, thread in block.threads():
                warpIndexInBlock = tid >> cp.int32(5)
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                if cp.assume_uniform(warpIndexInBlock < cp.int32(NUM_WARPS // 2)):
                    tempVal = sRadixSum[tid] + cp.int32(0)

                    # Inclusive scan within warp (NUM_WARPS=32=WARP_SIZE elements per group)
                    val = tempVal
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(1))
                    if threadIndexInWarp >= cp.int32(1):
                        val = val + n
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(2))
                    if threadIndexInWarp >= cp.int32(2):
                        val = val + n
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(4))
                    if threadIndexInWarp >= cp.int32(4):
                        val = val + n
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(8))
                    if threadIndexInWarp >= cp.int32(8):
                        val = val + n
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(16))
                    if threadIndexInWarp >= cp.int32(16):
                        val = val + n

                    # scanRadixWarps returns val - tempVal (exclusive)
                    sRadixSum[tid] = cp.disjoint(val - tempVal)

            block.barrier()

            # Phase 2d: Compute sRadixSumSum (cross-radix prefix sum of per-radix totals)
            for tid, thread in block.threads():
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    # Total for radix tid = last warp's exclusive + last warp's count
                    value = sRadixSum[tid * cp.int32(NUM_WARPS) + cp.int32(NUM_WARPS - 1)] + cp.int32(0)
                    value = value + sRadixCount[tid]
                    sRadixCount[tid] = cp.disjoint(value)
                    sRadixSumSum[tid] = cp.disjoint(value)

            block.barrier()

            # warpScanAddWriteToSharedMem<RADIX_SIZE> on sRadixSumSum
            for tid, thread in block.threads():
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    value = sRadixSumSum[tid] + cp.int32(0)
                    val = value
                    n = thread.shfl_up(val, 1)
                    if threadIndexInWarp >= cp.int32(1):
                        val = val + n
                    n = thread.shfl_up(val, 2)
                    if threadIndexInWarp >= cp.int32(2):
                        val = val + n
                    n = thread.shfl_up(val, 4)
                    if threadIndexInWarp >= cp.int32(4):
                        val = val + n
                    n = thread.shfl_up(val, 8)
                    if threadIndexInWarp >= cp.int32(8):
                        val = val + n
                    # Exclusive = inclusive - original
                    sRadixSumSum[tid] = cp.disjoint(val - value)

            block.barrier()

            # Phase 2e: Add sRadixSumSum to sRadixSum
            # if idx < NUM_WARPS * RADIX_SIZE: sRadixSum[idx] += sRadixSumSum[idx / NUM_WARPS]
            for tid, thread in block.threads():
                if tid < cp.int32(NUM_WARPS * RADIX_SIZE):
                    radixIdx = tid // cp.int32(NUM_WARPS)
                    addVal = sRadixSumSum[radixIdx] + cp.int32(0)
                    curVal = sRadixSum[tid] + cp.int32(0)
                    sRadixSum[tid] = cp.disjoint(curVal + addVal)

            block.barrier()

            # Phase 2f: Compute global offsets
            # sRadixSumSum[idx] = sRadixSumBetweenBlocks[idx] - sRadixSumSum[idx]
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    betweenBlocks = sRadixSumBetweenBlocks[tid] + cp.int32(0)
                    sumSum = sRadixSumSum[tid] + cp.int32(0)
                    sRadixSumSum[tid] = cp.disjoint(betweenBlocks - sumSum)

            block.barrier()

            # Phase 2g: Scatter keys and ranks to shared memory
            for tid, thread in block.threads():
                inputKeyIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                k0 = cp.int32(0)
                k1 = cp.int32(0)
                k2 = cp.int32(0)
                k3 = cp.int32(0)
                ki0 = cp.int32(0)
                ki1 = cp.int32(0)
                ki2 = cp.int32(0)
                ki3 = cp.int32(0)
                r0 = cp.int32(0xFF)
                r1 = cp.int32(0xFF)
                r2 = cp.int32(0xFF)
                r3 = cp.int32(0xFF)

                if inputKeyIndex < numKeysU4:
                    ki0 = inputRanks[inputKeyIndex, 0] + cp.int32(0)
                    ki1 = inputRanks[inputKeyIndex, 1] + cp.int32(0)
                    ki2 = inputRanks[inputKeyIndex, 2] + cp.int32(0)
                    ki3 = inputRanks[inputKeyIndex, 3] + cp.int32(0)

                    k0 = inputKeys[inputKeyIndex, 0] + cp.int32(0)
                    k1 = inputKeys[inputKeyIndex, 1] + cp.int32(0)
                    k2 = inputKeys[inputKeyIndex, 2] + cp.int32(0)
                    k3 = inputKeys[inputKeyIndex, 3] + cp.int32(0)

                    baseScalarIdx = inputKeyIndex * cp.int32(4)
                    goodVals = numKeys - baseScalarIdx
                    if goodVals < cp.int32(4):
                        k3 = cp.int32(-1)
                    if goodVals < cp.int32(3):
                        k2 = cp.int32(-1)
                    if goodVals < cp.int32(2):
                        k1 = cp.int32(-1)

                    r0 = (k0 >> startBit) & cp.int32(0xF)
                    r1 = (k1 >> startBit) & cp.int32(0xF)
                    r2 = (k2 >> startBit) & cp.int32(0xF)
                    r3 = (k3 >> startBit) & cp.int32(0xF)

                    warpIndexInBlock = tid >> cp.int32(5)

                    # Reload radix offsets from sBuckets scratch
                    ro0 = sBuckets[tid * cp.int32(4) + cp.int32(0)] + cp.int32(0)
                    ro1 = sBuckets[tid * cp.int32(4) + cp.int32(1)] + cp.int32(0)
                    ro2 = sBuckets[tid * cp.int32(4) + cp.int32(2)] + cp.int32(0)
                    ro3 = sBuckets[tid * cp.int32(4) + cp.int32(3)] + cp.int32(0)

                    # Add cross-warp offset: sRadixSum[radix * NUM_WARPS + warpIndexInBlock]
                    ro0 = ro0 + sRadixSum[r0 * cp.int32(NUM_WARPS) + warpIndexInBlock]
                    ro1 = ro1 + sRadixSum[r1 * cp.int32(NUM_WARPS) + warpIndexInBlock]
                    ro2 = ro2 + sRadixSum[r2 * cp.int32(NUM_WARPS) + warpIndexInBlock]
                    ro3 = ro3 + sRadixSum[r3 * cp.int32(NUM_WARPS) + warpIndexInBlock]

                    sKeys[ro0] = cp.disjoint(k0)
                    sKeys[ro1] = cp.disjoint(k1)
                    sKeys[ro2] = cp.disjoint(k2)
                    sKeys[ro3] = cp.disjoint(k3)

                    sRanks[ro0] = cp.disjoint(ki0)
                    sRanks[ro1] = cp.disjoint(ki1)
                    sRanks[ro2] = cp.disjoint(ki2)
                    sRanks[ro3] = cp.disjoint(ki3)

            block.barrier()

            # Phase 2h: Write from shared memory to global output
            for tid, thread in block.threads():
                inputKeyIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)
                baseInputKeyIndex = inputKeyIndex - tid

                if baseInputKeyIndex < numKeysU4:
                    keysToProcess_raw = numKeys - baseInputKeyIndex * cp.int32(4)
                    keysToProcess_max = cp.int32(WARP_SIZE * NUM_WARPS * 4)
                    keysToProcess = keysToProcess_raw if keysToProcess_raw < keysToProcess_max else keysToProcess_max

                    # Grid-stride write: for a = tid; a < keysToProcess; a += BLOCK_SIZE
                    # Since keysToProcess <= BLOCK_SIZE * 4 and stride = BLOCK_SIZE,
                    # each thread writes up to 4 elements.
                    a0 = tid
                    if a0 < keysToProcess:
                        key = sKeys[a0] + cp.int32(0)
                        radix = (key >> startBit) & cp.int32(0xF)
                        writeIndex = a0 + sRadixSumSum[radix]
                        outputKeys[writeIndex] = key
                        outputRanks[writeIndex] = sRanks[a0] + cp.int32(0)

                    a1 = tid + cp.int32(BLOCK_SIZE)
                    if a1 < keysToProcess:
                        key = sKeys[a1] + cp.int32(0)
                        radix = (key >> startBit) & cp.int32(0xF)
                        writeIndex = a1 + sRadixSumSum[radix]
                        outputKeys[writeIndex] = key
                        outputRanks[writeIndex] = sRanks[a1] + cp.int32(0)

                    a2 = tid + cp.int32(BLOCK_SIZE * 2)
                    if a2 < keysToProcess:
                        key = sKeys[a2] + cp.int32(0)
                        radix = (key >> startBit) & cp.int32(0xF)
                        writeIndex = a2 + sRadixSumSum[radix]
                        outputKeys[writeIndex] = key
                        outputRanks[writeIndex] = sRanks[a2] + cp.int32(0)

                    a3 = tid + cp.int32(BLOCK_SIZE * 3)
                    if a3 < keysToProcess:
                        key = sKeys[a3] + cp.int32(0)
                        radix = (key >> startBit) & cp.int32(0xF)
                        writeIndex = a3 + sRadixSumSum[radix]
                        outputKeys[writeIndex] = key
                        outputRanks[writeIndex] = sRanks[a3] + cp.int32(0)

            block.barrier()

            # Phase 2i: Update sRadixSumBetweenBlocks for next iteration
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    betweenBlocks = sRadixSumBetweenBlocks[tid] + cp.int32(0)
                    radixCnt = sRadixCount[tid] + cp.int32(0)
                    sRadixSumBetweenBlocks[tid] = cp.disjoint(betweenBlocks + radixCnt)

            block.barrier()


@cp.kernel
def radixSortMultiCalculateRanksLaunchWithCount(inputKeys, inputRanks, numKeys, startBit,
                                                 radixCount, outputKeys, outputRanks,
                                                 BLOCK_SIZE: cp.constexpr = 1024,
                                                 NUM_WARPS: cp.constexpr = 32,
                                                 RADIX_SIZE: cp.constexpr = 16,
                                                 NB_BLOCKS: cp.constexpr = 32):
    with cp.Kernel(NB_BLOCKS, threads=BLOCK_SIZE) as (bx, block):
        # Shared memory allocations matching CUDA:
        # sRadixSumBetweenBlocks[RADIX_SIZE] — global prefix sums per radix
        # sBuckets[NUM_WARPS * WARP_SIZE] — scratch for warp scans
        # sRadixSum[RADIX_SIZE * NUM_WARPS] — per-warp radix counts within block
        # sRadixSumSum[RADIX_SIZE] — prefix sum of radix totals
        # sRadixCount[RADIX_SIZE] — radix totals for current iteration
        # sKeys[WARP_SIZE * NUM_WARPS * 4] — scattered keys in shared mem
        # sRanks[WARP_SIZE * NUM_WARPS * 4] — scattered ranks in shared mem
        sRadixSumBetweenBlocks = block.alloc((RADIX_SIZE,), dtype=cp.int32)
        sBuckets = block.alloc((NUM_WARPS * WARP_SIZE,), dtype=cp.int32)
        sRadixSum = block.alloc((RADIX_SIZE * NUM_WARPS,), dtype=cp.int32)
        sRadixSumSum = block.alloc((RADIX_SIZE,), dtype=cp.int32)
        sRadixCount = block.alloc((RADIX_SIZE,), dtype=cp.int32)
        sKeys = block.alloc((WARP_SIZE * NUM_WARPS * 4,), dtype=cp.int32)
        sRanks = block.alloc((WARP_SIZE * NUM_WARPS * 4,), dtype=cp.int32)

        numKeysU4 = (numKeys + cp.int32(3)) >> cp.int32(2)
        totalBlockRequired = (numKeysU4 + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterationPerBlock = (totalBlockRequired + cp.int32(NB_BLOCKS - 1)) // cp.int32(NB_BLOCKS)

        # ---------------------------------------------------------------
        # Part 1: scanRadixes — scan radixCount across blocks per radix
        # ---------------------------------------------------------------
        # scanRadixes<NUM_WARPS>(warpIndexInBlock, threadIndexInWarp, gRadixCount,
        #                        sRadixCountBetweenBlocks=sBuckets, sRadixSumBetweenBlocks)
        #
        # For each radix i (i = warpIndexInBlock; i < RADIX_SIZE; i += NUM_WARPS):
        #   value = gRadixCount[i * gridDim.x + threadIndexInWarp]
        #   output = warpScanAddWriteToSharedMem<WARP_SIZE>(index=radixSumIndex, tiw, sData=sBuckets, value, value)
        #   if tiw == WARP_SIZE-1: sRadixSumBetweenBlocks[i] = output + value
        #
        # Since NUM_WARPS=32 and RADIX_SIZE=16, each warp handles at most 1 radix
        # (i = warpIndexInBlock, step NUM_WARPS=32 > 16, so only warpIndexInBlock < 16 runs).
        # threadIndexInWarp indexes across NB_BLOCKS=32 blocks.
        # warpScanAddWriteToSharedMem<WARP_SIZE> does inclusive scan and writes
        # exclusive result (val - value) to sData[index], returns exclusive result.
        # So sBuckets[i*gridDim.x + tiw] = exclusive prefix sum of radixCount for radix i.
        # sRadixSumBetweenBlocks[i] = total count for radix i across all blocks.

        for tid, thread in block.threads():
            warpIndexInBlock = tid >> cp.int32(5)
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
            if cp.assume_uniform(warpIndexInBlock < cp.int32(RADIX_SIZE)):
                radixSumIndex = warpIndexInBlock * cp.int32(NB_BLOCKS) + threadIndexInWarp
                value = radixCount[radixSumIndex] + cp.int32(0)

                # warpScanAddWriteToSharedMem<WARP_SIZE>: inclusive scan, write exclusive
                val = value
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n

                # Exclusive result = inclusive - original
                exclusive = val - value
                sBuckets[radixSumIndex] = cp.disjoint(exclusive)
                if threadIndexInWarp == cp.int32(WARP_SIZE - 1):
                    # Total = exclusive + value = val
                    sRadixSumBetweenBlocks[warpIndexInBlock] = cp.disjoint(val)

        block.barrier()

        # Part 1b: Exclusive scan of sRadixSumBetweenBlocks[RADIX_SIZE=16]
        # if idx < RADIX_SIZE:
        #   value = sRadixSumBetweenBlocks[idx]
        #   output = warpScanAdd<RADIX_SIZE>(value, value) -> inclusive - value = exclusive
        #   sRadixSumBetweenBlocks[idx] = output + sBuckets[idx * NB_BLOCKS + bx]
        # This gives each block's global starting offset for each radix.
        for tid, thread in block.threads():
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
            if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                value = sRadixSumBetweenBlocks[tid] + cp.int32(0)

                # warpScanAdd<RADIX_SIZE>: inclusive scan among first 16 threads
                val = value
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n

                # warpScanAdd returns val - value (exclusive prefix sum)
                output = val - value
                # Add this block's offset from the cross-block scan
                sRadixSumBetweenBlocks[tid] = cp.disjoint(output + sBuckets[tid * cp.int32(NB_BLOCKS) + bx])

        block.barrier()

        # ---------------------------------------------------------------
        # Part 2: Per-iteration reorder loop
        # ---------------------------------------------------------------
        for _iter in range(64):
            # Phase 2a: Load keys, compute radix, warp-level rank computation
            for tid, thread in block.threads():
                warpIndexInBlock = tid >> cp.int32(5)
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)

                inputKeyIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                # Load keys and ranks (or pad with 0xFF radix if out of bounds)
                k0 = cp.int32(0)
                k1 = cp.int32(0)
                k2 = cp.int32(0)
                k3 = cp.int32(0)
                ki0 = cp.int32(0)
                ki1 = cp.int32(0)
                ki2 = cp.int32(0)
                ki3 = cp.int32(0)
                r0 = cp.int32(0xFF)
                r1 = cp.int32(0xFF)
                r2 = cp.int32(0xFF)
                r3 = cp.int32(0xFF)

                if inputKeyIndex < numKeysU4:
                    ki0 = inputRanks[inputKeyIndex, 0] + cp.int32(0)
                    ki1 = inputRanks[inputKeyIndex, 1] + cp.int32(0)
                    ki2 = inputRanks[inputKeyIndex, 2] + cp.int32(0)
                    ki3 = inputRanks[inputKeyIndex, 3] + cp.int32(0)

                    k0 = inputKeys[inputKeyIndex, 0] + cp.int32(0)
                    k1 = inputKeys[inputKeyIndex, 1] + cp.int32(0)
                    k2 = inputKeys[inputKeyIndex, 2] + cp.int32(0)
                    k3 = inputKeys[inputKeyIndex, 3] + cp.int32(0)

                    # sanitizeKeys
                    baseScalarIdx = inputKeyIndex * cp.int32(4)
                    goodVals = numKeys - baseScalarIdx
                    if goodVals < cp.int32(4):
                        k3 = cp.int32(-1)
                    if goodVals < cp.int32(3):
                        k2 = cp.int32(-1)
                    if goodVals < cp.int32(2):
                        k1 = cp.int32(-1)

                    r0 = (k0 >> startBit) & cp.int32(0xF)
                    r1 = (k1 >> startBit) & cp.int32(0xF)
                    r2 = (k2 >> startBit) & cp.int32(0xF)
                    r3 = (k3 >> startBit) & cp.int32(0xF)

                # Compute radix offsets within warp using packed 8-bit counters.
                # For each group of 4 radixes (i=0,4,8,12):
                #   accum = sum of (1 << ((radix_j - i) << 3)) for j in {x,y,z,w}
                #   val = warpScanAdd<WARP_SIZE>(accum, 0) -> inclusive scan
                #   val2 = shfl(val, 31) -> total
                #   For tiw < 4: sRadixSum[(i+tiw)*NUM_WARPS + warpIdx] = (val2 >> (8*tiw)) & 0xFF
                #   val -= accum -> exclusive scan
                #   Then extract per-element offsets from packed val

                # radixOffset for each of the 4 keys
                ro0 = cp.int32(0)
                ro1 = cp.int32(0)
                ro2 = cp.int32(0)
                ro3 = cp.int32(0)

                # Group i=0 (radixes 0-3)
                accum = (cp.int32(1) << ((r0 - cp.int32(0)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r1 - cp.int32(0)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r2 - cp.int32(0)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r3 - cp.int32(0)) << cp.int32(3)))
                # Inclusive warp scan
                val = accum
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n
                val2 = thread.shfl_idx(val, 31)
                if threadIndexInWarp < cp.int32(4):
                    sRadixSum[(cp.int32(0) + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint((val2 >> (cp.int32(8) * threadIndexInWarp)) & cp.int32(0xFF))
                val = val - accum  # exclusive
                # Extract per-key offsets
                sb0 = (r0 - cp.int32(0)) << cp.int32(3)
                ro0 = ro0 | ((val >> sb0) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb0)
                sb1 = (r1 - cp.int32(0)) << cp.int32(3)
                ro1 = ro1 | ((val >> sb1) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb1)
                sb2 = (r2 - cp.int32(0)) << cp.int32(3)
                ro2 = ro2 | ((val >> sb2) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb2)
                sb3 = (r3 - cp.int32(0)) << cp.int32(3)
                ro3 = ro3 | ((val >> sb3) & cp.int32(0xFF))

                # Group i=4 (radixes 4-7)
                accum = (cp.int32(1) << ((r0 - cp.int32(4)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r1 - cp.int32(4)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r2 - cp.int32(4)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r3 - cp.int32(4)) << cp.int32(3)))
                val = accum
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n
                val2 = thread.shfl_idx(val, 31)
                if threadIndexInWarp < cp.int32(4):
                    sRadixSum[(cp.int32(4) + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint((val2 >> (cp.int32(8) * threadIndexInWarp)) & cp.int32(0xFF))
                val = val - accum
                sb0 = (r0 - cp.int32(4)) << cp.int32(3)
                ro0 = ro0 | ((val >> sb0) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb0)
                sb1 = (r1 - cp.int32(4)) << cp.int32(3)
                ro1 = ro1 | ((val >> sb1) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb1)
                sb2 = (r2 - cp.int32(4)) << cp.int32(3)
                ro2 = ro2 | ((val >> sb2) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb2)
                sb3 = (r3 - cp.int32(4)) << cp.int32(3)
                ro3 = ro3 | ((val >> sb3) & cp.int32(0xFF))

                # Group i=8 (radixes 8-11)
                accum = (cp.int32(1) << ((r0 - cp.int32(8)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r1 - cp.int32(8)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r2 - cp.int32(8)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r3 - cp.int32(8)) << cp.int32(3)))
                val = accum
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n
                val2 = thread.shfl_idx(val, 31)
                if threadIndexInWarp < cp.int32(4):
                    sRadixSum[(cp.int32(8) + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint((val2 >> (cp.int32(8) * threadIndexInWarp)) & cp.int32(0xFF))
                val = val - accum
                sb0 = (r0 - cp.int32(8)) << cp.int32(3)
                ro0 = ro0 | ((val >> sb0) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb0)
                sb1 = (r1 - cp.int32(8)) << cp.int32(3)
                ro1 = ro1 | ((val >> sb1) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb1)
                sb2 = (r2 - cp.int32(8)) << cp.int32(3)
                ro2 = ro2 | ((val >> sb2) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb2)
                sb3 = (r3 - cp.int32(8)) << cp.int32(3)
                ro3 = ro3 | ((val >> sb3) & cp.int32(0xFF))

                # Group i=12 (radixes 12-15)
                accum = (cp.int32(1) << ((r0 - cp.int32(12)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r1 - cp.int32(12)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r2 - cp.int32(12)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r3 - cp.int32(12)) << cp.int32(3)))
                val = accum
                n = thread.shfl_up(val, 1)
                if threadIndexInWarp >= cp.int32(1):
                    val = val + n
                n = thread.shfl_up(val, 2)
                if threadIndexInWarp >= cp.int32(2):
                    val = val + n
                n = thread.shfl_up(val, 4)
                if threadIndexInWarp >= cp.int32(4):
                    val = val + n
                n = thread.shfl_up(val, 8)
                if threadIndexInWarp >= cp.int32(8):
                    val = val + n
                n = thread.shfl_up(val, 16)
                if threadIndexInWarp >= cp.int32(16):
                    val = val + n
                val2 = thread.shfl_idx(val, 31)
                if threadIndexInWarp < cp.int32(4):
                    sRadixSum[(cp.int32(12) + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint((val2 >> (cp.int32(8) * threadIndexInWarp)) & cp.int32(0xFF))
                val = val - accum
                sb0 = (r0 - cp.int32(12)) << cp.int32(3)
                ro0 = ro0 | ((val >> sb0) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb0)
                sb1 = (r1 - cp.int32(12)) << cp.int32(3)
                ro1 = ro1 | ((val >> sb1) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb1)
                sb2 = (r2 - cp.int32(12)) << cp.int32(3)
                ro2 = ro2 | ((val >> sb2) & cp.int32(0xFF))
                val = val + (cp.int32(1) << sb2)
                sb3 = (r3 - cp.int32(12)) << cp.int32(3)
                ro3 = ro3 | ((val >> sb3) & cp.int32(0xFF))

                # Store radixOffset and key/rank data to per-thread shared mem slots
                # for use after the cross-warp scan. We reuse sBuckets as scratch.
                sBuckets[tid * cp.int32(4) + cp.int32(0)] = cp.disjoint(ro0)
                sBuckets[tid * cp.int32(4) + cp.int32(1)] = cp.disjoint(ro1)
                sBuckets[tid * cp.int32(4) + cp.int32(2)] = cp.disjoint(ro2)
                sBuckets[tid * cp.int32(4) + cp.int32(3)] = cp.disjoint(ro3)

            block.barrier()

            # Phase 2b: Save lastRadixSum before cross-warp scan overwrites sRadixSum
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    lastRS = sRadixSum[tid * cp.int32(NUM_WARPS) + cp.int32(NUM_WARPS - 1)] + cp.int32(0)
                    sRadixCount[tid] = cp.disjoint(lastRS)

            block.barrier()

            # Phase 2c: Cross-warp exclusive scan of sRadixSum (scanRadixWarps<NUM_WARPS>)
            # Only first NUM_WARPS/2 = 16 warps (512 threads).
            # sRadixSum[radix * NUM_WARPS + warp] -> exclusive prefix sum within each group of NUM_WARPS.
            for tid, thread in block.threads():
                warpIndexInBlock = tid >> cp.int32(5)
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                if cp.assume_uniform(warpIndexInBlock < cp.int32(NUM_WARPS // 2)):
                    tempVal = sRadixSum[tid] + cp.int32(0)

                    # Inclusive scan within warp (NUM_WARPS=32=WARP_SIZE elements per group)
                    val = tempVal
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(1))
                    if threadIndexInWarp >= cp.int32(1):
                        val = val + n
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(2))
                    if threadIndexInWarp >= cp.int32(2):
                        val = val + n
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(4))
                    if threadIndexInWarp >= cp.int32(4):
                        val = val + n
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(8))
                    if threadIndexInWarp >= cp.int32(8):
                        val = val + n
                    n = thread.shfl_idx(val, threadIndexInWarp - cp.int32(16))
                    if threadIndexInWarp >= cp.int32(16):
                        val = val + n

                    # scanRadixWarps returns val - tempVal (exclusive)
                    sRadixSum[tid] = cp.disjoint(val - tempVal)

            block.barrier()

            # Phase 2d: Compute sRadixSumSum (cross-radix prefix sum of per-radix totals)
            for tid, thread in block.threads():
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    # Total for radix tid = last warp's exclusive + last warp's count
                    value = sRadixSum[tid * cp.int32(NUM_WARPS) + cp.int32(NUM_WARPS - 1)] + cp.int32(0)
                    value = value + sRadixCount[tid]
                    sRadixCount[tid] = cp.disjoint(value)
                    sRadixSumSum[tid] = cp.disjoint(value)

            block.barrier()

            # warpScanAddWriteToSharedMem<RADIX_SIZE> on sRadixSumSum
            for tid, thread in block.threads():
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    value = sRadixSumSum[tid] + cp.int32(0)
                    val = value
                    n = thread.shfl_up(val, 1)
                    if threadIndexInWarp >= cp.int32(1):
                        val = val + n
                    n = thread.shfl_up(val, 2)
                    if threadIndexInWarp >= cp.int32(2):
                        val = val + n
                    n = thread.shfl_up(val, 4)
                    if threadIndexInWarp >= cp.int32(4):
                        val = val + n
                    n = thread.shfl_up(val, 8)
                    if threadIndexInWarp >= cp.int32(8):
                        val = val + n
                    # Exclusive = inclusive - original
                    sRadixSumSum[tid] = cp.disjoint(val - value)

            block.barrier()

            # Phase 2e: Add sRadixSumSum to sRadixSum
            # if idx < NUM_WARPS * RADIX_SIZE: sRadixSum[idx] += sRadixSumSum[idx / NUM_WARPS]
            for tid, thread in block.threads():
                if tid < cp.int32(NUM_WARPS * RADIX_SIZE):
                    radixIdx = tid // cp.int32(NUM_WARPS)
                    addVal = sRadixSumSum[radixIdx] + cp.int32(0)
                    curVal = sRadixSum[tid] + cp.int32(0)
                    sRadixSum[tid] = cp.disjoint(curVal + addVal)

            block.barrier()

            # Phase 2f: Compute global offsets
            # sRadixSumSum[idx] = sRadixSumBetweenBlocks[idx] - sRadixSumSum[idx]
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    betweenBlocks = sRadixSumBetweenBlocks[tid] + cp.int32(0)
                    sumSum = sRadixSumSum[tid] + cp.int32(0)
                    sRadixSumSum[tid] = cp.disjoint(betweenBlocks - sumSum)

            block.barrier()

            # Phase 2g: Scatter keys and ranks to shared memory
            for tid, thread in block.threads():
                inputKeyIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                k0 = cp.int32(0)
                k1 = cp.int32(0)
                k2 = cp.int32(0)
                k3 = cp.int32(0)
                ki0 = cp.int32(0)
                ki1 = cp.int32(0)
                ki2 = cp.int32(0)
                ki3 = cp.int32(0)
                r0 = cp.int32(0xFF)
                r1 = cp.int32(0xFF)
                r2 = cp.int32(0xFF)
                r3 = cp.int32(0xFF)

                if inputKeyIndex < numKeysU4:
                    ki0 = inputRanks[inputKeyIndex, 0] + cp.int32(0)
                    ki1 = inputRanks[inputKeyIndex, 1] + cp.int32(0)
                    ki2 = inputRanks[inputKeyIndex, 2] + cp.int32(0)
                    ki3 = inputRanks[inputKeyIndex, 3] + cp.int32(0)

                    k0 = inputKeys[inputKeyIndex, 0] + cp.int32(0)
                    k1 = inputKeys[inputKeyIndex, 1] + cp.int32(0)
                    k2 = inputKeys[inputKeyIndex, 2] + cp.int32(0)
                    k3 = inputKeys[inputKeyIndex, 3] + cp.int32(0)

                    baseScalarIdx = inputKeyIndex * cp.int32(4)
                    goodVals = numKeys - baseScalarIdx
                    if goodVals < cp.int32(4):
                        k3 = cp.int32(-1)
                    if goodVals < cp.int32(3):
                        k2 = cp.int32(-1)
                    if goodVals < cp.int32(2):
                        k1 = cp.int32(-1)

                    r0 = (k0 >> startBit) & cp.int32(0xF)
                    r1 = (k1 >> startBit) & cp.int32(0xF)
                    r2 = (k2 >> startBit) & cp.int32(0xF)
                    r3 = (k3 >> startBit) & cp.int32(0xF)

                    warpIndexInBlock = tid >> cp.int32(5)

                    # Reload radix offsets from sBuckets scratch
                    ro0 = sBuckets[tid * cp.int32(4) + cp.int32(0)] + cp.int32(0)
                    ro1 = sBuckets[tid * cp.int32(4) + cp.int32(1)] + cp.int32(0)
                    ro2 = sBuckets[tid * cp.int32(4) + cp.int32(2)] + cp.int32(0)
                    ro3 = sBuckets[tid * cp.int32(4) + cp.int32(3)] + cp.int32(0)

                    # Add cross-warp offset: sRadixSum[radix * NUM_WARPS + warpIndexInBlock]
                    ro0 = ro0 + sRadixSum[r0 * cp.int32(NUM_WARPS) + warpIndexInBlock]
                    ro1 = ro1 + sRadixSum[r1 * cp.int32(NUM_WARPS) + warpIndexInBlock]
                    ro2 = ro2 + sRadixSum[r2 * cp.int32(NUM_WARPS) + warpIndexInBlock]
                    ro3 = ro3 + sRadixSum[r3 * cp.int32(NUM_WARPS) + warpIndexInBlock]

                    sKeys[ro0] = cp.disjoint(k0)
                    sKeys[ro1] = cp.disjoint(k1)
                    sKeys[ro2] = cp.disjoint(k2)
                    sKeys[ro3] = cp.disjoint(k3)

                    sRanks[ro0] = cp.disjoint(ki0)
                    sRanks[ro1] = cp.disjoint(ki1)
                    sRanks[ro2] = cp.disjoint(ki2)
                    sRanks[ro3] = cp.disjoint(ki3)

            block.barrier()

            # Phase 2h: Write from shared memory to global output
            for tid, thread in block.threads():
                inputKeyIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)
                baseInputKeyIndex = inputKeyIndex - tid

                if baseInputKeyIndex < numKeysU4:
                    keysToProcess_raw = numKeys - baseInputKeyIndex * cp.int32(4)
                    keysToProcess_max = cp.int32(WARP_SIZE * NUM_WARPS * 4)
                    keysToProcess = keysToProcess_raw if keysToProcess_raw < keysToProcess_max else keysToProcess_max

                    # Grid-stride write: for a = tid; a < keysToProcess; a += BLOCK_SIZE
                    # Since keysToProcess <= BLOCK_SIZE * 4 and stride = BLOCK_SIZE,
                    # each thread writes up to 4 elements.
                    a0 = tid
                    if a0 < keysToProcess:
                        key = sKeys[a0] + cp.int32(0)
                        radix = (key >> startBit) & cp.int32(0xF)
                        writeIndex = a0 + sRadixSumSum[radix]
                        outputKeys[writeIndex] = key
                        outputRanks[writeIndex] = sRanks[a0] + cp.int32(0)

                    a1 = tid + cp.int32(BLOCK_SIZE)
                    if a1 < keysToProcess:
                        key = sKeys[a1] + cp.int32(0)
                        radix = (key >> startBit) & cp.int32(0xF)
                        writeIndex = a1 + sRadixSumSum[radix]
                        outputKeys[writeIndex] = key
                        outputRanks[writeIndex] = sRanks[a1] + cp.int32(0)

                    a2 = tid + cp.int32(BLOCK_SIZE * 2)
                    if a2 < keysToProcess:
                        key = sKeys[a2] + cp.int32(0)
                        radix = (key >> startBit) & cp.int32(0xF)
                        writeIndex = a2 + sRadixSumSum[radix]
                        outputKeys[writeIndex] = key
                        outputRanks[writeIndex] = sRanks[a2] + cp.int32(0)

                    a3 = tid + cp.int32(BLOCK_SIZE * 3)
                    if a3 < keysToProcess:
                        key = sKeys[a3] + cp.int32(0)
                        radix = (key >> startBit) & cp.int32(0xF)
                        writeIndex = a3 + sRadixSumSum[radix]
                        outputKeys[writeIndex] = key
                        outputRanks[writeIndex] = sRanks[a3] + cp.int32(0)

            block.barrier()

            # Phase 2i: Update sRadixSumBetweenBlocks for next iteration
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    betweenBlocks = sRadixSumBetweenBlocks[tid] + cp.int32(0)
                    radixCnt = sRadixCount[tid] + cp.int32(0)
                    sRadixSumBetweenBlocks[tid] = cp.disjoint(betweenBlocks + radixCnt)

            block.barrier()

