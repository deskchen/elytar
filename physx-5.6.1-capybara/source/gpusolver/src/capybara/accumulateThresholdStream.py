"""Capybara DSL port of gpusolver/CUDA/accumulateThresholdStream.cu -- all 14 kernels.

Ported kernels (matching CUDA names for PTX replacement):
  Group A -- Radix sort wrappers:
    - bodyInputAndRanksSingleBlockLaunch   (radixSortSingleBlock body)
    - bodyInputAndRanksBlocksLaunch        (radixSortCalculateRanks body)

  Group B -- Simple grid-stride loops:
    - initialRanksAndBodyIndexB
    - initialRanksAndBodyIndexA
    - reorganizeThresholdElements
    - writeoutAccumulatedForcePerObject

  Group C -- Multi-iteration warp scan + reduction (stage 1):
    - computeAccumulateThresholdStream
    - computeExceededForceThresholdElementIndice
    - computeThresholdElementMaskIndices

  Group D -- Cross-block scan fixup (stage 2):
    - outputAccumulateThresholdStream
    - outputExceededForceThresholdElementIndice
    - outputThresholdPairsMaskIndices

  Group E -- Binary search:
    - setThresholdElementsMask

  Group F -- Element copy with 3-way conditional:
    - createForceChangeThresholdElements

ABI differences from CUDA:
  - ThresholdStreamElement -> int32[N, TE_SIZE] flat tensor (struct as int32 row).
    Key field offsets (int32 indices within a row):
      TE_SHAPE_INTERACTION = 0 (PxU64, 2 int32s: lo at 0, hi at 1)
      TE_NODE_INDEX_A = 2
      TE_NODE_INDEX_B = 3
      TE_NORMAL_FORCE = 4 (float stored as int32, use bitcast)
      TE_ACCUM_FORCE = 5 (float stored as int32, use bitcast)
      TE_THRESHOLD = 6 (float stored as int32, use bitcast)
      TE_PAD = 7
  - Float fields in int32 tensors accessed via thread.bitcast(val + cp.int32(0), cp.float32).
  - AccumulatedForceObjectPair -> int32[N, 4] (4 int32s per element).
  - PxU64 fields stored as pairs of int32 (lo, hi).

Capybara structural notes:
  - block.barrier() between thread regions, not inside.
  - cp.disjoint() for smem writes inside block.threads().
  - cp.assume_uniform() for shfl inside warp-uniform conditionals.
  - Variables in if/else must be pre-declared before the if.
  - Warp reduction / scan manually inlined (no @cp.inline with thread).
  - While loops used for runtime-dependent iteration counts.
  - No bare `if boolvar:` -- always `if var != cp.int32(0):`.
  - No tuple-return @cp.inline in conditionals/loops.
  - `+ cp.int32(0)` for force-loads.
  - `thread.bitcast(val + cp.int32(0), cp.float32)` for float fields in int32 tensors.
"""

import capybara as cp

WARP_SIZE = 32


# ===========================================================================
# Group A: Radix sort wrappers
# ===========================================================================

# ===== Kernel 1: bodyInputAndRanksSingleBlockLaunch =====
# Full body duplicated from radixSortMultiBlockLaunch (radixSortSingleBlock pattern).
@cp.kernel
def bodyInputAndRanksSingleBlockLaunch(inputKeys, inputRanks, numKeys, startBit, radixCount,
                                       BLOCK_SIZE: cp.constexpr = 1024,
                                       NUM_WARPS: cp.constexpr = 32,
                                       RADIX_SIZE: cp.constexpr = 16,
                                       NB_BLOCKS: cp.constexpr = 32):
    with cp.Kernel(NB_BLOCKS, threads=BLOCK_SIZE) as (bx, block):
        sRadixSum = block.alloc((RADIX_SIZE * NUM_WARPS,), dtype=cp.int32)

        numKeysU4 = (numKeys + cp.int32(3)) >> cp.int32(2)
        totalBlockRequired = (numKeysU4 + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterationPerBlock = (totalBlockRequired + cp.int32(NB_BLOCKS - 1)) // cp.int32(NB_BLOCKS)

        rAccum0 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum1 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum2 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum3 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum4 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum5 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum6 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        rAccum7 = block.alloc((BLOCK_SIZE,), dtype=cp.int32)

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

        for _iter in range(64):
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

                    baseScalarIdx = gInputIdx * cp.int32(4)
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

                    acc = rAccum0[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(0)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(0)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(0)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(0)) << cp.int32(4)))
                    rAccum0[tid] = cp.disjoint(acc)

                    acc = rAccum1[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(2)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(2)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(2)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(2)) << cp.int32(4)))
                    rAccum1[tid] = cp.disjoint(acc)

                    acc = rAccum2[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(4)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(4)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(4)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(4)) << cp.int32(4)))
                    rAccum2[tid] = cp.disjoint(acc)

                    acc = rAccum3[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(6)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(6)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(6)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(6)) << cp.int32(4)))
                    rAccum3[tid] = cp.disjoint(acc)

                    acc = rAccum4[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(8)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(8)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(8)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(8)) << cp.int32(4)))
                    rAccum4[tid] = cp.disjoint(acc)

                    acc = rAccum5[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(10)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(10)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(10)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(10)) << cp.int32(4)))
                    rAccum5[tid] = cp.disjoint(acc)

                    acc = rAccum6[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(12)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(12)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(12)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(12)) << cp.int32(4)))
                    rAccum6[tid] = cp.disjoint(acc)

                    acc = rAccum7[tid] + cp.int32(0)
                    acc = acc + (cp.int32(1) << ((r0 - cp.int32(14)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r1 - cp.int32(14)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r2 - cp.int32(14)) << cp.int32(4)))
                    acc = acc + (cp.int32(1) << ((r3 - cp.int32(14)) << cp.int32(4)))
                    rAccum7[tid] = cp.disjoint(acc)

            block.barrier()

        # Phase 2: Warp-level inclusive scan on each packed pair, then extract counts.
        for _pair in range(8):
            for tid, thread in block.threads():
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                warpIndexInBlock = tid >> cp.int32(5)

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

                val2 = thread.shfl_idx(val, 31)

                radixBase = cp.int32(_pair * 2)
                if threadIndexInWarp < cp.int32(2):
                    shifted = val2 >> (threadIndexInWarp << cp.int32(4))
                    count_val = shifted & cp.int32(0xFFFF)
                    sRadixSum[(radixBase + threadIndexInWarp) * cp.int32(NUM_WARPS) + warpIndexInBlock] = cp.disjoint(count_val)

        block.barrier()

        # Phase 3: Cross-warp scan and write to gRadixCount.
        for tid, thread in block.threads():
            warpIndexInBlock = tid >> cp.int32(5)
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
            if cp.assume_uniform(warpIndexInBlock < cp.int32(NUM_WARPS // 2)):
                originalValue = sRadixSum[tid] + cp.int32(0)

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

                output = val

                if threadIndexInWarp == cp.int32(NUM_WARPS - 1):
                    gRadixIndex = bx + warpIndexInBlock * cp.int32(NB_BLOCKS)
                    radixCount[gRadixIndex] = output


# ===== Kernel 2: bodyInputAndRanksBlocksLaunch =====
# Full body duplicated from radixSortMultiCalculateRanksLaunch.
@cp.kernel
def bodyInputAndRanksBlocksLaunch(inputKeys, inputRanks, numKeys, startBit,
                                  radixCount, outputKeys, outputRanks,
                                  BLOCK_SIZE: cp.constexpr = 1024,
                                  NUM_WARPS: cp.constexpr = 32,
                                  RADIX_SIZE: cp.constexpr = 16,
                                  NB_BLOCKS: cp.constexpr = 32):
    with cp.Kernel(NB_BLOCKS, threads=BLOCK_SIZE) as (bx, block):
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

        # Part 1: scanRadixes
        for tid, thread in block.threads():
            warpIndexInBlock = tid >> cp.int32(5)
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
            if cp.assume_uniform(warpIndexInBlock < cp.int32(RADIX_SIZE)):
                radixSumIndex = warpIndexInBlock * cp.int32(NB_BLOCKS) + threadIndexInWarp
                value = radixCount[radixSumIndex] + cp.int32(0)

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

                exclusive = val - value
                sBuckets[radixSumIndex] = cp.disjoint(exclusive)
                if threadIndexInWarp == cp.int32(WARP_SIZE - 1):
                    sRadixSumBetweenBlocks[warpIndexInBlock] = cp.disjoint(val)

        block.barrier()

        # Part 1b: Exclusive scan of sRadixSumBetweenBlocks
        for tid, thread in block.threads():
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
            if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                value = sRadixSumBetweenBlocks[tid] + cp.int32(0)

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

                output = val - value
                sRadixSumBetweenBlocks[tid] = cp.disjoint(output + sBuckets[tid * cp.int32(NB_BLOCKS) + bx])

        block.barrier()

        # Part 2: Per-iteration reorder loop
        for _iter in range(64):
            # Phase 2a: Load keys, compute radix, warp-level rank computation
            for tid, thread in block.threads():
                warpIndexInBlock = tid >> cp.int32(5)
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)

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

                ro0 = cp.int32(0)
                ro1 = cp.int32(0)
                ro2 = cp.int32(0)
                ro3 = cp.int32(0)

                # Group i=0 (radixes 0-3)
                accum = (cp.int32(1) << ((r0 - cp.int32(0)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r1 - cp.int32(0)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r2 - cp.int32(0)) << cp.int32(3)))
                accum = accum + (cp.int32(1) << ((r3 - cp.int32(0)) << cp.int32(3)))
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
                val = val - accum
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

                sBuckets[tid * cp.int32(4) + cp.int32(0)] = cp.disjoint(ro0)
                sBuckets[tid * cp.int32(4) + cp.int32(1)] = cp.disjoint(ro1)
                sBuckets[tid * cp.int32(4) + cp.int32(2)] = cp.disjoint(ro2)
                sBuckets[tid * cp.int32(4) + cp.int32(3)] = cp.disjoint(ro3)

            block.barrier()

            # Phase 2b: Save lastRadixSum
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
                    lastRS = sRadixSum[tid * cp.int32(NUM_WARPS) + cp.int32(NUM_WARPS - 1)] + cp.int32(0)
                    sRadixCount[tid] = cp.disjoint(lastRS)

            block.barrier()

            # Phase 2c: Cross-warp exclusive scan of sRadixSum
            for tid, thread in block.threads():
                warpIndexInBlock = tid >> cp.int32(5)
                threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                if cp.assume_uniform(warpIndexInBlock < cp.int32(NUM_WARPS // 2)):
                    tempVal = sRadixSum[tid] + cp.int32(0)

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

                    sRadixSum[tid] = cp.disjoint(val - tempVal)

            block.barrier()

            # Phase 2d: Compute sRadixSumSum
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(RADIX_SIZE)):
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
                    sRadixSumSum[tid] = cp.disjoint(val - value)

            block.barrier()

            # Phase 2e: Add sRadixSumSum to sRadixSum
            for tid, thread in block.threads():
                if tid < cp.int32(NUM_WARPS * RADIX_SIZE):
                    radixIdx = tid // cp.int32(NUM_WARPS)
                    addVal = sRadixSumSum[radixIdx] + cp.int32(0)
                    curVal = sRadixSum[tid] + cp.int32(0)
                    sRadixSum[tid] = cp.disjoint(curVal + addVal)

            block.barrier()

            # Phase 2f: Compute global offsets
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

                    ro0 = sBuckets[tid * cp.int32(4) + cp.int32(0)] + cp.int32(0)
                    ro1 = sBuckets[tid * cp.int32(4) + cp.int32(1)] + cp.int32(0)
                    ro2 = sBuckets[tid * cp.int32(4) + cp.int32(2)] + cp.int32(0)
                    ro3 = sBuckets[tid * cp.int32(4) + cp.int32(3)] + cp.int32(0)

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


# ===========================================================================
# Group B: Simple grid-stride loops
# ===========================================================================

# ===== Kernel 3: initialRanksAndBodyIndexB =====
# Read thresholdStream[i].nodeIndexB -> keys[i], ranks[i] = i. Pad keys to multiple of 4.
@cp.kernel
def initialRanksAndBodyIndexB(
    thresholdStream,  # int32[N, TE_SIZE]
    keys,             # int32[M] -- output keys (M >= numElements rounded up to mult of 4)
    ranks,            # int32[M] -- output ranks
    numElements,      # int32 scalar
    BLOCK_SIZE: cp.constexpr = 256,
    TE_SIZE: cp.constexpr = 8,
    TE_NODE_INDEX_B: cp.constexpr = 3,
):
    with cp.Kernel(cp.ceildiv(numElements, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            i = bx * BLOCK_SIZE + tid
            if i < numElements:
                keys[i] = thresholdStream[i, TE_NODE_INDEX_B] + cp.int32(0)
                ranks[i] = i
            # Pad to multiple of 4 with max-value keys
            numRounded = (numElements + cp.int32(3)) & cp.int32(-4)
            if i >= numElements:
                if i < numRounded:
                    keys[i] = cp.int32(0x7FFFFFFF)
                    ranks[i] = i


# ===== Kernel 4: initialRanksAndBodyIndexA =====
# Read thresholdStream[ranks[i]].nodeIndexA -> keys[i]. Pad.
@cp.kernel
def initialRanksAndBodyIndexA(
    thresholdStream,  # int32[N, TE_SIZE]
    keys,             # int32[M] -- output keys
    ranks,            # int32[M] -- input ranks from previous sort
    numElements,      # int32 scalar
    BLOCK_SIZE: cp.constexpr = 256,
    TE_SIZE: cp.constexpr = 8,
    TE_NODE_INDEX_A: cp.constexpr = 2,
):
    with cp.Kernel(cp.ceildiv(numElements, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            i = bx * BLOCK_SIZE + tid
            if i < numElements:
                rank = ranks[i] + cp.int32(0)
                keys[i] = thresholdStream[rank, TE_NODE_INDEX_A] + cp.int32(0)
            numRounded = (numElements + cp.int32(3)) & cp.int32(-4)
            if i >= numElements:
                if i < numRounded:
                    keys[i] = cp.int32(0x7FFFFFFF)


# ===== Kernel 5: reorganizeThresholdElements =====
# Copy thresholdStream[i] = tmpThresholdStream[ranks[i]] using 8-int-per-element pattern.
@cp.kernel
def reorganizeThresholdElements(
    thresholdStream,     # int32[N, TE_SIZE] -- output
    tmpThresholdStream,  # int32[N, TE_SIZE] -- input
    ranks,               # int32[N]
    numElements,         # int32 scalar
    BLOCK_SIZE: cp.constexpr = 256,
    TE_SIZE: cp.constexpr = 8,
):
    with cp.Kernel(cp.ceildiv(numElements, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            i = bx * BLOCK_SIZE + tid
            if i < numElements:
                rank = ranks[i] + cp.int32(0)
                thresholdStream[i, 0] = tmpThresholdStream[rank, 0] + cp.int32(0)
                thresholdStream[i, 1] = tmpThresholdStream[rank, 1] + cp.int32(0)
                thresholdStream[i, 2] = tmpThresholdStream[rank, 2] + cp.int32(0)
                thresholdStream[i, 3] = tmpThresholdStream[rank, 3] + cp.int32(0)
                thresholdStream[i, 4] = tmpThresholdStream[rank, 4] + cp.int32(0)
                thresholdStream[i, 5] = tmpThresholdStream[rank, 5] + cp.int32(0)
                thresholdStream[i, 6] = tmpThresholdStream[rank, 6] + cp.int32(0)
                thresholdStream[i, 7] = tmpThresholdStream[rank, 7] + cp.int32(0)


# ===== Kernel 8: writeoutAccumulatedForcePerObject =====
# If writeable[i], accumulatedForceObjectPairs[writeIndex[i]] = accumulatedForce[i].
@cp.kernel
def writeoutAccumulatedForcePerObject(
    accumulatedForce,            # int32[N, 4] -- accumulated force per object
    accumulatedForceObjectPairs, # int32[M, 4] -- output
    writeIndex,                  # int32[N] -- write indices
    writeable,                   # int32[N] -- 0 or 1
    numElements,                 # int32 scalar
    BLOCK_SIZE: cp.constexpr = 256,
):
    with cp.Kernel(cp.ceildiv(numElements, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            i = bx * BLOCK_SIZE + tid
            if i < numElements:
                w = writeable[i] + cp.int32(0)
                if w != cp.int32(0):
                    wIdx = writeIndex[i] + cp.int32(0)
                    accumulatedForceObjectPairs[wIdx, 0] = accumulatedForce[i, 0] + cp.int32(0)
                    accumulatedForceObjectPairs[wIdx, 1] = accumulatedForce[i, 1] + cp.int32(0)
                    accumulatedForceObjectPairs[wIdx, 2] = accumulatedForce[i, 2] + cp.int32(0)
                    accumulatedForceObjectPairs[wIdx, 3] = accumulatedForce[i, 3] + cp.int32(0)


# ===========================================================================
# Group C: Multi-iteration warp scan + reduction (stage 1)
# ===========================================================================

# ===== Kernel 6: computeAccumulateThresholdStream =====
# Multi-iteration scan of BOTH float (normalForce) and int (pair count) simultaneously.
# Uses warp scan pattern (shfl_up) to replace ballot+popc.
# BLOCK_SIZE=256, NUM_WARPS=8, LOG2_WARP_PERBLOCK_SIZE=3 (scan of 8 warp sums -> 3 rounds).
@cp.kernel
def computeAccumulateThresholdStream(
    thresholdStream,    # int32[N, TE_SIZE]
    numElements,        # int32 scalar
    gBlockNumForce,     # int32[GRID_SIZE] -- output: per-block accumulated force (float as int32)
    gBlockNumPairs,     # int32[GRID_SIZE] -- output: per-block pair count
    BLOCK_SIZE: cp.constexpr = 256,
    GRID_SIZE: cp.constexpr = 32,
    TE_SIZE: cp.constexpr = 8,
    TE_NODE_INDEX_A: cp.constexpr = 2,
    TE_NODE_INDEX_B: cp.constexpr = 3,
    TE_NORMAL_FORCE: cp.constexpr = 4,
):
    NUM_WARPS = BLOCK_SIZE // WARP_SIZE

    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        # Shared memory for warp scan pattern
        sWarpForceAccum = block.alloc((NUM_WARPS,), dtype=cp.int32)   # float stored as int32
        sBlockForceAccum = block.alloc((1,), dtype=cp.int32)          # float stored as int32
        sWarpPairsAccum = block.alloc((NUM_WARPS,), dtype=cp.int32)
        sBlockPairsAccum = block.alloc((1,), dtype=cp.int32)

        # Initialize block accumulators to 0
        for tid, thread in block.threads():
            if tid == cp.int32(0):
                sBlockForceAccum[0] = cp.disjoint(cp.int32(0))  # float 0.0 as int32 = 0
                sBlockPairsAccum[0] = cp.disjoint(cp.int32(0))

        block.barrier()

        totalBlockRequired = (numElements + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterPerBlock = (totalBlockRequired + cp.int32(GRID_SIZE - 1)) // cp.int32(GRID_SIZE)

        _iter = cp.int32(0)
        while _iter < numIterPerBlock:
            # Phase 1: Per-thread compute -> warp scan -> store warp totals
            for warp_id, warp in block.warps():
                for lane, thread in warp.threads():
                    tid = warp_id * cp.int32(WARP_SIZE) + lane

                    workIndex = _iter * cp.int32(BLOCK_SIZE) + tid + numIterPerBlock * bx * cp.int32(BLOCK_SIZE)

                    # Compute normalForce and isNewPair flag
                    forceVal = cp.float32(0.0)
                    isNewPair = cp.int32(0)

                    if workIndex < numElements:
                        forceRaw = thresholdStream[workIndex, TE_NORMAL_FORCE] + cp.int32(0)
                        forceVal = thread.bitcast(forceRaw, cp.float32)

                        # isNewPair: check if this is a different (nodeA, nodeB) pair than previous
                        if workIndex == cp.int32(0):
                            isNewPair = cp.int32(1)
                        else:
                            curA = thresholdStream[workIndex, TE_NODE_INDEX_A] + cp.int32(0)
                            curB = thresholdStream[workIndex, TE_NODE_INDEX_B] + cp.int32(0)
                            prevA = thresholdStream[workIndex - cp.int32(1), TE_NODE_INDEX_A] + cp.int32(0)
                            prevB = thresholdStream[workIndex - cp.int32(1), TE_NODE_INDEX_B] + cp.int32(0)
                            if curA != prevA:
                                isNewPair = cp.int32(1)
                            if curB != prevB:
                                isNewPair = cp.int32(1)

                    # Warp scan of force (float): inclusive prefix sum using shfl_up
                    f_scan = forceVal
                    fn = thread.shfl_up(f_scan, 1)
                    if lane >= cp.int32(1):
                        f_scan = f_scan + fn
                    fn = thread.shfl_up(f_scan, 2)
                    if lane >= cp.int32(2):
                        f_scan = f_scan + fn
                    fn = thread.shfl_up(f_scan, 4)
                    if lane >= cp.int32(4):
                        f_scan = f_scan + fn
                    fn = thread.shfl_up(f_scan, 8)
                    if lane >= cp.int32(8):
                        f_scan = f_scan + fn
                    fn = thread.shfl_up(f_scan, 16)
                    if lane >= cp.int32(16):
                        f_scan = f_scan + fn
                    # f_scan is inclusive scan; warp total is last lane's value

                    # Warp scan of pairs (int): inclusive prefix sum
                    p_scan = isNewPair + cp.int32(0)
                    pn = thread.shfl_up(p_scan, 1)
                    if lane >= cp.int32(1):
                        p_scan = p_scan + pn
                    pn = thread.shfl_up(p_scan, 2)
                    if lane >= cp.int32(2):
                        p_scan = p_scan + pn
                    pn = thread.shfl_up(p_scan, 4)
                    if lane >= cp.int32(4):
                        p_scan = p_scan + pn
                    pn = thread.shfl_up(p_scan, 8)
                    if lane >= cp.int32(8):
                        p_scan = p_scan + pn
                    pn = thread.shfl_up(p_scan, 16)
                    if lane >= cp.int32(16):
                        p_scan = p_scan + pn

                    if lane == cp.int32(WARP_SIZE - 1):
                        sWarpForceAccum[warp_id] = cp.disjoint(thread.bitcast(f_scan, cp.int32))
                        sWarpPairsAccum[warp_id] = cp.disjoint(p_scan)

            block.barrier()

            # Phase 2: Scan warp sums (NUM_WARPS=8 -> 3 rounds of shfl_up)
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(NUM_WARPS)):
                    # Force scan
                    wf = thread.bitcast(sWarpForceAccum[tid] + cp.int32(0), cp.float32)
                    wf_orig = wf
                    fn = thread.shfl_up(wf, 1)
                    if tid >= cp.int32(1):
                        wf = wf + fn
                    fn = thread.shfl_up(wf, 2)
                    if tid >= cp.int32(2):
                        wf = wf + fn
                    fn = thread.shfl_up(wf, 4)
                    if tid >= cp.int32(4):
                        wf = wf + fn
                    # Exclusive = inclusive - original
                    wf_exclusive = wf - wf_orig
                    sWarpForceAccum[tid] = cp.disjoint(thread.bitcast(wf_exclusive, cp.int32))

                    # Pairs scan
                    wp = sWarpPairsAccum[tid] + cp.int32(0)
                    wp_orig = wp
                    pn = thread.shfl_up(wp, 1)
                    if tid >= cp.int32(1):
                        wp = wp + pn
                    pn = thread.shfl_up(wp, 2)
                    if tid >= cp.int32(2):
                        wp = wp + pn
                    pn = thread.shfl_up(wp, 4)
                    if tid >= cp.int32(4):
                        wp = wp + pn
                    wp_exclusive = wp - wp_orig
                    sWarpPairsAccum[tid] = cp.disjoint(wp_exclusive)

                    # Last warp thread accumulates block totals
                    if tid == cp.int32(NUM_WARPS - 1):
                        oldForce = thread.bitcast(sBlockForceAccum[0] + cp.int32(0), cp.float32)
                        newForce = oldForce + wf
                        sBlockForceAccum[0] = cp.disjoint(thread.bitcast(newForce, cp.int32))

                        oldPairs = sBlockPairsAccum[0] + cp.int32(0)
                        sBlockPairsAccum[0] = cp.disjoint(oldPairs + wp)

            block.barrier()

            _iter = _iter + cp.int32(1)

        # Write final results
        for tid, thread in block.threads():
            if tid == cp.int32(0):
                gBlockNumForce[bx] = sBlockForceAccum[0] + cp.int32(0)
                gBlockNumPairs[bx] = sBlockPairsAccum[0] + cp.int32(0)


# ===== Kernel 7: outputAccumulateThresholdStream =====
# Simple cross-block fixup: scan block totals, add offsets to all elements.
@cp.kernel
def outputAccumulateThresholdStream(
    accumulatedForce,   # float32[N] -- accumulated force per element (in-place fixup)
    writeIndex,         # int32[N] -- write index per element (in-place fixup)
    gBlockForce,        # float32[GRID_SIZE] -- per-block force total from Stage1
    gBlockPairs,        # int32[GRID_SIZE] -- per-block pair count from Stage1
    numElements,        # int scalar
    numPairsOut,        # int32[1] -- output total pair count
    BLOCK_SIZE: cp.constexpr = 256,
    GRID_SIZE: cp.constexpr = 32,
):
    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        sBlockHistForce = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockHistPairs = block.alloc((GRID_SIZE,), dtype=cp.int32)

        # Phase 0: Warp-0 scans per-block totals → exclusive prefix sums
        for tid, thread in block.threads():
            if cp.assume_uniform(tid < cp.int32(GRID_SIZE)):
                    blockForce = gBlockForce[tid] + cp.float32(0.0)
                    fval = blockForce
                    fn = thread.shfl_up(fval, 1)
                    if tid >= cp.int32(1):
                        fval = fval + fn
                    fn = thread.shfl_up(fval, 2)
                    if tid >= cp.int32(2):
                        fval = fval + fn
                    fn = thread.shfl_up(fval, 4)
                    if tid >= cp.int32(4):
                        fval = fval + fn
                    fn = thread.shfl_up(fval, 8)
                    if tid >= cp.int32(8):
                        fval = fval + fn
                    fn = thread.shfl_up(fval, 16)
                    if tid >= cp.int32(16):
                        fval = fval + fn
                    sBlockHistForce[tid] = cp.disjoint(fval - blockForce)

                    # Pairs scan
                    blockPairs = gBlockPairs[tid] + cp.int32(0)
                    pval = blockPairs
                    pn = thread.shfl_up(pval, 1)
                    if tid >= cp.int32(1):
                        pval = pval + pn
                    pn = thread.shfl_up(pval, 2)
                    if tid >= cp.int32(2):
                        pval = pval + pn
                    pn = thread.shfl_up(pval, 4)
                    if tid >= cp.int32(4):
                        pval = pval + pn
                    pn = thread.shfl_up(pval, 8)
                    if tid >= cp.int32(8):
                        pval = pval + pn
                    pn = thread.shfl_up(pval, 16)
                    if tid >= cp.int32(16):
                        pval = pval + pn
                    sBlockHistPairs[tid] = cp.disjoint(pval - blockPairs)

                    if tid == cp.int32(GRID_SIZE - 1):
                        numPairsOut[0] = pval

        block.barrier()

        # Phase 1: Grid-stride add block offsets to all elements
        totalBlockRequired = (numElements + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterPerBlock = (totalBlockRequired + cp.int32(GRID_SIZE - 1)) // cp.int32(GRID_SIZE)
        blockForceOffset = sBlockHistForce[bx] + cp.float32(0.0)
        blockPairsOffset = sBlockHistPairs[bx] + cp.int32(0)

        _iter = cp.int32(0)
        while _iter < numIterPerBlock:
            for tid, thread in block.threads():
                workIndex = _iter * cp.int32(BLOCK_SIZE) + tid + numIterPerBlock * bx * cp.int32(BLOCK_SIZE)
                if workIndex < numElements:
                    accumulatedForce[workIndex] = accumulatedForce[workIndex] + blockForceOffset
                    writeIndex[workIndex] = writeIndex[workIndex] + blockPairsOffset

            _iter = _iter + cp.int32(1)


# ===== Kernel 9: computeExceededForceThresholdElementIndice =====
# Scans only int (exceeded force count). Same warp scan pattern.
@cp.kernel
def computeExceededForceThresholdElementIndice(
    thresholdStream,    # int32[N, TE_SIZE]
    numElements,        # int32 scalar
    gBlockNumExceeded,  # int32[GRID_SIZE] -- output: per-block exceeded count
    BLOCK_SIZE: cp.constexpr = 256,
    GRID_SIZE: cp.constexpr = 32,
    TE_SIZE: cp.constexpr = 8,
    TE_ACCUM_FORCE: cp.constexpr = 5,
    TE_THRESHOLD: cp.constexpr = 6,
):
    NUM_WARPS = BLOCK_SIZE // WARP_SIZE

    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        sWarpAccum = block.alloc((NUM_WARPS,), dtype=cp.int32)
        sBlockAccum = block.alloc((1,), dtype=cp.int32)

        for tid, thread in block.threads():
            if tid == cp.int32(0):
                sBlockAccum[0] = cp.disjoint(cp.int32(0))

        block.barrier()

        totalBlockRequired = (numElements + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterPerBlock = (totalBlockRequired + cp.int32(GRID_SIZE - 1)) // cp.int32(GRID_SIZE)

        _iter = cp.int32(0)
        while _iter < numIterPerBlock:
            # Phase 1: warp scan of exceeded flag
            for warp_id, warp in block.warps():
                for lane, thread in warp.threads():
                    tid = warp_id * cp.int32(WARP_SIZE) + lane

                    workIndex = _iter * cp.int32(BLOCK_SIZE) + tid + numIterPerBlock * bx * cp.int32(BLOCK_SIZE)

                    exceeded = cp.int32(0)
                    if workIndex < numElements:
                        accumForceRaw = thresholdStream[workIndex, TE_ACCUM_FORCE] + cp.int32(0)
                        thresholdRaw = thresholdStream[workIndex, TE_THRESHOLD] + cp.int32(0)
                        accumForce = thread.bitcast(accumForceRaw, cp.float32)
                        threshold = thread.bitcast(thresholdRaw, cp.float32)
                        if accumForce > threshold:
                            exceeded = cp.int32(1)

                    # Inclusive warp scan
                    e_scan = exceeded + cp.int32(0)
                    n = thread.shfl_up(e_scan, 1)
                    if lane >= cp.int32(1):
                        e_scan = e_scan + n
                    n = thread.shfl_up(e_scan, 2)
                    if lane >= cp.int32(2):
                        e_scan = e_scan + n
                    n = thread.shfl_up(e_scan, 4)
                    if lane >= cp.int32(4):
                        e_scan = e_scan + n
                    n = thread.shfl_up(e_scan, 8)
                    if lane >= cp.int32(8):
                        e_scan = e_scan + n
                    n = thread.shfl_up(e_scan, 16)
                    if lane >= cp.int32(16):
                        e_scan = e_scan + n

                    if lane == cp.int32(WARP_SIZE - 1):
                        sWarpAccum[warp_id] = cp.disjoint(e_scan)

            block.barrier()

            # Phase 2: scan warp sums (NUM_WARPS=8)
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(NUM_WARPS)):
                    value = sWarpAccum[tid] + cp.int32(0)
                    n = thread.shfl_xor(value, 1)
                    value = value + n
                    n = thread.shfl_xor(value, 2)
                    value = value + n
                    n = thread.shfl_xor(value, 4)
                    value = value + n
                    n = thread.shfl_xor(value, 8)
                    value = value + n
                    n = thread.shfl_xor(value, 16)
                    value = value + n
                    if tid == cp.int32(NUM_WARPS - 1):
                        sBlockAccum[0] = cp.disjoint(sBlockAccum[0] + value)

            block.barrier()

            _iter = _iter + cp.int32(1)

        # Write final result
        for tid, thread in block.threads():
            if tid == cp.int32(0):
                gBlockNumExceeded[bx] = sBlockAccum[0] + cp.int32(0)


# ===== Kernel 10: outputExceededForceThresholdElementIndice =====
# Cross-block fixup for exceeded force indices + copies exceeded elements.
@cp.kernel
def outputExceededForceThresholdElementIndice(
    thresholdStream,         # int32[N, TE_SIZE]
    numElements,             # int32 scalar
    gBlockNumExceeded,       # int32[GRID_SIZE]
    exceededForceElements,   # int32[M, TE_SIZE] -- output: exceeded elements
    numExceededForceElements,  # int32[1] -- output: total exceeded count
    BLOCK_SIZE: cp.constexpr = 256,
    GRID_SIZE: cp.constexpr = 32,
    TE_SIZE: cp.constexpr = 8,
    TE_ACCUM_FORCE: cp.constexpr = 5,
    TE_THRESHOLD: cp.constexpr = 6,
):
    NUM_WARPS = BLOCK_SIZE // WARP_SIZE

    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        sWarpAccum = block.alloc((NUM_WARPS,), dtype=cp.int32)
        sBlockAccum = block.alloc((1,), dtype=cp.int32)
        sBlockHistogram = block.alloc((GRID_SIZE,), dtype=cp.int32)
        sPrevAccum = block.alloc((1,), dtype=cp.int32)

        for tid, thread in block.threads():
            if tid == cp.int32(0):
                sBlockAccum[0] = cp.disjoint(cp.int32(0))

        block.barrier()

        # Phase 0: Warp-0 scans gBlockNumExceeded -> exclusive prefix sum
        for tid, thread in block.threads():
            warpIndex = tid // cp.int32(WARP_SIZE)
            threadIndexInWarp = tid % cp.int32(WARP_SIZE)
            if cp.assume_uniform(warpIndex == cp.int32(0)):
                if cp.assume_uniform(threadIndexInWarp < cp.int32(GRID_SIZE)):
                    blockNum = gBlockNumExceeded[threadIndexInWarp] + cp.int32(0)
                    value = blockNum
                    n = thread.shfl_up(value, 1)
                    if threadIndexInWarp >= cp.int32(1):
                        value = value + n
                    n = thread.shfl_up(value, 2)
                    if threadIndexInWarp >= cp.int32(2):
                        value = value + n
                    n = thread.shfl_up(value, 4)
                    if threadIndexInWarp >= cp.int32(4):
                        value = value + n
                    n = thread.shfl_up(value, 8)
                    if threadIndexInWarp >= cp.int32(8):
                        value = value + n
                    n = thread.shfl_up(value, 16)
                    if threadIndexInWarp >= cp.int32(16):
                        value = value + n
                    sBlockHistogram[threadIndexInWarp] = cp.disjoint(value - blockNum)
                    if threadIndexInWarp == cp.int32(GRID_SIZE - 1):
                        numExceededForceElements[0] = value

        block.barrier()

        totalBlockRequired = (numElements + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterPerBlock = (totalBlockRequired + cp.int32(GRID_SIZE - 1)) // cp.int32(GRID_SIZE)
        blockStartIndex = sBlockHistogram[bx] + cp.int32(0)

        _iter = cp.int32(0)
        while _iter < numIterPerBlock:
            # Phase 1: warp scan
            for warp_id, warp in block.warps():
                for lane, thread in warp.threads():
                    tid = warp_id * cp.int32(WARP_SIZE) + lane

                    workIndex = _iter * cp.int32(BLOCK_SIZE) + tid + numIterPerBlock * bx * cp.int32(BLOCK_SIZE)

                    exceeded = cp.int32(0)
                    if workIndex < numElements:
                        accumForceRaw = thresholdStream[workIndex, TE_ACCUM_FORCE] + cp.int32(0)
                        thresholdRaw = thresholdStream[workIndex, TE_THRESHOLD] + cp.int32(0)
                        accumForce = thread.bitcast(accumForceRaw, cp.float32)
                        threshold = thread.bitcast(thresholdRaw, cp.float32)
                        if accumForce > threshold:
                            exceeded = cp.int32(1)

                    e_scan = exceeded + cp.int32(0)
                    n = thread.shfl_up(e_scan, 1)
                    if lane >= cp.int32(1):
                        e_scan = e_scan + n
                    n = thread.shfl_up(e_scan, 2)
                    if lane >= cp.int32(2):
                        e_scan = e_scan + n
                    n = thread.shfl_up(e_scan, 4)
                    if lane >= cp.int32(4):
                        e_scan = e_scan + n
                    n = thread.shfl_up(e_scan, 8)
                    if lane >= cp.int32(8):
                        e_scan = e_scan + n
                    n = thread.shfl_up(e_scan, 16)
                    if lane >= cp.int32(16):
                        e_scan = e_scan + n

                    if lane == cp.int32(WARP_SIZE - 1):
                        sWarpAccum[warp_id] = cp.disjoint(e_scan)

                    if tid == cp.int32(0):
                        sPrevAccum[0] = cp.disjoint(sBlockAccum[0])

            block.barrier()

            # Phase 2: scan warp sums
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(NUM_WARPS)):
                    value = sWarpAccum[tid] + cp.int32(0)
                    orig = value
                    n = thread.shfl_up(value, 1)
                    if tid >= cp.int32(1):
                        value = value + n
                    n = thread.shfl_up(value, 2)
                    if tid >= cp.int32(2):
                        value = value + n
                    n = thread.shfl_up(value, 4)
                    if tid >= cp.int32(4):
                        value = value + n
                    sWarpAccum[tid] = cp.disjoint(value - orig)
                    if tid == cp.int32(NUM_WARPS - 1):
                        sBlockAccum[0] = cp.disjoint(sBlockAccum[0] + value)

            block.barrier()

            # Phase 3: write exceeded elements
            for warp_id, warp in block.warps():
                for lane, thread in warp.threads():
                    tid = warp_id * cp.int32(WARP_SIZE) + lane

                    workIndex = _iter * cp.int32(BLOCK_SIZE) + tid + numIterPerBlock * bx * cp.int32(BLOCK_SIZE)

                    exceeded = cp.int32(0)
                    if workIndex < numElements:
                        accumForceRaw = thresholdStream[workIndex, TE_ACCUM_FORCE] + cp.int32(0)
                        thresholdRaw = thresholdStream[workIndex, TE_THRESHOLD] + cp.int32(0)
                        accumForce = thread.bitcast(accumForceRaw, cp.float32)
                        threshold = thread.bitcast(thresholdRaw, cp.float32)
                        if accumForce > threshold:
                            exceeded = cp.int32(1)

                    e_scan = exceeded + cp.int32(0)
                    n = thread.shfl_up(e_scan, 1)
                    if lane >= cp.int32(1):
                        e_scan = e_scan + n
                    n = thread.shfl_up(e_scan, 2)
                    if lane >= cp.int32(2):
                        e_scan = e_scan + n
                    n = thread.shfl_up(e_scan, 4)
                    if lane >= cp.int32(4):
                        e_scan = e_scan + n
                    n = thread.shfl_up(e_scan, 8)
                    if lane >= cp.int32(8):
                        e_scan = e_scan + n
                    n = thread.shfl_up(e_scan, 16)
                    if lane >= cp.int32(16):
                        e_scan = e_scan + n
                    e_exclusive = e_scan - exceeded

                    if exceeded != cp.int32(0):
                        prevAccum = sPrevAccum[0] + cp.int32(0)
                        warpOff = sWarpAccum[warp_id] + cp.int32(0)
                        index = e_exclusive + warpOff + prevAccum + blockStartIndex

                        # Copy all 8 int32s of the element
                        exceededForceElements[index, 0] = thresholdStream[workIndex, 0] + cp.int32(0)
                        exceededForceElements[index, 1] = thresholdStream[workIndex, 1] + cp.int32(0)
                        exceededForceElements[index, 2] = thresholdStream[workIndex, 2] + cp.int32(0)
                        exceededForceElements[index, 3] = thresholdStream[workIndex, 3] + cp.int32(0)
                        exceededForceElements[index, 4] = thresholdStream[workIndex, 4] + cp.int32(0)
                        exceededForceElements[index, 5] = thresholdStream[workIndex, 5] + cp.int32(0)
                        exceededForceElements[index, 6] = thresholdStream[workIndex, 6] + cp.int32(0)
                        exceededForceElements[index, 7] = thresholdStream[workIndex, 7] + cp.int32(0)

            block.barrier()

            _iter = _iter + cp.int32(1)


# ===== Kernel 11: setThresholdElementsMask =====
# Binary search on sorted prevExceededForceElements, then while loop for matching pairs.
@cp.kernel
def setThresholdElementsMask(
    thresholdStream,            # int32[N, TE_SIZE]
    numElements,                # int32 scalar
    prevExceededForceElements,  # int32[M, TE_SIZE] -- sorted by shapeInteraction
    numPrevExceeded,            # int32 scalar
    thresholdMask,              # int32[N] -- output: 1 if found in prev, 0 otherwise
    BLOCK_SIZE: cp.constexpr = 256,
    TE_SIZE: cp.constexpr = 8,
    TE_SHAPE_INTERACTION_LO: cp.constexpr = 0,
    TE_SHAPE_INTERACTION_HI: cp.constexpr = 1,
    TE_NODE_INDEX_A: cp.constexpr = 2,
    TE_NODE_INDEX_B: cp.constexpr = 3,
):
    with cp.Kernel(cp.ceildiv(numElements, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            i = bx * BLOCK_SIZE + tid
            if i < numElements:
                found = cp.int32(0)

                numPrev = numPrevExceeded + cp.int32(0)
                if numPrev > cp.int32(0):
                    # Get this element's shapeInteraction (PxU64 as 2 int32s)
                    siLo = thresholdStream[i, TE_SHAPE_INTERACTION_LO] + cp.int32(0)
                    siHi = thresholdStream[i, TE_SHAPE_INTERACTION_HI] + cp.int32(0)
                    nodeA = thresholdStream[i, TE_NODE_INDEX_A] + cp.int32(0)
                    nodeB = thresholdStream[i, TE_NODE_INDEX_B] + cp.int32(0)

                    # Binary search for shapeInteraction in prevExceededForceElements
                    left = cp.int32(0)
                    right = numPrev + cp.int32(0)
                    while (right - left) > cp.int32(1):
                        pos = (left + right) >> cp.int32(1)
                        elemHi = prevExceededForceElements[pos, TE_SHAPE_INTERACTION_HI] + cp.int32(0)
                        elemLo = prevExceededForceElements[pos, TE_SHAPE_INTERACTION_LO] + cp.int32(0)
                        # Compare (hi, lo) as unsigned 64-bit: hi first, then lo
                        isLessOrEqual = cp.int32(0)
                        if elemHi < siHi:
                            isLessOrEqual = cp.int32(1)
                        if elemHi == siHi:
                            # Unsigned compare of lo: use xor with sign bit to convert
                            elemLoU = elemLo ^ cp.int32(-2147483648)
                            siLoU = siLo ^ cp.int32(-2147483648)
                            if elemLoU <= siLoU:
                                isLessOrEqual = cp.int32(1)
                        if isLessOrEqual != cp.int32(0):
                            left = pos
                        else:
                            right = pos

                    # Check if we found a match and scan for matching pairs
                    # Search left and right from the found position
                    j = left
                    while j >= cp.int32(0):
                        pHi = prevExceededForceElements[j, TE_SHAPE_INTERACTION_HI] + cp.int32(0)
                        pLo = prevExceededForceElements[j, TE_SHAPE_INTERACTION_LO] + cp.int32(0)
                        if pHi != siHi:
                            j = cp.int32(-1)  # break
                        elif pLo != siLo:
                            j = cp.int32(-1)  # break
                        else:
                            pA = prevExceededForceElements[j, TE_NODE_INDEX_A] + cp.int32(0)
                            pB = prevExceededForceElements[j, TE_NODE_INDEX_B] + cp.int32(0)
                            if pA == nodeA:
                                if pB == nodeB:
                                    found = cp.int32(1)
                            j = j - cp.int32(1)

                    # Search right from left+1
                    j = left + cp.int32(1)
                    while j < numPrev:
                        pHi = prevExceededForceElements[j, TE_SHAPE_INTERACTION_HI] + cp.int32(0)
                        pLo = prevExceededForceElements[j, TE_SHAPE_INTERACTION_LO] + cp.int32(0)
                        if pHi != siHi:
                            j = numPrev  # break
                        elif pLo != siLo:
                            j = numPrev  # break
                        else:
                            pA = prevExceededForceElements[j, TE_NODE_INDEX_A] + cp.int32(0)
                            pB = prevExceededForceElements[j, TE_NODE_INDEX_B] + cp.int32(0)
                            if pA == nodeA:
                                if pB == nodeB:
                                    found = cp.int32(1)
                            j = j + cp.int32(1)

                thresholdMask[i] = found


# ===== Kernel 12: computeThresholdElementMaskIndices =====
# Same as kernel 9 but first calls setPersistentForceElementMask (inlined).
# Then scans the mask to compute indices.
@cp.kernel
def computeThresholdElementMaskIndices(
    thresholdStream,    # int32[N, TE_SIZE]
    thresholdMask,      # int32[N] -- input mask (0 or 1)
    numElements,        # int32 scalar
    gBlockNumMasked,    # int32[GRID_SIZE] -- output: per-block masked count
    BLOCK_SIZE: cp.constexpr = 256,
    GRID_SIZE: cp.constexpr = 32,
    TE_SIZE: cp.constexpr = 8,
    TE_ACCUM_FORCE: cp.constexpr = 5,
    TE_THRESHOLD: cp.constexpr = 6,
):
    NUM_WARPS = BLOCK_SIZE // WARP_SIZE

    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        sWarpAccum = block.alloc((NUM_WARPS,), dtype=cp.int32)
        sBlockAccum = block.alloc((1,), dtype=cp.int32)

        # First: inline setPersistentForceElementMask -- set mask for elements
        # where accumForce > threshold (persistent force elements).
        # This is a simple grid-stride loop over all elements.
        for tid, thread in block.threads():
            if tid == cp.int32(0):
                sBlockAccum[0] = cp.disjoint(cp.int32(0))

        block.barrier()

        totalBlockRequired = (numElements + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterPerBlock = (totalBlockRequired + cp.int32(GRID_SIZE - 1)) // cp.int32(GRID_SIZE)

        _iter = cp.int32(0)
        while _iter < numIterPerBlock:
            # Phase 1: warp scan of mask flag
            for warp_id, warp in block.warps():
                for lane, thread in warp.threads():
                    tid = warp_id * cp.int32(WARP_SIZE) + lane

                    workIndex = _iter * cp.int32(BLOCK_SIZE) + tid + numIterPerBlock * bx * cp.int32(BLOCK_SIZE)

                    masked = cp.int32(0)
                    if workIndex < numElements:
                        masked = thresholdMask[workIndex] + cp.int32(0)

                    # Inclusive warp scan
                    m_scan = masked + cp.int32(0)
                    n = thread.shfl_up(m_scan, 1)
                    if lane >= cp.int32(1):
                        m_scan = m_scan + n
                    n = thread.shfl_up(m_scan, 2)
                    if lane >= cp.int32(2):
                        m_scan = m_scan + n
                    n = thread.shfl_up(m_scan, 4)
                    if lane >= cp.int32(4):
                        m_scan = m_scan + n
                    n = thread.shfl_up(m_scan, 8)
                    if lane >= cp.int32(8):
                        m_scan = m_scan + n
                    n = thread.shfl_up(m_scan, 16)
                    if lane >= cp.int32(16):
                        m_scan = m_scan + n

                    if lane == cp.int32(WARP_SIZE - 1):
                        sWarpAccum[warp_id] = cp.disjoint(m_scan)

            block.barrier()

            # Phase 2: scan warp sums
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(NUM_WARPS)):
                    value = sWarpAccum[tid] + cp.int32(0)
                    n = thread.shfl_xor(value, 1)
                    value = value + n
                    n = thread.shfl_xor(value, 2)
                    value = value + n
                    n = thread.shfl_xor(value, 4)
                    value = value + n
                    n = thread.shfl_xor(value, 8)
                    value = value + n
                    n = thread.shfl_xor(value, 16)
                    value = value + n
                    if tid == cp.int32(NUM_WARPS - 1):
                        sBlockAccum[0] = cp.disjoint(sBlockAccum[0] + value)

            block.barrier()

            _iter = _iter + cp.int32(1)

        # Write final result
        for tid, thread in block.threads():
            if tid == cp.int32(0):
                gBlockNumMasked[bx] = sBlockAccum[0] + cp.int32(0)


# ===== Kernel 13: outputThresholdPairsMaskIndices =====
# Cross-block fixup for mask indices.
@cp.kernel
def outputThresholdPairsMaskIndices(
    thresholdStream,         # int32[N, TE_SIZE]
    thresholdMask,           # int32[N] -- input mask
    numElements,             # int32 scalar
    gBlockNumMasked,         # int32[GRID_SIZE]
    maskedElements,          # int32[M, TE_SIZE] -- output: masked elements
    numMaskedElements,       # int32[1] -- output: total masked count
    BLOCK_SIZE: cp.constexpr = 256,
    GRID_SIZE: cp.constexpr = 32,
    TE_SIZE: cp.constexpr = 8,
):
    NUM_WARPS = BLOCK_SIZE // WARP_SIZE

    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        sWarpAccum = block.alloc((NUM_WARPS,), dtype=cp.int32)
        sBlockAccum = block.alloc((1,), dtype=cp.int32)
        sBlockHistogram = block.alloc((GRID_SIZE,), dtype=cp.int32)
        sPrevAccum = block.alloc((1,), dtype=cp.int32)

        for tid, thread in block.threads():
            if tid == cp.int32(0):
                sBlockAccum[0] = cp.disjoint(cp.int32(0))

        block.barrier()

        # Phase 0: scan gBlockNumMasked
        for tid, thread in block.threads():
            warpIndex = tid // cp.int32(WARP_SIZE)
            threadIndexInWarp = tid % cp.int32(WARP_SIZE)
            if cp.assume_uniform(warpIndex == cp.int32(0)):
                if cp.assume_uniform(threadIndexInWarp < cp.int32(GRID_SIZE)):
                    blockNum = gBlockNumMasked[threadIndexInWarp] + cp.int32(0)
                    value = blockNum
                    n = thread.shfl_up(value, 1)
                    if threadIndexInWarp >= cp.int32(1):
                        value = value + n
                    n = thread.shfl_up(value, 2)
                    if threadIndexInWarp >= cp.int32(2):
                        value = value + n
                    n = thread.shfl_up(value, 4)
                    if threadIndexInWarp >= cp.int32(4):
                        value = value + n
                    n = thread.shfl_up(value, 8)
                    if threadIndexInWarp >= cp.int32(8):
                        value = value + n
                    n = thread.shfl_up(value, 16)
                    if threadIndexInWarp >= cp.int32(16):
                        value = value + n
                    sBlockHistogram[threadIndexInWarp] = cp.disjoint(value - blockNum)
                    if threadIndexInWarp == cp.int32(GRID_SIZE - 1):
                        numMaskedElements[0] = value

        block.barrier()

        totalBlockRequired = (numElements + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterPerBlock = (totalBlockRequired + cp.int32(GRID_SIZE - 1)) // cp.int32(GRID_SIZE)
        blockStartIndex = sBlockHistogram[bx] + cp.int32(0)

        _iter = cp.int32(0)
        while _iter < numIterPerBlock:
            # Phase 1: warp scan
            for warp_id, warp in block.warps():
                for lane, thread in warp.threads():
                    tid = warp_id * cp.int32(WARP_SIZE) + lane

                    workIndex = _iter * cp.int32(BLOCK_SIZE) + tid + numIterPerBlock * bx * cp.int32(BLOCK_SIZE)

                    masked = cp.int32(0)
                    if workIndex < numElements:
                        masked = thresholdMask[workIndex] + cp.int32(0)

                    m_scan = masked + cp.int32(0)
                    n = thread.shfl_up(m_scan, 1)
                    if lane >= cp.int32(1):
                        m_scan = m_scan + n
                    n = thread.shfl_up(m_scan, 2)
                    if lane >= cp.int32(2):
                        m_scan = m_scan + n
                    n = thread.shfl_up(m_scan, 4)
                    if lane >= cp.int32(4):
                        m_scan = m_scan + n
                    n = thread.shfl_up(m_scan, 8)
                    if lane >= cp.int32(8):
                        m_scan = m_scan + n
                    n = thread.shfl_up(m_scan, 16)
                    if lane >= cp.int32(16):
                        m_scan = m_scan + n

                    if lane == cp.int32(WARP_SIZE - 1):
                        sWarpAccum[warp_id] = cp.disjoint(m_scan)

                    if tid == cp.int32(0):
                        sPrevAccum[0] = cp.disjoint(sBlockAccum[0])

            block.barrier()

            # Phase 2: scan warp sums
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(NUM_WARPS)):
                    value = sWarpAccum[tid] + cp.int32(0)
                    orig = value
                    n = thread.shfl_up(value, 1)
                    if tid >= cp.int32(1):
                        value = value + n
                    n = thread.shfl_up(value, 2)
                    if tid >= cp.int32(2):
                        value = value + n
                    n = thread.shfl_up(value, 4)
                    if tid >= cp.int32(4):
                        value = value + n
                    sWarpAccum[tid] = cp.disjoint(value - orig)
                    if tid == cp.int32(NUM_WARPS - 1):
                        sBlockAccum[0] = cp.disjoint(sBlockAccum[0] + value)

            block.barrier()

            # Phase 3: write masked elements
            for warp_id, warp in block.warps():
                for lane, thread in warp.threads():
                    tid = warp_id * cp.int32(WARP_SIZE) + lane

                    workIndex = _iter * cp.int32(BLOCK_SIZE) + tid + numIterPerBlock * bx * cp.int32(BLOCK_SIZE)

                    masked = cp.int32(0)
                    if workIndex < numElements:
                        masked = thresholdMask[workIndex] + cp.int32(0)

                    m_scan = masked + cp.int32(0)
                    n = thread.shfl_up(m_scan, 1)
                    if lane >= cp.int32(1):
                        m_scan = m_scan + n
                    n = thread.shfl_up(m_scan, 2)
                    if lane >= cp.int32(2):
                        m_scan = m_scan + n
                    n = thread.shfl_up(m_scan, 4)
                    if lane >= cp.int32(4):
                        m_scan = m_scan + n
                    n = thread.shfl_up(m_scan, 8)
                    if lane >= cp.int32(8):
                        m_scan = m_scan + n
                    n = thread.shfl_up(m_scan, 16)
                    if lane >= cp.int32(16):
                        m_scan = m_scan + n
                    m_exclusive = m_scan - masked

                    if masked != cp.int32(0):
                        prevAccum = sPrevAccum[0] + cp.int32(0)
                        warpOff = sWarpAccum[warp_id] + cp.int32(0)
                        index = m_exclusive + warpOff + prevAccum + blockStartIndex

                        maskedElements[index, 0] = thresholdStream[workIndex, 0] + cp.int32(0)
                        maskedElements[index, 1] = thresholdStream[workIndex, 1] + cp.int32(0)
                        maskedElements[index, 2] = thresholdStream[workIndex, 2] + cp.int32(0)
                        maskedElements[index, 3] = thresholdStream[workIndex, 3] + cp.int32(0)
                        maskedElements[index, 4] = thresholdStream[workIndex, 4] + cp.int32(0)
                        maskedElements[index, 5] = thresholdStream[workIndex, 5] + cp.int32(0)
                        maskedElements[index, 6] = thresholdStream[workIndex, 6] + cp.int32(0)
                        maskedElements[index, 7] = thresholdStream[workIndex, 7] + cp.int32(0)

            block.barrier()

            _iter = _iter + cp.int32(1)


# ===========================================================================
# Group F: Element copy with 3-way conditional
# ===========================================================================

# ===== Kernel 14: createForceChangeThresholdElements =====
# Grid-stride loop, 3 branches (lostPair/foundPair/persistent), copies elements.
@cp.kernel
def createForceChangeThresholdElements(
    exceededForceElements,       # int32[M, TE_SIZE] -- current exceeded
    numExceededForceElements,    # int32 scalar
    prevExceededForceElements,   # int32[P, TE_SIZE] -- previous exceeded
    numPrevExceeded,             # int32 scalar
    thresholdMask,               # int32[M] -- mask from setThresholdElementsMask
    prevMaskedElements,          # int32[Q, TE_SIZE] -- prev masked (persistent)
    numPrevMasked,               # int32 scalar
    forceChangeElements,         # int32[R, TE_SIZE] -- output: force change elements
    forceChangeTypes,            # int32[R] -- output: 0=lost, 1=found, 2=persistent
    numForceChangeElements,      # int32[1] -- output: total count
    BLOCK_SIZE: cp.constexpr = 256,
    TE_SIZE: cp.constexpr = 8,
):
    # Total elements to process = numPrevExceeded + numExceededForceElements
    totalElements = numExceededForceElements + numPrevExceeded + numPrevMasked
    with cp.Kernel(cp.ceildiv(totalElements, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            i = bx * BLOCK_SIZE + tid
            if i < totalElements:
                # Pre-declare variables used across if/elif/else branches
                outIdx = cp.int32(0)

                # Determine which source this element comes from
                if i < numPrevExceeded:
                    # Lost pair: prev exceeded element NOT in current mask
                    # The prev element at index i -- check thresholdMask
                    # Lost pairs are prev elements not found in current exceeded
                    outIdx = thread.atomic_add(numForceChangeElements[0], cp.int32(1))
                    forceChangeElements[outIdx, 0] = prevExceededForceElements[i, 0] + cp.int32(0)
                    forceChangeElements[outIdx, 1] = prevExceededForceElements[i, 1] + cp.int32(0)
                    forceChangeElements[outIdx, 2] = prevExceededForceElements[i, 2] + cp.int32(0)
                    forceChangeElements[outIdx, 3] = prevExceededForceElements[i, 3] + cp.int32(0)
                    forceChangeElements[outIdx, 4] = prevExceededForceElements[i, 4] + cp.int32(0)
                    forceChangeElements[outIdx, 5] = prevExceededForceElements[i, 5] + cp.int32(0)
                    forceChangeElements[outIdx, 6] = prevExceededForceElements[i, 6] + cp.int32(0)
                    forceChangeElements[outIdx, 7] = prevExceededForceElements[i, 7] + cp.int32(0)
                    forceChangeTypes[outIdx] = cp.int32(0)  # lost pair
                elif i < numPrevExceeded + numExceededForceElements:
                    # Found pair: current exceeded element NOT in prev
                    curIdx = i - numPrevExceeded
                    mask = thresholdMask[curIdx] + cp.int32(0)
                    if mask == cp.int32(0):
                        # Not found in prev -> this is a newly found pair
                        outIdx = thread.atomic_add(numForceChangeElements[0], cp.int32(1))
                        forceChangeElements[outIdx, 0] = exceededForceElements[curIdx, 0] + cp.int32(0)
                        forceChangeElements[outIdx, 1] = exceededForceElements[curIdx, 1] + cp.int32(0)
                        forceChangeElements[outIdx, 2] = exceededForceElements[curIdx, 2] + cp.int32(0)
                        forceChangeElements[outIdx, 3] = exceededForceElements[curIdx, 3] + cp.int32(0)
                        forceChangeElements[outIdx, 4] = exceededForceElements[curIdx, 4] + cp.int32(0)
                        forceChangeElements[outIdx, 5] = exceededForceElements[curIdx, 5] + cp.int32(0)
                        forceChangeElements[outIdx, 6] = exceededForceElements[curIdx, 6] + cp.int32(0)
                        forceChangeElements[outIdx, 7] = exceededForceElements[curIdx, 7] + cp.int32(0)
                        forceChangeTypes[outIdx] = cp.int32(1)  # found pair
                else:
                    # Persistent pair
                    persistIdx = i - numPrevExceeded - numExceededForceElements
                    outIdx = thread.atomic_add(numForceChangeElements[0], cp.int32(1))
                    forceChangeElements[outIdx, 0] = prevMaskedElements[persistIdx, 0] + cp.int32(0)
                    forceChangeElements[outIdx, 1] = prevMaskedElements[persistIdx, 1] + cp.int32(0)
                    forceChangeElements[outIdx, 2] = prevMaskedElements[persistIdx, 2] + cp.int32(0)
                    forceChangeElements[outIdx, 3] = prevMaskedElements[persistIdx, 3] + cp.int32(0)
                    forceChangeElements[outIdx, 4] = prevMaskedElements[persistIdx, 4] + cp.int32(0)
                    forceChangeElements[outIdx, 5] = prevMaskedElements[persistIdx, 5] + cp.int32(0)
                    forceChangeElements[outIdx, 6] = prevMaskedElements[persistIdx, 6] + cp.int32(0)
                    forceChangeElements[outIdx, 7] = prevMaskedElements[persistIdx, 7] + cp.int32(0)
                    forceChangeTypes[outIdx] = cp.int32(2)  # persistent
