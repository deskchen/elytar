"""Capybara DSL port of gpunarrowphase/CUDA/pairManagement.cu — all 7 kernels.

Ported kernels (matching CUDA names for PTX replacement):
  - removeContactManagers_Stage1  — initialize keep/drop buffer
  - removeContactManagers_Stage2  — mark keep/drop buffer with remove indices
  - removeContactManagers_Stage3  — exclusive prefix scan (scanKernel1of2)
  - removeContactManagers_Stage4  — cross-block accumulation (scanKernel2of2)
  - removeContactManagers_Stage5  — scatter copy with binary search (PxgPersistentContactManifold)
  - removeContactManagers_Stage5_CvxTri — scatter copy (PxgPersistentContactMultiManifold)
  - initializeManifolds            — shared-memory template replication

ABI differences from CUDA:
  - PxgPairManagementData decomposed: each kernel receives only the fields it
    accesses, as separate tensor/scalar args.
  - PxgContactManagerInput  -> int32[N, CMI_SIZE]  flat tensor (struct as int32 row)
  - PxsContactManagerOutput -> int32[N, CMO_SIZE]  flat tensor
  - PxsContactManager**     -> int64[N] tensor (raw pointers)
  - Sc::ShapeInteraction**  -> int64[N] tensor (raw pointers)
  - PxReal*                 -> float32[N] tensor
  - PxsTorsionalFrictionData -> int32[N, TORSION_SIZE] flat tensor
  - PxgPersistentContactManifold   -> float32[N, MANIFOLD_F4_SIZE * 4] flat tensor
  - PxgPersistentContactMultiManifold -> float32[N, MULTI_MANIFOLD_F4_SIZE * 4] flat tensor
  - mBlockSharedAccumulator -> int32[GRID_SIZE] tensor
  - Stage3/Stage4 use BLOCK_SIZE=512, GRID_SIZE=32 matching CUDA launch config.
  - Stage5/Stage5_CvxTri: struct sizes passed as constexprs.
  - initializeManifolds: source is float32[dataF4Size, 4], destination is
    float32[nbTimesToReplicate * dataF4Size, 4].

Capybara structural notes:
  - block.barrier() between thread regions, not inside.
  - cp.disjoint() for smem writes inside block.threads().
  - cp.assume_uniform() for shfl_up inside warp-uniform conditionals.
  - Ternary safe-load pattern for shfl_up (see algorithms.py).
  - Variables assigned in if/else must be pre-declared.
  - No method chaining on structs; use intermediate variables.
"""

import capybara as cp

WARP_SIZE = 32


# ===== Inline helpers =====

@cp.inline
def binarySearch(data, numElements, value):
    """Binary search: returns largest index such that data[index] <= value.
    If no such index exists (data[0] > value), returns 0.
    Matches CUDA binarySearch from reduction.cuh.
    """
    left = cp.int32(0)
    right = numElements
    while left < right:
        pos = (left + right) >> cp.int32(1)
        element = data[pos] + cp.int32(0)
        if element <= value:
            left = pos + cp.int32(1)
        else:
            right = pos
    # return left ? left - 1 : 0
    result = cp.int32(0)
    if left > cp.int32(0):
        result = left - cp.int32(1)
    return result


# ===== Kernel 1: removeContactManagers_Stage1 =====
# Inlines initializeKeepDropBuffer from SparseRemove.cuh.
# Simple grid-stride loop: first (N-K) elements = 0, last K elements = 1.
@cp.kernel
def removeContactManagers_Stage1(tempAccumulator, nbPairs, nbToRemove,
                                 BLOCK_SIZE: cp.constexpr = 256):
    with cp.Kernel(cp.ceildiv(nbPairs, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            i = bx * BLOCK_SIZE + tid
            if i < nbPairs:
                newSize = nbPairs - nbToRemove
                val = cp.int32(0)
                if i >= newSize:
                    val = cp.int32(1)
                tempAccumulator[i] = val


# ===== Kernel 2: removeContactManagers_Stage2 =====
# Inlines markKeepDropBuff from SparseRemove.cuh.
# For each remove index: if index < newSize, mark 1 (needs swap); else mark 0.
@cp.kernel
def removeContactManagers_Stage2(removeIndices, nbToRemove, tempAccumulator, nbPairs,
                                 BLOCK_SIZE: cp.constexpr = 256):
    with cp.Kernel(cp.ceildiv(nbToRemove, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            i = bx * BLOCK_SIZE + tid
            if i < nbToRemove:
                newSize = nbPairs - nbToRemove
                index = removeIndices[i] + cp.int32(0)
                mask = cp.int32(0)
                if index < newSize:
                    mask = cp.int32(1)
                tempAccumulator[index] = mask


# ===== Kernel 3: removeContactManagers_Stage3 =====
# Inlines scanKernel1of2 from reduction.cuh (exclusive prefix scan, AddOp<PxU32>).
# CUDA launch: gridDim=GRID_SIZE=32, blockDim=(WARP_SIZE, BLOCK_SIZE/WARP_SIZE)
#   = (32, 16) = 512 threads total.
#
# This performs an in-place exclusive scan on tempAccumulator with a grid-stride
# outer loop. Each block processes multiple chunks of BLOCK_SIZE elements,
# accumulating a running total. At the end, each block writes its total to
# crossBlockAccumulator.
#
# The CUDA exclusive warp scan computes: for each thread, the sum of all
# preceding elements in the warp (excluding the thread's own value).
# We implement this by tracking both inclusive (inp) and exclusive (exc) sums.
@cp.kernel
def removeContactManagers_Stage3(tempAccumulator, totalCount, crossBlockAccumulator,
                                 BLOCK_SIZE: cp.constexpr = 512,
                                 NUM_WARPS: cp.constexpr = 16,
                                 GRID_SIZE: cp.constexpr = 32):
    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        crossWarpAccum = block.alloc((NUM_WARPS,), dtype=cp.int32)
        blockAccum = block.alloc((1,), dtype=cp.int32)
        prevAccumSmem = block.alloc((1,), dtype=cp.int32)
        vals = block.alloc((BLOCK_SIZE,), dtype=cp.int32)

        nbBlocksRequired = cp.ceildiv(totalCount, BLOCK_SIZE)
        nbBlocksPerBlock = cp.ceildiv(nbBlocksRequired, GRID_SIZE)
        blockStartIndex = bx * nbBlocksPerBlock

        # Initialize block accumulator to 0 (AddOp default)
        for tid, thread in block.threads():
            if tid == cp.int32(0):
                blockAccum[0] = cp.int32(0)

        block.barrier()

        for iterIdx in range(4096):
            # Phase 1: Load + exclusive warp scan + store warp totals
            # Also save prevAccum (value of blockAccum before this iteration)
            for tid, thread in block.threads():
                if cp.assume_uniform(iterIdx < nbBlocksPerBlock):
                    threadIndex = tid + (blockStartIndex + cp.int32(iterIdx)) * BLOCK_SIZE
                    lane_id = tid % cp.int32(WARP_SIZE)
                    warp_id = tid // cp.int32(WARP_SIZE)

                    safe_idx = threadIndex if threadIndex < totalCount else cp.int32(0)
                    raw = tempAccumulator[safe_idx] + cp.int32(0)
                    val = raw if threadIndex < totalCount else cp.int32(0)

                    # Exclusive warp scan: exc accumulates the prefix sum
                    # excluding the current thread's value
                    exc = cp.int32(0)
                    inp = val
                    n = thread.shfl_up(inp, 1)
                    if lane_id >= cp.int32(1):
                        inp = inp + n
                        exc = exc + n
                    n = thread.shfl_up(inp, 2)
                    if lane_id >= cp.int32(2):
                        inp = inp + n
                        exc = exc + n
                    n = thread.shfl_up(inp, 4)
                    if lane_id >= cp.int32(4):
                        inp = inp + n
                        exc = exc + n
                    n = thread.shfl_up(inp, 8)
                    if lane_id >= cp.int32(8):
                        inp = inp + n
                        exc = exc + n
                    n = thread.shfl_up(inp, 16)
                    if lane_id >= cp.int32(16):
                        inp = inp + n
                        exc = exc + n

                    vals[tid] = cp.disjoint(exc)
                    # Last lane of each warp: write inclusive scan (= warp total)
                    if lane_id == cp.int32(WARP_SIZE - 1):
                        crossWarpAccum[warp_id] = cp.disjoint(inp)

                    # Thread 0 saves the previous block accumulator
                    if tid == cp.int32(0):
                        prevAccumSmem[0] = cp.disjoint(blockAccum[0])

            block.barrier()

            # Phase 2: Warp 0 exclusive-scans the cross-warp accumulators
            for tid, thread in block.threads():
                if cp.assume_uniform(iterIdx < nbBlocksPerBlock):
                    if cp.assume_uniform(tid < cp.int32(NUM_WARPS)):
                        cw = crossWarpAccum[tid] + cp.int32(0)
                        exc2 = cp.int32(0)
                        inp2 = cw
                        n = thread.shfl_up(inp2, 1)
                        if tid >= cp.int32(1):
                            inp2 = inp2 + n
                            exc2 = exc2 + n
                        n = thread.shfl_up(inp2, 2)
                        if tid >= cp.int32(2):
                            inp2 = inp2 + n
                            exc2 = exc2 + n
                        n = thread.shfl_up(inp2, 4)
                        if tid >= cp.int32(4):
                            inp2 = inp2 + n
                            exc2 = exc2 + n
                        n = thread.shfl_up(inp2, 8)
                        if tid >= cp.int32(8):
                            inp2 = inp2 + n
                            exc2 = exc2 + n
                        n = thread.shfl_up(inp2, 16)
                        if tid >= cp.int32(16):
                            inp2 = inp2 + n
                            exc2 = exc2 + n

                        crossWarpAccum[tid] = cp.disjoint(exc2)

                        # Last cross-warp thread: update block accum
                        # accum = prevAccum + inclusive_total_of_all_warps
                        if tid == cp.int32(NUM_WARPS - 1):
                            blockAccum[0] = cp.disjoint(prevAccumSmem[0] + inp2)

            block.barrier()

            # Phase 3: Combine per-thread exclusive + cross-warp offset + prev accum
            for tid, thread in block.threads():
                if cp.assume_uniform(iterIdx < nbBlocksPerBlock):
                    threadIndex = tid + (blockStartIndex + cp.int32(iterIdx)) * BLOCK_SIZE
                    warp_id = tid // cp.int32(WARP_SIZE)
                    if threadIndex < totalCount:
                        excVal = vals[tid] + cp.int32(0)
                        cwVal = crossWarpAccum[warp_id] + cp.int32(0)
                        pa = prevAccumSmem[0] + cp.int32(0)
                        tempAccumulator[threadIndex] = excVal + cwVal + pa

            block.barrier()

        # After all iterations, write the final block accumulator
        for tid, thread in block.threads():
            if tid == cp.int32(NUM_WARPS - 1):
                crossBlockAccumulator[bx] = blockAccum[0]


# ===== Kernel 4: removeContactManagers_Stage4 =====
# Inlines scanKernel2of2 from reduction.cuh.
# Phase 1: Warp 0 exclusive-scans crossBlockAccumulator (GRID_SIZE entries).
# Phase 2: All threads add the per-block offset to their values (grid-stride).
@cp.kernel
def removeContactManagers_Stage4(tempAccumulator, totalCount, crossBlockAccumulator,
                                 BLOCK_SIZE: cp.constexpr = 512,
                                 NUM_WARPS: cp.constexpr = 16,
                                 GRID_SIZE: cp.constexpr = 32):
    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        blockAccumSmem = block.alloc((GRID_SIZE,), dtype=cp.int32)

        nbBlocksRequired = cp.ceildiv(totalCount, BLOCK_SIZE)
        nbBlocksPerBlock = cp.ceildiv(nbBlocksRequired, GRID_SIZE)
        blockStartIndex = bx * nbBlocksPerBlock

        # Phase 1: Warp 0 exclusive-scans crossBlockAccumulator
        for tid, thread in block.threads():
            warp_id = tid // cp.int32(WARP_SIZE)
            if cp.assume_uniform(warp_id == cp.int32(0)):
                safe_tid = tid if tid < cp.int32(GRID_SIZE) else cp.int32(0)
                raw = crossBlockAccumulator[safe_tid] + cp.int32(0)
                val = raw if tid < cp.int32(GRID_SIZE) else cp.int32(0)

                # Exclusive warp scan
                exc = cp.int32(0)
                inp = val
                n = thread.shfl_up(inp, 1)
                if tid >= cp.int32(1):
                    inp = inp + n
                    exc = exc + n
                n = thread.shfl_up(inp, 2)
                if tid >= cp.int32(2):
                    inp = inp + n
                    exc = exc + n
                n = thread.shfl_up(inp, 4)
                if tid >= cp.int32(4):
                    inp = inp + n
                    exc = exc + n
                n = thread.shfl_up(inp, 8)
                if tid >= cp.int32(8):
                    inp = inp + n
                    exc = exc + n
                n = thread.shfl_up(inp, 16)
                if tid >= cp.int32(16):
                    inp = inp + n
                    exc = exc + n

                if tid < cp.int32(GRID_SIZE):
                    blockAccumSmem[tid] = cp.disjoint(exc)

        block.barrier()

        # Phase 2: Grid-stride add of per-block accumulation to each element
        for iterIdx in range(4096):
            for tid, thread in block.threads():
                if cp.assume_uniform(iterIdx < nbBlocksPerBlock):
                    threadIndex = tid + (blockStartIndex + cp.int32(iterIdx)) * BLOCK_SIZE
                    if threadIndex < totalCount:
                        accumulation = blockAccumSmem[bx] + cp.int32(0)
                        val = tempAccumulator[threadIndex] + cp.int32(0)
                        tempAccumulator[threadIndex] = val + accumulation

            block.barrier()


# ===== Kernel 5: removeContactManagers_Stage5 =====
# Scatter copy using binary search for swap indices.
# Uses PxgPersistentContactManifold (16 float4 = 64 floats).
# CUDA uses 16 threads per swap item (half-warp), but the scalar fields are
# only copied by thread 0 of each group. In Capybara, each thread handles
# one full swap item (all fields + manifold) for simplicity.
@cp.kernel
def removeContactManagers_Stage5(tempAccumulator, nbPairs, nbToRemove,
                                 inputData, outputData, cms, sis,
                                 rest, torsional, manifolds,
                                 copyManifold,
                                 CMI_SIZE: cp.constexpr = 1,
                                 CMO_SIZE: cp.constexpr = 1,
                                 TORSION_SIZE: cp.constexpr = 1,
                                 MANIFOLD_F4_SIZE: cp.constexpr = 16,
                                 BLOCK_SIZE: cp.constexpr = 512,
                                 GRID_SIZE: cp.constexpr = 32):
    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            newSize = nbPairs - nbToRemove
            nbToSwap = tempAccumulator[newSize] + cp.int32(0)

            totalThreads = BLOCK_SIZE * GRID_SIZE
            globalTid = bx * BLOCK_SIZE + tid

            for swapIdx in range(1048576):
                i = globalTid + cp.int32(swapIdx) * cp.int32(totalThreads)
                if i < nbToSwap:
                    # getSwapIndices via binary search
                    dstIndex = binarySearch(tempAccumulator, nbPairs, i)
                    srcIndex = binarySearch(tempAccumulator, nbPairs, i + nbToSwap)

                    # Copy all struct fields
                    for c in range(CMI_SIZE):
                        inputData[dstIndex, c] = inputData[srcIndex, c]
                    for c in range(CMO_SIZE):
                        outputData[dstIndex, c] = outputData[srcIndex, c]
                    cms[dstIndex] = cms[srcIndex]
                    sis[dstIndex] = sis[srcIndex]
                    rest[dstIndex] = rest[srcIndex]
                    for c in range(TORSION_SIZE):
                        torsional[dstIndex, c] = torsional[srcIndex, c]

                    # Copy manifold (PxgPersistentContactManifold) if requested
                    if copyManifold != cp.int32(0):
                        for f4 in range(MANIFOLD_F4_SIZE):
                            manifolds[dstIndex, f4 * 4 + 0] = manifolds[srcIndex, f4 * 4 + 0]
                            manifolds[dstIndex, f4 * 4 + 1] = manifolds[srcIndex, f4 * 4 + 1]
                            manifolds[dstIndex, f4 * 4 + 2] = manifolds[srcIndex, f4 * 4 + 2]
                            manifolds[dstIndex, f4 * 4 + 3] = manifolds[srcIndex, f4 * 4 + 3]


# ===== Kernel 6: removeContactManagers_Stage5_CvxTri =====
# Same as Stage5 but with PxgPersistentContactMultiManifold (76 float4 = 304 floats).
# CUDA uses WARP_SIZE threads per swap item for the larger manifold copy.
@cp.kernel
def removeContactManagers_Stage5_CvxTri(tempAccumulator, nbPairs, nbToRemove,
                                         inputData, outputData, cms, sis,
                                         rest, torsional, manifolds,
                                         copyManifold,
                                         CMI_SIZE: cp.constexpr = 1,
                                         CMO_SIZE: cp.constexpr = 1,
                                         TORSION_SIZE: cp.constexpr = 1,
                                         MULTI_MANIFOLD_F4_SIZE: cp.constexpr = 76,
                                         BLOCK_SIZE: cp.constexpr = 512,
                                         GRID_SIZE: cp.constexpr = 32):
    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            newSize = nbPairs - nbToRemove
            nbToSwap = tempAccumulator[newSize] + cp.int32(0)

            totalThreads = BLOCK_SIZE * GRID_SIZE
            globalTid = bx * BLOCK_SIZE + tid

            for swapIdx in range(1048576):
                i = globalTid + cp.int32(swapIdx) * cp.int32(totalThreads)
                if i < nbToSwap:
                    dstIndex = binarySearch(tempAccumulator, nbPairs, i)
                    srcIndex = binarySearch(tempAccumulator, nbPairs, i + nbToSwap)

                    # Copy struct fields
                    for c in range(CMI_SIZE):
                        inputData[dstIndex, c] = inputData[srcIndex, c]
                    for c in range(CMO_SIZE):
                        outputData[dstIndex, c] = outputData[srcIndex, c]
                    cms[dstIndex] = cms[srcIndex]
                    sis[dstIndex] = sis[srcIndex]
                    rest[dstIndex] = rest[srcIndex]
                    for c in range(TORSION_SIZE):
                        torsional[dstIndex, c] = torsional[srcIndex, c]

                    # Copy manifold (PxgPersistentContactMultiManifold) if requested
                    if copyManifold != cp.int32(0):
                        for f4 in range(MULTI_MANIFOLD_F4_SIZE):
                            manifolds[dstIndex, f4 * 4 + 0] = manifolds[srcIndex, f4 * 4 + 0]
                            manifolds[dstIndex, f4 * 4 + 1] = manifolds[srcIndex, f4 * 4 + 1]
                            manifolds[dstIndex, f4 * 4 + 2] = manifolds[srcIndex, f4 * 4 + 2]
                            manifolds[dstIndex, f4 * 4 + 3] = manifolds[srcIndex, f4 * 4 + 3]


# ===== Kernel 7: initializeManifolds =====
# Loads a template float4 structure into shared memory, then replicates it to
# all destination slots using grid-stride blocks.
#
# CUDA signature: initializeManifolds(float4* destination, const float4* source,
#                                      PxU32 dataSize, PxU32 nbTimesToReplicate)
# Capybara ABI:
#   source      — float32[dataF4Size, 4]   (the template structure as float4 rows)
#   destination — float32[nbTimesToReplicate * dataF4Size, 4]  (output)
#   dataF4Size  — scalar (= dataSize / sizeof(float4) = number of float4 elements)
#   nbTimesToReplicate — scalar
#
# Shared memory stores the source data deinterleaved into 4 float channels
# (x, y, z, w). MaxStructureSize = 4096 bytes = 256 float4 max = 1024 floats.
@cp.kernel
def initializeManifolds(destination, source, dataF4Size, nbTimesToReplicate,
                        BLOCK_SIZE: cp.constexpr = 256,
                        MAX_SMEM_FLOATS: cp.constexpr = 1024):
    with cp.Kernel(cp.ceildiv(nbTimesToReplicate, BLOCK_SIZE // dataF4Size),
                   threads=BLOCK_SIZE) as (bx, block):
        sourceData = block.alloc((MAX_SMEM_FLOATS,), dtype=cp.float32)

        nbThreadsPerElement = dataF4Size
        nbPerBlock = BLOCK_SIZE // nbThreadsPerElement
        maxThreadIdx = nbPerBlock * nbThreadsPerElement
        nbBlocksRequired = cp.ceildiv(nbTimesToReplicate, nbPerBlock)

        # Phase 1: Load source float4 data into shared memory (deinterleaved)
        for tid, thread in block.threads():
            if tid < nbThreadsPerElement:
                sx = source[tid, 0]
                sy = source[tid, 1]
                sz = source[tid, 2]
                sw = source[tid, 3]
                sourceData[tid] = cp.disjoint(sx)
                sourceData[tid + nbThreadsPerElement] = cp.disjoint(sy)
                sourceData[tid + cp.int32(2) * nbThreadsPerElement] = cp.disjoint(sz)
                sourceData[tid + cp.int32(3) * nbThreadsPerElement] = cp.disjoint(sw)

        block.barrier()

        # Phase 2: Replicate to destination using grid-stride over blocks
        for iterIdx in range(1048576):
            for tid, thread in block.threads():
                a = bx + cp.int32(iterIdx) * nbBlocksRequired
                if a < nbBlocksRequired:
                    threadReadIdx = tid % nbThreadsPerElement
                    startBlockIdx = nbPerBlock * a
                    outBase = startBlockIdx * nbThreadsPerElement
                    idx = startBlockIdx + tid // nbThreadsPerElement

                    if tid < maxThreadIdx:
                        if idx < nbTimesToReplicate:
                            fx = sourceData[threadReadIdx] + cp.float32(0.0)
                            fy = sourceData[threadReadIdx + nbThreadsPerElement] + cp.float32(0.0)
                            fz = sourceData[threadReadIdx + cp.int32(2) * nbThreadsPerElement] + cp.float32(0.0)
                            fw = sourceData[threadReadIdx + cp.int32(3) * nbThreadsPerElement] + cp.float32(0.0)
                            outIdx = outBase + tid
                            destination[outIdx, 0] = fx
                            destination[outIdx, 1] = fy
                            destination[outIdx, 2] = fz
                            destination[outIdx, 3] = fw

            block.barrier()
