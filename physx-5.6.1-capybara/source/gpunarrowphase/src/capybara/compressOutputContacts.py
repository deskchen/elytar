"""Capybara DSL port of gpunarrowphase/CUDA/compressOutputContacts.cu -- all 3 kernels.

Ported kernels (matching CUDA names for PTX replacement):
  - compressContactStage1       -- block reduction counting pairs with contacts
  - compressContactStage2       -- prefix-sum + scatter to PxGpuContactPair output
  - updateFrictionPatches       -- simple per-thread friction patch pointer update

ABI differences from CUDA:
  - PxsContactManagerOutput -> int32[N, CMO_SIZE] flat tensor (struct as int32 row).
    Key field offsets (int32 indices within a row):
      CMO_NB_CONTACTS  = word containing nbContacts (PxU16, low 16 bits)
      CMO_PATCH_OFFSET = contactPatches stored as pre-computed byte offset (int32)
      CMO_CONTACT_OFFSET = contactPoints stored as pre-computed byte offset (int32)
      CMO_FORCE_OFFSET = contactForces stored as byte offset or 0xFFFFFFFF if null
      CMO_NB_PATCHES   = word containing nbPatches (byte 1 of packed word)
      CMO_FRICTION_PATCH_OFFSET = frictionPatches stored as byte offset (int32)
  - PxgContactManagerInput -> int32[N, CMI_SIZE] flat tensor.
      CMI_TRANSFORM_REF0 = offset of transformCacheRef0
      CMI_TRANSFORM_REF1 = offset of transformCacheRef1
  - PxGpuContactPair output -> int32[M, PAIR_SIZE] flat tensor.
    Pointer fields stored as pairs of int32 (lo, hi) for 64-bit GPU addresses.
    Host pre-computes byte offsets; kernel adds GPU base addresses.
  - PxNodeIndex shapeToRigidRemapTable -> int64[N] tensor (PxNodeIndex is 8 bytes).
  - transformCacheIdToActorTable -> int64[N] tensor (PxActor* is 8 bytes).
  - GPU base pointers passed as int64 scalars.
  - sizeof(PxContactPatch)=64 and sizeof(PxFrictionPatch)=52 as constexprs.

Capybara structural notes:
  - block.barrier() between thread regions, not inside.
  - cp.disjoint() for smem writes inside block.threads().
  - cp.assume_uniform() for shfl inside warp-uniform conditionals.
  - Variables in if/else must be pre-declared before the if.
  - Warp reduction / scan manually inlined (no @cp.inline with thread).
  - While loops used for runtime-dependent iteration counts.
"""

import capybara as cp

WARP_SIZE = 32


# ===== Kernel 1: compressContactStage1 =====
# Multi-iteration grid-stride block reduction counting pairs with contacts.
# Each block counts how many of its assigned pairs have nbContacts > 0.
@cp.kernel
def compressContactStage1(
    outputData,          # int32[N, CMO_SIZE]
    numTotalPairs,       # int32 scalar
    gBlockNumPairs,      # int32[GRID_SIZE] -- output: per-block contact count
    BLOCK_SIZE: cp.constexpr = 256,
    GRID_SIZE: cp.constexpr = 32,
    CMO_SIZE: cp.constexpr = 12,
    CMO_NB_CONTACTS: cp.constexpr = 9,
):
    NUM_WARPS = BLOCK_SIZE // WARP_SIZE

    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        sWarpAccum = block.alloc((NUM_WARPS,), dtype=cp.int32)
        sAccum = block.alloc((1,), dtype=cp.int32)

        # Initialize sAccum to 0
        for tid, thread in block.threads():
            if tid == cp.int32(WARP_SIZE - 1):
                sAccum[0] = cp.disjoint(cp.int32(0))

        block.barrier()

        totalBlockRequired = (numTotalPairs + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterationPerBlock = (totalBlockRequired + cp.int32(GRID_SIZE - 1)) // cp.int32(GRID_SIZE)

        _iter = cp.int32(0)
        while _iter < numIterationPerBlock:
            # Phase 1: ballot+popc -> write per-warp count to sWarpAccum
            for warp_id, warp in block.warps():
                for lane, thread in warp.threads():
                    tid = warp_id * cp.int32(WARP_SIZE) + lane

                    workIndex = _iter * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                    nbContacts = cp.int32(0)
                    if workIndex < numTotalPairs:
                        raw = outputData[workIndex, CMO_NB_CONTACTS] + cp.int32(0)
                        nbContacts = raw & cp.int32(0xFFFF)

                    hasContact = cp.int32(1) if nbContacts > cp.int32(0) else cp.int32(0)
                    contactMask = thread.coll.ballot(hasContact != cp.int32(0))
                    contactAccum = thread.popcount(contactMask)

                    if lane == cp.int32(WARP_SIZE - 1):
                        sWarpAccum[warp_id] = cp.disjoint(contactAccum)

            block.barrier()

            # Phase 2: warp reduction across warp sums -> accumulate into sAccum
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(NUM_WARPS)):
                    value = sWarpAccum[tid] + cp.int32(0)
                    # Warp reduction using shfl_xor (5 rounds)
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
                    # All active threads have the sum; last thread accumulates.
                    if tid == cp.int32(NUM_WARPS - 1):
                        sAccum[0] = cp.disjoint(sAccum[0] + value)

            block.barrier()

            _iter = _iter + cp.int32(1)

        # Write final result
        for tid, thread in block.threads():
            if tid == cp.int32(NUM_WARPS - 1):
                gBlockNumPairs[bx] = sAccum[0]


# ===== Kernel 2: compressContactStage2 =====
# Prefix-sum of per-block counts, then scatter contact pairs to output.
# GRID_SIZE == 32, so warp-0 can scan all block counts in one warp scan.
@cp.kernel
def compressContactStage2(
    inputData,                    # int32[N, CMI_SIZE]
    outputData,                   # int32[N, CMO_SIZE]
    numTotalPairs,                # int32 scalar
    shapeToRigidRemapTable,       # int64[N]
    transformCacheIdToActorTable,  # int64[N]
    gBlockNumPairs,               # int32[GRID_SIZE]
    gpuCompressedPatchesBase,     # int64 scalar
    gpuCompressedContactsBase,    # int64 scalar
    gpuForceBufferBase,           # int64 scalar
    gpuFrictionPatchesBase,       # int64 scalar
    numPairs,                     # int32[1] -- output: total pair count
    outputPairs,                  # int32[M, PAIR_SIZE]
    maxOutputPatches,             # int32 scalar
    BLOCK_SIZE: cp.constexpr = 256,
    GRID_SIZE: cp.constexpr = 32,
    CMI_SIZE: cp.constexpr = 4,
    CMO_SIZE: cp.constexpr = 12,
    PAIR_SIZE: cp.constexpr = 19,
    CMI_TRANSFORM_REF0: cp.constexpr = 2,
    CMI_TRANSFORM_REF1: cp.constexpr = 3,
    CMO_NB_CONTACTS: cp.constexpr = 9,
    CMO_PATCH_OFFSET: cp.constexpr = 0,
    CMO_CONTACT_OFFSET: cp.constexpr = 1,
    CMO_FORCE_OFFSET: cp.constexpr = 2,
    CMO_NB_PATCHES: cp.constexpr = 8,
    PAIR_PATCH_OFFSET: cp.constexpr = 0,
    PAIR_CONTACT_OFFSET: cp.constexpr = 2,
    PAIR_FORCE_OFFSET: cp.constexpr = 4,
    PAIR_FRICTION_OFFSET: cp.constexpr = 6,
    PAIR_TRANSFORM_REF0: cp.constexpr = 8,
    PAIR_TRANSFORM_REF1: cp.constexpr = 9,
    PAIR_NODE_INDEX0_LO: cp.constexpr = 10,
    PAIR_NODE_INDEX0_HI: cp.constexpr = 11,
    PAIR_NODE_INDEX1_LO: cp.constexpr = 12,
    PAIR_NODE_INDEX1_HI: cp.constexpr = 13,
    PAIR_ACTOR0_LO: cp.constexpr = 14,
    PAIR_ACTOR0_HI: cp.constexpr = 15,
    PAIR_ACTOR1_LO: cp.constexpr = 16,
    PAIR_ACTOR1_HI: cp.constexpr = 17,
    PAIR_NB_CONTACTS_PATCHES: cp.constexpr = 18,
    SIZEOF_CONTACT_PATCH: cp.constexpr = 64,
    SIZEOF_FRICTION_PATCH: cp.constexpr = 52,
):
    NUM_WARPS = BLOCK_SIZE // WARP_SIZE

    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        sWarpAccum = block.alloc((NUM_WARPS,), dtype=cp.int32)
        sBlockHistogram = block.alloc((GRID_SIZE,), dtype=cp.int32)
        sAccum = block.alloc((1,), dtype=cp.int32)
        sPrevAccum = block.alloc((1,), dtype=cp.int32)

        # Initialize sAccum
        for tid, thread in block.threads():
            if tid == cp.int32(WARP_SIZE - 1):
                sAccum[0] = cp.disjoint(cp.int32(0))

        block.barrier()

        # Phase 0: Warp-0 scans gBlockNumPairs -> exclusive prefix sum
        for tid, thread in block.threads():
            warpIndex = tid // cp.int32(WARP_SIZE)
            threadIndexInWarp = tid % cp.int32(WARP_SIZE)
            if cp.assume_uniform(warpIndex == cp.int32(0)):
                if cp.assume_uniform(threadIndexInWarp < cp.int32(GRID_SIZE)):
                    blockNumPairs = gBlockNumPairs[threadIndexInWarp] + cp.int32(0)
                    value = blockNumPairs
                    # Inclusive warp scan (shfl_up, 5 rounds)
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
                    # Exclusive prefix sum = inclusive - original
                    sBlockHistogram[threadIndexInWarp] = cp.disjoint(value - blockNumPairs)
                    if threadIndexInWarp == cp.int32(GRID_SIZE - 1):
                        numPairs[0] = value

        block.barrier()

        totalBlockRequired = (numTotalPairs + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterationPerBlock = (totalBlockRequired + cp.int32(GRID_SIZE - 1)) // cp.int32(GRID_SIZE)
        blockStartIndex = sBlockHistogram[bx] + cp.int32(0)

        _iter = cp.int32(0)
        while _iter < numIterationPerBlock:
            # Phase 1: ballot+popc, write warp counts, save prevAccum
            for warp_id, warp in block.warps():
                for lane, thread in warp.threads():
                    tid = warp_id * cp.int32(WARP_SIZE) + lane

                    workIndex = _iter * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                    nbContacts = cp.int32(0)
                    if workIndex < numTotalPairs:
                        raw = outputData[workIndex, CMO_NB_CONTACTS] + cp.int32(0)
                        nbContacts = raw & cp.int32(0xFFFF)

                    hasContact = cp.int32(1) if nbContacts > cp.int32(0) else cp.int32(0)
                    contactMask = thread.coll.ballot(hasContact != cp.int32(0))
                    contactAccum = thread.popcount(contactMask)

                    if lane == cp.int32(WARP_SIZE - 1):
                        sWarpAccum[warp_id] = cp.disjoint(contactAccum)

                    # Save prevAccum (sAccum before this iteration's update)
                    if tid == cp.int32(0):
                        sPrevAccum[0] = cp.disjoint(sAccum[0])

            block.barrier()

            # Phase 2: warp scan of warp counts -> exclusive prefix; update sAccum
            for tid, thread in block.threads():
                if cp.assume_uniform(tid < cp.int32(NUM_WARPS)):
                    value = sWarpAccum[tid] + cp.int32(0)
                    orig = value
                    # Inclusive warp scan (shfl_up)
                    n = thread.shfl_up(value, 1)
                    if tid >= cp.int32(1):
                        value = value + n
                    n = thread.shfl_up(value, 2)
                    if tid >= cp.int32(2):
                        value = value + n
                    n = thread.shfl_up(value, 4)
                    if tid >= cp.int32(4):
                        value = value + n
                    n = thread.shfl_up(value, 8)
                    if tid >= cp.int32(8):
                        value = value + n
                    n = thread.shfl_up(value, 16)
                    if tid >= cp.int32(16):
                        value = value + n
                    # Exclusive scan: inclusive - original
                    sWarpAccum[tid] = cp.disjoint(value - orig)
                    if tid == cp.int32(NUM_WARPS - 1):
                        sAccum[0] = cp.disjoint(sAccum[0] + value)

            block.barrier()

            # Phase 3: write output for threads with contacts
            for warp_id, warp in block.warps():
                for lane, thread in warp.threads():
                    tid = warp_id * cp.int32(WARP_SIZE) + lane

                    workIndex = _iter * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                    nbContacts = cp.int32(0)
                    if workIndex < numTotalPairs:
                        raw = outputData[workIndex, CMO_NB_CONTACTS] + cp.int32(0)
                        nbContacts = raw & cp.int32(0xFFFF)

                    hasContact = cp.int32(1) if nbContacts > cp.int32(0) else cp.int32(0)
                    contactMask = thread.coll.ballot(hasContact != cp.int32(0))
                    threadMask = (cp.int32(1) << lane) - cp.int32(1)
                    offset = thread.popcount(contactMask & threadMask)

                    prevAccum = sPrevAccum[0] + cp.int32(0)
                    warpOff = sWarpAccum[warp_id] + cp.int32(0)

                    if nbContacts > cp.int32(0):
                        index = offset + prevAccum + warpOff + blockStartIndex

                        if index < maxOutputPatches:
                            # Read PxsContactManagerOutput fields (pre-computed byte offsets)
                            patchOffset = outputData[workIndex, CMO_PATCH_OFFSET] + cp.int32(0)
                            contactOffset = outputData[workIndex, CMO_CONTACT_OFFSET] + cp.int32(0)
                            forceOffset = outputData[workIndex, CMO_FORCE_OFFSET] + cp.int32(0)
                            nbPatchesWord = outputData[workIndex, CMO_NB_PATCHES] + cp.int32(0)
                            # nbPatches is byte 1 (bits 8-15) of the packed word:
                            # [allflagsStart(u8), nbPatches(u8), statusFlag(u8), prevPatches(u8)]
                            nbPatches = (nbPatchesWord >> cp.int32(8)) & cp.int32(0xFF)

                            # Read PxgContactManagerInput fields
                            transformRef0 = inputData[workIndex, CMI_TRANSFORM_REF0] + cp.int32(0)
                            transformRef1 = inputData[workIndex, CMI_TRANSFORM_REF1] + cp.int32(0)

                            # Look up nodeIndex (int64) from remap table
                            nodeIndex0 = shapeToRigidRemapTable[transformRef0] + cp.int64(0)
                            nodeIndex1 = shapeToRigidRemapTable[transformRef1] + cp.int64(0)

                            # Look up actor pointers (int64)
                            actor0 = transformCacheIdToActorTable[transformRef0] + cp.int64(0)
                            actor1 = transformCacheIdToActorTable[transformRef1] + cp.int64(0)

                            # Compute friction patch byte offset from contact patch byte offset
                            frictionOffset = (patchOffset // cp.int32(SIZEOF_CONTACT_PATCH)) * cp.int32(SIZEOF_FRICTION_PATCH)

                            # Compute GPU pointers as int64
                            gpuPatchPtr = gpuCompressedPatchesBase + cp.int64(patchOffset)
                            gpuContactPtr = gpuCompressedContactsBase + cp.int64(contactOffset)
                            gpuForcePtr = gpuForceBufferBase + cp.int64(forceOffset)
                            gpuFrictionPtr = gpuFrictionPatchesBase + cp.int64(frictionOffset)

                            # Write PxGpuContactPair output (pointers as lo/hi int32 pairs)
                            outputPairs[index, PAIR_PATCH_OFFSET] = cp.int32(gpuPatchPtr & cp.int64(0xFFFFFFFF))
                            outputPairs[index, PAIR_PATCH_OFFSET + 1] = cp.int32((gpuPatchPtr >> cp.int64(32)) & cp.int64(0xFFFFFFFF))
                            outputPairs[index, PAIR_CONTACT_OFFSET] = cp.int32(gpuContactPtr & cp.int64(0xFFFFFFFF))
                            outputPairs[index, PAIR_CONTACT_OFFSET + 1] = cp.int32((gpuContactPtr >> cp.int64(32)) & cp.int64(0xFFFFFFFF))
                            outputPairs[index, PAIR_FORCE_OFFSET] = cp.int32(gpuForcePtr & cp.int64(0xFFFFFFFF))
                            outputPairs[index, PAIR_FORCE_OFFSET + 1] = cp.int32((gpuForcePtr >> cp.int64(32)) & cp.int64(0xFFFFFFFF))
                            outputPairs[index, PAIR_FRICTION_OFFSET] = cp.int32(gpuFrictionPtr & cp.int64(0xFFFFFFFF))
                            outputPairs[index, PAIR_FRICTION_OFFSET + 1] = cp.int32((gpuFrictionPtr >> cp.int64(32)) & cp.int64(0xFFFFFFFF))

                            outputPairs[index, PAIR_TRANSFORM_REF0] = transformRef0
                            outputPairs[index, PAIR_TRANSFORM_REF1] = transformRef1

                            outputPairs[index, PAIR_NODE_INDEX0_LO] = cp.int32(nodeIndex0 & cp.int64(0xFFFFFFFF))
                            outputPairs[index, PAIR_NODE_INDEX0_HI] = cp.int32((nodeIndex0 >> cp.int64(32)) & cp.int64(0xFFFFFFFF))
                            outputPairs[index, PAIR_NODE_INDEX1_LO] = cp.int32(nodeIndex1 & cp.int64(0xFFFFFFFF))
                            outputPairs[index, PAIR_NODE_INDEX1_HI] = cp.int32((nodeIndex1 >> cp.int64(32)) & cp.int64(0xFFFFFFFF))

                            outputPairs[index, PAIR_ACTOR0_LO] = cp.int32(actor0 & cp.int64(0xFFFFFFFF))
                            outputPairs[index, PAIR_ACTOR0_HI] = cp.int32((actor0 >> cp.int64(32)) & cp.int64(0xFFFFFFFF))
                            outputPairs[index, PAIR_ACTOR1_LO] = cp.int32(actor1 & cp.int64(0xFFFFFFFF))
                            outputPairs[index, PAIR_ACTOR1_HI] = cp.int32((actor1 >> cp.int64(32)) & cp.int64(0xFFFFFFFF))

                            # Pack nbContacts (u16) and nbPatches (u16) into one int32
                            outputPairs[index, PAIR_NB_CONTACTS_PATCHES] = nbContacts | (nbPatches << cp.int32(16))

            block.barrier()

            _iter = _iter + cp.int32(1)


# ===== Kernel 3: updateFrictionPatches =====
# Simple per-thread kernel: compute friction patch byte offset from contact patch byte offset.
# Original CUDA:
#   output.frictionPatches = startFrictionPatches +
#       (output.contactPatches - startContactPatches) * sizeof(PxFrictionPatch) / sizeof(PxContactPatch)
# With index-based ABI: contactPatches and frictionPatches stored as byte offsets.
# Converts patch byte offset to patch index, multiplies by friction patch size.
@cp.kernel
def updateFrictionPatches(
    outputs,              # int32[N, CMO_SIZE]
    pairCount,            # int32 scalar
    BLOCK_SIZE: cp.constexpr = 256,
    CMO_SIZE: cp.constexpr = 12,
    CMO_PATCH_OFFSET: cp.constexpr = 0,
    CMO_FRICTION_PATCH_OFFSET: cp.constexpr = 3,
    SIZEOF_CONTACT_PATCH: cp.constexpr = 64,
    SIZEOF_FRICTION_PATCH: cp.constexpr = 52,
):
    with cp.Kernel(cp.ceildiv(pairCount, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            threadIndex = bx * BLOCK_SIZE + tid
            if threadIndex < pairCount:
                patchOffset = outputs[threadIndex, CMO_PATCH_OFFSET] + cp.int32(0)
                frictionOffset = (patchOffset // cp.int32(SIZEOF_CONTACT_PATCH)) * cp.int32(SIZEOF_FRICTION_PATCH)
                outputs[threadIndex, CMO_FRICTION_PATCH_OFFSET] = frictionOffset
