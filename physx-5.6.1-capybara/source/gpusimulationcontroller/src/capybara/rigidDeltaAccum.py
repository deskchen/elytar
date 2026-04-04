"""Capybara DSL port of rigidDeltaAccum.cu -- all 5 kernels.

Ported kernels:
  1. accumulateDeltaVRigidFirstLaunch
  2. accumulateDeltaVRigidSecondLaunch
  3. clearDeltaVRigidSecondLaunchMulti
  4. accumulateDeltaVRigidSecondLaunchMultiStage1
  5. accumulateDeltaVRigidSecondLaunchMultiStage2

ABI differences from CUDA:
  - PxU64 sortedRigidIds stored as int32[N, 2] (lo=mID, hi=mLinkID) since PxNodeIndex
    is a union { struct { PxU32 mID; PxU32 mLinkID; }; PxU64 mInd; }.
    Comparisons done on both lo and hi components (nested ifs).
  - PxU64 blockRigidId stored as int32[GRID_SIZE, 2] (lo, hi).
  - float4 deltaV stored as float32[2*N, 4] where [0..N) = lin, [N..2N) = ang.
  - float4 blockDeltaV stored as float32[2*GRID_SIZE, 4].
  - numContacts as int32[1] tensor (pointer to single PxU32).
  - PxgPrePrepDesc decomposed: solverBodyIndices as int32[M] tensor.
  - PxgSolverCoreDesc + PxgSolverSharedDesc decomposed: solverBodyDeltaVel as
    float32[2*S, 4] tensor (host pre-computes base = velPool + accumulatedBodyDeltaVOffset),
    numSolverBodies as int scalar.
  - PxgArticulationCoreDesc decomposed:
      artiStateDirty: int32[numArtiBlocks, 8]
        -- mStateDirty is PxU8[32], packed as 8 int32s (4 bytes each).
      artiLinkScratchImpulseTop: float32[totalLinkSlots, 32, 4]
        -- mScratchImpulse.mTopxyz_bx per link slot. [linkSlot, artiIdx, xyzw].
      artiLinkScratchImpulseByz: float32[totalLinkSlots, 32, 2]
        -- mScratchImpulse.mbyz per link slot. [linkSlot, artiIdx, yz].
      artiLinkDeltaScale: float32[totalLinkSlots, 32]
        -- mDeltaScale per link slot. [linkSlot, artiIdx].
      maxLinks: int scalar.
  - isTGS: int32 scalar (bool as 0/1).
  - useLocalRelax: int32 scalar (bool as 0/1).
  - globalRelaxationCoefficient: float32 scalar.
  - tempDenom: float32[S] tensor.

Capybara structural notes:
  - No `elif` -- use `if` only (MergeFlatIfToSwitchPass crashes).
  - Pre-declare ALL variables before branches.
  - No `(A > B) & (C < D)` -- use nested ifs.
  - `+ cp.int32(0)` / `+ cp.float32(0.0)` for force-loads.
  - `thread.bitcast()` for int/float conversion.
  - `thread.shfl_idx` for indexed shuffle.
  - `thread.atomic_add(tensor[idx], value)` for atomics.
  - Boolean flags as cp.int32(0/1).
  - No tuple assignments.
  - No Python `/` on int32.
  - Warp shuffle of float4 decomposed to 4 separate float shuffles.
  - PxU64 comparisons decomposed: two int32 equal checks via nested ifs.
  - Capybara limitation: no int64 shuffle support; use int32[N,2] for PxU64 and
    shared memory as two int32 arrays (lo, hi).
  - Per-thread register state does not persist across block.barrier() in Capybara,
    so warp-reduced per-thread values are stored in shared memory arrays of
    BLOCK_SIZE to be re-read after barriers.
"""

import capybara as cp

WARP_SIZE = 32

# Sentinel value for invalid rigidId: 0x8fffffffffffffff
# lo = 0xFFFFFFFF = -1 as int32, hi = 0x8FFFFFFF = -1879048193 as int32
SENTINEL_LO = -1            # 0xFFFFFFFF
SENTINEL_HI = -1879048193   # 0x8FFFFFFF

# PxNodeIndex bit layout (from PxNodeIndex.h):
# mID = lo 32 bits; isStaticBody = (mID == 0xFFFFFFFF)
# mLinkID = hi 32 bits; isArticulation = mLinkID & 1; articulationLinkId = mLinkID >> 1
INVALID_NODE = -1  # 0xFFFFFFFF as int32

# PxgArtiStateDirtyFlag::eHAS_IMPULSES = 1 << 1 = 2
eHAS_IMPULSES = 2


# ===== Kernel 1: accumulateDeltaVRigidFirstLaunch =====
@cp.kernel
def accumulateDeltaVRigidFirstLaunch(
    sortedRigidIds,    # int32[N, 2] -- (lo, hi) per entry
    numContacts,       # int32[1]
    deltaV,            # float32[2*N, 4] -- lin at [0..N), ang at [N..2N)
    blockDeltaV,       # float32[2*GRID_SIZE, 4] -- output
    blockRigidId,      # int32[GRID_SIZE, 2] -- output (lo, hi)
    BLOCK_SIZE: cp.constexpr = 512,
    GRID_SIZE: cp.constexpr = 32,
):
    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        NUM_WARPS = BLOCK_SIZE // WARP_SIZE

        # Shared memory for rigidIds per thread
        sRigidIdLo = block.alloc((BLOCK_SIZE + 1,), dtype=cp.int32)
        sRigidIdHi = block.alloc((BLOCK_SIZE + 1,), dtype=cp.int32)

        # Per-warp accumulators (last thread of each warp)
        sLinWarpAccX = block.alloc((WARP_SIZE,), dtype=cp.float32)
        sLinWarpAccY = block.alloc((WARP_SIZE,), dtype=cp.float32)
        sLinWarpAccZ = block.alloc((WARP_SIZE,), dtype=cp.float32)
        sLinWarpAccW = block.alloc((WARP_SIZE,), dtype=cp.float32)
        sAngWarpAccX = block.alloc((WARP_SIZE,), dtype=cp.float32)
        sAngWarpAccY = block.alloc((WARP_SIZE,), dtype=cp.float32)
        sAngWarpAccZ = block.alloc((WARP_SIZE,), dtype=cp.float32)
        sAngWarpAccW = block.alloc((WARP_SIZE,), dtype=cp.float32)
        sWarpRigidIdLo = block.alloc((WARP_SIZE,), dtype=cp.int32)
        sWarpRigidIdHi = block.alloc((WARP_SIZE,), dtype=cp.int32)

        # Block-level accumulators (scalar)
        sLinBlockAccX = block.alloc((1,), dtype=cp.float32)
        sLinBlockAccY = block.alloc((1,), dtype=cp.float32)
        sLinBlockAccZ = block.alloc((1,), dtype=cp.float32)
        sLinBlockAccW = block.alloc((1,), dtype=cp.float32)
        sAngBlockAccX = block.alloc((1,), dtype=cp.float32)
        sAngBlockAccY = block.alloc((1,), dtype=cp.float32)
        sAngBlockAccZ = block.alloc((1,), dtype=cp.float32)
        sAngBlockAccW = block.alloc((1,), dtype=cp.float32)
        sBlockRigidIdLo = block.alloc((1,), dtype=cp.int32)
        sBlockRigidIdHi = block.alloc((1,), dtype=cp.int32)

        # Per-thread storage for warp-reduced values (persisted across barriers)
        sRedLinX = block.alloc((BLOCK_SIZE,), dtype=cp.float32)
        sRedLinY = block.alloc((BLOCK_SIZE,), dtype=cp.float32)
        sRedLinZ = block.alloc((BLOCK_SIZE,), dtype=cp.float32)
        sRedLinW = block.alloc((BLOCK_SIZE,), dtype=cp.float32)
        sRedAngX = block.alloc((BLOCK_SIZE,), dtype=cp.float32)
        sRedAngY = block.alloc((BLOCK_SIZE,), dtype=cp.float32)
        sRedAngZ = block.alloc((BLOCK_SIZE,), dtype=cp.float32)
        sRedAngW = block.alloc((BLOCK_SIZE,), dtype=cp.float32)

        # Previous block accumulator storage
        sPrevLinBlockAccX = block.alloc((1,), dtype=cp.float32)
        sPrevLinBlockAccY = block.alloc((1,), dtype=cp.float32)
        sPrevLinBlockAccZ = block.alloc((1,), dtype=cp.float32)
        sPrevLinBlockAccW = block.alloc((1,), dtype=cp.float32)
        sPrevAngBlockAccX = block.alloc((1,), dtype=cp.float32)
        sPrevAngBlockAccY = block.alloc((1,), dtype=cp.float32)
        sPrevAngBlockAccZ = block.alloc((1,), dtype=cp.float32)
        sPrevAngBlockAccW = block.alloc((1,), dtype=cp.float32)
        sPrevBlockRigidIdLo = block.alloc((1,), dtype=cp.int32)
        sPrevBlockRigidIdHi = block.alloc((1,), dtype=cp.int32)

        # Initialize block accumulators
        for tid, thread in block.threads():
            if tid == cp.int32(0):
                sLinBlockAccX[0] = cp.disjoint(cp.float32(0.0))
                sLinBlockAccY[0] = cp.disjoint(cp.float32(0.0))
                sLinBlockAccZ[0] = cp.disjoint(cp.float32(0.0))
                sLinBlockAccW[0] = cp.disjoint(cp.float32(0.0))
                sAngBlockAccX[0] = cp.disjoint(cp.float32(0.0))
                sAngBlockAccY[0] = cp.disjoint(cp.float32(0.0))
                sAngBlockAccZ[0] = cp.disjoint(cp.float32(0.0))
                sAngBlockAccW[0] = cp.disjoint(cp.float32(0.0))
                sBlockRigidIdLo[0] = cp.disjoint(cp.int32(SENTINEL_LO))
                sBlockRigidIdHi[0] = cp.disjoint(cp.int32(SENTINEL_HI))

        block.barrier()

        tNumContacts = numContacts[0] + cp.int32(0)
        nbBlocksRequired = (tNumContacts + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        nbIterationsPerBlock = (nbBlocksRequired + cp.int32(GRID_SIZE - 1)) // cp.int32(GRID_SIZE)

        for _iter in range(4096):
            if _iter < nbIterationsPerBlock:

                # Phase 1: Load rigidIds and deltaV, store rigidIds in shared memory
                for tid, thread in block.threads():
                    workIndex = cp.int32(BLOCK_SIZE) * (bx * nbIterationsPerBlock + cp.int32(_iter)) + tid

                    rLo = cp.int32(SENTINEL_LO)
                    rHi = cp.int32(SENTINEL_HI)
                    lx = cp.float32(0.0)
                    ly = cp.float32(0.0)
                    lz = cp.float32(0.0)
                    lw = cp.float32(0.0)
                    ax = cp.float32(0.0)
                    ay = cp.float32(0.0)
                    az = cp.float32(0.0)
                    aw = cp.float32(0.0)

                    if workIndex < tNumContacts:
                        rLo = sortedRigidIds[workIndex, 0] + cp.int32(0)
                        rHi = sortedRigidIds[workIndex, 1] + cp.int32(0)
                        sRigidIdLo[tid] = cp.disjoint(rLo)
                        sRigidIdHi[tid] = cp.disjoint(rHi)
                        lx = deltaV[workIndex, 0] + cp.float32(0.0)
                        ly = deltaV[workIndex, 1] + cp.float32(0.0)
                        lz = deltaV[workIndex, 2] + cp.float32(0.0)
                        lw = deltaV[workIndex, 3] + cp.float32(0.0)
                        ax = deltaV[workIndex + tNumContacts, 0] + cp.float32(0.0)
                        ay = deltaV[workIndex + tNumContacts, 1] + cp.float32(0.0)
                        az = deltaV[workIndex + tNumContacts, 2] + cp.float32(0.0)
                        aw = deltaV[workIndex + tNumContacts, 3] + cp.float32(0.0)

                    # Store initial values in per-thread shared memory
                    sRedLinX[tid] = cp.disjoint(lx)
                    sRedLinY[tid] = cp.disjoint(ly)
                    sRedLinZ[tid] = cp.disjoint(lz)
                    sRedLinW[tid] = cp.disjoint(lw)
                    sRedAngX[tid] = cp.disjoint(ax)
                    sRedAngY[tid] = cp.disjoint(ay)
                    sRedAngZ[tid] = cp.disjoint(az)
                    sRedAngW[tid] = cp.disjoint(aw)

                block.barrier()

                # Phase 2: Warp reduction with 5 steps (1, 2, 4, 8, 16)
                # After this, each thread has accumulated values from prior threads
                # with matching rigidId in its warp. Last warp thread writes to
                # sWarpAccumulator. All threads write their reduced values to sRed*.
                for tid, thread in block.threads():
                    threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                    warpIndex = tid >> cp.int32(5)

                    workIndex = cp.int32(BLOCK_SIZE) * (bx * nbIterationsPerBlock + cp.int32(_iter)) + tid

                    rigidIdLo = sRigidIdLo[tid] + cp.int32(0)
                    rigidIdHi = sRigidIdHi[tid] + cp.int32(0)

                    linDvX = sRedLinX[tid] + cp.float32(0.0)
                    linDvY = sRedLinY[tid] + cp.float32(0.0)
                    linDvZ = sRedLinZ[tid] + cp.float32(0.0)
                    linDvW = sRedLinW[tid] + cp.float32(0.0)
                    angDvX = sRedAngX[tid] + cp.float32(0.0)
                    angDvY = sRedAngY[tid] + cp.float32(0.0)
                    angDvZ = sRedAngZ[tid] + cp.float32(0.0)
                    angDvW = sRedAngW[tid] + cp.float32(0.0)

                    # reductionRadius = 1
                    srcLane = threadIndexInWarp - cp.int32(1)
                    safeLane = srcLane if srcLane >= cp.int32(0) else cp.int32(0)
                    oLx = thread.shfl_idx(linDvX, safeLane)
                    oLy = thread.shfl_idx(linDvY, safeLane)
                    oLz = thread.shfl_idx(linDvZ, safeLane)
                    oLw = thread.shfl_idx(linDvW, safeLane)
                    oAx = thread.shfl_idx(angDvX, safeLane)
                    oAy = thread.shfl_idx(angDvY, safeLane)
                    oAz = thread.shfl_idx(angDvZ, safeLane)
                    oAw = thread.shfl_idx(angDvW, safeLane)
                    if threadIndexInWarp >= cp.int32(1):
                        if workIndex < tNumContacts:
                            sIdx = cp.int32(WARP_SIZE) * warpIndex + srcLane
                            oIdLo = sRigidIdLo[sIdx] + cp.int32(0)
                            oIdHi = sRigidIdHi[sIdx] + cp.int32(0)
                            if rigidIdLo == oIdLo:
                                if rigidIdHi == oIdHi:
                                    linDvX = linDvX + oLx
                                    linDvY = linDvY + oLy
                                    linDvZ = linDvZ + oLz
                                    linDvW = linDvW + oLw
                                    angDvX = angDvX + oAx
                                    angDvY = angDvY + oAy
                                    angDvZ = angDvZ + oAz
                                    angDvW = angDvW + oAw

                    # reductionRadius = 2
                    srcLane = threadIndexInWarp - cp.int32(2)
                    safeLane = srcLane if srcLane >= cp.int32(0) else cp.int32(0)
                    oLx = thread.shfl_idx(linDvX, safeLane)
                    oLy = thread.shfl_idx(linDvY, safeLane)
                    oLz = thread.shfl_idx(linDvZ, safeLane)
                    oLw = thread.shfl_idx(linDvW, safeLane)
                    oAx = thread.shfl_idx(angDvX, safeLane)
                    oAy = thread.shfl_idx(angDvY, safeLane)
                    oAz = thread.shfl_idx(angDvZ, safeLane)
                    oAw = thread.shfl_idx(angDvW, safeLane)
                    if threadIndexInWarp >= cp.int32(2):
                        if workIndex < tNumContacts:
                            sIdx = cp.int32(WARP_SIZE) * warpIndex + srcLane
                            oIdLo = sRigidIdLo[sIdx] + cp.int32(0)
                            oIdHi = sRigidIdHi[sIdx] + cp.int32(0)
                            if rigidIdLo == oIdLo:
                                if rigidIdHi == oIdHi:
                                    linDvX = linDvX + oLx
                                    linDvY = linDvY + oLy
                                    linDvZ = linDvZ + oLz
                                    linDvW = linDvW + oLw
                                    angDvX = angDvX + oAx
                                    angDvY = angDvY + oAy
                                    angDvZ = angDvZ + oAz
                                    angDvW = angDvW + oAw

                    # reductionRadius = 4
                    srcLane = threadIndexInWarp - cp.int32(4)
                    safeLane = srcLane if srcLane >= cp.int32(0) else cp.int32(0)
                    oLx = thread.shfl_idx(linDvX, safeLane)
                    oLy = thread.shfl_idx(linDvY, safeLane)
                    oLz = thread.shfl_idx(linDvZ, safeLane)
                    oLw = thread.shfl_idx(linDvW, safeLane)
                    oAx = thread.shfl_idx(angDvX, safeLane)
                    oAy = thread.shfl_idx(angDvY, safeLane)
                    oAz = thread.shfl_idx(angDvZ, safeLane)
                    oAw = thread.shfl_idx(angDvW, safeLane)
                    if threadIndexInWarp >= cp.int32(4):
                        if workIndex < tNumContacts:
                            sIdx = cp.int32(WARP_SIZE) * warpIndex + srcLane
                            oIdLo = sRigidIdLo[sIdx] + cp.int32(0)
                            oIdHi = sRigidIdHi[sIdx] + cp.int32(0)
                            if rigidIdLo == oIdLo:
                                if rigidIdHi == oIdHi:
                                    linDvX = linDvX + oLx
                                    linDvY = linDvY + oLy
                                    linDvZ = linDvZ + oLz
                                    linDvW = linDvW + oLw
                                    angDvX = angDvX + oAx
                                    angDvY = angDvY + oAy
                                    angDvZ = angDvZ + oAz
                                    angDvW = angDvW + oAw

                    # reductionRadius = 8
                    srcLane = threadIndexInWarp - cp.int32(8)
                    safeLane = srcLane if srcLane >= cp.int32(0) else cp.int32(0)
                    oLx = thread.shfl_idx(linDvX, safeLane)
                    oLy = thread.shfl_idx(linDvY, safeLane)
                    oLz = thread.shfl_idx(linDvZ, safeLane)
                    oLw = thread.shfl_idx(linDvW, safeLane)
                    oAx = thread.shfl_idx(angDvX, safeLane)
                    oAy = thread.shfl_idx(angDvY, safeLane)
                    oAz = thread.shfl_idx(angDvZ, safeLane)
                    oAw = thread.shfl_idx(angDvW, safeLane)
                    if threadIndexInWarp >= cp.int32(8):
                        if workIndex < tNumContacts:
                            sIdx = cp.int32(WARP_SIZE) * warpIndex + srcLane
                            oIdLo = sRigidIdLo[sIdx] + cp.int32(0)
                            oIdHi = sRigidIdHi[sIdx] + cp.int32(0)
                            if rigidIdLo == oIdLo:
                                if rigidIdHi == oIdHi:
                                    linDvX = linDvX + oLx
                                    linDvY = linDvY + oLy
                                    linDvZ = linDvZ + oLz
                                    linDvW = linDvW + oLw
                                    angDvX = angDvX + oAx
                                    angDvY = angDvY + oAy
                                    angDvZ = angDvZ + oAz
                                    angDvW = angDvW + oAw

                    # reductionRadius = 16
                    srcLane = threadIndexInWarp - cp.int32(16)
                    safeLane = srcLane if srcLane >= cp.int32(0) else cp.int32(0)
                    oLx = thread.shfl_idx(linDvX, safeLane)
                    oLy = thread.shfl_idx(linDvY, safeLane)
                    oLz = thread.shfl_idx(linDvZ, safeLane)
                    oLw = thread.shfl_idx(linDvW, safeLane)
                    oAx = thread.shfl_idx(angDvX, safeLane)
                    oAy = thread.shfl_idx(angDvY, safeLane)
                    oAz = thread.shfl_idx(angDvZ, safeLane)
                    oAw = thread.shfl_idx(angDvW, safeLane)
                    if threadIndexInWarp >= cp.int32(16):
                        if workIndex < tNumContacts:
                            sIdx = cp.int32(WARP_SIZE) * warpIndex + srcLane
                            oIdLo = sRigidIdLo[sIdx] + cp.int32(0)
                            oIdHi = sRigidIdHi[sIdx] + cp.int32(0)
                            if rigidIdLo == oIdLo:
                                if rigidIdHi == oIdHi:
                                    linDvX = linDvX + oLx
                                    linDvY = linDvY + oLy
                                    linDvZ = linDvZ + oLz
                                    linDvW = linDvW + oLw
                                    angDvX = angDvX + oAx
                                    angDvY = angDvY + oAy
                                    angDvZ = angDvZ + oAz
                                    angDvW = angDvW + oAw

                    # Store per-thread warp-reduced values for later writeback
                    sRedLinX[tid] = cp.disjoint(linDvX)
                    sRedLinY[tid] = cp.disjoint(linDvY)
                    sRedLinZ[tid] = cp.disjoint(linDvZ)
                    sRedLinW[tid] = cp.disjoint(linDvW)
                    sRedAngX[tid] = cp.disjoint(angDvX)
                    sRedAngY[tid] = cp.disjoint(angDvY)
                    sRedAngZ[tid] = cp.disjoint(angDvZ)
                    sRedAngW[tid] = cp.disjoint(angDvW)

                    # Last thread in warp writes to warp accumulator
                    if threadIndexInWarp == cp.int32(WARP_SIZE - 1):
                        sLinWarpAccX[warpIndex] = cp.disjoint(linDvX)
                        sLinWarpAccY[warpIndex] = cp.disjoint(linDvY)
                        sLinWarpAccZ[warpIndex] = cp.disjoint(linDvZ)
                        sLinWarpAccW[warpIndex] = cp.disjoint(linDvW)
                        sAngWarpAccX[warpIndex] = cp.disjoint(angDvX)
                        sAngWarpAccY[warpIndex] = cp.disjoint(angDvY)
                        sAngWarpAccZ[warpIndex] = cp.disjoint(angDvZ)
                        sAngWarpAccW[warpIndex] = cp.disjoint(angDvW)
                        sWarpRigidIdLo[warpIndex] = cp.disjoint(rigidIdLo)
                        sWarpRigidIdHi[warpIndex] = cp.disjoint(rigidIdHi)

                # Save previous block accumulators and sync
                for tid, thread in block.threads():
                    if tid == cp.int32(0):
                        sPrevLinBlockAccX[0] = cp.disjoint(sLinBlockAccX[0] + cp.float32(0.0))
                        sPrevLinBlockAccY[0] = cp.disjoint(sLinBlockAccY[0] + cp.float32(0.0))
                        sPrevLinBlockAccZ[0] = cp.disjoint(sLinBlockAccZ[0] + cp.float32(0.0))
                        sPrevLinBlockAccW[0] = cp.disjoint(sLinBlockAccW[0] + cp.float32(0.0))
                        sPrevAngBlockAccX[0] = cp.disjoint(sAngBlockAccX[0] + cp.float32(0.0))
                        sPrevAngBlockAccY[0] = cp.disjoint(sAngBlockAccY[0] + cp.float32(0.0))
                        sPrevAngBlockAccZ[0] = cp.disjoint(sAngBlockAccZ[0] + cp.float32(0.0))
                        sPrevAngBlockAccW[0] = cp.disjoint(sAngBlockAccW[0] + cp.float32(0.0))
                        sPrevBlockRigidIdLo[0] = cp.disjoint(sBlockRigidIdLo[0] + cp.int32(0))
                        sPrevBlockRigidIdHi[0] = cp.disjoint(sBlockRigidIdHi[0] + cp.int32(0))

                block.barrier()

                # Phase 3: Cross-warp reduction (warp 0 only)
                for tid, thread in block.threads():
                    threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                    warpIndex = tid >> cp.int32(5)

                    if cp.assume_uniform(warpIndex == cp.int32(0)):
                        wLinX = cp.float32(0.0)
                        wLinY = cp.float32(0.0)
                        wLinZ = cp.float32(0.0)
                        wLinW = cp.float32(0.0)
                        wAngX = cp.float32(0.0)
                        wAngY = cp.float32(0.0)
                        wAngZ = cp.float32(0.0)
                        wAngW = cp.float32(0.0)
                        wRigidIdLo = cp.int32(SENTINEL_LO)
                        wRigidIdHi = cp.int32(SENTINEL_HI)

                        if threadIndexInWarp < cp.int32(NUM_WARPS):
                            wLinX = sLinWarpAccX[threadIndexInWarp] + cp.float32(0.0)
                            wLinY = sLinWarpAccY[threadIndexInWarp] + cp.float32(0.0)
                            wLinZ = sLinWarpAccZ[threadIndexInWarp] + cp.float32(0.0)
                            wLinW = sLinWarpAccW[threadIndexInWarp] + cp.float32(0.0)
                            wAngX = sAngWarpAccX[threadIndexInWarp] + cp.float32(0.0)
                            wAngY = sAngWarpAccY[threadIndexInWarp] + cp.float32(0.0)
                            wAngZ = sAngWarpAccZ[threadIndexInWarp] + cp.float32(0.0)
                            wAngW = sAngWarpAccW[threadIndexInWarp] + cp.float32(0.0)
                            wRigidIdLo = sWarpRigidIdLo[threadIndexInWarp] + cp.int32(0)
                            wRigidIdHi = sWarpRigidIdHi[threadIndexInWarp] + cp.int32(0)

                        tLx = wLinX
                        tLy = wLinY
                        tLz = wLinZ
                        tLw = wLinW
                        tAx = wAngX
                        tAy = wAngY
                        tAz = wAngZ
                        tAw = wAngW

                        # Cross-warp reduction: numWarpsPerBlock = 16, so radii 1,2,4,8
                        # reductionRadius = 1
                        sl = threadIndexInWarp - cp.int32(1)
                        safe = sl if sl >= cp.int32(0) else cp.int32(0)
                        oLx = thread.shfl_idx(tLx, safe)
                        oLy = thread.shfl_idx(tLy, safe)
                        oLz = thread.shfl_idx(tLz, safe)
                        oLw = thread.shfl_idx(tLw, safe)
                        oAx = thread.shfl_idx(tAx, safe)
                        oAy = thread.shfl_idx(tAy, safe)
                        oAz = thread.shfl_idx(tAz, safe)
                        oAw = thread.shfl_idx(tAw, safe)
                        if threadIndexInWarp >= cp.int32(1):
                            oIdLo = sWarpRigidIdLo[sl] + cp.int32(0)
                            oIdHi = sWarpRigidIdHi[sl] + cp.int32(0)
                            if wRigidIdLo == oIdLo:
                                if wRigidIdHi == oIdHi:
                                    tLx = tLx + oLx
                                    tLy = tLy + oLy
                                    tLz = tLz + oLz
                                    tLw = tLw + oLw
                                    tAx = tAx + oAx
                                    tAy = tAy + oAy
                                    tAz = tAz + oAz
                                    tAw = tAw + oAw

                        # reductionRadius = 2
                        sl = threadIndexInWarp - cp.int32(2)
                        safe = sl if sl >= cp.int32(0) else cp.int32(0)
                        oLx = thread.shfl_idx(tLx, safe)
                        oLy = thread.shfl_idx(tLy, safe)
                        oLz = thread.shfl_idx(tLz, safe)
                        oLw = thread.shfl_idx(tLw, safe)
                        oAx = thread.shfl_idx(tAx, safe)
                        oAy = thread.shfl_idx(tAy, safe)
                        oAz = thread.shfl_idx(tAz, safe)
                        oAw = thread.shfl_idx(tAw, safe)
                        if threadIndexInWarp >= cp.int32(2):
                            oIdLo = sWarpRigidIdLo[sl] + cp.int32(0)
                            oIdHi = sWarpRigidIdHi[sl] + cp.int32(0)
                            if wRigidIdLo == oIdLo:
                                if wRigidIdHi == oIdHi:
                                    tLx = tLx + oLx
                                    tLy = tLy + oLy
                                    tLz = tLz + oLz
                                    tLw = tLw + oLw
                                    tAx = tAx + oAx
                                    tAy = tAy + oAy
                                    tAz = tAz + oAz
                                    tAw = tAw + oAw

                        # reductionRadius = 4
                        sl = threadIndexInWarp - cp.int32(4)
                        safe = sl if sl >= cp.int32(0) else cp.int32(0)
                        oLx = thread.shfl_idx(tLx, safe)
                        oLy = thread.shfl_idx(tLy, safe)
                        oLz = thread.shfl_idx(tLz, safe)
                        oLw = thread.shfl_idx(tLw, safe)
                        oAx = thread.shfl_idx(tAx, safe)
                        oAy = thread.shfl_idx(tAy, safe)
                        oAz = thread.shfl_idx(tAz, safe)
                        oAw = thread.shfl_idx(tAw, safe)
                        if threadIndexInWarp >= cp.int32(4):
                            oIdLo = sWarpRigidIdLo[sl] + cp.int32(0)
                            oIdHi = sWarpRigidIdHi[sl] + cp.int32(0)
                            if wRigidIdLo == oIdLo:
                                if wRigidIdHi == oIdHi:
                                    tLx = tLx + oLx
                                    tLy = tLy + oLy
                                    tLz = tLz + oLz
                                    tLw = tLw + oLw
                                    tAx = tAx + oAx
                                    tAy = tAy + oAy
                                    tAz = tAz + oAz
                                    tAw = tAw + oAw

                        # reductionRadius = 8
                        sl = threadIndexInWarp - cp.int32(8)
                        safe = sl if sl >= cp.int32(0) else cp.int32(0)
                        oLx = thread.shfl_idx(tLx, safe)
                        oLy = thread.shfl_idx(tLy, safe)
                        oLz = thread.shfl_idx(tLz, safe)
                        oLw = thread.shfl_idx(tLw, safe)
                        oAx = thread.shfl_idx(tAx, safe)
                        oAy = thread.shfl_idx(tAy, safe)
                        oAz = thread.shfl_idx(tAz, safe)
                        oAw = thread.shfl_idx(tAw, safe)
                        if threadIndexInWarp >= cp.int32(8):
                            oIdLo = sWarpRigidIdLo[sl] + cp.int32(0)
                            oIdHi = sWarpRigidIdHi[sl] + cp.int32(0)
                            if wRigidIdLo == oIdLo:
                                if wRigidIdHi == oIdHi:
                                    tLx = tLx + oLx
                                    tLy = tLy + oLy
                                    tLz = tLz + oLz
                                    tLw = tLw + oLw
                                    tAx = tAx + oAx
                                    tAy = tAy + oAy
                                    tAz = tAz + oAz
                                    tAw = tAw + oAw

                        # Last warp thread (NUM_WARPS-1) updates block accumulators
                        if threadIndexInWarp == cp.int32(NUM_WARPS - 1):
                            curBlockLo = sBlockRigidIdLo[0] + cp.int32(0)
                            curBlockHi = sBlockRigidIdHi[0] + cp.int32(0)
                            matchBlock = cp.int32(0)
                            if curBlockLo == wRigidIdLo:
                                if curBlockHi == wRigidIdHi:
                                    matchBlock = cp.int32(1)
                            if matchBlock == cp.int32(0):
                                sLinBlockAccX[0] = cp.disjoint(cp.float32(0.0))
                                sLinBlockAccY[0] = cp.disjoint(cp.float32(0.0))
                                sLinBlockAccZ[0] = cp.disjoint(cp.float32(0.0))
                                sLinBlockAccW[0] = cp.disjoint(cp.float32(0.0))
                                sAngBlockAccX[0] = cp.disjoint(cp.float32(0.0))
                                sAngBlockAccY[0] = cp.disjoint(cp.float32(0.0))
                                sAngBlockAccZ[0] = cp.disjoint(cp.float32(0.0))
                                sAngBlockAccW[0] = cp.disjoint(cp.float32(0.0))
                            oldLx = sLinBlockAccX[0] + cp.float32(0.0)
                            oldLy = sLinBlockAccY[0] + cp.float32(0.0)
                            oldLz = sLinBlockAccZ[0] + cp.float32(0.0)
                            oldLw = sLinBlockAccW[0] + cp.float32(0.0)
                            oldAx = sAngBlockAccX[0] + cp.float32(0.0)
                            oldAy = sAngBlockAccY[0] + cp.float32(0.0)
                            oldAz = sAngBlockAccZ[0] + cp.float32(0.0)
                            oldAw = sAngBlockAccW[0] + cp.float32(0.0)
                            sLinBlockAccX[0] = cp.disjoint(oldLx + tLx)
                            sLinBlockAccY[0] = cp.disjoint(oldLy + tLy)
                            sLinBlockAccZ[0] = cp.disjoint(oldLz + tLz)
                            sLinBlockAccW[0] = cp.disjoint(oldLw + tLw)
                            sAngBlockAccX[0] = cp.disjoint(oldAx + tAx)
                            sAngBlockAccY[0] = cp.disjoint(oldAy + tAy)
                            sAngBlockAccZ[0] = cp.disjoint(oldAz + tAz)
                            sAngBlockAccW[0] = cp.disjoint(oldAw + tAw)
                            sBlockRigidIdLo[0] = cp.disjoint(wRigidIdLo)
                            sBlockRigidIdHi[0] = cp.disjoint(wRigidIdHi)

                        # Write cross-warp reduced values back to sWarpAcc for readback
                        sLinWarpAccX[threadIndexInWarp] = cp.disjoint(tLx)
                        sLinWarpAccY[threadIndexInWarp] = cp.disjoint(tLy)
                        sLinWarpAccZ[threadIndexInWarp] = cp.disjoint(tLz)
                        sLinWarpAccW[threadIndexInWarp] = cp.disjoint(tLw)
                        sAngWarpAccX[threadIndexInWarp] = cp.disjoint(tAx)
                        sAngWarpAccY[threadIndexInWarp] = cp.disjoint(tAy)
                        sAngWarpAccZ[threadIndexInWarp] = cp.disjoint(tAz)
                        sAngWarpAccW[threadIndexInWarp] = cp.disjoint(tAw)

                block.barrier()

                # Phase 4: Writeback -- combine warp-reduced value + cross-warp + cross-block
                for tid, thread in block.threads():
                    threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)
                    warpIndex = tid >> cp.int32(5)

                    workIndex = cp.int32(BLOCK_SIZE) * (bx * nbIterationsPerBlock + cp.int32(_iter)) + tid

                    if workIndex < tNumContacts:
                        rigidIdLo = sRigidIdLo[tid] + cp.int32(0)
                        rigidIdHi = sRigidIdHi[tid] + cp.int32(0)

                        # Per-thread warp-reduced values
                        linDvX = sRedLinX[tid] + cp.float32(0.0)
                        linDvY = sRedLinY[tid] + cp.float32(0.0)
                        linDvZ = sRedLinZ[tid] + cp.float32(0.0)
                        linDvW = sRedLinW[tid] + cp.float32(0.0)
                        angDvX = sRedAngX[tid] + cp.float32(0.0)
                        angDvY = sRedAngY[tid] + cp.float32(0.0)
                        angDvZ = sRedAngZ[tid] + cp.float32(0.0)
                        angDvW = sRedAngW[tid] + cp.float32(0.0)

                        accumLinX = cp.float32(0.0)
                        accumLinY = cp.float32(0.0)
                        accumLinZ = cp.float32(0.0)
                        accumLinW = cp.float32(0.0)
                        accumAngX = cp.float32(0.0)
                        accumAngY = cp.float32(0.0)
                        accumAngZ = cp.float32(0.0)
                        accumAngW = cp.float32(0.0)

                        # Add previous warp's cross-warp-reduced accumulator if rigidId matches
                        if warpIndex > cp.int32(0):
                            prevWarpIdx = warpIndex - cp.int32(1)
                            prevLo = sWarpRigidIdLo[prevWarpIdx] + cp.int32(0)
                            prevHi = sWarpRigidIdHi[prevWarpIdx] + cp.int32(0)
                            if rigidIdLo == prevLo:
                                if rigidIdHi == prevHi:
                                    accumLinX = sLinWarpAccX[prevWarpIdx] + cp.float32(0.0)
                                    accumLinY = sLinWarpAccY[prevWarpIdx] + cp.float32(0.0)
                                    accumLinZ = sLinWarpAccZ[prevWarpIdx] + cp.float32(0.0)
                                    accumLinW = sLinWarpAccW[prevWarpIdx] + cp.float32(0.0)
                                    accumAngX = sAngWarpAccX[prevWarpIdx] + cp.float32(0.0)
                                    accumAngY = sAngWarpAccY[prevWarpIdx] + cp.float32(0.0)
                                    accumAngZ = sAngWarpAccZ[prevWarpIdx] + cp.float32(0.0)
                                    accumAngW = sAngWarpAccW[prevWarpIdx] + cp.float32(0.0)

                        # Add previous iteration's block accumulator if rigidId matches
                        if cp.int32(_iter) != cp.int32(0):
                            prevBLo = sPrevBlockRigidIdLo[0] + cp.int32(0)
                            prevBHi = sPrevBlockRigidIdHi[0] + cp.int32(0)
                            if rigidIdLo == prevBLo:
                                if rigidIdHi == prevBHi:
                                    accumLinX = accumLinX + sPrevLinBlockAccX[0] + cp.float32(0.0)
                                    accumLinY = accumLinY + sPrevLinBlockAccY[0] + cp.float32(0.0)
                                    accumLinZ = accumLinZ + sPrevLinBlockAccZ[0] + cp.float32(0.0)
                                    accumLinW = accumLinW + sPrevLinBlockAccW[0] + cp.float32(0.0)
                                    accumAngX = accumAngX + sPrevAngBlockAccX[0] + cp.float32(0.0)
                                    accumAngY = accumAngY + sPrevAngBlockAccY[0] + cp.float32(0.0)
                                    accumAngZ = accumAngZ + sPrevAngBlockAccZ[0] + cp.float32(0.0)
                                    accumAngW = accumAngW + sPrevAngBlockAccW[0] + cp.float32(0.0)

                        deltaV[workIndex, 0] = linDvX + accumLinX
                        deltaV[workIndex, 1] = linDvY + accumLinY
                        deltaV[workIndex, 2] = linDvZ + accumLinZ
                        deltaV[workIndex, 3] = linDvW + accumLinW
                        deltaV[workIndex + tNumContacts, 0] = angDvX + accumAngX
                        deltaV[workIndex + tNumContacts, 1] = angDvY + accumAngY
                        deltaV[workIndex + tNumContacts, 2] = angDvZ + accumAngZ
                        deltaV[workIndex + tNumContacts, 3] = angDvW + accumAngW

                block.barrier()

        # Final: thread 0 writes block-level accumulators to global memory
        for tid, thread in block.threads():
            if tid == cp.int32(0):
                blockDeltaV[bx, 0] = sLinBlockAccX[0] + cp.float32(0.0)
                blockDeltaV[bx, 1] = sLinBlockAccY[0] + cp.float32(0.0)
                blockDeltaV[bx, 2] = sLinBlockAccZ[0] + cp.float32(0.0)
                blockDeltaV[bx, 3] = sLinBlockAccW[0] + cp.float32(0.0)
                blockDeltaV[bx + cp.int32(GRID_SIZE), 0] = sAngBlockAccX[0] + cp.float32(0.0)
                blockDeltaV[bx + cp.int32(GRID_SIZE), 1] = sAngBlockAccY[0] + cp.float32(0.0)
                blockDeltaV[bx + cp.int32(GRID_SIZE), 2] = sAngBlockAccZ[0] + cp.float32(0.0)
                blockDeltaV[bx + cp.int32(GRID_SIZE), 3] = sAngBlockAccW[0] + cp.float32(0.0)
                blockRigidId[bx, 0] = sBlockRigidIdLo[0] + cp.int32(0)
                blockRigidId[bx, 1] = sBlockRigidIdHi[0] + cp.int32(0)


# ===== Kernel 2: accumulateDeltaVRigidSecondLaunch =====
@cp.kernel
def accumulateDeltaVRigidSecondLaunch(
    sortedRigidIds,          # int32[N, 2]
    numContacts,             # int32[1]
    deltaV,                  # float32[2*N, 4] -- input
    blockDeltaV,             # float32[2*GRID_SIZE, 4] -- input
    blockRigidId,            # int32[GRID_SIZE, 2] -- input
    solverBodyIndices,       # int32[M] -- prePrepDesc->solverBodyIndices
    solverBodyDeltaVel,      # float32[2*S, 4] -- solverBodyVelPool + offset
    numSolverBodies,         # int scalar
    artiStateDirty,          # int32[numArtiBlocks, 8] -- packed PxU8[32]
    artiLinkScratchImpTop,   # float32[totalLinkSlots, 32, 4] -- mScratchImpulse.mTopxyz_bx
    artiLinkScratchImpByz,   # float32[totalLinkSlots, 32, 2] -- mScratchImpulse.mbyz
    maxLinks,                # int scalar
    isTGS,                   # int scalar (0/1)
    BLOCK_SIZE: cp.constexpr = 512,
    GRID_SIZE: cp.constexpr = 32,
):
    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        # Shared memory
        sBlockLinDvX = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockLinDvY = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockLinDvZ = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockLinDvW = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockAngDvX = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockAngDvY = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockAngDvZ = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockAngDvW = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockRigidIdLo = block.alloc((GRID_SIZE,), dtype=cp.int32)
        sBlockRigidIdHi = block.alloc((GRID_SIZE,), dtype=cp.int32)
        sRigidIdLo = block.alloc((BLOCK_SIZE + 1,), dtype=cp.int32)
        sRigidIdHi = block.alloc((BLOCK_SIZE + 1,), dtype=cp.int32)

        tNumContacts = numContacts[0] + cp.int32(0)
        nbBlocksRequired = (tNumContacts + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        nbIterationsPerBlock = (nbBlocksRequired + cp.int32(GRID_SIZE - 1)) // cp.int32(GRID_SIZE)

        # Phase 1: Load block-level deltaV and rigidId, do cross-block reduction
        for tid, thread in block.threads():
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)

            linBdX = cp.float32(0.0)
            linBdY = cp.float32(0.0)
            linBdZ = cp.float32(0.0)
            linBdW = cp.float32(0.0)
            angBdX = cp.float32(0.0)
            angBdY = cp.float32(0.0)
            angBdZ = cp.float32(0.0)
            angBdW = cp.float32(0.0)
            tBlockRigidIdLo = cp.int32(SENTINEL_LO)
            tBlockRigidIdHi = cp.int32(SENTINEL_HI)

            if tid < cp.int32(GRID_SIZE):
                linBdX = blockDeltaV[tid, 0] + cp.float32(0.0)
                linBdY = blockDeltaV[tid, 1] + cp.float32(0.0)
                linBdZ = blockDeltaV[tid, 2] + cp.float32(0.0)
                linBdW = blockDeltaV[tid, 3] + cp.float32(0.0)
                angBdX = blockDeltaV[tid + cp.int32(GRID_SIZE), 0] + cp.float32(0.0)
                angBdY = blockDeltaV[tid + cp.int32(GRID_SIZE), 1] + cp.float32(0.0)
                angBdZ = blockDeltaV[tid + cp.int32(GRID_SIZE), 2] + cp.float32(0.0)
                angBdW = blockDeltaV[tid + cp.int32(GRID_SIZE), 3] + cp.float32(0.0)
                tBlockRigidIdLo = blockRigidId[tid, 0] + cp.int32(0)
                tBlockRigidIdHi = blockRigidId[tid, 1] + cp.int32(0)

                sBlockLinDvX[tid] = cp.disjoint(linBdX)
                sBlockLinDvY[tid] = cp.disjoint(linBdY)
                sBlockLinDvZ[tid] = cp.disjoint(linBdZ)
                sBlockLinDvW[tid] = cp.disjoint(linBdW)
                sBlockAngDvX[tid] = cp.disjoint(angBdX)
                sBlockAngDvY[tid] = cp.disjoint(angBdY)
                sBlockAngDvZ[tid] = cp.disjoint(angBdZ)
                sBlockAngDvW[tid] = cp.disjoint(angBdW)
                sBlockRigidIdLo[tid] = cp.disjoint(tBlockRigidIdLo)
                sBlockRigidIdHi[tid] = cp.disjoint(tBlockRigidIdHi)

        block.barrier()

        # Cross-block reduction on block-level accumulators (GRID_SIZE=32, so radii 1..16)
        # NOTE: shuffles must be unconditional (all warp lanes active).
        # Safe-load pattern: all threads load, but out-of-bounds threads get zeros.
        # Conditional write: only threads with tid < GRID_SIZE write results.
        for tid, thread in block.threads():
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)

            tLx = cp.float32(0.0)
            tLy = cp.float32(0.0)
            tLz = cp.float32(0.0)
            tLw = cp.float32(0.0)
            tAx = cp.float32(0.0)
            tAy = cp.float32(0.0)
            tAz = cp.float32(0.0)
            tAw = cp.float32(0.0)
            tBlockRigidIdLo = cp.int32(SENTINEL_LO)
            tBlockRigidIdHi = cp.int32(SENTINEL_HI)

            # Safe load: all threads read, out-of-bounds get zeros/sentinel
            safeIdx = tid
            if tid >= cp.int32(GRID_SIZE):
                safeIdx = cp.int32(0)
            tLx = sBlockLinDvX[safeIdx] + cp.float32(0.0)
            tLy = sBlockLinDvY[safeIdx] + cp.float32(0.0)
            tLz = sBlockLinDvZ[safeIdx] + cp.float32(0.0)
            tLw = sBlockLinDvW[safeIdx] + cp.float32(0.0)
            tAx = sBlockAngDvX[safeIdx] + cp.float32(0.0)
            tAy = sBlockAngDvY[safeIdx] + cp.float32(0.0)
            tAz = sBlockAngDvZ[safeIdx] + cp.float32(0.0)
            tAw = sBlockAngDvW[safeIdx] + cp.float32(0.0)
            tBlockRigidIdLo = sBlockRigidIdLo[safeIdx] + cp.int32(0)
            tBlockRigidIdHi = sBlockRigidIdHi[safeIdx] + cp.int32(0)
            if tid >= cp.int32(GRID_SIZE):
                tLx = cp.float32(0.0)
                tLy = cp.float32(0.0)
                tLz = cp.float32(0.0)
                tLw = cp.float32(0.0)
                tAx = cp.float32(0.0)
                tAy = cp.float32(0.0)
                tAz = cp.float32(0.0)
                tAw = cp.float32(0.0)
                tBlockRigidIdLo = cp.int32(SENTINEL_LO)
                tBlockRigidIdHi = cp.int32(SENTINEL_HI)

            # reductionRadius = 1
            sl = threadIndexInWarp - cp.int32(1)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oLx = thread.shfl_idx(tLx, safe)
            oLy = thread.shfl_idx(tLy, safe)
            oLz = thread.shfl_idx(tLz, safe)
            oLw = thread.shfl_idx(tLw, safe)
            oAx = thread.shfl_idx(tAx, safe)
            oAy = thread.shfl_idx(tAy, safe)
            oAz = thread.shfl_idx(tAz, safe)
            oAw = thread.shfl_idx(tAw, safe)
            if threadIndexInWarp >= cp.int32(1):
                if tid < cp.int32(GRID_SIZE):
                    oIdLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oIdHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oIdLo:
                        if tBlockRigidIdHi == oIdHi:
                            tLx = tLx + oLx
                            tLy = tLy + oLy
                            tLz = tLz + oLz
                            tLw = tLw + oLw
                            tAx = tAx + oAx
                            tAy = tAy + oAy
                            tAz = tAz + oAz
                            tAw = tAw + oAw

            # reductionRadius = 2
            sl = threadIndexInWarp - cp.int32(2)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oLx = thread.shfl_idx(tLx, safe)
            oLy = thread.shfl_idx(tLy, safe)
            oLz = thread.shfl_idx(tLz, safe)
            oLw = thread.shfl_idx(tLw, safe)
            oAx = thread.shfl_idx(tAx, safe)
            oAy = thread.shfl_idx(tAy, safe)
            oAz = thread.shfl_idx(tAz, safe)
            oAw = thread.shfl_idx(tAw, safe)
            if threadIndexInWarp >= cp.int32(2):
                if tid < cp.int32(GRID_SIZE):
                    oIdLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oIdHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oIdLo:
                        if tBlockRigidIdHi == oIdHi:
                            tLx = tLx + oLx
                            tLy = tLy + oLy
                            tLz = tLz + oLz
                            tLw = tLw + oLw
                            tAx = tAx + oAx
                            tAy = tAy + oAy
                            tAz = tAz + oAz
                            tAw = tAw + oAw

            # reductionRadius = 4
            sl = threadIndexInWarp - cp.int32(4)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oLx = thread.shfl_idx(tLx, safe)
            oLy = thread.shfl_idx(tLy, safe)
            oLz = thread.shfl_idx(tLz, safe)
            oLw = thread.shfl_idx(tLw, safe)
            oAx = thread.shfl_idx(tAx, safe)
            oAy = thread.shfl_idx(tAy, safe)
            oAz = thread.shfl_idx(tAz, safe)
            oAw = thread.shfl_idx(tAw, safe)
            if threadIndexInWarp >= cp.int32(4):
                if tid < cp.int32(GRID_SIZE):
                    oIdLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oIdHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oIdLo:
                        if tBlockRigidIdHi == oIdHi:
                            tLx = tLx + oLx
                            tLy = tLy + oLy
                            tLz = tLz + oLz
                            tLw = tLw + oLw
                            tAx = tAx + oAx
                            tAy = tAy + oAy
                            tAz = tAz + oAz
                            tAw = tAw + oAw

            # reductionRadius = 8
            sl = threadIndexInWarp - cp.int32(8)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oLx = thread.shfl_idx(tLx, safe)
            oLy = thread.shfl_idx(tLy, safe)
            oLz = thread.shfl_idx(tLz, safe)
            oLw = thread.shfl_idx(tLw, safe)
            oAx = thread.shfl_idx(tAx, safe)
            oAy = thread.shfl_idx(tAy, safe)
            oAz = thread.shfl_idx(tAz, safe)
            oAw = thread.shfl_idx(tAw, safe)
            if threadIndexInWarp >= cp.int32(8):
                if tid < cp.int32(GRID_SIZE):
                    oIdLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oIdHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oIdLo:
                        if tBlockRigidIdHi == oIdHi:
                            tLx = tLx + oLx
                            tLy = tLy + oLy
                            tLz = tLz + oLz
                            tLw = tLw + oLw
                            tAx = tAx + oAx
                            tAy = tAy + oAy
                            tAz = tAz + oAz
                            tAw = tAw + oAw

            # reductionRadius = 16
            sl = threadIndexInWarp - cp.int32(16)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oLx = thread.shfl_idx(tLx, safe)
            oLy = thread.shfl_idx(tLy, safe)
            oLz = thread.shfl_idx(tLz, safe)
            oLw = thread.shfl_idx(tLw, safe)
            oAx = thread.shfl_idx(tAx, safe)
            oAy = thread.shfl_idx(tAy, safe)
            oAz = thread.shfl_idx(tAz, safe)
            oAw = thread.shfl_idx(tAw, safe)
            if threadIndexInWarp >= cp.int32(16):
                if tid < cp.int32(GRID_SIZE):
                    oIdLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oIdHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oIdLo:
                        if tBlockRigidIdHi == oIdHi:
                            tLx = tLx + oLx
                            tLy = tLy + oLy
                            tLz = tLz + oLz
                            tLw = tLw + oLw
                            tAx = tAx + oAx
                            tAy = tAy + oAy
                            tAz = tAz + oAz
                            tAw = tAw + oAw

            # Conditional write: only in-bounds threads store results
            if tid < cp.int32(GRID_SIZE):
                sBlockLinDvX[tid] = cp.disjoint(tLx)
                sBlockLinDvY[tid] = cp.disjoint(tLy)
                sBlockLinDvZ[tid] = cp.disjoint(tLz)
                sBlockLinDvW[tid] = cp.disjoint(tLw)
                sBlockAngDvX[tid] = cp.disjoint(tAx)
                sBlockAngDvY[tid] = cp.disjoint(tAy)
                sBlockAngDvZ[tid] = cp.disjoint(tAz)
                sBlockAngDvW[tid] = cp.disjoint(tAw)
                # Re-store original blockRigidId (not the reduced one)
                sBlockRigidIdLo[tid] = cp.disjoint(blockRigidId[tid, 0] + cp.int32(0))
                sBlockRigidIdHi[tid] = cp.disjoint(blockRigidId[tid, 1] + cp.int32(0))

        block.barrier()

        # Phase 2: For each contact, check if it's the last for its rigidId,
        # then apply the accumulated delta velocity to the solver body
        for _iter in range(4096):
            if _iter < nbIterationsPerBlock:
                # Load rigidIds for neighbor detection
                for tid, thread in block.threads():
                    workIndex = cp.int32(BLOCK_SIZE) * (bx * nbIterationsPerBlock + cp.int32(_iter)) + tid

                    rigidIdLo = cp.int32(SENTINEL_LO)
                    rigidIdHi = cp.int32(SENTINEL_HI)
                    if workIndex < tNumContacts:
                        rigidIdLo = sortedRigidIds[workIndex, 0] + cp.int32(0)
                        rigidIdHi = sortedRigidIds[workIndex, 1] + cp.int32(0)
                        if tid > cp.int32(0):
                            sRigidIdLo[tid - cp.int32(1)] = cp.disjoint(rigidIdLo)
                            sRigidIdHi[tid - cp.int32(1)] = cp.disjoint(rigidIdHi)

                        if workIndex == tNumContacts - cp.int32(1):
                            sRigidIdLo[tid] = cp.disjoint(cp.int32(SENTINEL_LO))
                            sRigidIdHi[tid] = cp.disjoint(cp.int32(SENTINEL_HI))
                        if tid == cp.int32(BLOCK_SIZE - 1):
                            if workIndex < tNumContacts - cp.int32(1):
                                nextLo = sortedRigidIds[workIndex + cp.int32(1), 0] + cp.int32(0)
                                nextHi = sortedRigidIds[workIndex + cp.int32(1), 1] + cp.int32(0)
                                sRigidIdLo[tid] = cp.disjoint(nextLo)
                                sRigidIdHi[tid] = cp.disjoint(nextHi)

                block.barrier()

                for tid, thread in block.threads():
                    workIndex = cp.int32(BLOCK_SIZE) * (bx * nbIterationsPerBlock + cp.int32(_iter)) + tid

                    if workIndex < tNumContacts:
                        rigidIdLo = sortedRigidIds[workIndex, 0] + cp.int32(0)
                        rigidIdHi = sortedRigidIds[workIndex, 1] + cp.int32(0)

                        # Check if this is the last entry for this rigidId
                        nextLo = sRigidIdLo[tid] + cp.int32(0)
                        nextHi = sRigidIdHi[tid] + cp.int32(0)
                        isLast = cp.int32(1)
                        if rigidIdLo == nextLo:
                            if rigidIdHi == nextHi:
                                isLast = cp.int32(0)

                        if isLast != cp.int32(0):
                            linVx = deltaV[workIndex, 0] + cp.float32(0.0)
                            linVy = deltaV[workIndex, 1] + cp.float32(0.0)
                            linVz = deltaV[workIndex, 2] + cp.float32(0.0)
                            linVw = deltaV[workIndex, 3] + cp.float32(0.0)
                            angVx = deltaV[workIndex + tNumContacts, 0] + cp.float32(0.0)
                            angVy = deltaV[workIndex + tNumContacts, 1] + cp.float32(0.0)
                            angVz = deltaV[workIndex + tNumContacts, 2] + cp.float32(0.0)
                            angVw = deltaV[workIndex + tNumContacts, 3] + cp.float32(0.0)

                            # Add previous block's accumulator if matching
                            preBlockLo = cp.int32(SENTINEL_LO)
                            preBlockHi = cp.int32(SENTINEL_HI)
                            if bx > cp.int32(0):
                                preBlockLo = sBlockRigidIdLo[bx - cp.int32(1)] + cp.int32(0)
                                preBlockHi = sBlockRigidIdHi[bx - cp.int32(1)] + cp.int32(0)
                            if rigidIdLo == preBlockLo:
                                if rigidIdHi == preBlockHi:
                                    linVx = linVx + sBlockLinDvX[bx - cp.int32(1)] + cp.float32(0.0)
                                    linVy = linVy + sBlockLinDvY[bx - cp.int32(1)] + cp.float32(0.0)
                                    linVz = linVz + sBlockLinDvZ[bx - cp.int32(1)] + cp.float32(0.0)
                                    linVw = linVw + sBlockLinDvW[bx - cp.int32(1)] + cp.float32(0.0)
                                    angVx = angVx + sBlockAngDvX[bx - cp.int32(1)] + cp.float32(0.0)
                                    angVy = angVy + sBlockAngDvY[bx - cp.int32(1)] + cp.float32(0.0)
                                    angVz = angVz + sBlockAngDvZ[bx - cp.int32(1)] + cp.float32(0.0)
                                    angVw = angVw + sBlockAngDvW[bx - cp.int32(1)] + cp.float32(0.0)

                            # Decode PxNodeIndex from rigidId (lo=mID, hi=mLinkID)
                            nodeId = rigidIdLo  # mID
                            nodeLinkId = rigidIdHi  # mLinkID
                            isStatic = cp.int32(1) if nodeId == cp.int32(INVALID_NODE) else cp.int32(0)

                            if isStatic == cp.int32(0):
                                nodeIndex = nodeId
                                solverBodyIndex = solverBodyIndices[nodeIndex] + cp.int32(0)
                                isArticulation = nodeLinkId & cp.int32(1)

                                if isArticulation != cp.int32(0):
                                    blockIndex = solverBodyIndex >> cp.int32(5)  # / WARP_SIZE
                                    artiIndexInBlock = solverBodyIndex & cp.int32(31)  # % WARP_SIZE

                                    # Set mStateDirty = eHAS_IMPULSES
                                    # mStateDirty is PxU8[32], packed as int32[8]
                                    byteSlot = artiIndexInBlock >> cp.int32(2)  # / 4
                                    byteOff = (artiIndexInBlock & cp.int32(3)) << cp.int32(3)  # (% 4) * 8
                                    # Set the byte to eHAS_IMPULSES = 2
                                    oldPacked = artiStateDirty[blockIndex, byteSlot] + cp.int32(0)
                                    clearMask = ~(cp.int32(0xFF) << byteOff)
                                    newPacked = (oldPacked & clearMask) | (cp.int32(eHAS_IMPULSES) << byteOff)
                                    artiStateDirty[blockIndex, byteSlot] = newPacked

                                    linkID = nodeLinkId >> cp.int32(1)  # articulationLinkId

                                    denom_val = linVw if linVw > cp.float32(1.0) else cp.float32(1.0)
                                    ratio = cp.float32(1.0) / denom_val
                                    linVw = cp.float32(0.0)

                                    # storeSpatialVector: -impulse
                                    linkSlot = blockIndex * maxLinks + linkID
                                    artiLinkScratchImpTop[linkSlot, artiIndexInBlock, 0] = cp.float32(0.0) - linVx * ratio
                                    artiLinkScratchImpTop[linkSlot, artiIndexInBlock, 1] = cp.float32(0.0) - linVy * ratio
                                    artiLinkScratchImpTop[linkSlot, artiIndexInBlock, 2] = cp.float32(0.0) - linVz * ratio
                                    artiLinkScratchImpTop[linkSlot, artiIndexInBlock, 3] = cp.float32(0.0) - angVx * ratio
                                    artiLinkScratchImpByz[linkSlot, artiIndexInBlock, 0] = cp.float32(0.0) - angVy * ratio
                                    artiLinkScratchImpByz[linkSlot, artiIndexInBlock, 1] = cp.float32(0.0) - angVz * ratio

                                if isArticulation == cp.int32(0):
                                    linearVx = solverBodyDeltaVel[solverBodyIndex, 0] + cp.float32(0.0)
                                    linearVy = solverBodyDeltaVel[solverBodyIndex, 1] + cp.float32(0.0)
                                    linearVz = solverBodyDeltaVel[solverBodyIndex, 2] + cp.float32(0.0)
                                    linearVw = solverBodyDeltaVel[solverBodyIndex, 3] + cp.float32(0.0)
                                    angularVx = solverBodyDeltaVel[solverBodyIndex + numSolverBodies, 0] + cp.float32(0.0)
                                    angularVy = solverBodyDeltaVel[solverBodyIndex + numSolverBodies, 1] + cp.float32(0.0)
                                    angularVz = solverBodyDeltaVel[solverBodyIndex + numSolverBodies, 2] + cp.float32(0.0)
                                    angularVw = solverBodyDeltaVel[solverBodyIndex + numSolverBodies, 3] + cp.float32(0.0)

                                    denom_val = linVw if linVw > cp.float32(1.0) else cp.float32(1.0)
                                    ratio = cp.float32(1.0) / denom_val
                                    linVw = cp.float32(0.0)

                                    if isTGS != cp.int32(0):
                                        linearVx = linearVx + linVx * ratio
                                        linearVy = linearVy + linVy * ratio
                                        linearVz = linearVz + linVz * ratio
                                        linearVw = linearVw + angVx * ratio
                                        angularVx = angularVx + angVy * ratio
                                        angularVy = angularVy + angVz * ratio
                                    if isTGS == cp.int32(0):
                                        linearVx = linearVx + linVx * ratio
                                        linearVy = linearVy + linVy * ratio
                                        linearVz = linearVz + linVz * ratio
                                        linearVw = linearVw + linVw * ratio
                                        angularVx = angularVx + angVx * ratio
                                        angularVy = angularVy + angVy * ratio
                                        angularVz = angularVz + angVz * ratio
                                        angularVw = angularVw + angVw * ratio

                                    solverBodyDeltaVel[solverBodyIndex, 0] = linearVx
                                    solverBodyDeltaVel[solverBodyIndex, 1] = linearVy
                                    solverBodyDeltaVel[solverBodyIndex, 2] = linearVz
                                    solverBodyDeltaVel[solverBodyIndex, 3] = linearVw
                                    solverBodyDeltaVel[solverBodyIndex + numSolverBodies, 0] = angularVx
                                    solverBodyDeltaVel[solverBodyIndex + numSolverBodies, 1] = angularVy
                                    solverBodyDeltaVel[solverBodyIndex + numSolverBodies, 2] = angularVz
                                    solverBodyDeltaVel[solverBodyIndex + numSolverBodies, 3] = angularVw

                block.barrier()


# ===== Kernel 3: clearDeltaVRigidSecondLaunchMulti =====
@cp.kernel
def clearDeltaVRigidSecondLaunchMulti(
    sortedRigidIds,          # int32[N, 2]
    numContacts,             # int32[1]
    solverBodyIndices,       # int32[M]
    artiLinkDeltaScale,      # float32[totalLinkSlots, 32]
    maxLinks,                # int scalar
    tempDenom,               # float32[S]
    BLOCK_SIZE: cp.constexpr = 512,
    GRID_SIZE: cp.constexpr = 32,
):
    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        sRigidIdLo = block.alloc((BLOCK_SIZE + 1,), dtype=cp.int32)
        sRigidIdHi = block.alloc((BLOCK_SIZE + 1,), dtype=cp.int32)

        tNumContacts = numContacts[0] + cp.int32(0)
        totalBlockRequired = (tNumContacts + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)
        numIterationPerBlock = (totalBlockRequired + cp.int32(GRID_SIZE - 1)) // cp.int32(GRID_SIZE)

        for _iter in range(4096):
            if _iter < numIterationPerBlock:
                for tid, thread in block.threads():
                    workIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                    rigidIdLo = cp.int32(SENTINEL_LO)
                    rigidIdHi = cp.int32(SENTINEL_HI)
                    if workIndex < tNumContacts:
                        rigidIdLo = sortedRigidIds[workIndex, 0] + cp.int32(0)
                        rigidIdHi = sortedRigidIds[workIndex, 1] + cp.int32(0)
                        if tid > cp.int32(0):
                            sRigidIdLo[tid - cp.int32(1)] = cp.disjoint(rigidIdLo)
                            sRigidIdHi[tid - cp.int32(1)] = cp.disjoint(rigidIdHi)

                        if workIndex == tNumContacts - cp.int32(1):
                            sRigidIdLo[tid] = cp.disjoint(cp.int32(SENTINEL_LO))
                            sRigidIdHi[tid] = cp.disjoint(cp.int32(SENTINEL_HI))
                        if tid == cp.int32(BLOCK_SIZE - 1):
                            if workIndex < tNumContacts - cp.int32(1):
                                nextLo = sortedRigidIds[workIndex + cp.int32(1), 0] + cp.int32(0)
                                nextHi = sortedRigidIds[workIndex + cp.int32(1), 1] + cp.int32(0)
                                sRigidIdLo[tid] = cp.disjoint(nextLo)
                                sRigidIdHi[tid] = cp.disjoint(nextHi)

                block.barrier()

                for tid, thread in block.threads():
                    workIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                    if workIndex < tNumContacts:
                        rigidIdLo = sortedRigidIds[workIndex, 0] + cp.int32(0)
                        rigidIdHi = sortedRigidIds[workIndex, 1] + cp.int32(0)

                        nextLo = sRigidIdLo[tid] + cp.int32(0)
                        nextHi = sRigidIdHi[tid] + cp.int32(0)
                        isLast = cp.int32(1)
                        if rigidIdLo == nextLo:
                            if rigidIdHi == nextHi:
                                isLast = cp.int32(0)

                        if isLast != cp.int32(0):
                            nodeId = rigidIdLo
                            nodeLinkId = rigidIdHi
                            isStatic = cp.int32(1) if nodeId == cp.int32(INVALID_NODE) else cp.int32(0)

                            if isStatic == cp.int32(0):
                                nodeIndex = nodeId
                                solverBodyIndex = solverBodyIndices[nodeIndex] + cp.int32(0)
                                isArticulation = nodeLinkId & cp.int32(1)

                                if isArticulation != cp.int32(0):
                                    blockIndex = solverBodyIndex >> cp.int32(5)
                                    artiIndexInBlock = solverBodyIndex & cp.int32(31)
                                    linkID = nodeLinkId >> cp.int32(1)
                                    linkSlot = blockIndex * maxLinks + linkID
                                    artiLinkDeltaScale[linkSlot, artiIndexInBlock] = cp.float32(0.0)

                                if isArticulation == cp.int32(0):
                                    tempDenom[solverBodyIndex] = cp.float32(0.0)

                block.barrier()


# ===== Kernel 4: accumulateDeltaVRigidSecondLaunchMultiStage1 =====
@cp.kernel
def accumulateDeltaVRigidSecondLaunchMultiStage1(
    sortedRigidIds,          # int32[N, 2]
    numContacts,             # int32[1]
    deltaV,                  # float32[2*N, 4]
    blockDeltaV,             # float32[2*GRID_SIZE, 4]
    blockRigidId,            # int32[GRID_SIZE, 2]
    solverBodyIndices,       # int32[M]
    artiLinkDeltaScale,      # float32[totalLinkSlots, 32]
    maxLinks,                # int scalar
    tempDenom,               # float32[S]
    useLocalRelax,           # int scalar (0/1)
    globalRelaxationCoefficient,  # float32 scalar
    BLOCK_SIZE: cp.constexpr = 512,
    GRID_SIZE: cp.constexpr = 32,
):
    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        sBlockLinDvW = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockRigidIdLo = block.alloc((GRID_SIZE,), dtype=cp.int32)
        sBlockRigidIdHi = block.alloc((GRID_SIZE,), dtype=cp.int32)
        sRigidIdLo = block.alloc((BLOCK_SIZE + 1,), dtype=cp.int32)
        sRigidIdHi = block.alloc((BLOCK_SIZE + 1,), dtype=cp.int32)

        tNumContacts = numContacts[0] + cp.int32(0)
        totalBlockRequired = (tNumContacts + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)

        # Phase 1: Load block-level w component and reduce
        for tid, thread in block.threads():
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)

            tBlockRigidIdLo = cp.int32(SENTINEL_LO)
            tBlockRigidIdHi = cp.int32(SENTINEL_HI)
            linDvW = cp.float32(0.0)

            if tid < cp.int32(GRID_SIZE):
                linDvW = blockDeltaV[tid, 3] + cp.float32(0.0)
                tBlockRigidIdLo = blockRigidId[tid, 0] + cp.int32(0)
                tBlockRigidIdHi = blockRigidId[tid, 1] + cp.int32(0)
                sBlockLinDvW[tid] = cp.disjoint(linDvW)
                sBlockRigidIdLo[tid] = cp.disjoint(tBlockRigidIdLo)
                sBlockRigidIdHi[tid] = cp.disjoint(tBlockRigidIdHi)

        block.barrier()

        # Cross-block reduction on w component only
        # NOTE: shuffles must be unconditional (all warp lanes active).
        # Safe-load pattern: all threads load, but out-of-bounds threads get zeros.
        # Conditional write: only threads with tid < GRID_SIZE write results.
        for tid, thread in block.threads():
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)

            tW = cp.float32(0.0)
            tBlockRigidIdLo = cp.int32(SENTINEL_LO)
            tBlockRigidIdHi = cp.int32(SENTINEL_HI)

            # Safe load: all threads read, out-of-bounds get zeros/sentinel
            safeIdx = tid
            if tid >= cp.int32(GRID_SIZE):
                safeIdx = cp.int32(0)
            tW = sBlockLinDvW[safeIdx] + cp.float32(0.0)
            tBlockRigidIdLo = sBlockRigidIdLo[safeIdx] + cp.int32(0)
            tBlockRigidIdHi = sBlockRigidIdHi[safeIdx] + cp.int32(0)
            if tid >= cp.int32(GRID_SIZE):
                tW = cp.float32(0.0)
                tBlockRigidIdLo = cp.int32(SENTINEL_LO)
                tBlockRigidIdHi = cp.int32(SENTINEL_HI)

            # reductionRadius = 1
            sl = threadIndexInWarp - cp.int32(1)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oW = thread.shfl_idx(tW, safe)
            if threadIndexInWarp >= cp.int32(1):
                if tid < cp.int32(GRID_SIZE):
                    oLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oLo:
                        if tBlockRigidIdHi == oHi:
                            tW = tW + oW

            # reductionRadius = 2
            sl = threadIndexInWarp - cp.int32(2)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oW = thread.shfl_idx(tW, safe)
            if threadIndexInWarp >= cp.int32(2):
                if tid < cp.int32(GRID_SIZE):
                    oLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oLo:
                        if tBlockRigidIdHi == oHi:
                            tW = tW + oW

            # reductionRadius = 4
            sl = threadIndexInWarp - cp.int32(4)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oW = thread.shfl_idx(tW, safe)
            if threadIndexInWarp >= cp.int32(4):
                if tid < cp.int32(GRID_SIZE):
                    oLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oLo:
                        if tBlockRigidIdHi == oHi:
                            tW = tW + oW

            # reductionRadius = 8
            sl = threadIndexInWarp - cp.int32(8)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oW = thread.shfl_idx(tW, safe)
            if threadIndexInWarp >= cp.int32(8):
                if tid < cp.int32(GRID_SIZE):
                    oLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oLo:
                        if tBlockRigidIdHi == oHi:
                            tW = tW + oW

            # reductionRadius = 16
            sl = threadIndexInWarp - cp.int32(16)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oW = thread.shfl_idx(tW, safe)
            if threadIndexInWarp >= cp.int32(16):
                if tid < cp.int32(GRID_SIZE):
                    oLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oLo:
                        if tBlockRigidIdHi == oHi:
                            tW = tW + oW

            # Conditional write: only in-bounds threads store results
            if tid < cp.int32(GRID_SIZE):
                sBlockLinDvW[tid] = cp.disjoint(tW)
                sBlockRigidIdLo[tid] = cp.disjoint(blockRigidId[tid, 0] + cp.int32(0))
                sBlockRigidIdHi[tid] = cp.disjoint(blockRigidId[tid, 1] + cp.int32(0))

        numIterationPerBlock = (totalBlockRequired + cp.int32(GRID_SIZE - 1)) // cp.int32(GRID_SIZE)

        block.barrier()

        # Phase 2: Process contacts
        for _iter in range(4096):
            if _iter < numIterationPerBlock:
                for tid, thread in block.threads():
                    workIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                    rigidIdLo = cp.int32(SENTINEL_LO)
                    rigidIdHi = cp.int32(SENTINEL_HI)
                    if workIndex < tNumContacts:
                        rigidIdLo = sortedRigidIds[workIndex, 0] + cp.int32(0)
                        rigidIdHi = sortedRigidIds[workIndex, 1] + cp.int32(0)
                        if tid > cp.int32(0):
                            sRigidIdLo[tid - cp.int32(1)] = cp.disjoint(rigidIdLo)
                            sRigidIdHi[tid - cp.int32(1)] = cp.disjoint(rigidIdHi)
                        if workIndex == tNumContacts - cp.int32(1):
                            sRigidIdLo[tid] = cp.disjoint(cp.int32(SENTINEL_LO))
                            sRigidIdHi[tid] = cp.disjoint(cp.int32(SENTINEL_HI))
                        if tid == cp.int32(BLOCK_SIZE - 1):
                            if workIndex < tNumContacts - cp.int32(1):
                                nextLo = sortedRigidIds[workIndex + cp.int32(1), 0] + cp.int32(0)
                                nextHi = sortedRigidIds[workIndex + cp.int32(1), 1] + cp.int32(0)
                                sRigidIdLo[tid] = cp.disjoint(nextLo)
                                sRigidIdHi[tid] = cp.disjoint(nextHi)

                block.barrier()

                for tid, thread in block.threads():
                    workIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                    if workIndex < tNumContacts:
                        rigidIdLo = sortedRigidIds[workIndex, 0] + cp.int32(0)
                        rigidIdHi = sortedRigidIds[workIndex, 1] + cp.int32(0)

                        nextLo = sRigidIdLo[tid] + cp.int32(0)
                        nextHi = sRigidIdHi[tid] + cp.int32(0)
                        isLast = cp.int32(1)
                        if rigidIdLo == nextLo:
                            if rigidIdHi == nextHi:
                                isLast = cp.int32(0)

                        if isLast != cp.int32(0):
                            linVelW = deltaV[workIndex, 3] + cp.float32(0.0)

                            preBlockLo = cp.int32(SENTINEL_LO)
                            preBlockHi = cp.int32(SENTINEL_HI)
                            if bx > cp.int32(0):
                                preBlockLo = sBlockRigidIdLo[bx - cp.int32(1)] + cp.int32(0)
                                preBlockHi = sBlockRigidIdHi[bx - cp.int32(1)] + cp.int32(0)
                            if rigidIdLo == preBlockLo:
                                if rigidIdHi == preBlockHi:
                                    linVelW = linVelW + sBlockLinDvW[bx - cp.int32(1)] + cp.float32(0.0)

                            nodeId = rigidIdLo
                            nodeLinkId = rigidIdHi
                            isStatic = cp.int32(1) if nodeId == cp.int32(INVALID_NODE) else cp.int32(0)

                            if isStatic == cp.int32(0):
                                nodeIndex = nodeId
                                solverBodyIndex = solverBodyIndices[nodeIndex] + cp.int32(0)
                                isArticulation = nodeLinkId & cp.int32(1)

                                denom = globalRelaxationCoefficient
                                if useLocalRelax != cp.int32(0):
                                    denom = denom if denom > linVelW else linVelW

                                if isArticulation != cp.int32(0):
                                    blockIndex = solverBodyIndex >> cp.int32(5)
                                    artiIndexInBlock = solverBodyIndex & cp.int32(31)
                                    linkID = nodeLinkId >> cp.int32(1)
                                    linkSlot = blockIndex * maxLinks + linkID
                                    thread.atomic_add(artiLinkDeltaScale[linkSlot, artiIndexInBlock], denom)

                                if isArticulation == cp.int32(0):
                                    thread.atomic_add(tempDenom[solverBodyIndex], denom)

                block.barrier()


# ===== Kernel 5: accumulateDeltaVRigidSecondLaunchMultiStage2 =====
@cp.kernel
def accumulateDeltaVRigidSecondLaunchMultiStage2(
    sortedRigidIds,          # int32[N, 2]
    numContacts,             # int32[1]
    deltaV,                  # float32[2*N, 4]
    blockDeltaV,             # float32[2*GRID_SIZE, 4]
    blockRigidId,            # int32[GRID_SIZE, 2]
    solverBodyIndices,       # int32[M]
    solverBodyDeltaVel,      # float32[2*S, 4]
    numSolverBodies,         # int scalar
    artiStateDirty,          # int32[numArtiBlocks, 8]
    artiLinkScratchImpTop,   # float32[totalLinkSlots, 32, 4]
    artiLinkScratchImpByz,   # float32[totalLinkSlots, 32, 2]
    artiLinkDeltaScale,      # float32[totalLinkSlots, 32]
    maxLinks,                # int scalar
    tempDenom,               # float32[S]
    isTGS,                   # int scalar (0/1)
    BLOCK_SIZE: cp.constexpr = 512,
    GRID_SIZE: cp.constexpr = 32,
):
    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        sBlockLinDvX = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockLinDvY = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockLinDvZ = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockLinDvW = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockAngDvX = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockAngDvY = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockAngDvZ = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockAngDvW = block.alloc((GRID_SIZE,), dtype=cp.float32)
        sBlockRigidIdLo = block.alloc((GRID_SIZE,), dtype=cp.int32)
        sBlockRigidIdHi = block.alloc((GRID_SIZE,), dtype=cp.int32)
        sRigidIdLo = block.alloc((BLOCK_SIZE + 1,), dtype=cp.int32)
        sRigidIdHi = block.alloc((BLOCK_SIZE + 1,), dtype=cp.int32)

        tNumContacts = numContacts[0] + cp.int32(0)
        totalBlockRequired = (tNumContacts + cp.int32(BLOCK_SIZE - 1)) // cp.int32(BLOCK_SIZE)

        # Phase 1: Load and reduce block-level accumulators (same as SecondLaunch)
        for tid, thread in block.threads():
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)

            tBlockRigidIdLo = cp.int32(SENTINEL_LO)
            tBlockRigidIdHi = cp.int32(SENTINEL_HI)
            lx = cp.float32(0.0)
            ly = cp.float32(0.0)
            lz = cp.float32(0.0)
            lw = cp.float32(0.0)
            ax = cp.float32(0.0)
            ay = cp.float32(0.0)
            az = cp.float32(0.0)
            aw = cp.float32(0.0)

            if tid < cp.int32(GRID_SIZE):
                lx = blockDeltaV[tid, 0] + cp.float32(0.0)
                ly = blockDeltaV[tid, 1] + cp.float32(0.0)
                lz = blockDeltaV[tid, 2] + cp.float32(0.0)
                lw = blockDeltaV[tid, 3] + cp.float32(0.0)
                ax = blockDeltaV[tid + cp.int32(GRID_SIZE), 0] + cp.float32(0.0)
                ay = blockDeltaV[tid + cp.int32(GRID_SIZE), 1] + cp.float32(0.0)
                az = blockDeltaV[tid + cp.int32(GRID_SIZE), 2] + cp.float32(0.0)
                aw = blockDeltaV[tid + cp.int32(GRID_SIZE), 3] + cp.float32(0.0)
                tBlockRigidIdLo = blockRigidId[tid, 0] + cp.int32(0)
                tBlockRigidIdHi = blockRigidId[tid, 1] + cp.int32(0)

                sBlockLinDvX[tid] = cp.disjoint(lx)
                sBlockLinDvY[tid] = cp.disjoint(ly)
                sBlockLinDvZ[tid] = cp.disjoint(lz)
                sBlockLinDvW[tid] = cp.disjoint(lw)
                sBlockAngDvX[tid] = cp.disjoint(ax)
                sBlockAngDvY[tid] = cp.disjoint(ay)
                sBlockAngDvZ[tid] = cp.disjoint(az)
                sBlockAngDvW[tid] = cp.disjoint(aw)
                sBlockRigidIdLo[tid] = cp.disjoint(tBlockRigidIdLo)
                sBlockRigidIdHi[tid] = cp.disjoint(tBlockRigidIdHi)

        block.barrier()

        # Cross-block reduction
        # NOTE: shuffles must be unconditional (all warp lanes active).
        # Safe-load pattern: all threads load, but out-of-bounds threads get zeros.
        # Conditional write: only threads with tid < GRID_SIZE write results.
        for tid, thread in block.threads():
            threadIndexInWarp = tid & cp.int32(WARP_SIZE - 1)

            tLx = cp.float32(0.0)
            tLy = cp.float32(0.0)
            tLz = cp.float32(0.0)
            tLw = cp.float32(0.0)
            tAx = cp.float32(0.0)
            tAy = cp.float32(0.0)
            tAz = cp.float32(0.0)
            tAw = cp.float32(0.0)
            tBlockRigidIdLo = cp.int32(SENTINEL_LO)
            tBlockRigidIdHi = cp.int32(SENTINEL_HI)

            # Safe load: all threads read, out-of-bounds get zeros/sentinel
            safeIdx = tid
            if tid >= cp.int32(GRID_SIZE):
                safeIdx = cp.int32(0)
            tLx = sBlockLinDvX[safeIdx] + cp.float32(0.0)
            tLy = sBlockLinDvY[safeIdx] + cp.float32(0.0)
            tLz = sBlockLinDvZ[safeIdx] + cp.float32(0.0)
            tLw = sBlockLinDvW[safeIdx] + cp.float32(0.0)
            tAx = sBlockAngDvX[safeIdx] + cp.float32(0.0)
            tAy = sBlockAngDvY[safeIdx] + cp.float32(0.0)
            tAz = sBlockAngDvZ[safeIdx] + cp.float32(0.0)
            tAw = sBlockAngDvW[safeIdx] + cp.float32(0.0)
            tBlockRigidIdLo = sBlockRigidIdLo[safeIdx] + cp.int32(0)
            tBlockRigidIdHi = sBlockRigidIdHi[safeIdx] + cp.int32(0)
            if tid >= cp.int32(GRID_SIZE):
                tLx = cp.float32(0.0)
                tLy = cp.float32(0.0)
                tLz = cp.float32(0.0)
                tLw = cp.float32(0.0)
                tAx = cp.float32(0.0)
                tAy = cp.float32(0.0)
                tAz = cp.float32(0.0)
                tAw = cp.float32(0.0)
                tBlockRigidIdLo = cp.int32(SENTINEL_LO)
                tBlockRigidIdHi = cp.int32(SENTINEL_HI)

            # Unrolled reduction for radius 1, 2, 4, 8, 16
            # reductionRadius = 1
            sl = threadIndexInWarp - cp.int32(1)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oLx = thread.shfl_idx(tLx, safe)
            oLy = thread.shfl_idx(tLy, safe)
            oLz = thread.shfl_idx(tLz, safe)
            oLw = thread.shfl_idx(tLw, safe)
            oAx = thread.shfl_idx(tAx, safe)
            oAy = thread.shfl_idx(tAy, safe)
            oAz = thread.shfl_idx(tAz, safe)
            oAw = thread.shfl_idx(tAw, safe)
            if threadIndexInWarp >= cp.int32(1):
                if tid < cp.int32(GRID_SIZE):
                    oLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oLo:
                        if tBlockRigidIdHi == oHi:
                            tLx = tLx + oLx
                            tLy = tLy + oLy
                            tLz = tLz + oLz
                            tLw = tLw + oLw
                            tAx = tAx + oAx
                            tAy = tAy + oAy
                            tAz = tAz + oAz
                            tAw = tAw + oAw

            # reductionRadius = 2
            sl = threadIndexInWarp - cp.int32(2)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oLx = thread.shfl_idx(tLx, safe)
            oLy = thread.shfl_idx(tLy, safe)
            oLz = thread.shfl_idx(tLz, safe)
            oLw = thread.shfl_idx(tLw, safe)
            oAx = thread.shfl_idx(tAx, safe)
            oAy = thread.shfl_idx(tAy, safe)
            oAz = thread.shfl_idx(tAz, safe)
            oAw = thread.shfl_idx(tAw, safe)
            if threadIndexInWarp >= cp.int32(2):
                if tid < cp.int32(GRID_SIZE):
                    oLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oLo:
                        if tBlockRigidIdHi == oHi:
                            tLx = tLx + oLx
                            tLy = tLy + oLy
                            tLz = tLz + oLz
                            tLw = tLw + oLw
                            tAx = tAx + oAx
                            tAy = tAy + oAy
                            tAz = tAz + oAz
                            tAw = tAw + oAw

            # reductionRadius = 4
            sl = threadIndexInWarp - cp.int32(4)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oLx = thread.shfl_idx(tLx, safe)
            oLy = thread.shfl_idx(tLy, safe)
            oLz = thread.shfl_idx(tLz, safe)
            oLw = thread.shfl_idx(tLw, safe)
            oAx = thread.shfl_idx(tAx, safe)
            oAy = thread.shfl_idx(tAy, safe)
            oAz = thread.shfl_idx(tAz, safe)
            oAw = thread.shfl_idx(tAw, safe)
            if threadIndexInWarp >= cp.int32(4):
                if tid < cp.int32(GRID_SIZE):
                    oLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oLo:
                        if tBlockRigidIdHi == oHi:
                            tLx = tLx + oLx
                            tLy = tLy + oLy
                            tLz = tLz + oLz
                            tLw = tLw + oLw
                            tAx = tAx + oAx
                            tAy = tAy + oAy
                            tAz = tAz + oAz
                            tAw = tAw + oAw

            # reductionRadius = 8
            sl = threadIndexInWarp - cp.int32(8)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oLx = thread.shfl_idx(tLx, safe)
            oLy = thread.shfl_idx(tLy, safe)
            oLz = thread.shfl_idx(tLz, safe)
            oLw = thread.shfl_idx(tLw, safe)
            oAx = thread.shfl_idx(tAx, safe)
            oAy = thread.shfl_idx(tAy, safe)
            oAz = thread.shfl_idx(tAz, safe)
            oAw = thread.shfl_idx(tAw, safe)
            if threadIndexInWarp >= cp.int32(8):
                if tid < cp.int32(GRID_SIZE):
                    oLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oLo:
                        if tBlockRigidIdHi == oHi:
                            tLx = tLx + oLx
                            tLy = tLy + oLy
                            tLz = tLz + oLz
                            tLw = tLw + oLw
                            tAx = tAx + oAx
                            tAy = tAy + oAy
                            tAz = tAz + oAz
                            tAw = tAw + oAw

            # reductionRadius = 16
            sl = threadIndexInWarp - cp.int32(16)
            safe = sl if sl >= cp.int32(0) else cp.int32(0)
            oLx = thread.shfl_idx(tLx, safe)
            oLy = thread.shfl_idx(tLy, safe)
            oLz = thread.shfl_idx(tLz, safe)
            oLw = thread.shfl_idx(tLw, safe)
            oAx = thread.shfl_idx(tAx, safe)
            oAy = thread.shfl_idx(tAy, safe)
            oAz = thread.shfl_idx(tAz, safe)
            oAw = thread.shfl_idx(tAw, safe)
            if threadIndexInWarp >= cp.int32(16):
                if tid < cp.int32(GRID_SIZE):
                    oLo = sBlockRigidIdLo[sl] + cp.int32(0)
                    oHi = sBlockRigidIdHi[sl] + cp.int32(0)
                    if tBlockRigidIdLo == oLo:
                        if tBlockRigidIdHi == oHi:
                            tLx = tLx + oLx
                            tLy = tLy + oLy
                            tLz = tLz + oLz
                            tLw = tLw + oLw
                            tAx = tAx + oAx
                            tAy = tAy + oAy
                            tAz = tAz + oAz
                            tAw = tAw + oAw

            # Conditional write: only in-bounds threads store results
            if tid < cp.int32(GRID_SIZE):
                sBlockLinDvX[tid] = cp.disjoint(tLx)
                sBlockLinDvY[tid] = cp.disjoint(tLy)
                sBlockLinDvZ[tid] = cp.disjoint(tLz)
                sBlockLinDvW[tid] = cp.disjoint(tLw)
                sBlockAngDvX[tid] = cp.disjoint(tAx)
                sBlockAngDvY[tid] = cp.disjoint(tAy)
                sBlockAngDvZ[tid] = cp.disjoint(tAz)
                sBlockAngDvW[tid] = cp.disjoint(tAw)
                sBlockRigidIdLo[tid] = cp.disjoint(blockRigidId[tid, 0] + cp.int32(0))
                sBlockRigidIdHi[tid] = cp.disjoint(blockRigidId[tid, 1] + cp.int32(0))

        numIterationPerBlock = (totalBlockRequired + cp.int32(GRID_SIZE - 1)) // cp.int32(GRID_SIZE)

        block.barrier()

        # Phase 2: Process contacts with atomics
        for _iter in range(4096):
            if _iter < numIterationPerBlock:
                for tid, thread in block.threads():
                    workIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                    rigidIdLo = cp.int32(SENTINEL_LO)
                    rigidIdHi = cp.int32(SENTINEL_HI)
                    if workIndex < tNumContacts:
                        rigidIdLo = sortedRigidIds[workIndex, 0] + cp.int32(0)
                        rigidIdHi = sortedRigidIds[workIndex, 1] + cp.int32(0)
                        if tid > cp.int32(0):
                            sRigidIdLo[tid - cp.int32(1)] = cp.disjoint(rigidIdLo)
                            sRigidIdHi[tid - cp.int32(1)] = cp.disjoint(rigidIdHi)
                        if workIndex == tNumContacts - cp.int32(1):
                            sRigidIdLo[tid] = cp.disjoint(cp.int32(SENTINEL_LO))
                            sRigidIdHi[tid] = cp.disjoint(cp.int32(SENTINEL_HI))
                        if tid == cp.int32(BLOCK_SIZE - 1):
                            if workIndex < tNumContacts - cp.int32(1):
                                nextLo = sortedRigidIds[workIndex + cp.int32(1), 0] + cp.int32(0)
                                nextHi = sortedRigidIds[workIndex + cp.int32(1), 1] + cp.int32(0)
                                sRigidIdLo[tid] = cp.disjoint(nextLo)
                                sRigidIdHi[tid] = cp.disjoint(nextHi)

                block.barrier()

                for tid, thread in block.threads():
                    workIndex = cp.int32(_iter) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                    if workIndex < tNumContacts:
                        rigidIdLo = sortedRigidIds[workIndex, 0] + cp.int32(0)
                        rigidIdHi = sortedRigidIds[workIndex, 1] + cp.int32(0)

                        nextLo = sRigidIdLo[tid] + cp.int32(0)
                        nextHi = sRigidIdHi[tid] + cp.int32(0)
                        isLast = cp.int32(1)
                        if rigidIdLo == nextLo:
                            if rigidIdHi == nextHi:
                                isLast = cp.int32(0)

                        if isLast != cp.int32(0):
                            linVx = deltaV[workIndex, 0] + cp.float32(0.0)
                            linVy = deltaV[workIndex, 1] + cp.float32(0.0)
                            linVz = deltaV[workIndex, 2] + cp.float32(0.0)
                            linVw = deltaV[workIndex, 3] + cp.float32(0.0)
                            angVx = deltaV[workIndex + tNumContacts, 0] + cp.float32(0.0)
                            angVy = deltaV[workIndex + tNumContacts, 1] + cp.float32(0.0)
                            angVz = deltaV[workIndex + tNumContacts, 2] + cp.float32(0.0)
                            angVw = deltaV[workIndex + tNumContacts, 3] + cp.float32(0.0)

                            preBlockLo = cp.int32(SENTINEL_LO)
                            preBlockHi = cp.int32(SENTINEL_HI)
                            if bx > cp.int32(0):
                                preBlockLo = sBlockRigidIdLo[bx - cp.int32(1)] + cp.int32(0)
                                preBlockHi = sBlockRigidIdHi[bx - cp.int32(1)] + cp.int32(0)
                            if rigidIdLo == preBlockLo:
                                if rigidIdHi == preBlockHi:
                                    linVx = linVx + sBlockLinDvX[bx - cp.int32(1)] + cp.float32(0.0)
                                    linVy = linVy + sBlockLinDvY[bx - cp.int32(1)] + cp.float32(0.0)
                                    linVz = linVz + sBlockLinDvZ[bx - cp.int32(1)] + cp.float32(0.0)
                                    linVw = linVw + sBlockLinDvW[bx - cp.int32(1)] + cp.float32(0.0)
                                    angVx = angVx + sBlockAngDvX[bx - cp.int32(1)] + cp.float32(0.0)
                                    angVy = angVy + sBlockAngDvY[bx - cp.int32(1)] + cp.float32(0.0)
                                    angVz = angVz + sBlockAngDvZ[bx - cp.int32(1)] + cp.float32(0.0)
                                    angVw = angVw + sBlockAngDvW[bx - cp.int32(1)] + cp.float32(0.0)

                            nodeId = rigidIdLo
                            nodeLinkId = rigidIdHi
                            isStatic = cp.int32(1) if nodeId == cp.int32(INVALID_NODE) else cp.int32(0)

                            if isStatic == cp.int32(0):
                                nodeIndex = nodeId
                                solverBodyIndex = solverBodyIndices[nodeIndex] + cp.int32(0)
                                isArticulation = nodeLinkId & cp.int32(1)

                                if isArticulation != cp.int32(0):
                                    blockIndex = solverBodyIndex >> cp.int32(5)
                                    artiIndexInBlock = solverBodyIndex & cp.int32(31)

                                    # Set mStateDirty = eHAS_IMPULSES
                                    byteSlot = artiIndexInBlock >> cp.int32(2)
                                    byteOff = (artiIndexInBlock & cp.int32(3)) << cp.int32(3)
                                    oldPacked = artiStateDirty[blockIndex, byteSlot] + cp.int32(0)
                                    clearMask = ~(cp.int32(0xFF) << byteOff)
                                    newPacked = (oldPacked & clearMask) | (cp.int32(eHAS_IMPULSES) << byteOff)
                                    artiStateDirty[blockIndex, byteSlot] = newPacked

                                    linkID = nodeLinkId >> cp.int32(1)
                                    linkSlot = blockIndex * maxLinks + linkID

                                    denom_val = artiLinkDeltaScale[linkSlot, artiIndexInBlock] + cp.float32(0.0)
                                    ratio = cp.float32(1.0) / denom_val

                                    # atomicAddSpatialVector: -impulse * ratio
                                    negLinX = cp.float32(0.0) - linVx * ratio
                                    negLinY = cp.float32(0.0) - linVy * ratio
                                    negLinZ = cp.float32(0.0) - linVz * ratio
                                    negAngX = cp.float32(0.0) - angVx * ratio
                                    negAngY = cp.float32(0.0) - angVy * ratio
                                    negAngZ = cp.float32(0.0) - angVz * ratio

                                    thread.atomic_add(artiLinkScratchImpTop[linkSlot, artiIndexInBlock, 0], negLinX)
                                    thread.atomic_add(artiLinkScratchImpTop[linkSlot, artiIndexInBlock, 1], negLinY)
                                    thread.atomic_add(artiLinkScratchImpTop[linkSlot, artiIndexInBlock, 2], negLinZ)
                                    thread.atomic_add(artiLinkScratchImpTop[linkSlot, artiIndexInBlock, 3], negAngX)
                                    thread.atomic_add(artiLinkScratchImpByz[linkSlot, artiIndexInBlock, 0], negAngY)
                                    thread.atomic_add(artiLinkScratchImpByz[linkSlot, artiIndexInBlock, 1], negAngZ)

                                if isArticulation == cp.int32(0):
                                    denom_val = tempDenom[solverBodyIndex] + cp.float32(0.0)
                                    ratio = cp.float32(1.0) / denom_val

                                    if isTGS != cp.int32(0):
                                        thread.atomic_add(solverBodyDeltaVel[solverBodyIndex, 0], linVx * ratio)
                                        thread.atomic_add(solverBodyDeltaVel[solverBodyIndex, 1], linVy * ratio)
                                        thread.atomic_add(solverBodyDeltaVel[solverBodyIndex, 2], linVz * ratio)
                                        thread.atomic_add(solverBodyDeltaVel[solverBodyIndex, 3], angVx * ratio)
                                        thread.atomic_add(solverBodyDeltaVel[solverBodyIndex + numSolverBodies, 0], angVy * ratio)
                                        thread.atomic_add(solverBodyDeltaVel[solverBodyIndex + numSolverBodies, 1], angVz * ratio)

                                    if isTGS == cp.int32(0):
                                        thread.atomic_add(solverBodyDeltaVel[solverBodyIndex, 0], linVx * ratio)
                                        thread.atomic_add(solverBodyDeltaVel[solverBodyIndex, 1], linVy * ratio)
                                        thread.atomic_add(solverBodyDeltaVel[solverBodyIndex, 2], linVz * ratio)
                                        thread.atomic_add(solverBodyDeltaVel[solverBodyIndex + numSolverBodies, 0], angVx * ratio)
                                        thread.atomic_add(solverBodyDeltaVel[solverBodyIndex + numSolverBodies, 1], angVy * ratio)
                                        thread.atomic_add(solverBodyDeltaVel[solverBodyIndex + numSolverBodies, 2], angVz * ratio)

                block.barrier()
