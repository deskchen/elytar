"""Capybara DSL port of gpusimulationcontroller/CUDA/diffuseParticles.cu — all 7 kernels.

Ported kernels:
  - ps_diffuseParticleCopy
  - ps_diffuseParticleSum
  - ps_updateUnsortedDiffuseArrayLaunch
  - ps_diffuseParticleOneWayCollision
  - ps_diffuseParticleCreate
  - ps_diffuseParticleUpdatePBF
  - ps_diffuseParticleCompact

ABI differences from CUDA:
  - PxgParticleSystem struct decomposed to per-kernel flat tensor/scalar args.
  - blockCopy<uint2> to shared memory eliminated entirely.
  - All mDiffuseSimBuffers[bufferIndex] indirection resolved on host side.
  - PxVec3 gravity decomposed to 3 float scalars.
  - PxgParticleContactInfo decomposed to float32[N, 4] (normal_pen).
  - float4 positions/velocities as float32[N, 4] tensors.
  - float2 potentials as float32[N, 2] tensors.
  - Warp operations (warpReduction, warpScanExclusive, ballot, popc, shfl)
    implemented as @cp.inline helpers or direct thread intrinsics.
  - goto in ps_diffuseParticleUpdatePBF replaced with done-flag pattern.
  - Host must pass grid dimensions matching CUDA blockIdx.y (bufferIndex)
    and blockIdx.z (particleSystemId) by launching per-buffer/per-system.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', 'gpucommon', 'src', 'capybara'))
from physx_math import PxVec3

import capybara as cp

WARP_SIZE = 32
EMPTY_CELL = 0xFFFFFFFF


# ===== Warp helpers =====

## warp_reduction_add_u32 must be inlined manually in each kernel
## (cannot pass `thread` to @cp.inline functions).
## Pattern: 5 rounds of shfl_xor with radii 16,8,4,2,1.


@cp.inline
def WDiffuse(h, invR):
    """Diffuse weighting kernel: 1 - h*invR."""
    return cp.float32(1.0) - h * invR


# ===== Grid calculation helpers (from gridCal.cuh) =====

@cp.inline
def floor_to_int(val):
    """Floor a float to int32 (toward negative infinity)."""
    truncated = cp.int32(val)
    ft = cp.float32(truncated)
    correction = cp.int32(0)
    if val < ft:
        correction = cp.int32(1)
    return truncated - correction


@cp.inline
def calcGridPos(px, py, pz, cellWidth):
    """Returns (gx, gy, gz) grid cell coords from position."""
    invW = cp.float32(1.0) / cellWidth
    gx = floor_to_int(px * invW)
    gy = floor_to_int(py * invW)
    gz = floor_to_int(pz * invW)
    return gx, gy, gz


@cp.inline
def calcGridHash(gx, gy, gz, gridSizeX, gridSizeY, gridSizeZ):
    """Returns linear grid hash from cell coords (wrapped)."""
    wx = gx & (gridSizeX - cp.int32(1))
    wy = gy & (gridSizeY - cp.int32(1))
    wz = gz & (gridSizeZ - cp.int32(1))
    return wz * gridSizeY * gridSizeX + wy * gridSizeX + wx


# ===== Kernel 1: ps_diffuseParticleCopy =====
# Host launches this per-buffer (blockIdx.y resolved on host).
# Single-threaded per buffer (only thread 0 does work).
@cp.kernel
def ps_diffuseParticleCopy(
    numDiffuseParticles, maxNumParticles, numActiveDiffuseParticles,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Copy diffuse particle count from slot [1] to slot [0], clamped to max.
    numDiffuseParticles: int32[2] tensor (slot 0 = current, slot 1 = staging).
    numActiveDiffuseParticles: int32[1] tensor (pinned memory output).
    maxNumParticles: scalar.
    """
    with cp.Kernel(1, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            if tid == cp.int32(0):
                staged = numDiffuseParticles[1] + cp.int32(0)
                maxP = maxNumParticles + cp.int32(0)
                clamped = staged + cp.int32(0)
                if staged > maxP:
                    clamped = maxP
                numActiveDiffuseParticles[0] = clamped
                numDiffuseParticles[0] = clamped
                numDiffuseParticles[1] = cp.int32(0)


# ===== Kernel 2: ps_diffuseParticleSum =====
# Host resolves per-particle-system. Sums diffuse counts across buffers.
# numDiffuseCountsPerBuffer: int32[maxBuffers] tensor with mNumDiffuseParticles[0] per buffer.
@cp.kernel
def ps_diffuseParticleSum(
    numDiffuseCountsPerBuffer, numDiffuseBuffers, totalDiffuseOut,
    BLOCK_SIZE: cp.constexpr = 32
):
    """Sum diffuse particle counts across all buffers into totalDiffuseOut[0]."""
    with cp.Kernel(1, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            localSum = cp.int32(0)
            # Each lane accumulates a strided portion
            idx = tid + cp.int32(0)
            while idx < numDiffuseBuffers:
                localSum = localSum + numDiffuseCountsPerBuffer[idx]
                idx = idx + cp.int32(WARP_SIZE)
            # Inline warp reduction (shfl_xor, 5 rounds)
            v = localSum + cp.int32(0)
            n = thread.shfl_xor(v, 16)
            v = v + n
            n = thread.shfl_xor(v, 8)
            v = v + n
            n = thread.shfl_xor(v, 4)
            v = v + n
            n = thread.shfl_xor(v, 2)
            v = v + n
            n = thread.shfl_xor(v, 1)
            v = v + n
            if tid == cp.int32(0):
                totalDiffuseOut[0] = v


# ===== Kernel 3: ps_updateUnsortedDiffuseArrayLaunch =====
# Host resolves per-buffer. bufferOffset is computed via warp reduction
# over mNumDiffuseParticles[0] of all previous buffers.
# For simplicity, host pre-computes bufferOffset and passes it directly.
@cp.kernel
def ps_updateUnsortedDiffuseArrayLaunch(
    unsortedPositions, unsortedVels,
    bufferPositions, bufferVelocities,
    numDiffuseParticles, bufferOffset, bufferStartIndex,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Copy from per-buffer sorted arrays to the unsorted global arrays."""
    with cp.Kernel(cp.ceildiv(numDiffuseParticles, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gid = bx * BLOCK_SIZE + tid
            if gid < numDiffuseParticles:
                if gid == cp.int32(0):
                    bufferStartIndex[0] = bufferOffset
                ind = bufferOffset + gid
                # Copy float4 position
                unsortedPositions[ind, 0] = bufferPositions[gid, 0]
                unsortedPositions[ind, 1] = bufferPositions[gid, 1]
                unsortedPositions[ind, 2] = bufferPositions[gid, 2]
                unsortedPositions[ind, 3] = bufferPositions[gid, 3]
                # Copy float4 velocity
                unsortedVels[ind, 0] = bufferVelocities[gid, 0]
                unsortedVels[ind, 1] = bufferVelocities[gid, 1]
                unsortedVels[ind, 2] = bufferVelocities[gid, 2]
                unsortedVels[ind, 3] = bufferVelocities[gid, 3]


# ===== Kernel 4: ps_diffuseParticleOneWayCollision =====
# Host decomposes PxgParticleSystem fields to flat args.
@cp.kernel
def ps_diffuseParticleOneWayCollision(
    newPos, contactNormalPen, contactCounts,
    numParticles, maxContactsPerParticle,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Apply one-way collision corrections to diffuse particle positions.
    newPos: float32[N, 4] — sorted diffuse positions (read/write).
    contactNormalPen: float32[N*maxC, 4] — contact normals + penetration.
    contactCounts: int32[N] — number of contacts per particle.
    """
    with cp.Kernel(cp.ceildiv(numParticles, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            pi = bx * BLOCK_SIZE + tid
            if pi < numParticles:
                contactCount = contactCounts[pi] + cp.int32(0)
                maxC = maxContactsPerParticle + cp.int32(0)
                if contactCount > maxC:
                    contactCount = maxC

                if contactCount > cp.int32(0):
                    # Load current position
                    px = newPos[pi, 0] + cp.float32(0.0)
                    py = newPos[pi, 1] + cp.float32(0.0)
                    pz = newPos[pi, 2] + cp.float32(0.0)

                    offset = pi + cp.int32(0)
                    for _c in range(16):  # max unroll; guard by contactCount
                        if cp.int32(_c) < contactCount:
                            nx = contactNormalPen[offset, 0] + cp.float32(0.0)
                            ny = contactNormalPen[offset, 1] + cp.float32(0.0)
                            nz = contactNormalPen[offset, 2] + cp.float32(0.0)
                            pen = contactNormalPen[offset, 3] + cp.float32(0.0)
                            # deltaP = -surfaceNormal * penetration
                            px = px - nx * pen
                            py = py - ny * pen
                            pz = pz - nz * pen
                            offset = offset + numParticles

                    newPos[pi, 0] = px
                    newPos[pi, 1] = py
                    newPos[pi, 2] = pz


# ===== Kernel 5: ps_diffuseParticleCreate =====
# Host decomposes PxgParticleSystem + PxgParticleDiffuseSimBuffer fields.
@cp.kernel
def ps_diffuseParticleCreate(
    sortedPositions, sortedVelocities, sortedPhases,
    diffusePotentials,
    diffusePositionsNew, diffuseVelocitiesNew,
    numDiffuseParticles,
    unsortedToSortedMapping, particleBufferRunsumOffset,
    randomTable, randomTableSize,
    numParticles, maxNumDiffuseParticles,
    threshold, kineticEnergyWeight, divergenceWeight, pressureWeight,
    lifetime, restOffset, dt,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Create new diffuse particles based on kinetic energy and pressure potentials."""
    with cp.Kernel(cp.ceildiv(numParticles, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            pi = bx * BLOCK_SIZE + tid
            if pi < numParticles:
                sortedInd = unsortedToSortedMapping[pi + particleBufferRunsumOffset] + cp.int32(0)

                # Check phase (PxGetFluid: bit 20)
                phase = sortedPhases[sortedInd] + cp.int32(0)
                isFluid = phase & cp.int32(0x100000)  # ePARTICLE_PHASE_FLUID = 1 << 20

                if isFluid != cp.int32(0):
                    # Get potentials
                    div_val = diffusePotentials[sortedInd, 0] + cp.float32(0.0)
                    pres_val = diffusePotentials[sortedInd, 1] + cp.float32(0.0)

                    # Kinetic energy
                    vx = sortedVelocities[sortedInd, 0] + cp.float32(0.0)
                    vy = sortedVelocities[sortedInd, 1] + cp.float32(0.0)
                    vz = sortedVelocities[sortedInd, 2] + cp.float32(0.0)
                    ke = (vx * vx + vy * vy + vz * vz) * kineticEnergyWeight
                    divergence = divergenceWeight * div_val
                    pressure = pressureWeight * pres_val
                    intensity = pressure - divergence + ke

                    r0_idx = sortedInd % randomTableSize
                    r0 = randomTable[r0_idx] + cp.float32(0.0)

                    if r0 * intensity > threshold:
                        newIndex = thread.atomic_add(numDiffuseParticles[1], cp.int32(1))

                        if newIndex < maxNumDiffuseParticles:
                            xi_x = sortedPositions[sortedInd, 0] + cp.float32(0.0)
                            xi_y = sortedPositions[sortedInd, 1] + cp.float32(0.0)
                            xi_z = sortedPositions[sortedInd, 2] + cp.float32(0.0)

                            r1_idx = (sortedInd + cp.int32(1)) % randomTableSize
                            r2_idx = (sortedInd + cp.int32(2)) % randomTableSize
                            r3_idx = (sortedInd + cp.int32(3)) % randomTableSize
                            r1 = randomTable[r1_idx] + cp.float32(0.0)
                            r2 = randomTable[r2_idx] + cp.float32(0.0)
                            r3 = randomTable[r3_idx] + cp.float32(0.0)

                            lifeMin = cp.float32(1.0)
                            lifeMax = lifetime
                            intRatio = intensity / threshold
                            if intRatio > cp.float32(1.0):
                                intRatio = cp.float32(1.0)
                            lifeScale = intRatio * r1
                            lt = lifeMin + lifeScale * (lifeMax - lifeMin)

                            qx = xi_x - r2 * vx * dt + r1 * restOffset * cp.float32(0.25)
                            qy = xi_y - r2 * vy * dt + r2 * restOffset * cp.float32(0.25)
                            qz = xi_z - r2 * vz * dt + r3 * restOffset * cp.float32(0.25)

                            diffusePositionsNew[newIndex, 0] = qx
                            diffusePositionsNew[newIndex, 1] = qy
                            diffusePositionsNew[newIndex, 2] = qz
                            diffusePositionsNew[newIndex, 3] = lt
                            diffuseVelocitiesNew[newIndex, 0] = vx
                            diffuseVelocitiesNew[newIndex, 1] = vy
                            diffuseVelocitiesNew[newIndex, 2] = vz
                            diffuseVelocitiesNew[newIndex, 3] = cp.float32(0.0)


# ===== Kernel 6: ps_diffuseParticleUpdatePBF =====
# Grid neighbor iteration with goto replaced by done-flag.
@cp.kernel
def ps_diffuseParticleUpdatePBF(
    cellStarts, cellEnds,
    sortedPositions, sortedVelocities,
    diffusePositions, newVelOut,
    numDiffuse,
    cellWidth, contactDistanceSq, contactDistanceInv,
    gridSizeX, gridSizeY, gridSizeZ,
    fullAdvectionEnd,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Interpolate velocity from neighbors for diffuse particles (PBF-style).
    cellStarts/cellEnds: int32[gridCells] — cell start/end indices.
    sortedPositions/sortedVelocities: float32[N, 4] — sorted particle data.
    diffusePositions: float32[M, 4] — diffuse particle positions (read).
    newVelOut: float32[M, 4] — output velocity average + neighbor count.
    fullAdvectionEnd: int (3 if full advection, 1 if not).
    """
    with cp.Kernel(cp.ceildiv(numDiffuse, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            pi = bx * BLOCK_SIZE + tid
            if pi < numDiffuse:
                xi_x = diffusePositions[pi, 0] + cp.float32(0.0)
                xi_y = diffusePositions[pi, 1] + cp.float32(0.0)
                xi_z = diffusePositions[pi, 2] + cp.float32(0.0)

                weightSum = cp.float32(0.0)
                velSumX = cp.float32(0.0)
                velSumY = cp.float32(0.0)
                velSumZ = cp.float32(0.0)
                numNeighbors = cp.int32(0)
                done = cp.int32(0)

                gx, gy, gz = calcGridPos(xi_x, xi_y, xi_z, cellWidth)

                # Offsets: [0, -1, 1]
                for dz in range(3):
                    for dy in range(3):
                        for dx in range(3):
                            if done == cp.int32(0):
                                if dx < fullAdvectionEnd:
                                    if dy < fullAdvectionEnd:
                                        if dz < fullAdvectionEnd:
                                            ox = cp.int32(0)
                                            if dx == cp.int32(1):
                                                ox = cp.int32(-1)
                                            if dx == cp.int32(2):
                                                ox = cp.int32(1)
                                            oy = cp.int32(0)
                                            if dy == cp.int32(1):
                                                oy = cp.int32(-1)
                                            if dy == cp.int32(2):
                                                oy = cp.int32(1)
                                            oz = cp.int32(0)
                                            if dz == cp.int32(1):
                                                oz = cp.int32(-1)
                                            if dz == cp.int32(2):
                                                oz = cp.int32(1)

                                            nx = gx + ox
                                            ny = gy + oy
                                            nz = gz + oz
                                            gridHash = calcGridHash(nx, ny, nz, gridSizeX, gridSizeY, gridSizeZ)
                                            startIndex = cellStarts[gridHash] + cp.int32(0)

                                            if startIndex != cp.int32(EMPTY_CELL):
                                                endIndex = cellEnds[gridHash] + cp.int32(0)
                                                q = startIndex
                                                while q < endIndex:
                                                    if done == cp.int32(0):
                                                        xj_x = sortedPositions[q, 0] + cp.float32(0.0)
                                                        xj_y = sortedPositions[q, 1] + cp.float32(0.0)
                                                        xj_z = sortedPositions[q, 2] + cp.float32(0.0)
                                                        dxx = xi_x - xj_x
                                                        dyy = xi_y - xj_y
                                                        dzz = xi_z - xj_z
                                                        dSq = dxx * dxx + dyy * dyy + dzz * dzz

                                                        if dSq < contactDistanceSq:
                                                            vj_x = sortedVelocities[q, 0] + cp.float32(0.0)
                                                            vj_y = sortedVelocities[q, 1] + cp.float32(0.0)
                                                            vj_z = sortedVelocities[q, 2] + cp.float32(0.0)
                                                            dist = thread.sqrt(dSq)
                                                            w = WDiffuse(dist, contactDistanceInv)
                                                            weightSum = weightSum + w
                                                            velSumX = velSumX + vj_x * w
                                                            velSumY = velSumY + vj_y * w
                                                            velSumZ = velSumZ + vj_z * w
                                                            numNeighbors = numNeighbors + cp.int32(1)
                                                            if numNeighbors == cp.int32(16):
                                                                done = cp.int32(1)
                                                    q = q + cp.int32(1)

                # Compute average velocity
                avgX = cp.float32(0.0)
                avgY = cp.float32(0.0)
                avgZ = cp.float32(0.0)
                if weightSum > cp.float32(0.0):
                    invW = cp.float32(1.0) / weightSum
                    avgX = velSumX * invW
                    avgY = velSumY * invW
                    avgZ = velSumZ * invW

                newVelOut[pi, 0] = avgX
                newVelOut[pi, 1] = avgY
                newVelOut[pi, 2] = avgZ
                newVelOut[pi, 3] = cp.float32(numNeighbors)


# ===== Kernel 7: ps_diffuseParticleCompact =====
# Stream compaction with ballot/popc/shfl for live diffuse particles.
# Host resolves per-buffer. Warp-level operations for compaction.
@cp.kernel
def ps_diffuseParticleCompact(
    diffusePositionsNew, diffuseVelocitiesNew,
    velAvgs, diffusePositions, diffusePositionsOld,
    reverseLookup, numDiffuseParticles,
    bufferStartIndex, maxVelocity,
    airDrag, buoyancy, bubbleDrag,
    grav_x, grav_y, grav_z, dt,
    BLOCK_SIZE: cp.constexpr = 256,
    NUM_WARPS: cp.constexpr = 8
):
    """Compact live diffuse particles: integrate, apply physics, write survivors.
    diffusePositionsNew/diffuseVelocitiesNew: float32[M, 4] — output arrays.
    velAvgs: float32[N, 4] — velocity averages from PBF (w = neighbor count).
    diffusePositions: float32[N, 4] — current sorted positions.
    diffusePositionsOld: float32[N, 4] — original unsorted positions.
    reverseLookup: int32[N] — unsorted-to-sorted mapping.
    numDiffuseParticles: int32[2] — [0]=current count, [1]=staging count.
    """
    with cp.Kernel(cp.ceildiv(numDiffuseParticles[0], BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        # Need warp scope for ballot
        for warp_id, warp in block.warps():
            for lane, thread in warp.threads():
                tid = warp_id * cp.int32(WARP_SIZE) + lane
                pi = bx * BLOCK_SIZE + tid
                numDiffuse = numDiffuseParticles[0] + cp.int32(0)

                # Default values for out-of-range threads
                newPosX = cp.float32(0.0)
                newPosY = cp.float32(0.0)
                newPosZ = cp.float32(0.0)
                newVelX = cp.float32(0.0)
                newVelY = cp.float32(0.0)
                newVelZ = cp.float32(0.0)
                lifeTime = cp.float32(0.0)

                if pi < numDiffuse:
                    index = pi + bufferStartIndex
                    sortedInd = reverseLookup[index] + cp.int32(0)

                    # Load data
                    xi_x = diffusePositions[sortedInd, 0] + cp.float32(0.0)
                    xi_y = diffusePositions[sortedInd, 1] + cp.float32(0.0)
                    xi_z = diffusePositions[sortedInd, 2] + cp.float32(0.0)
                    xi_w = diffusePositions[sortedInd, 3] + cp.float32(0.0)

                    old_x = diffusePositionsOld[index, 0] + cp.float32(0.0)
                    old_y = diffusePositionsOld[index, 1] + cp.float32(0.0)
                    old_z = diffusePositionsOld[index, 2] + cp.float32(0.0)

                    va_x = velAvgs[sortedInd, 0] + cp.float32(0.0)
                    va_y = velAvgs[sortedInd, 1] + cp.float32(0.0)
                    va_z = velAvgs[sortedInd, 2] + cp.float32(0.0)
                    va_w = velAvgs[sortedInd, 3] + cp.float32(0.0)

                    invDt = cp.float32(1.0) / dt
                    vel_x = (xi_x - old_x) * invDt
                    vel_y = (xi_y - old_y) * invDt
                    vel_z = (xi_z - old_z) * invDt

                    # Physics integration based on neighbor count (va_w)
                    nv_x = cp.float32(0.0)
                    nv_y = cp.float32(0.0)
                    nv_z = cp.float32(0.0)

                    if va_w < cp.float32(4.0):
                        # Spray (ballistic)
                        drag = cp.float32(1.0) - airDrag * dt
                        nv_x = vel_x * drag
                        nv_y = vel_y * drag
                        nv_z = vel_z * drag
                    elif va_w < cp.float32(8.0):
                        # Foam
                        nv_x = va_x
                        nv_y = va_y
                        nv_z = va_z
                    else:
                        # Bubble
                        bfactor = cp.float32(1.0) + buoyancy
                        nv_x = vel_x - bfactor * grav_x * dt + bubbleDrag * (va_x - vel_x)
                        nv_y = vel_y - bfactor * grav_y * dt + bubbleDrag * (va_y - vel_y)
                        nv_z = vel_z - bfactor * grav_z * dt + bubbleDrag * (va_z - vel_z)

                    # Clamp velocity
                    magSq = nv_x * nv_x + nv_y * nv_y + nv_z * nv_z
                    if magSq > cp.float32(0.0):
                        mag = thread.sqrt(magSq)
                        clampedMag = mag
                        if mag > maxVelocity:
                            clampedMag = maxVelocity
                        scale = clampedMag / mag
                        nv_x = nv_x * scale
                        nv_y = nv_y * scale
                        nv_z = nv_z * scale

                    newPosX = xi_x + (nv_x - vel_x) * dt
                    newPosY = xi_y + (nv_y - vel_y) * dt
                    newPosZ = xi_z + (nv_z - vel_z) * dt
                    newVelX = nv_x
                    newVelY = nv_y
                    newVelZ = nv_z

                    lifeTime = xi_w - dt
                    if lifeTime < cp.float32(0.0):
                        lifeTime = cp.float32(0.0)

                # Ballot for alive particles
                alive = cp.int32(1) if lifeTime > cp.float32(0.0) else cp.int32(0)
                # In-range check
                if pi >= numDiffuse:
                    alive = cp.int32(0)

                res = thread.coll.ballot(alive != cp.int32(0))
                popcount = thread.popcount(res)

                # Lane 0 atomically reserves space
                offset = cp.int32(0)
                if lane == cp.int32(0):
                    offset = thread.atomic_add(numDiffuseParticles[1], popcount)
                offset = thread.shfl_idx(offset, 0)

                # Compute exclusive scan within warp via popcount of mask
                laneMask = (cp.int32(1) << lane) - cp.int32(1)
                warpOffset = thread.popcount(res & laneMask)

                if alive != cp.int32(0):
                    newIndex = offset + warpOffset
                    diffusePositionsNew[newIndex, 0] = newPosX
                    diffusePositionsNew[newIndex, 1] = newPosY
                    diffusePositionsNew[newIndex, 2] = newPosZ
                    diffusePositionsNew[newIndex, 3] = lifeTime
                    diffuseVelocitiesNew[newIndex, 0] = newVelX
                    diffuseVelocitiesNew[newIndex, 1] = newVelY
                    diffuseVelocitiesNew[newIndex, 2] = newVelZ
                    diffuseVelocitiesNew[newIndex, 3] = cp.float32(0.0)
