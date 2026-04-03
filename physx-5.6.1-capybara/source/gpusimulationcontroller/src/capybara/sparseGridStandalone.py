"""Capybara DSL port of gpusimulationcontroller/CUDA/sparseGridStandalone.cu — all 10 kernels.

Ported kernels:
  - sg_SparseGridClearDensity
  - sg_MarkSubgridEndIndices
  - sg_SparseGridSortedArrayToDelta
  - sg_SparseGridCalcSubgridHashes
  - sg_SparseGridGetUniqueValues
  - sg_SparseGridBuildSubgridNeighbors
  - sg_ReuseSubgrids
  - sg_AddReleasedSubgridsToUnusedStack
  - sg_AllocateNewSubgrids
  - sg_SparseGridMarkRequiredNeighbors

ABI differences from CUDA:
  - PxSparseGridParams struct decomposed to 5 scalar args:
    sg_maxNumSubgrids, sg_gridSpacing, sg_subgridSizeX/Y/Z.
    haloSize hardcoded to 0 (matches all CUDA call sites).
  - NULL pointer args replaced by integer flags + always-valid dummy tensors.
  - PxVec4* passed as float32[N, 4] tensors.
  - int3/int4 device helpers return tuples of scalars.
  - PxU32* pointing to single device values passed as int32[1] tensors.
"""

import capybara as cp

# ===== Constants =====
MAX_SPARSEGRID_DIM = 1024
MIN_SPARSEGRID_ID = -512
MAX_SPARSEGRID_ID = 511
EMPTY_SUBGRID = 0xFFFFFFFF
NEW_SUBGRID = 0xFFFFFFFE
REUSED_SUBGRID = 0xFFFFFFFD
SUBGRID_CENTER_IDX = 13
WARP_SIZE = 32


# ===== Inlined helper functions from sparseGridStandalone.cuh =====

@cp.inline
def getSubgridDomainSize(sg_gridSpacing, sg_subgridSizeX, sg_subgridSizeY, sg_subgridSizeZ):
    """Returns (domX, domY, domZ) — subgrid domain size with haloSize=0."""
    dx = sg_gridSpacing
    domX = dx * cp.float32(sg_subgridSizeX)
    domY = dx * cp.float32(sg_subgridSizeY)
    domZ = dx * cp.float32(sg_subgridSizeZ)
    return domX, domY, domZ


@cp.inline
def floor_to_int(val):
    """Floor a float to int32 (toward negative infinity)."""
    truncated = cp.int32(val)
    ft = cp.float32(truncated)
    # If val < truncated (negative non-integer), subtract 1
    correction = cp.int32(0)
    if val < ft:
        correction = cp.int32(1)
    return truncated - correction


@cp.inline
def calcSubgridId(pos_x, pos_y, pos_z, domX, domY, domZ):
    """Returns (ix, iy, iz) — subgrid id from position and domain size."""
    ix = floor_to_int(pos_x / domX)
    iy = floor_to_int(pos_y / domY)
    iz = floor_to_int(pos_z / domZ)
    return ix, iy, iz


@cp.inline
def calcSubgridHash(sx, sy, sz):
    """Returns hash from subgrid id (sx, sy, sz)."""
    shifted_x = sx - cp.int32(MIN_SPARSEGRID_ID)
    shifted_y = sy - cp.int32(MIN_SPARSEGRID_ID)
    shifted_z = sz - cp.int32(MIN_SPARSEGRID_ID)
    return cp.int32(MAX_SPARSEGRID_DIM) * cp.int32(MAX_SPARSEGRID_DIM) * shifted_z + cp.int32(MAX_SPARSEGRID_DIM) * shifted_y + shifted_x


@cp.inline
def subgridHashOffset(sx, sy, sz, offX, offY, offZ):
    """Returns hash of (sx+offX, sy+offY, sz+offZ)."""
    return calcSubgridHash(sx + offX, sy + offY, sz + offZ)


@cp.inline
def subgridHashToId(hashKey):
    """Returns (ix, iy, iz, iw=0) from hash key."""
    ih = cp.int32(hashKey)
    ix = ih % cp.int32(MAX_SPARSEGRID_DIM) + cp.int32(MIN_SPARSEGRID_ID)
    iy = (ih // cp.int32(MAX_SPARSEGRID_DIM)) % cp.int32(MAX_SPARSEGRID_DIM) + cp.int32(MIN_SPARSEGRID_ID)
    iz = ih // (cp.int32(MAX_SPARSEGRID_DIM) * cp.int32(MAX_SPARSEGRID_DIM)) + cp.int32(MIN_SPARSEGRID_ID)
    return ix, iy, iz, cp.int32(0)


@cp.inline
def subgridNeighborIndex(x, y, z):
    """Returns linear neighbor index from offsets (-1/0/1)."""
    return (x + cp.int32(1)) + cp.int32(3) * (y + cp.int32(1)) + cp.int32(9) * (z + cp.int32(1))


@cp.inline
def isSubgridInsideRange(sx, sy, sz):
    """Returns 1 if (sx,sy,sz) is within valid sparse grid range, else 0."""
    inside = cp.int32(1)
    if sx < cp.int32(MIN_SPARSEGRID_ID):
        inside = cp.int32(0)
    if sx > cp.int32(MAX_SPARSEGRID_ID):
        inside = cp.int32(0)
    if sy < cp.int32(MIN_SPARSEGRID_ID):
        inside = cp.int32(0)
    if sy > cp.int32(MAX_SPARSEGRID_ID):
        inside = cp.int32(0)
    if sz < cp.int32(MIN_SPARSEGRID_ID):
        inside = cp.int32(0)
    if sz > cp.int32(MAX_SPARSEGRID_ID):
        inside = cp.int32(0)
    return inside


@cp.inline
def searchSorted(data, numElements, value):
    """Binary search: returns index of last element <= value.
    Equivalent to CUDA searchSorted in sparseGridStandalone.cuh.
    """
    left = cp.int32(0)
    right = cp.int32(numElements)
    while (right - left) > cp.int32(1):
        pos = (left + right) >> cp.int32(1)
        element = data[pos] + cp.int32(0)
        if element <= value:
            left = pos
        else:
            right = pos
    return left


@cp.inline
def tryFindHashkey(sortedHashkey, numSubgrids, hashToFind):
    """Returns (found, result_idx). found=1 if hash is present, 0 otherwise."""
    result = searchSorted(sortedHashkey, numSubgrids, hashToFind)
    found_val = sortedHashkey[result] + cp.int32(0)
    found = cp.int32(1) if found_val == hashToFind else cp.int32(0)
    return found, result


# ===== Kernel 1: sg_SparseGridClearDensity =====
@cp.kernel
def sg_SparseGridClearDensity(density, clearValue, numActiveSubgrids, subgridSize,
                               BLOCK_SIZE: cp.constexpr = 256):
    with cp.Kernel(cp.ceildiv(numActiveSubgrids[0] * subgridSize, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            idx = bx * BLOCK_SIZE + tid
            total = numActiveSubgrids[0] * subgridSize
            if idx < total:
                density[idx] = clearValue


# ===== Kernel 2: sg_MarkSubgridEndIndices =====
@cp.kernel
def sg_MarkSubgridEndIndices(sortedParticleToSubgrid, numParticles, subgridEndIndices,
                              BLOCK_SIZE: cp.constexpr = 256):
    with cp.Kernel(cp.ceildiv(numParticles, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            threadIndex = bx * BLOCK_SIZE + tid
            if threadIndex < numParticles:
                cur = sortedParticleToSubgrid[threadIndex] + cp.int32(0)
                if threadIndex < numParticles - cp.int32(1):
                    nxt = sortedParticleToSubgrid[threadIndex + cp.int32(1)] + cp.int32(0)
                    if cur != nxt:
                        subgridEndIndices[cur] = threadIndex + cp.int32(1)
                else:
                    subgridEndIndices[cur] = numParticles


# ===== Kernel 3: sg_SparseGridSortedArrayToDelta =====
@cp.kernel
def sg_SparseGridSortedArrayToDelta(in_data, mask, out, n, has_mask,
                                     BLOCK_SIZE: cp.constexpr = 256):
    with cp.Kernel(cp.ceildiv(n, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            i = bx * BLOCK_SIZE + tid
            if i < n:
                result = cp.int32(0)
                if i < n - cp.int32(1):
                    cur = in_data[i] + cp.int32(0)
                    nxt = in_data[i + cp.int32(1)] + cp.int32(0)
                    if cur != nxt:
                        result = cp.int32(1)
                        if has_mask != cp.int32(0):
                            result = mask[i] + cp.int32(0)
                else:
                    result = cp.int32(1)
                    if has_mask != cp.int32(0):
                        result = mask[i] + cp.int32(0)
                out[i] = result


# ===== Kernel 4: sg_SparseGridCalcSubgridHashes =====
@cp.kernel
def sg_SparseGridCalcSubgridHashes(
    sg_maxNumSubgrids, sg_gridSpacing, sg_subgridSizeX, sg_subgridSizeY, sg_subgridSizeZ,
    indices, hashkeyPerParticle, positions,
    numParticles, phases, validPhaseMask, activeIndices,
    has_phases, has_activeIndices,
    BLOCK_SIZE: cp.constexpr = 256
):
    with cp.Kernel(cp.ceildiv(numParticles, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            p = bx * BLOCK_SIZE + tid
            if p < numParticles:
                # Resolve active index
                actual_p = p + cp.int32(0)
                if has_activeIndices != cp.int32(0):
                    actual_p = activeIndices[p] + cp.int32(0)

                domX, domY, domZ = getSubgridDomainSize(sg_gridSpacing, sg_subgridSizeX, sg_subgridSizeY, sg_subgridSizeZ)

                pos_x = positions[actual_p, 0]
                pos_y = positions[actual_p, 1]
                pos_z = positions[actual_p, 2]

                sx, sy, sz = calcSubgridId(pos_x, pos_y, pos_z, domX, domY, domZ)

                isValidPhase = cp.int32(1)
                if has_phases != cp.int32(0):
                    phase_val = phases[actual_p] + cp.int32(0)
                    if (phase_val & validPhaseMask) == cp.int32(0):
                        isValidPhase = cp.int32(0)

                indices[actual_p] = actual_p
                if isValidPhase != cp.int32(0):
                    hashkeyPerParticle[actual_p] = calcSubgridHash(sx, sy, sz)
                else:
                    hashkeyPerParticle[actual_p] = cp.int32(EMPTY_SUBGRID)


# ===== Kernel 5: sg_SparseGridGetUniqueValues =====
@cp.kernel
def sg_SparseGridGetUniqueValues(
    sortedData, scan_indices, uniqueValues, n,
    subgridNeighborCollector, uniqueValuesSize,
    has_subgridNeighborCollector,
    BLOCK_SIZE: cp.constexpr = 256
):
    with cp.Kernel(cp.ceildiv(n, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            i = bx * BLOCK_SIZE + tid
            if i < n:
                cur_idx = scan_indices[i] + cp.int32(0)
                is_boundary = cp.int32(0)
                if i == n - cp.int32(1):
                    is_boundary = cp.int32(1)
                else:
                    nxt_idx = scan_indices[i + cp.int32(1)] + cp.int32(0)
                    if cur_idx != nxt_idx:
                        is_boundary = cp.int32(1)

                if is_boundary != cp.int32(0):
                    if cur_idx < uniqueValuesSize:
                        sd = sortedData[i] + cp.int32(0)
                        uniqueValues[cur_idx] = sd

                        if has_subgridNeighborCollector != cp.int32(0):
                            id_x, id_y, id_z, _id_w = subgridHashToId(sd)
                            indexer = cp.int32(27) * cur_idx
                            # 3x3x3 neighbor loop (unrolled with range)
                            for di in range(3):
                                for dj in range(3):
                                    for dk in range(3):
                                        ni = id_x + cp.int32(di - 1)
                                        nj = id_y + cp.int32(dj - 1)
                                        nk = id_z + cp.int32(dk - 1)
                                        subgridNeighborCollector[indexer] = calcSubgridHash(ni, nj, nk)
                                        indexer = indexer + cp.int32(1)


# ===== Kernel 6: sg_SparseGridBuildSubgridNeighbors =====
@cp.kernel
def sg_SparseGridBuildSubgridNeighbors(
    uniqueSortedHashkey, numActiveSubgrids, maxNumSubgrids, subgridNeighbors,
    BLOCK_SIZE: cp.constexpr = 256
):
    with cp.Kernel(cp.ceildiv(maxNumSubgrids, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            si = bx * BLOCK_SIZE + tid
            if si < maxNumSubgrids:
                hash_val = uniqueSortedHashkey[si] + cp.int32(0)
                s_x, s_y, s_z, _s_w = subgridHashToId(hash_val)

                subgridNeighbors[cp.int32(27) * si + cp.int32(SUBGRID_CENTER_IDX)] = si

                numActive = numActiveSubgrids[0] + cp.int32(0)

                for dz in range(3):
                    for dy in range(3):
                        for dx in range(3):
                            ox = cp.int32(dx - 1)
                            oy = cp.int32(dy - 1)
                            oz = cp.int32(dz - 1)
                            n_x = s_x + ox
                            n_y = s_y + oy
                            n_z = s_z + oz
                            nHash = calcSubgridHash(n_x, n_y, n_z)

                            n_val = cp.int32(EMPTY_SUBGRID)
                            if isSubgridInsideRange(n_x, n_y, n_z) != cp.int32(0):
                                found, nSortedIdx = tryFindHashkey(uniqueSortedHashkey, numActive, nHash)
                                if found != cp.int32(0):
                                    n_val = nSortedIdx

                            nidx = subgridNeighborIndex(ox, oy, oz)
                            subgridNeighbors[cp.int32(27) * si + nidx] = n_val


# ===== Kernel 7: sg_ReuseSubgrids =====
@cp.kernel
def sg_ReuseSubgrids(
    sg_maxNumSubgrids,
    uniqueHashkeysPerSubgridPreviousUpdate, numActiveSubgridsPreviousUpdate,
    subgridOrderMapPreviousUpdate,
    uniqueHashkeysPerSubgrid, numActiveSubgrids,
    subgridOrderMap,
    BLOCK_SIZE: cp.constexpr = 256
):
    with cp.Kernel(cp.ceildiv(sg_maxNumSubgrids, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            threadIndex = bx * BLOCK_SIZE + tid
            if threadIndex < sg_maxNumSubgrids:
                numActive = numActiveSubgrids[0] + cp.int32(0)

                if threadIndex >= numActive:
                    subgridOrderMap[threadIndex] = cp.int32(EMPTY_SUBGRID)
                else:
                    hashkey = uniqueHashkeysPerSubgrid[threadIndex] + cp.int32(0)
                    numActivePrev = numActiveSubgridsPreviousUpdate[0] + cp.int32(0)
                    found, sortedIdx = tryFindHashkey(uniqueHashkeysPerSubgridPreviousUpdate, numActivePrev, hashkey)
                    if found == cp.int32(0):
                        subgridOrderMap[threadIndex] = cp.int32(NEW_SUBGRID)
                    else:
                        prev_order = subgridOrderMapPreviousUpdate[sortedIdx] + cp.int32(0)
                        subgridOrderMap[threadIndex] = prev_order
                        subgridOrderMapPreviousUpdate[sortedIdx] = cp.int32(REUSED_SUBGRID)


# ===== Kernel 8: sg_AddReleasedSubgridsToUnusedStack =====
# Note: host must pass numActivePrev as a scalar (read from numActiveSubgridsPreviousUpdate[0]).
@cp.kernel
def sg_AddReleasedSubgridsToUnusedStack(
    numActivePrev, subgridOrderMapPreviousUpdate,
    unusedSubgridStackSize, unusedSubgridStack,
    BLOCK_SIZE: cp.constexpr = 256
):
    with cp.Kernel(cp.ceildiv(numActivePrev, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            threadIndex = bx * BLOCK_SIZE + tid
            if threadIndex < numActivePrev:
                orderVal = subgridOrderMapPreviousUpdate[threadIndex] + cp.int32(0)
                if orderVal != cp.int32(REUSED_SUBGRID):
                    slot = thread.atomic_add(unusedSubgridStackSize[0], cp.int32(1))
                    unusedSubgridStack[slot] = orderVal


# ===== Kernel 9: sg_AllocateNewSubgrids =====
# Note: host must pass numActive and numActivePrev as scalars.
@cp.kernel
def sg_AllocateNewSubgrids(
    numActive, subgridOrderMap,
    unusedSubgridStackSize, unusedSubgridStack,
    numActivePrev, maxNumSubgrids,
    BLOCK_SIZE: cp.constexpr = 256
):
    with cp.Kernel(cp.ceildiv(numActive, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            threadIndex = bx * BLOCK_SIZE + tid
            if threadIndex < numActive:
                if numActivePrev == cp.int32(0):
                    # Special first-frame case: deterministic assignment
                    numActiveSubgridsClamped = numActive
                    if numActive > maxNumSubgrids:
                        numActiveSubgridsClamped = maxNumSubgrids
                    subgridOrderMap[threadIndex] = unusedSubgridStack[maxNumSubgrids - numActiveSubgridsClamped + threadIndex]
                    if threadIndex == cp.int32(0):
                        unusedSubgridStackSize[0] = unusedSubgridStackSize[0] - numActiveSubgridsClamped
                else:
                    orderVal = subgridOrderMap[threadIndex] + cp.int32(0)
                    if orderVal == cp.int32(NEW_SUBGRID):
                        # Atomic decrement to pop from stack
                        slot = thread.atomic_add(unusedSubgridStackSize[0], cp.int32(-1))
                        subgridOrderMap[threadIndex] = unusedSubgridStack[slot - cp.int32(1)]


# ===== Kernel 10: sg_SparseGridMarkRequiredNeighbors =====
# Inlined device function applyMask from sparseGridStandalone.cu.
# CUDA version uses while loops to mark duplicate hash entries.
# Capybara version: same while loop structure.
@cp.inline
def applyMask(requiredNeighborMask, uniqueSortedHashkey, hashkey, maxNumSubgrids):
    """Mark all entries in the sorted hashkey array matching hashkey."""
    if hashkey != cp.int32(EMPTY_SUBGRID):
        searchLimit = cp.int32(27) * maxNumSubgrids
        found, sortedIdx = tryFindHashkey(uniqueSortedHashkey, searchLimit, hashkey)
        if found != cp.int32(0):
            already = requiredNeighborMask[sortedIdx] + cp.int32(0)
            if already != cp.int32(1):
                requiredNeighborMask[sortedIdx] = cp.int32(1)

                # Mark duplicates to the left
                j = sortedIdx - cp.int32(1)
                while j >= cp.int32(0):
                    v = uniqueSortedHashkey[j] + cp.int32(0)
                    if v != hashkey:
                        j = cp.int32(-1)  # break
                    else:
                        requiredNeighborMask[j] = cp.int32(1)
                        j = j - cp.int32(1)

                # Mark duplicates to the right
                k = sortedIdx + cp.int32(1)
                while k < searchLimit:
                    v = uniqueSortedHashkey[k] + cp.int32(0)
                    if v != hashkey:
                        k = searchLimit  # break
                    else:
                        requiredNeighborMask[k] = cp.int32(1)
                        k = k + cp.int32(1)


@cp.kernel
def sg_SparseGridMarkRequiredNeighbors(
    requiredNeighborMask, uniqueSortedHashkey,
    sg_maxNumSubgrids, sg_gridSpacing, sg_subgridSizeX, sg_subgridSizeY, sg_subgridSizeZ,
    neighborhoodSize,
    particlePositions, numParticles,
    phases, validPhaseMask, activeIndices,
    has_phases, has_activeIndices,
    BLOCK_SIZE: cp.constexpr = 256
):
    with cp.Kernel(cp.ceildiv(numParticles, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            i = bx * BLOCK_SIZE + tid
            if i < numParticles:
                actual_i = i + cp.int32(0)
                if has_activeIndices != cp.int32(0):
                    actual_i = activeIndices[i] + cp.int32(0)

                # Phase check
                skip = cp.int32(0)
                if has_phases != cp.int32(0):
                    phase_val = phases[actual_i] + cp.int32(0)
                    if (phase_val & validPhaseMask) == cp.int32(0):
                        skip = cp.int32(1)

                if skip == cp.int32(0):
                    xp_x = particlePositions[actual_i, 0]
                    xp_y = particlePositions[actual_i, 1]
                    xp_z = particlePositions[actual_i, 2]

                    domX, domY, domZ = getSubgridDomainSize(sg_gridSpacing, sg_subgridSizeX, sg_subgridSizeY, sg_subgridSizeZ)
                    sx, sy, sz = calcSubgridId(xp_x, xp_y, xp_z, domX, domY, domZ)
                    dx = sg_gridSpacing
                    invDx = cp.float32(1.0) / dx

                    # Compute subgrid origin (haloSize=0)
                    originX = cp.float32(sx) * dx * cp.float32(sg_subgridSizeX)
                    originY = cp.float32(sy) * dx * cp.float32(sg_subgridSizeY)
                    originZ = cp.float32(sz) * dx * cp.float32(sg_subgridSizeZ)
                    localX = xp_x - originX
                    localY = xp_y - originY
                    localZ = xp_z - originZ

                    # Grid base coord (clamped)
                    baseX_raw = floor_to_int(localX * invDx)
                    baseY_raw = floor_to_int(localY * invDx)
                    baseZ_raw = floor_to_int(localZ * invDx)
                    sizeX_m1 = sg_subgridSizeX - cp.int32(1)
                    sizeY_m1 = sg_subgridSizeY - cp.int32(1)
                    sizeZ_m1 = sg_subgridSizeZ - cp.int32(1)
                    # Clamp baseX to [0, sizeX_m1]
                    baseX = baseX_raw
                    if baseX < cp.int32(0):
                        baseX = cp.int32(0)
                    if baseX > sizeX_m1:
                        baseX = sizeX_m1
                    baseY = baseY_raw
                    if baseY < cp.int32(0):
                        baseY = cp.int32(0)
                    if baseY > sizeY_m1:
                        baseY = sizeY_m1
                    baseZ = baseZ_raw
                    if baseZ < cp.int32(0):
                        baseZ = cp.int32(0)
                    if baseZ > sizeZ_m1:
                        baseZ = sizeZ_m1

                    # Determine step in each axis (-1, 0, or 1)
                    stepX = cp.int32(0)
                    if baseX < neighborhoodSize:
                        stepX = cp.int32(-1)
                    if baseX >= sg_subgridSizeX - neighborhoodSize:
                        stepX = cp.int32(1)
                    stepY = cp.int32(0)
                    if baseY < neighborhoodSize:
                        stepY = cp.int32(-1)
                    if baseY >= sg_subgridSizeY - neighborhoodSize:
                        stepY = cp.int32(1)
                    stepZ = cp.int32(0)
                    if baseZ < neighborhoodSize:
                        stepZ = cp.int32(-1)
                    if baseZ >= sg_subgridSizeZ - neighborhoodSize:
                        stepZ = cp.int32(1)

                    # Build buffer of up to 8 hashes (using scalars instead of array)
                    buf0 = calcSubgridHash(sx, sy, sz)
                    count = cp.int32(1)

                    buf1 = cp.int32(0)
                    buf2 = cp.int32(0)
                    buf3 = cp.int32(0)
                    buf4 = cp.int32(0)
                    buf5 = cp.int32(0)
                    buf6 = cp.int32(0)
                    buf7 = cp.int32(0)

                    # Corner neighbor (all 3 axes differ)
                    if stepX != cp.int32(0):
                        if stepY != cp.int32(0):
                            if stepZ != cp.int32(0):
                                buf1 = subgridHashOffset(sx, sy, sz, stepX, stepY, stepZ)
                                count = cp.int32(2)

                    # Edge neighbors (2 axes differ)
                    if stepX != cp.int32(0):
                        if stepY != cp.int32(0):
                            if count == cp.int32(2):
                                buf2 = subgridHashOffset(sx, sy, sz, stepX, stepY, cp.int32(0))
                            else:
                                buf1 = subgridHashOffset(sx, sy, sz, stepX, stepY, cp.int32(0))
                            count = count + cp.int32(1)
                    if stepX != cp.int32(0):
                        if stepZ != cp.int32(0):
                            if count == cp.int32(3):
                                buf3 = subgridHashOffset(sx, sy, sz, stepX, cp.int32(0), stepZ)
                            elif count == cp.int32(2):
                                buf2 = subgridHashOffset(sx, sy, sz, stepX, cp.int32(0), stepZ)
                            else:
                                buf1 = subgridHashOffset(sx, sy, sz, stepX, cp.int32(0), stepZ)
                            count = count + cp.int32(1)
                    if stepY != cp.int32(0):
                        if stepZ != cp.int32(0):
                            if count == cp.int32(4):
                                buf4 = subgridHashOffset(sx, sy, sz, cp.int32(0), stepY, stepZ)
                            elif count == cp.int32(3):
                                buf3 = subgridHashOffset(sx, sy, sz, cp.int32(0), stepY, stepZ)
                            elif count == cp.int32(2):
                                buf2 = subgridHashOffset(sx, sy, sz, cp.int32(0), stepY, stepZ)
                            else:
                                buf1 = subgridHashOffset(sx, sy, sz, cp.int32(0), stepY, stepZ)
                            count = count + cp.int32(1)

                    # Face neighbors (1 axis differs)
                    if stepX != cp.int32(0):
                        if count == cp.int32(5):
                            buf5 = subgridHashOffset(sx, sy, sz, stepX, cp.int32(0), cp.int32(0))
                        elif count == cp.int32(4):
                            buf4 = subgridHashOffset(sx, sy, sz, stepX, cp.int32(0), cp.int32(0))
                        elif count == cp.int32(3):
                            buf3 = subgridHashOffset(sx, sy, sz, stepX, cp.int32(0), cp.int32(0))
                        elif count == cp.int32(2):
                            buf2 = subgridHashOffset(sx, sy, sz, stepX, cp.int32(0), cp.int32(0))
                        else:
                            buf1 = subgridHashOffset(sx, sy, sz, stepX, cp.int32(0), cp.int32(0))
                        count = count + cp.int32(1)
                    if stepY != cp.int32(0):
                        if count == cp.int32(6):
                            buf6 = subgridHashOffset(sx, sy, sz, cp.int32(0), stepY, cp.int32(0))
                        elif count == cp.int32(5):
                            buf5 = subgridHashOffset(sx, sy, sz, cp.int32(0), stepY, cp.int32(0))
                        elif count == cp.int32(4):
                            buf4 = subgridHashOffset(sx, sy, sz, cp.int32(0), stepY, cp.int32(0))
                        elif count == cp.int32(3):
                            buf3 = subgridHashOffset(sx, sy, sz, cp.int32(0), stepY, cp.int32(0))
                        elif count == cp.int32(2):
                            buf2 = subgridHashOffset(sx, sy, sz, cp.int32(0), stepY, cp.int32(0))
                        else:
                            buf1 = subgridHashOffset(sx, sy, sz, cp.int32(0), stepY, cp.int32(0))
                        count = count + cp.int32(1)
                    if stepZ != cp.int32(0):
                        if count == cp.int32(7):
                            buf7 = subgridHashOffset(sx, sy, sz, cp.int32(0), cp.int32(0), stepZ)
                        elif count == cp.int32(6):
                            buf6 = subgridHashOffset(sx, sy, sz, cp.int32(0), cp.int32(0), stepZ)
                        elif count == cp.int32(5):
                            buf5 = subgridHashOffset(sx, sy, sz, cp.int32(0), cp.int32(0), stepZ)
                        elif count == cp.int32(4):
                            buf4 = subgridHashOffset(sx, sy, sz, cp.int32(0), cp.int32(0), stepZ)
                        elif count == cp.int32(3):
                            buf3 = subgridHashOffset(sx, sy, sz, cp.int32(0), cp.int32(0), stepZ)
                        elif count == cp.int32(2):
                            buf2 = subgridHashOffset(sx, sy, sz, cp.int32(0), cp.int32(0), stepZ)
                        else:
                            buf1 = subgridHashOffset(sx, sy, sz, cp.int32(0), cp.int32(0), stepZ)
                        count = count + cp.int32(1)

                    # Apply mask for each buffered hash
                    applyMask(requiredNeighborMask, uniqueSortedHashkey, buf0, sg_maxNumSubgrids)
                    if count > cp.int32(1):
                        applyMask(requiredNeighborMask, uniqueSortedHashkey, buf1, sg_maxNumSubgrids)
                    if count > cp.int32(2):
                        applyMask(requiredNeighborMask, uniqueSortedHashkey, buf2, sg_maxNumSubgrids)
                    if count > cp.int32(3):
                        applyMask(requiredNeighborMask, uniqueSortedHashkey, buf3, sg_maxNumSubgrids)
                    if count > cp.int32(4):
                        applyMask(requiredNeighborMask, uniqueSortedHashkey, buf4, sg_maxNumSubgrids)
                    if count > cp.int32(5):
                        applyMask(requiredNeighborMask, uniqueSortedHashkey, buf5, sg_maxNumSubgrids)
                    if count > cp.int32(6):
                        applyMask(requiredNeighborMask, uniqueSortedHashkey, buf6, sg_maxNumSubgrids)
                    if count > cp.int32(7):
                        applyMask(requiredNeighborMask, uniqueSortedHashkey, buf7, sg_maxNumSubgrids)
