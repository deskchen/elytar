"""Capybara DSL port of gpusimulationcontroller/CUDA/anisotropy.cu — all 4 kernels.

Ported kernels:
  - smoothPositionsLaunch
  - calculateAnisotropyLaunch
  - anisotropyKernel
  - smoothPositionsKernel

ABI differences from CUDA:
  - PxGpuParticleSystem decomposed to per-kernel flat tensor/scalar args
    (kernels 1,2). Host resolves pointer members before launch.
  - PxSmoothedPositionData decomposed: mPositions -> tensor, mSmoothing -> scalar.
  - PxAnisotropyData decomposed: q1/q2/q3 -> tensors, scalars for bounds.
  - PxMat33 represented as 9 named float scalars (not a struct).
  - eigenDecomposition (Jacobi rotation) inlined as @cp.inline operating on
    9 scalars. 3 template instantiations of jacobiRotateT unrolled.
  - float4 positions/velocities as float32[N, 4] tensors.
  - NULL phases pointer -> has_phases int flag + dummy tensor.
  - PxGetFluid(phase) = (phase & 0x100000) -> inline bit test.
  - subgridNeighborOffset reused from sparseGridStandalone.cuh (re-inlined).
"""

import capybara as cp

WARP_SIZE = 32
EMPTY_SUBGRID = 0xFFFFFFFF


# ===== Inline helpers =====

@cp.inline
def cube(x):
    return x * x * x


@cp.inline
def Wa(x, invr):
    """Weighting kernel: 1 - (x*invr)^3."""
    return cp.float32(1.0) - cube(x * invr)


@cp.inline
def clamp_f(v, lo, hi):
    """Clamp float to [lo, hi]."""
    r = v
    if r < lo:
        r = lo
    if r > hi:
        r = hi
    return r


# ===== subgridNeighborOffset (re-inlined from sparseGridStandalone.cuh) =====

@cp.inline
def subgridNeighborIndex(x, y, z):
    return (x + cp.int32(1)) + cp.int32(3) * (y + cp.int32(1)) + cp.int32(9) * (z + cp.int32(1))


@cp.inline
def subgridNeighborOffset(subgridNeighbors, si, offsetX, offsetY, offsetZ):
    idx = cp.int32(27) * si + subgridNeighborIndex(offsetX, offsetY, offsetZ)
    return subgridNeighbors[idx] + cp.int32(0)


# ===== Kernel 1: smoothPositionsLaunch =====
# Host decomposes PxGpuParticleSystem + PxSmoothedPositionData fields.
@cp.kernel
def smoothPositionsLaunch(
    sortedPositions, sortedPhases, collisionIndex,
    gridParticleIndices, particleSelfCollisionCount,
    smoothPosOrig, numParticles,
    particleContactDistanceSq, particleContactDistanceInv, particleContactDistance,
    smoothing,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Smooth particle positions using neighbor weighted averaging.
    sortedPositions: float32[N, 4] — sorted particle positions + invMass.
    smoothPosOrig: float32[N, 4] — output smoothed positions (API order).
    collisionIndex: int32[N * maxNeighbors] — neighbor indices.
    """
    with cp.Kernel(cp.ceildiv(numParticles, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            p = bx * BLOCK_SIZE + tid
            if p < numParticles:
                origIdx = gridParticleIndices[p] + cp.int32(0)

                phase = sortedPhases[p] + cp.int32(0)
                isFluid = phase & cp.int32(0x100000)

                # Non-fluid: just copy position
                if isFluid == cp.int32(0):
                    smoothPosOrig[origIdx, 0] = sortedPositions[p, 0]
                    smoothPosOrig[origIdx, 1] = sortedPositions[p, 1]
                    smoothPosOrig[origIdx, 2] = sortedPositions[p, 2]
                    smoothPosOrig[origIdx, 3] = sortedPositions[p, 3]
                else:
                    xi_x = sortedPositions[p, 0] + cp.float32(0.0)
                    xi_y = sortedPositions[p, 1] + cp.float32(0.0)
                    xi_z = sortedPositions[p, 2] + cp.float32(0.0)
                    xi_w = sortedPositions[p, 3] + cp.float32(0.0)

                    contactCount = particleSelfCollisionCount[p] + cp.int32(0)

                    xs_x = cp.float32(0.0)
                    xs_y = cp.float32(0.0)
                    xs_z = cp.float32(0.0)
                    ws = cp.float32(0.0)

                    offset = p + cp.int32(0)
                    i = cp.int32(0)
                    while i < contactCount:
                        q = collisionIndex[offset] + cp.int32(0)
                        qphase = sortedPhases[q] + cp.int32(0)
                        qFluid = qphase & cp.int32(0x100000)
                        if qFluid != cp.int32(0):
                            xj_x = sortedPositions[q, 0] + cp.float32(0.0)
                            xj_y = sortedPositions[q, 1] + cp.float32(0.0)
                            xj_z = sortedPositions[q, 2] + cp.float32(0.0)
                            dx = xi_x - xj_x
                            dy = xi_y - xj_y
                            dz = xi_z - xj_z
                            dsq = dx * dx + dy * dy + dz * dz
                            if dsq > cp.float32(0.0):
                                if dsq < particleContactDistanceSq:
                                    w = Wa(thread.sqrt(dsq), particleContactDistanceInv)
                                    ws = ws + w
                                    xs_x = xs_x + xj_x * w
                                    xs_y = xs_y + xj_y * w
                                    xs_z = xs_z + xj_z * w
                        offset = offset + numParticles
                        i = i + cp.int32(1)

                    out_x = xi_x
                    out_y = xi_y
                    out_z = xi_z
                    if ws > cp.float32(0.0):
                        f = cp.float32(4.0) * Wa(particleContactDistance * cp.float32(0.5), particleContactDistanceInv)
                        ratio = ws / f
                        if ratio > cp.float32(1.0):
                            ratio = cp.float32(1.0)
                        smooth = ratio * smoothing
                        invWs = cp.float32(1.0) / ws
                        avg_x = xs_x * invWs
                        avg_y = xs_y * invWs
                        avg_z = xs_z * invWs
                        out_x = xi_x + (avg_x - xi_x) * smooth
                        out_y = xi_y + (avg_y - xi_y) * smooth
                        out_z = xi_z + (avg_z - xi_z) * smooth

                    smoothPosOrig[origIdx, 0] = out_x
                    smoothPosOrig[origIdx, 1] = out_y
                    smoothPosOrig[origIdx, 2] = out_z
                    smoothPosOrig[origIdx, 3] = xi_w


# ===== Kernel 2: calculateAnisotropyLaunch =====
# Host decomposes PxGpuParticleSystem + PxAnisotropyData fields.
@cp.kernel
def calculateAnisotropyLaunch(
    sortedPositions, sortedPhases, collisionIndex,
    gridParticleIndices, particleSelfCollisionCount,
    q1, q2, q3, numParticles,
    particleContactDistanceSq, particleContactDistanceInv, particleContactDistance,
    anisotropyScale, anisotropyMin, anisotropyMax,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Compute anisotropy ellipsoids via covariance eigen decomposition.
    q1/q2/q3: float32[N, 4] — output anisotropy direction+magnitude per axis.
    """
    with cp.Kernel(cp.ceildiv(numParticles, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            p = bx * BLOCK_SIZE + tid
            if p < numParticles:
                origIdx = gridParticleIndices[p] + cp.int32(0)
                phase = sortedPhases[p] + cp.int32(0)
                isFluid = phase & cp.int32(0x100000)

                kmin = anisotropyMin * particleContactDistance
                kmax = anisotropyMax * particleContactDistance

                if isFluid == cp.int32(0):
                    # Non-fluid: identity axes with min radius
                    q1[origIdx, 0] = cp.float32(1.0)
                    q1[origIdx, 1] = cp.float32(0.0)
                    q1[origIdx, 2] = cp.float32(0.0)
                    q1[origIdx, 3] = kmin
                    q2[origIdx, 0] = cp.float32(0.0)
                    q2[origIdx, 1] = cp.float32(1.0)
                    q2[origIdx, 2] = cp.float32(0.0)
                    q2[origIdx, 3] = kmin
                    q3[origIdx, 0] = cp.float32(0.0)
                    q3[origIdx, 1] = cp.float32(0.0)
                    q3[origIdx, 2] = cp.float32(1.0)
                    q3[origIdx, 3] = kmin
                else:
                    xi_x = sortedPositions[p, 0] + cp.float32(0.0)
                    xi_y = sortedPositions[p, 1] + cp.float32(0.0)
                    xi_z = sortedPositions[p, 2] + cp.float32(0.0)

                    contactCount = particleSelfCollisionCount[p] + cp.int32(0)

                    # Pass 1: weighted average position
                    xs_x = cp.float32(0.0)
                    xs_y = cp.float32(0.0)
                    xs_z = cp.float32(0.0)
                    ws = cp.float32(0.0)

                    offset = p + cp.int32(0)
                    i = cp.int32(0)
                    while i < contactCount:
                        q_idx = collisionIndex[offset] + cp.int32(0)
                        qphase = sortedPhases[q_idx] + cp.int32(0)
                        if (qphase & cp.int32(0x100000)) != cp.int32(0):
                            xj_x = sortedPositions[q_idx, 0] + cp.float32(0.0)
                            xj_y = sortedPositions[q_idx, 1] + cp.float32(0.0)
                            xj_z = sortedPositions[q_idx, 2] + cp.float32(0.0)
                            dx = xi_x - xj_x
                            dy = xi_y - xj_y
                            dz = xi_z - xj_z
                            dsq = dx * dx + dy * dy + dz * dz
                            if dsq > cp.float32(0.0):
                                if dsq < particleContactDistanceSq:
                                    w = Wa(thread.sqrt(dsq), particleContactDistanceInv)
                                    ws = ws + w
                                    xs_x = xs_x + xj_x * w
                                    xs_y = xs_y + xj_y * w
                                    xs_z = xs_z + xj_z * w
                        offset = offset + numParticles
                        i = i + cp.int32(1)

                    if ws == cp.float32(0.0):
                        # Isolated particle: identity axes
                        q1[origIdx, 0] = cp.float32(1.0)
                        q1[origIdx, 1] = cp.float32(0.0)
                        q1[origIdx, 2] = cp.float32(0.0)
                        q1[origIdx, 3] = kmin
                        q2[origIdx, 0] = cp.float32(0.0)
                        q2[origIdx, 1] = cp.float32(1.0)
                        q2[origIdx, 2] = cp.float32(0.0)
                        q2[origIdx, 3] = kmin
                        q3[origIdx, 0] = cp.float32(0.0)
                        q3[origIdx, 1] = cp.float32(0.0)
                        q3[origIdx, 2] = cp.float32(1.0)
                        q3[origIdx, 3] = kmin
                    else:
                        invWs = cp.float32(1.0) / ws
                        xs_x = xs_x * invWs
                        xs_y = xs_y * invWs
                        xs_z = xs_z * invWs

                        # Pass 2: covariance matrix (9 scalars, symmetric)
                        c00 = cp.float32(0.0)
                        c01 = cp.float32(0.0)
                        c02 = cp.float32(0.0)
                        c10 = cp.float32(0.0)
                        c11 = cp.float32(0.0)
                        c12 = cp.float32(0.0)
                        c20 = cp.float32(0.0)
                        c21 = cp.float32(0.0)
                        c22 = cp.float32(0.0)

                        offset2 = p + cp.int32(0)
                        j = cp.int32(0)
                        while j < contactCount:
                            q_idx2 = collisionIndex[offset2] + cp.int32(0)
                            qphase2 = sortedPhases[q_idx2] + cp.int32(0)
                            if (qphase2 & cp.int32(0x100000)) != cp.int32(0):
                                xj_x2 = sortedPositions[q_idx2, 0] + cp.float32(0.0)
                                xj_y2 = sortedPositions[q_idx2, 1] + cp.float32(0.0)
                                xj_z2 = sortedPositions[q_idx2, 2] + cp.float32(0.0)
                                dx2 = xi_x - xj_x2
                                dy2 = xi_y - xj_y2
                                dz2 = xi_z - xj_z2
                                dsq2 = dx2 * dx2 + dy2 * dy2 + dz2 * dz2
                                if dsq2 > cp.float32(0.0):
                                    if dsq2 < particleContactDistanceSq:
                                        w2 = Wa(thread.sqrt(dsq2), particleContactDistanceInv)
                                        # xjs = xj - xs (weighted avg)
                                        sx = xj_x2 - xs_x
                                        sy = xj_y2 - xs_y
                                        sz = xj_z2 - xs_z
                                        # outer product: w * xjs * xjs^T
                                        wsx = w2 * sx
                                        c00 = c00 + wsx * sx
                                        c01 = c01 + wsx * sy
                                        c02 = c02 + wsx * sz
                                        wsy = w2 * sy
                                        c10 = c10 + wsy * sx
                                        c11 = c11 + wsy * sy
                                        c12 = c12 + wsy * sz
                                        wsz = w2 * sz
                                        c20 = c20 + wsz * sx
                                        c21 = c21 + wsz * sy
                                        c22 = c22 + wsz * sz
                            offset2 = offset2 + numParticles
                            j = j + cp.int32(1)

                        # Scale covariance by invWs
                        c00 = c00 * invWs
                        c01 = c01 * invWs
                        c02 = c02 * invWs
                        c10 = c10 * invWs
                        c11 = c11 * invWs
                        c12 = c12 * invWs
                        c20 = c20 * invWs
                        c21 = c21 * invWs
                        c22 = c22 * invWs

                        # Eigen decomposition (4 Jacobi iterations)
                        # R starts as identity
                        r00 = cp.float32(1.0)
                        r01 = cp.float32(0.0)
                        r02 = cp.float32(0.0)
                        r10 = cp.float32(0.0)
                        r11 = cp.float32(1.0)
                        r12 = cp.float32(0.0)
                        r20 = cp.float32(0.0)
                        r21 = cp.float32(0.0)
                        r22 = cp.float32(1.0)

                        epsilon = cp.float32(1e-15)

                        for _iter in range(4):
                            # Find largest off-diagonal |A(0,1)|, |A(0,2)|, |A(1,2)|
                            abs01 = c10
                            if c10 < cp.float32(0.0):
                                abs01 = -c10
                            abs02 = c20
                            if c20 < cp.float32(0.0):
                                abs02 = -c20
                            abs12 = c21
                            if c21 < cp.float32(0.0):
                                abs12 = -c21

                            max_val = abs01
                            best_j = cp.int32(0)
                            if abs02 > max_val:
                                best_j = cp.int32(1)
                                max_val = abs02
                            if abs12 > max_val:
                                best_j = cp.int32(2)
                                max_val = abs12

                            # Pre-declare variables used across if/elif/else branches
                            apq = cp.float32(0.0)
                            d = cp.float32(0.0)
                            abs_d = cp.float32(0.0)
                            t_val = cp.float32(0.0)
                            c_rot = cp.float32(0.0)
                            s_rot = cp.float32(0.0)
                            Akp = cp.float32(0.0)
                            Akq = cp.float32(0.0)
                            Rkp0 = cp.float32(0.0)
                            Rkq0 = cp.float32(0.0)
                            Rkp1 = cp.float32(0.0)
                            Rkq1 = cp.float32(0.0)
                            Rkp2 = cp.float32(0.0)
                            Rkq2 = cp.float32(0.0)

                            if max_val >= epsilon:
                                if best_j == cp.int32(0):
                                    # jacobiRotateT<0,1,2>: apq=c10, app=c00, aqq=c11
                                    apq = c10 + cp.float32(0.0)
                                    if apq != cp.float32(0.0):
                                        d = (c00 - c11) / (cp.float32(2.0) * apq)
                                        abs_d = d
                                        if d < cp.float32(0.0):
                                            abs_d = -d
                                        t_val = cp.float32(1.0) / (abs_d + thread.sqrt(d * d + cp.float32(1.0)))
                                        if d < cp.float32(0.0):
                                            t_val = -t_val
                                        c_rot = thread.rsqrt(t_val * t_val + cp.float32(1.0))
                                        s_rot = t_val * c_rot
                                        c00 = c00 + t_val * apq
                                        c11 = c11 - t_val * apq
                                        c10 = cp.float32(0.0)
                                        c01 = cp.float32(0.0)
                                        # Transform A: k=2
                                        Akp = c_rot * c02 + s_rot * c12
                                        Akq = -s_rot * c02 + c_rot * c12
                                        c02 = Akp
                                        c20 = Akp
                                        c12 = Akq
                                        c21 = Akq
                                        # Rotate R columns p=0, q=1
                                        Rkp0 = c_rot * r00 + s_rot * r10
                                        Rkq0 = -s_rot * r00 + c_rot * r10
                                        r00 = Rkp0
                                        r10 = Rkq0
                                        Rkp1 = c_rot * r01 + s_rot * r11
                                        Rkq1 = -s_rot * r01 + c_rot * r11
                                        r01 = Rkp1
                                        r11 = Rkq1
                                        Rkp2 = c_rot * r02 + s_rot * r12
                                        Rkq2 = -s_rot * r02 + c_rot * r12
                                        r02 = Rkp2
                                        r12 = Rkq2
                                elif best_j == cp.int32(1):
                                    # jacobiRotateT<0,2,1>: apq=c20, app=c00, aqq=c22
                                    apq = c20 + cp.float32(0.0)
                                    if apq != cp.float32(0.0):
                                        d = (c00 - c22) / (cp.float32(2.0) * apq)
                                        abs_d = d
                                        if d < cp.float32(0.0):
                                            abs_d = -d
                                        t_val = cp.float32(1.0) / (abs_d + thread.sqrt(d * d + cp.float32(1.0)))
                                        if d < cp.float32(0.0):
                                            t_val = -t_val
                                        c_rot = thread.rsqrt(t_val * t_val + cp.float32(1.0))
                                        s_rot = t_val * c_rot
                                        c00 = c00 + t_val * apq
                                        c22 = c22 - t_val * apq
                                        c20 = cp.float32(0.0)
                                        c02 = cp.float32(0.0)
                                        # Transform A: k=1
                                        Akp = c_rot * c01 + s_rot * c21
                                        Akq = -s_rot * c01 + c_rot * c21
                                        c01 = Akp
                                        c10 = Akp
                                        c21 = Akq
                                        c12 = Akq
                                        # Rotate R columns p=0, q=2
                                        Rkp0 = c_rot * r00 + s_rot * r20
                                        Rkq0 = -s_rot * r00 + c_rot * r20
                                        r00 = Rkp0
                                        r20 = Rkq0
                                        Rkp1 = c_rot * r01 + s_rot * r21
                                        Rkq1 = -s_rot * r01 + c_rot * r21
                                        r01 = Rkp1
                                        r21 = Rkq1
                                        Rkp2 = c_rot * r02 + s_rot * r22
                                        Rkq2 = -s_rot * r02 + c_rot * r22
                                        r02 = Rkp2
                                        r22 = Rkq2
                                else:
                                    # jacobiRotateT<1,2,0>: apq=c21, app=c11, aqq=c22
                                    apq = c21 + cp.float32(0.0)
                                    if apq != cp.float32(0.0):
                                        d = (c11 - c22) / (cp.float32(2.0) * apq)
                                        abs_d = d
                                        if d < cp.float32(0.0):
                                            abs_d = -d
                                        t_val = cp.float32(1.0) / (abs_d + thread.sqrt(d * d + cp.float32(1.0)))
                                        if d < cp.float32(0.0):
                                            t_val = -t_val
                                        c_rot = thread.rsqrt(t_val * t_val + cp.float32(1.0))
                                        s_rot = t_val * c_rot
                                        c11 = c11 + t_val * apq
                                        c22 = c22 - t_val * apq
                                        c21 = cp.float32(0.0)
                                        c12 = cp.float32(0.0)
                                        # Transform A: k=0
                                        Akp = c_rot * c10 + s_rot * c20
                                        Akq = -s_rot * c10 + c_rot * c20
                                        c10 = Akp
                                        c01 = Akp
                                        c20 = Akq
                                        c02 = Akq
                                        # Rotate R columns p=1, q=2
                                        Rkp0 = c_rot * r10 + s_rot * r20
                                        Rkq0 = -s_rot * r10 + c_rot * r20
                                        r10 = Rkp0
                                        r20 = Rkq0
                                        Rkp1 = c_rot * r11 + s_rot * r21
                                        Rkq1 = -s_rot * r11 + c_rot * r21
                                        r11 = Rkp1
                                        r21 = Rkq1
                                        Rkp2 = c_rot * r12 + s_rot * r22
                                        Rkq2 = -s_rot * r12 + c_rot * r22
                                        r12 = Rkp2
                                        r22 = Rkq2

                        # Eigenvalues are diagonal of covariance
                        if c00 < cp.float32(0.0):
                            c00 = cp.float32(0.0)
                        if c11 < cp.float32(0.0):
                            c11 = cp.float32(0.0)
                        if c22 < cp.float32(0.0):
                            c22 = cp.float32(0.0)

                        lam0 = thread.sqrt(c00) * anisotropyScale
                        lam1 = thread.sqrt(c11) * anisotropyScale
                        lam2 = thread.sqrt(c22) * anisotropyScale

                        lam0 = clamp_f(lam0, kmin, kmax)
                        lam1 = clamp_f(lam1, kmin, kmax)
                        lam2 = clamp_f(lam2, kmin, kmax)

                        # Write eigenvectors (columns of R) + eigenvalues
                        q1[origIdx, 0] = r00
                        q1[origIdx, 1] = r01
                        q1[origIdx, 2] = r02
                        q1[origIdx, 3] = lam0
                        q2[origIdx, 0] = r10
                        q2[origIdx, 1] = r11
                        q2[origIdx, 2] = r12
                        q2[origIdx, 3] = lam1
                        q3[origIdx, 0] = r20
                        q3[origIdx, 1] = r21
                        q3[origIdx, 2] = r22
                        q3[origIdx, 3] = lam2


# ===== Kernel 3: anisotropyKernel =====
# Uses flat args (no PxGpuParticleSystem). Uses subgrid neighbor iteration.
@cp.kernel
def anisotropyKernel(
    deviceParticlePos, sortedToOriginalParticleIndex,
    sortedParticleToSubgrid, maxNumSubgrids,
    subgridNeighbors, subgridEndIndices,
    numParticles, phases, validPhaseMask, has_phases,
    q1, q2, q3,
    anisotropyScale, anisotropyMin, anisotropyMax,
    particleContactDistance,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Anisotropy via subgrid neighbor iteration + Jacobi eigen decomposition."""
    with cp.Kernel(cp.ceildiv(numParticles, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            threadIndex = bx * BLOCK_SIZE + tid
            if threadIndex < numParticles:
                pNr = sortedToOriginalParticleIndex[threadIndex] + cp.int32(0)
                subgridIndex = sortedParticleToSubgrid[threadIndex] + cp.int32(0)

                kmin = anisotropyMin * particleContactDistance
                kmax = anisotropyMax * particleContactDistance
                pcdSq = particleContactDistance * particleContactDistance
                pcdInv = cp.float32(1.0) / particleContactDistance

                skipPhase = cp.int32(0)
                if has_phases != cp.int32(0):
                    pval = phases[pNr] + cp.int32(0)
                    if (pval & validPhaseMask) == cp.int32(0):
                        skipPhase = cp.int32(1)

                if subgridIndex >= maxNumSubgrids:
                    skipPhase = cp.int32(1)

                if skipPhase != cp.int32(0):
                    q1[pNr, 0] = cp.float32(1.0)
                    q1[pNr, 1] = cp.float32(0.0)
                    q1[pNr, 2] = cp.float32(0.0)
                    q1[pNr, 3] = kmin
                    q2[pNr, 0] = cp.float32(0.0)
                    q2[pNr, 1] = cp.float32(1.0)
                    q2[pNr, 2] = cp.float32(0.0)
                    q2[pNr, 3] = kmin
                    q3[pNr, 0] = cp.float32(0.0)
                    q3[pNr, 1] = cp.float32(0.0)
                    q3[pNr, 2] = cp.float32(1.0)
                    q3[pNr, 3] = kmin
                else:
                    xi_x = deviceParticlePos[pNr, 0] + cp.float32(0.0)
                    xi_y = deviceParticlePos[pNr, 1] + cp.float32(0.0)
                    xi_z = deviceParticlePos[pNr, 2] + cp.float32(0.0)

                    # Pass 1: weighted average
                    xs_x = cp.float32(0.0)
                    xs_y = cp.float32(0.0)
                    xs_z = cp.float32(0.0)
                    ws = cp.float32(0.0)

                    for dz in range(3):
                        for dy in range(3):
                            for dx in range(3):
                                n = subgridNeighborOffset(subgridNeighbors, subgridIndex,
                                                          cp.int32(dx - 1), cp.int32(dy - 1), cp.int32(dz - 1))
                                if n != cp.int32(EMPTY_SUBGRID):
                                    start = cp.int32(0)
                                    if n > cp.int32(0):
                                        start = subgridEndIndices[n - cp.int32(1)] + cp.int32(0)
                                    end = subgridEndIndices[n] + cp.int32(0)
                                    ii = start
                                    while ii < end:
                                        jj = sortedToOriginalParticleIndex[ii] + cp.int32(0)
                                        validJ = cp.int32(1)
                                        if has_phases != cp.int32(0):
                                            jp = phases[jj] + cp.int32(0)
                                            if (jp & validPhaseMask) == cp.int32(0):
                                                validJ = cp.int32(0)
                                        if validJ != cp.int32(0):
                                            xj_x = deviceParticlePos[jj, 0] + cp.float32(0.0)
                                            xj_y = deviceParticlePos[jj, 1] + cp.float32(0.0)
                                            xj_z = deviceParticlePos[jj, 2] + cp.float32(0.0)
                                            ddx = xi_x - xj_x
                                            ddy = xi_y - xj_y
                                            ddz = xi_z - xj_z
                                            dsq = ddx * ddx + ddy * ddy + ddz * ddz
                                            if dsq > cp.float32(0.0):
                                                if dsq < pcdSq:
                                                    w = Wa(thread.sqrt(dsq), pcdInv)
                                                    ws = ws + w
                                                    xs_x = xs_x + xj_x * w
                                                    xs_y = xs_y + xj_y * w
                                                    xs_z = xs_z + xj_z * w
                                        ii = ii + cp.int32(1)

                    if ws == cp.float32(0.0):
                        q1[pNr, 0] = cp.float32(1.0)
                        q1[pNr, 1] = cp.float32(0.0)
                        q1[pNr, 2] = cp.float32(0.0)
                        q1[pNr, 3] = kmin
                        q2[pNr, 0] = cp.float32(0.0)
                        q2[pNr, 1] = cp.float32(1.0)
                        q2[pNr, 2] = cp.float32(0.0)
                        q2[pNr, 3] = kmin
                        q3[pNr, 0] = cp.float32(0.0)
                        q3[pNr, 1] = cp.float32(0.0)
                        q3[pNr, 2] = cp.float32(1.0)
                        q3[pNr, 3] = kmin
                    else:
                        invWs = cp.float32(1.0) / ws
                        xs_x = xs_x * invWs
                        xs_y = xs_y * invWs
                        xs_z = xs_z * invWs

                        # Pass 2: covariance
                        c00 = cp.float32(0.0)
                        c01 = cp.float32(0.0)
                        c02 = cp.float32(0.0)
                        c10 = cp.float32(0.0)
                        c11 = cp.float32(0.0)
                        c12 = cp.float32(0.0)
                        c20 = cp.float32(0.0)
                        c21 = cp.float32(0.0)
                        c22 = cp.float32(0.0)

                        for dz2 in range(3):
                            for dy2 in range(3):
                                for dx2 in range(3):
                                    n2 = subgridNeighborOffset(subgridNeighbors, subgridIndex,
                                                               cp.int32(dx2 - 1), cp.int32(dy2 - 1), cp.int32(dz2 - 1))
                                    if n2 != cp.int32(EMPTY_SUBGRID):
                                        start2 = cp.int32(0)
                                        if n2 > cp.int32(0):
                                            start2 = subgridEndIndices[n2 - cp.int32(1)] + cp.int32(0)
                                        end2 = subgridEndIndices[n2] + cp.int32(0)
                                        ii2 = start2
                                        while ii2 < end2:
                                            jj2 = sortedToOriginalParticleIndex[ii2] + cp.int32(0)
                                            validJ2 = cp.int32(1)
                                            if has_phases != cp.int32(0):
                                                jp2 = phases[jj2] + cp.int32(0)
                                                if (jp2 & validPhaseMask) == cp.int32(0):
                                                    validJ2 = cp.int32(0)
                                            if validJ2 != cp.int32(0):
                                                xj_x2 = deviceParticlePos[jj2, 0] + cp.float32(0.0)
                                                xj_y2 = deviceParticlePos[jj2, 1] + cp.float32(0.0)
                                                xj_z2 = deviceParticlePos[jj2, 2] + cp.float32(0.0)
                                                ddx2 = xi_x - xj_x2
                                                ddy2 = xi_y - xj_y2
                                                ddz2 = xi_z - xj_z2
                                                dsq2 = ddx2 * ddx2 + ddy2 * ddy2 + ddz2 * ddz2
                                                if dsq2 > cp.float32(0.0):
                                                    if dsq2 < pcdSq:
                                                        w2 = Wa(thread.sqrt(dsq2), pcdInv)
                                                        sx = xj_x2 - xs_x
                                                        sy = xj_y2 - xs_y
                                                        sz = xj_z2 - xs_z
                                                        wsx = w2 * sx
                                                        c00 = c00 + wsx * sx
                                                        c01 = c01 + wsx * sy
                                                        c02 = c02 + wsx * sz
                                                        wsy = w2 * sy
                                                        c10 = c10 + wsy * sx
                                                        c11 = c11 + wsy * sy
                                                        c12 = c12 + wsy * sz
                                                        wsz = w2 * sz
                                                        c20 = c20 + wsz * sx
                                                        c21 = c21 + wsz * sy
                                                        c22 = c22 + wsz * sz
                                            ii2 = ii2 + cp.int32(1)

                        c00 = c00 * invWs
                        c01 = c01 * invWs
                        c02 = c02 * invWs
                        c10 = c10 * invWs
                        c11 = c11 * invWs
                        c12 = c12 * invWs
                        c20 = c20 * invWs
                        c21 = c21 * invWs
                        c22 = c22 * invWs

                        # Eigen decomposition
                        r00 = cp.float32(1.0)
                        r01 = cp.float32(0.0)
                        r02 = cp.float32(0.0)
                        r10 = cp.float32(0.0)
                        r11 = cp.float32(1.0)
                        r12 = cp.float32(0.0)
                        r20 = cp.float32(0.0)
                        r21 = cp.float32(0.0)
                        r22 = cp.float32(1.0)
                        epsilon = cp.float32(1e-15)

                        for _iter in range(4):
                            abs01 = c10
                            if c10 < cp.float32(0.0):
                                abs01 = -c10
                            abs02 = c20
                            if c20 < cp.float32(0.0):
                                abs02 = -c20
                            abs12 = c21
                            if c21 < cp.float32(0.0):
                                abs12 = -c21
                            max_val = abs01
                            best_j = cp.int32(0)
                            if abs02 > max_val:
                                best_j = cp.int32(1)
                                max_val = abs02
                            if abs12 > max_val:
                                best_j = cp.int32(2)
                                max_val = abs12
                            # Pre-declare variables used across if/elif/else branches
                            apq = cp.float32(0.0)
                            d = cp.float32(0.0)
                            abs_d = cp.float32(0.0)
                            t_val = cp.float32(0.0)
                            c_rot = cp.float32(0.0)
                            s_rot = cp.float32(0.0)
                            Akp = cp.float32(0.0)
                            Akq = cp.float32(0.0)
                            Rkp0 = cp.float32(0.0)
                            Rkq0 = cp.float32(0.0)
                            Rkp1 = cp.float32(0.0)
                            Rkq1 = cp.float32(0.0)
                            Rkp2 = cp.float32(0.0)
                            Rkq2 = cp.float32(0.0)

                            if max_val >= epsilon:
                                if best_j == cp.int32(0):
                                    # jacobiRotateT<0,1,2>: apq=c10, app=c00, aqq=c11
                                    apq = c10 + cp.float32(0.0)
                                    if apq != cp.float32(0.0):
                                        d = (c00 - c11) / (cp.float32(2.0) * apq)
                                        abs_d = d
                                        if d < cp.float32(0.0):
                                            abs_d = -d
                                        t_val = cp.float32(1.0) / (abs_d + thread.sqrt(d * d + cp.float32(1.0)))
                                        if d < cp.float32(0.0):
                                            t_val = -t_val
                                        c_rot = thread.rsqrt(t_val * t_val + cp.float32(1.0))
                                        s_rot = t_val * c_rot
                                        c00 = c00 + t_val * apq
                                        c11 = c11 - t_val * apq
                                        c10 = cp.float32(0.0)
                                        c01 = cp.float32(0.0)
                                        # Transform A: k=2
                                        Akp = c_rot * c02 + s_rot * c12
                                        Akq = -s_rot * c02 + c_rot * c12
                                        c02 = Akp
                                        c20 = Akp
                                        c12 = Akq
                                        c21 = Akq
                                        # Rotate R columns p=0, q=1
                                        Rkp0 = c_rot * r00 + s_rot * r10
                                        Rkq0 = -s_rot * r00 + c_rot * r10
                                        r00 = Rkp0
                                        r10 = Rkq0
                                        Rkp1 = c_rot * r01 + s_rot * r11
                                        Rkq1 = -s_rot * r01 + c_rot * r11
                                        r01 = Rkp1
                                        r11 = Rkq1
                                        Rkp2 = c_rot * r02 + s_rot * r12
                                        Rkq2 = -s_rot * r02 + c_rot * r12
                                        r02 = Rkp2
                                        r12 = Rkq2
                                elif best_j == cp.int32(1):
                                    # jacobiRotateT<0,2,1>: apq=c20, app=c00, aqq=c22
                                    apq = c20 + cp.float32(0.0)
                                    if apq != cp.float32(0.0):
                                        d = (c00 - c22) / (cp.float32(2.0) * apq)
                                        abs_d = d
                                        if d < cp.float32(0.0):
                                            abs_d = -d
                                        t_val = cp.float32(1.0) / (abs_d + thread.sqrt(d * d + cp.float32(1.0)))
                                        if d < cp.float32(0.0):
                                            t_val = -t_val
                                        c_rot = thread.rsqrt(t_val * t_val + cp.float32(1.0))
                                        s_rot = t_val * c_rot
                                        c00 = c00 + t_val * apq
                                        c22 = c22 - t_val * apq
                                        c20 = cp.float32(0.0)
                                        c02 = cp.float32(0.0)
                                        # Transform A: k=1
                                        Akp = c_rot * c01 + s_rot * c21
                                        Akq = -s_rot * c01 + c_rot * c21
                                        c01 = Akp
                                        c10 = Akp
                                        c21 = Akq
                                        c12 = Akq
                                        # Rotate R columns p=0, q=2
                                        Rkp0 = c_rot * r00 + s_rot * r20
                                        Rkq0 = -s_rot * r00 + c_rot * r20
                                        r00 = Rkp0
                                        r20 = Rkq0
                                        Rkp1 = c_rot * r01 + s_rot * r21
                                        Rkq1 = -s_rot * r01 + c_rot * r21
                                        r01 = Rkp1
                                        r21 = Rkq1
                                        Rkp2 = c_rot * r02 + s_rot * r22
                                        Rkq2 = -s_rot * r02 + c_rot * r22
                                        r02 = Rkp2
                                        r22 = Rkq2
                                else:
                                    # jacobiRotateT<1,2,0>: apq=c21, app=c11, aqq=c22
                                    apq = c21 + cp.float32(0.0)
                                    if apq != cp.float32(0.0):
                                        d = (c11 - c22) / (cp.float32(2.0) * apq)
                                        abs_d = d
                                        if d < cp.float32(0.0):
                                            abs_d = -d
                                        t_val = cp.float32(1.0) / (abs_d + thread.sqrt(d * d + cp.float32(1.0)))
                                        if d < cp.float32(0.0):
                                            t_val = -t_val
                                        c_rot = thread.rsqrt(t_val * t_val + cp.float32(1.0))
                                        s_rot = t_val * c_rot
                                        c11 = c11 + t_val * apq
                                        c22 = c22 - t_val * apq
                                        c21 = cp.float32(0.0)
                                        c12 = cp.float32(0.0)
                                        # Transform A: k=0
                                        Akp = c_rot * c10 + s_rot * c20
                                        Akq = -s_rot * c10 + c_rot * c20
                                        c10 = Akp
                                        c01 = Akp
                                        c20 = Akq
                                        c02 = Akq
                                        # Rotate R columns p=1, q=2
                                        Rkp0 = c_rot * r10 + s_rot * r20
                                        Rkq0 = -s_rot * r10 + c_rot * r20
                                        r10 = Rkp0
                                        r20 = Rkq0
                                        Rkp1 = c_rot * r11 + s_rot * r21
                                        Rkq1 = -s_rot * r11 + c_rot * r21
                                        r11 = Rkp1
                                        r21 = Rkq1
                                        Rkp2 = c_rot * r12 + s_rot * r22
                                        Rkq2 = -s_rot * r12 + c_rot * r22
                                        r12 = Rkp2
                                        r22 = Rkq2

                        if c00 < cp.float32(0.0):
                            c00 = cp.float32(0.0)
                        if c11 < cp.float32(0.0):
                            c11 = cp.float32(0.0)
                        if c22 < cp.float32(0.0):
                            c22 = cp.float32(0.0)

                        lam0 = clamp_f(thread.sqrt(c00) * anisotropyScale, kmin, kmax)
                        lam1 = clamp_f(thread.sqrt(c11) * anisotropyScale, kmin, kmax)
                        lam2 = clamp_f(thread.sqrt(c22) * anisotropyScale, kmin, kmax)

                        q1[pNr, 0] = r00
                        q1[pNr, 1] = r01
                        q1[pNr, 2] = r02
                        q1[pNr, 3] = lam0
                        q2[pNr, 0] = r10
                        q2[pNr, 1] = r11
                        q2[pNr, 2] = r12
                        q2[pNr, 3] = lam1
                        q3[pNr, 0] = r20
                        q3[pNr, 1] = r21
                        q3[pNr, 2] = r22
                        q3[pNr, 3] = lam2


# ===== Kernel 4: smoothPositionsKernel =====
# Uses flat args. Subgrid neighbor iteration for position smoothing.
@cp.kernel
def smoothPositionsKernel(
    deviceParticlePos, sortedToOriginalParticleIndex,
    sortedParticleToSubgrid, maxNumSubgrids,
    subgridNeighbors, subgridEndIndices,
    numParticles, phases, validPhaseMask, has_phases,
    smoothPos, smoothing, particleContactDistance,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Smooth positions via subgrid neighbor weighted averaging."""
    with cp.Kernel(cp.ceildiv(numParticles, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            threadIndex = bx * BLOCK_SIZE + tid
            if threadIndex < numParticles:
                pNr = sortedToOriginalParticleIndex[threadIndex] + cp.int32(0)
                subgridIndex = sortedParticleToSubgrid[threadIndex] + cp.int32(0)

                pcdSq = particleContactDistance * particleContactDistance
                pcdInv = cp.float32(1.0) / particleContactDistance

                skipPhase = cp.int32(0)
                if has_phases != cp.int32(0):
                    pval = phases[pNr] + cp.int32(0)
                    if (pval & validPhaseMask) == cp.int32(0):
                        skipPhase = cp.int32(1)
                if subgridIndex >= maxNumSubgrids:
                    skipPhase = cp.int32(1)

                if skipPhase != cp.int32(0):
                    smoothPos[pNr, 0] = deviceParticlePos[pNr, 0]
                    smoothPos[pNr, 1] = deviceParticlePos[pNr, 1]
                    smoothPos[pNr, 2] = deviceParticlePos[pNr, 2]
                    smoothPos[pNr, 3] = deviceParticlePos[pNr, 3]
                else:
                    xi_x = deviceParticlePos[pNr, 0] + cp.float32(0.0)
                    xi_y = deviceParticlePos[pNr, 1] + cp.float32(0.0)
                    xi_z = deviceParticlePos[pNr, 2] + cp.float32(0.0)
                    xi_w = deviceParticlePos[pNr, 3] + cp.float32(0.0)

                    xs_x = cp.float32(0.0)
                    xs_y = cp.float32(0.0)
                    xs_z = cp.float32(0.0)
                    ws = cp.float32(0.0)

                    for dz in range(3):
                        for dy in range(3):
                            for dx in range(3):
                                n = subgridNeighborOffset(subgridNeighbors, subgridIndex,
                                                          cp.int32(dx - 1), cp.int32(dy - 1), cp.int32(dz - 1))
                                if n != cp.int32(EMPTY_SUBGRID):
                                    start = cp.int32(0)
                                    if n > cp.int32(0):
                                        start = subgridEndIndices[n - cp.int32(1)] + cp.int32(0)
                                    end = subgridEndIndices[n] + cp.int32(0)
                                    ii = start
                                    while ii < end:
                                        jj = sortedToOriginalParticleIndex[ii] + cp.int32(0)
                                        validJ = cp.int32(1)
                                        if has_phases != cp.int32(0):
                                            jp = phases[jj] + cp.int32(0)
                                            if (jp & validPhaseMask) == cp.int32(0):
                                                validJ = cp.int32(0)
                                        if validJ != cp.int32(0):
                                            xj_x = deviceParticlePos[jj, 0] + cp.float32(0.0)
                                            xj_y = deviceParticlePos[jj, 1] + cp.float32(0.0)
                                            xj_z = deviceParticlePos[jj, 2] + cp.float32(0.0)
                                            ddx = xi_x - xj_x
                                            ddy = xi_y - xj_y
                                            ddz = xi_z - xj_z
                                            dsq = ddx * ddx + ddy * ddy + ddz * ddz
                                            if dsq > cp.float32(0.0):
                                                if dsq < pcdSq:
                                                    w = Wa(thread.sqrt(dsq), pcdInv)
                                                    ws = ws + w
                                                    xs_x = xs_x + xj_x * w
                                                    xs_y = xs_y + xj_y * w
                                                    xs_z = xs_z + xj_z * w
                                        ii = ii + cp.int32(1)

                    out_x = xi_x
                    out_y = xi_y
                    out_z = xi_z
                    if ws > cp.float32(0.0):
                        f = cp.float32(4.0) * Wa(particleContactDistance * cp.float32(0.5), pcdInv)
                        ratio = ws / f
                        if ratio > cp.float32(1.0):
                            ratio = cp.float32(1.0)
                        smooth = ratio * smoothing
                        invWs = cp.float32(1.0) / ws
                        avg_x = xs_x * invWs
                        avg_y = xs_y * invWs
                        avg_z = xs_z * invWs
                        out_x = xi_x + (avg_x - xi_x) * smooth
                        out_y = xi_y + (avg_y - xi_y) * smooth
                        out_z = xi_z + (avg_z - xi_z) * smooth

                    smoothPos[pNr, 0] = out_x
                    smoothPos[pNr, 1] = out_y
                    smoothPos[pNr, 2] = out_z
                    smoothPos[pNr, 3] = xi_w
