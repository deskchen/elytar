"""Capybara DSL port of updateTransformAndBoundArray.cu -- all 7 kernels.

Ported kernels (matching CUDA names for PTX replacement):
  1. mergeTransformCacheAndBoundArrayChanges
  2. updateTransformCacheAndBoundArrayLaunch
  3. updateChangedAABBMgrHandlesLaunch
  4. mergeChangedAABBMgrHandlesLaunch
  5. computeFrozenAndUnfrozenHistogramLaunch
  6. outputFrozenAndUnfrozenHistogram
  7. createFrozenAndUnfrozenArray

ABI differences from CUDA:
  - PxgSimulationCoreDesc decomposed: host resolves pointer members into flat
    tensor/scalar args.
  - PxgUpdateActorDataDesc similarly decomposed for mergeChangedAABBMgrHandlesLaunch.
  - PxgBodySim as int32[N, 60] (same layout as updateBodiesAndShapes.py).
  - PxgShapeSim as int32[N, SS_SIZE] with SS_SIZE=17 (includes full PxNodeIndex).
  - PxsCachedTransform as int32[N, 8] (PxTransform=7 floats + flags=1 int).
  - PxBounds3 as float32[N, 6] (min xyz, max xyz).
  - PxgSolverBodySleepData as int32[N, 2] (wakeCounter float, internalFlags uint32).
  - PxBoundTransformUpdate as int32[N, 2] (indexTo, indexFrom).
  - Articulation link poses: artiLinkBody2Worlds/artiLinkBody2Actors passed as
    float32[maxArticulations, maxLinksPerArti * 7] (7 floats per PxTransform).
  - PxgArticulation sleep data: int32[maxArticulations, 2] matching PxgSolverBodySleepData.
  - Convex mesh hull vertex iteration replaced by transformFast fallback (no pointer
    chasing in Capybara). This is a known limitation: the compiler cannot represent
    GPU pointer indirection into raw device buffers. The host would need to
    pre-compute convex bounds or accept the transformFast approximation.
  - For computeFrozenAndUnfrozenHistogram / outputFrozenAndUnfrozenHistogram, block
    and grid dims are passed as constexpr parameters.
"""

import capybara as cp

# ---------------------------------------------------------------------------
# PxgBodySim layout -- 240 bytes = 60 int32 slots
# ---------------------------------------------------------------------------
BS_B2W_QX, BS_B2W_QY, BS_B2W_QZ, BS_B2W_QW = 28, 29, 30, 31
BS_B2W_PX, BS_B2W_PY, BS_B2W_PZ, BS_B2W_PW = 32, 33, 34, 35
BS_B2A_QX, BS_B2A_QY, BS_B2A_QZ, BS_B2A_QW = 36, 37, 38, 39
BS_B2A_PX, BS_B2A_PY, BS_B2A_PZ, BS_B2A_MAX_IMPULSE = 40, 41, 42, 43
BS_ARTIC_REMAP_ID = 44
BS_INTERNAL_FLAGS = 45

# PxgShapeSim layout -- 68 bytes = 17 int32 slots
# PxTransform mTransform: q(xyzw) + p(xyz) = 7 floats [0..6]
# PxBounds3 mLocalBounds: min(xyz) + max(xyz) = 6 floats [7..12]
# PxNodeIndex mBodySimIndex: mID [13], mLinkID [14]
# PxU32 mHullDataIndex [15]
# PxU16 mShapeFlags + PxU16 mShapeType packed [16]
SS_XFORM_QX, SS_XFORM_QY, SS_XFORM_QZ, SS_XFORM_QW = 0, 1, 2, 3
SS_XFORM_PX, SS_XFORM_PY, SS_XFORM_PZ = 4, 5, 6
SS_LBOUNDS_MIN_X, SS_LBOUNDS_MIN_Y, SS_LBOUNDS_MIN_Z = 7, 8, 9
SS_LBOUNDS_MAX_X, SS_LBOUNDS_MAX_Y, SS_LBOUNDS_MAX_Z = 10, 11, 12
SS_BODY_SIM_INDEX = 13      # PxNodeIndex.mID
SS_BODY_SIM_LINK_ID = 14    # PxNodeIndex.mLinkID
SS_HULL_DATA_INDEX = 15
SS_SHAPE_FLAGS_TYPE = 16    # packed: lower 16 = shapeFlags, upper 16 = shapeType

# PxsCachedTransform layout -- 32 bytes = 8 int32 slots
CT_QX, CT_QY, CT_QZ, CT_QW = 0, 1, 2, 3
CT_PX, CT_PY, CT_PZ = 4, 5, 6
CT_FLAGS = 7

# PxBounds3 layout -- 24 bytes = 6 float32 slots
BD_MIN_X, BD_MIN_Y, BD_MIN_Z = 0, 1, 2
BD_MAX_X, BD_MAX_Y, BD_MAX_Z = 3, 4, 5

# PxgSolverBodySleepData layout -- 8 bytes = 2 int32 slots
SD_WAKE_COUNTER = 0   # float as int32
SD_INTERNAL_FLAGS = 1  # uint32

# PxBoundTransformUpdate layout -- 8 bytes = 2 int32 slots
BTU_INDEX_TO = 0
BTU_INDEX_FROM = 1

# PxsRigidBody internal flags
eFROZEN = 1 << 4                # 16
eFREEZE_THIS_FRAME = 1 << 5    # 32
eUNFREEZE_THIS_FRAME = 1 << 6  # 64
eACTIVATE_THIS_FRAME = 1 << 7  # 128
eDEACTIVATE_THIS_FRAME = 1 << 10  # 1024

# PxsTransformFlag
TRANSFORM_FLAG_FROZEN = 1

# PxShapeFlag
eSIMULATION_SHAPE = 1 << 0   # 1
eSCENE_QUERY_SHAPE = 1 << 1  # 2
eTRIGGER_SHAPE = 1 << 2      # 4

# PxGeometryType
GEO_SPHERE = 0
GEO_CAPSULE = 2
GEO_BOX = 3
GEO_CONVEXMESH = 5

INVALID_NODE = 0xFFFFFFFF

WARP_SIZE = 32


# =====================================================================
# Inline helpers
# =====================================================================

@cp.inline
def abs_f(x):
    r = x
    if r < cp.float32(0.0):
        r = -r
    return r


# =====================================================================
# Kernel 1: mergeTransformCacheAndBoundArrayChanges
# =====================================================================
@cp.kernel
def mergeTransformCacheAndBoundArrayChanges(
    deviceBounds,       # float32[boundsSize, 6]
    deviceTransforms,   # int32[cacheSize, 8]
    boundsArray,        # float32[boundsSize, 6]
    transformsArray,    # int32[cacheSize, 8]
    changes,            # int32[numChanges, 2] -- PxBoundTransformUpdate
    numChanges,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Copy cached transforms and bounds from update arrays to persistent arrays."""
    with cp.Kernel(cp.ceildiv(numChanges, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            idx = bx * BLOCK_SIZE + tid
            if idx < numChanges:
                indexTo = changes[idx, BTU_INDEX_TO] + cp.int32(0)
                indexFromRaw = changes[idx, BTU_INDEX_FROM] + cp.int32(0)
                indexFrom = indexFromRaw & cp.int32(0x7FFFFFFF)
                isNew = (indexFromRaw >> cp.int32(31)) & cp.int32(1)

                # Read source bounds and transforms
                # Pre-declare all variables for if/else
                b0 = cp.float32(0.0)
                b1 = cp.float32(0.0)
                b2 = cp.float32(0.0)
                b3 = cp.float32(0.0)
                b4 = cp.float32(0.0)
                b5 = cp.float32(0.0)
                t0 = cp.int32(0)
                t1 = cp.int32(0)
                t2 = cp.int32(0)
                t3 = cp.int32(0)
                t4 = cp.int32(0)
                t5 = cp.int32(0)
                t6 = cp.int32(0)
                t7 = cp.int32(0)

                if isNew != cp.int32(0):
                    # Copy from CPU-side staging arrays
                    b0 = boundsArray[indexFrom, 0] + cp.float32(0.0)
                    b1 = boundsArray[indexFrom, 1] + cp.float32(0.0)
                    b2 = boundsArray[indexFrom, 2] + cp.float32(0.0)
                    b3 = boundsArray[indexFrom, 3] + cp.float32(0.0)
                    b4 = boundsArray[indexFrom, 4] + cp.float32(0.0)
                    b5 = boundsArray[indexFrom, 5] + cp.float32(0.0)
                    t0 = transformsArray[indexFrom, 0] + cp.int32(0)
                    t1 = transformsArray[indexFrom, 1] + cp.int32(0)
                    t2 = transformsArray[indexFrom, 2] + cp.int32(0)
                    t3 = transformsArray[indexFrom, 3] + cp.int32(0)
                    t4 = transformsArray[indexFrom, 4] + cp.int32(0)
                    t5 = transformsArray[indexFrom, 5] + cp.int32(0)
                    t6 = transformsArray[indexFrom, 6] + cp.int32(0)
                    t7 = transformsArray[indexFrom, 7] + cp.int32(0)
                else:
                    # Copy from GPU-side persistent arrays
                    b0 = deviceBounds[indexFrom, 0] + cp.float32(0.0)
                    b1 = deviceBounds[indexFrom, 1] + cp.float32(0.0)
                    b2 = deviceBounds[indexFrom, 2] + cp.float32(0.0)
                    b3 = deviceBounds[indexFrom, 3] + cp.float32(0.0)
                    b4 = deviceBounds[indexFrom, 4] + cp.float32(0.0)
                    b5 = deviceBounds[indexFrom, 5] + cp.float32(0.0)
                    t0 = deviceTransforms[indexFrom, 0] + cp.int32(0)
                    t1 = deviceTransforms[indexFrom, 1] + cp.int32(0)
                    t2 = deviceTransforms[indexFrom, 2] + cp.int32(0)
                    t3 = deviceTransforms[indexFrom, 3] + cp.int32(0)
                    t4 = deviceTransforms[indexFrom, 4] + cp.int32(0)
                    t5 = deviceTransforms[indexFrom, 5] + cp.int32(0)
                    t6 = deviceTransforms[indexFrom, 6] + cp.int32(0)
                    t7 = deviceTransforms[indexFrom, 7] + cp.int32(0)

                # Write to destination
                deviceBounds[indexTo, 0] = b0
                deviceBounds[indexTo, 1] = b1
                deviceBounds[indexTo, 2] = b2
                deviceBounds[indexTo, 3] = b3
                deviceBounds[indexTo, 4] = b4
                deviceBounds[indexTo, 5] = b5
                deviceTransforms[indexTo, 0] = t0
                deviceTransforms[indexTo, 1] = t1
                deviceTransforms[indexTo, 2] = t2
                deviceTransforms[indexTo, 3] = t3
                deviceTransforms[indexTo, 4] = t4
                deviceTransforms[indexTo, 5] = t5
                deviceTransforms[indexTo, 6] = t6
                deviceTransforms[indexTo, 7] = t7


# =====================================================================
# Kernel 2: updateTransformCacheAndBoundArrayLaunch
# =====================================================================
@cp.kernel
def updateTransformCacheAndBoundArrayLaunch(
    # Sleep data
    sleepData,              # int32[sleepPoolSize, 2] -- PxgSolverBodySleepData
    bodyDataIndices,        # int32[bodyPoolSize] -- maps bodySimIndex -> activeNodeIndex
    # Body / shape pools
    bodySimPool,            # int32[bodyPoolSize, 60] -- PxgBodySim
    shapeSimPool,           # int32[shapePoolSize, SS_SIZE] -- PxgShapeSim
    numShapes,              # total number of shapes
    # Articulation data
    artiSleepData,          # int32[maxArticulations, 2] -- articulation sleep data
    artiLinkBody2Worlds,    # float32[maxArticulations, maxLinksPerArti * 7]
    artiLinkBody2Actors,    # float32[maxArticulations, maxLinksPerArti * 7]
    # Output arrays
    transformCache,         # int32[cacheSize, 8] -- PxsCachedTransform as int32
    bounds,                 # float32[boundsSize, 6] -- PxBounds3
    frozen,                 # int32[shapePoolSize] -- per-shape frozen flag
    unfrozen,               # int32[shapePoolSize] -- per-shape unfrozen flag
    updated,                # int32[shapePoolSize] -- per-shape updated flag
    active,                 # int32[bodyPoolSize] -- per-body active flag
    deactivate,             # int32[bodyPoolSize] -- per-body deactivate flag
    # Constexpr params
    LINKS_PER_ARTI: cp.constexpr = 64,
    SS_SIZE: cp.constexpr = 17,
    BLOCK_SIZE: cp.constexpr = 256,
    NUM_BLOCKS: cp.constexpr = 256
):
    """Per-thread: for each shape, compute absolute pose, update transform cache + bounds."""
    with cp.Kernel(NUM_BLOCKS, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gti = bx * BLOCK_SIZE + tid
            totalThreads = cp.int32(BLOCK_SIZE) * cp.int32(NUM_BLOCKS)
            i = gti
            while i < numShapes:
                # Read PxNodeIndex from shapeSim
                nodeID = shapeSimPool[i, SS_BODY_SIM_INDEX] + cp.int32(0)
                nodeLinkID = shapeSimPool[i, SS_BODY_SIM_LINK_ID] + cp.int32(0)

                # isStaticBody: mID == 0xFFFFFFFF
                isStatic = cp.int32(1) if nodeID == cp.int32(INVALID_NODE) else cp.int32(0)

                if isStatic == cp.int32(0):
                    elementIndex = i
                    bodySimIndex = nodeID  # PxNodeIndex.index() = mID

                    # Read shape flags
                    flagsTypePacked = shapeSimPool[i, SS_SHAPE_FLAGS_TYPE] + cp.int32(0)
                    shapeFlags = flagsTypePacked & cp.int32(0xFFFF)
                    shapeType = (flagsTypePacked >> cp.int32(16)) & cp.int32(0xFFFF)

                    isBP_mask = cp.int32(eSIMULATION_SHAPE | eTRIGGER_SHAPE)
                    isBP = cp.int32(0)
                    if (shapeFlags & isBP_mask) != cp.int32(0):
                        isBP = cp.int32(1)
                    isBPOrSq_mask = cp.int32(eSIMULATION_SHAPE | eTRIGGER_SHAPE | eSCENE_QUERY_SHAPE)
                    isBPOrSq = cp.int32(0)
                    if (shapeFlags & isBPOrSq_mask) != cp.int32(0):
                        isBPOrSq = cp.int32(1)

                    # isArticulation: mLinkID & 1
                    isArticulation = nodeLinkID & cp.int32(1)

                    if isArticulation == cp.int32(0):
                        # --- Rigid body path ---
                        activeNodeIndex = bodyDataIndices[bodySimIndex] + cp.int32(0)

                        if activeNodeIndex != cp.int32(INVALID_NODE):
                            internalFlags = sleepData[activeNodeIndex, SD_INTERNAL_FLAGS] + cp.int32(0)

                            # Read body2World from bodySimPool
                            b2w_qx = thread.bitcast(bodySimPool[bodySimIndex, BS_B2W_QX] + cp.int32(0), cp.float32)
                            b2w_qy = thread.bitcast(bodySimPool[bodySimIndex, BS_B2W_QY] + cp.int32(0), cp.float32)
                            b2w_qz = thread.bitcast(bodySimPool[bodySimIndex, BS_B2W_QZ] + cp.int32(0), cp.float32)
                            b2w_qw = thread.bitcast(bodySimPool[bodySimIndex, BS_B2W_QW] + cp.int32(0), cp.float32)
                            b2w_px = thread.bitcast(bodySimPool[bodySimIndex, BS_B2W_PX] + cp.int32(0), cp.float32)
                            b2w_py = thread.bitcast(bodySimPool[bodySimIndex, BS_B2W_PY] + cp.int32(0), cp.float32)
                            b2w_pz = thread.bitcast(bodySimPool[bodySimIndex, BS_B2W_PZ] + cp.int32(0), cp.float32)

                            # Freeze / unfreeze flags (if/elif structure)
                            hasFreeze = internalFlags & cp.int32(eFREEZE_THIS_FRAME)
                            hasFrozen = internalFlags & cp.int32(eFROZEN)
                            hasUnfreeze = internalFlags & cp.int32(eUNFREEZE_THIS_FRAME)

                            if hasFreeze != cp.int32(0):
                                if hasFrozen != cp.int32(0):
                                    frozen[i] = cp.int32(1)
                                    transformCache[elementIndex, CT_FLAGS] = cp.int32(TRANSFORM_FLAG_FROZEN)
                            if hasUnfreeze != cp.int32(0):
                                # Only set unfrozen if NOT (freeze && frozen)
                                freezeAndFrozen = cp.int32(0)
                                if hasFreeze != cp.int32(0):
                                    if hasFrozen != cp.int32(0):
                                        freezeAndFrozen = cp.int32(1)
                                if freezeAndFrozen == cp.int32(0):
                                    unfrozen[i] = cp.int32(1)

                            # Update bounds if not frozen or just freezing
                            shouldUpdate = cp.int32(0)
                            if hasFrozen == cp.int32(0):
                                shouldUpdate = cp.int32(1)
                            if hasFreeze != cp.int32(0):
                                shouldUpdate = cp.int32(1)

                            if shouldUpdate != cp.int32(0):
                                if isBP != cp.int32(0):
                                    updated[elementIndex] = cp.int32(1)

                                # Read shape2Actor
                                s2a_qx = thread.bitcast(shapeSimPool[i, SS_XFORM_QX] + cp.int32(0), cp.float32)
                                s2a_qy = thread.bitcast(shapeSimPool[i, SS_XFORM_QY] + cp.int32(0), cp.float32)
                                s2a_qz = thread.bitcast(shapeSimPool[i, SS_XFORM_QZ] + cp.int32(0), cp.float32)
                                s2a_qw = thread.bitcast(shapeSimPool[i, SS_XFORM_QW] + cp.int32(0), cp.float32)
                                s2a_px = thread.bitcast(shapeSimPool[i, SS_XFORM_PX] + cp.int32(0), cp.float32)
                                s2a_py = thread.bitcast(shapeSimPool[i, SS_XFORM_PY] + cp.int32(0), cp.float32)
                                s2a_pz = thread.bitcast(shapeSimPool[i, SS_XFORM_PZ] + cp.int32(0), cp.float32)

                                # Read body2Actor
                                b2a_qx = thread.bitcast(bodySimPool[bodySimIndex, BS_B2A_QX] + cp.int32(0), cp.float32)
                                b2a_qy = thread.bitcast(bodySimPool[bodySimIndex, BS_B2A_QY] + cp.int32(0), cp.float32)
                                b2a_qz = thread.bitcast(bodySimPool[bodySimIndex, BS_B2A_QZ] + cp.int32(0), cp.float32)
                                b2a_qw = thread.bitcast(bodySimPool[bodySimIndex, BS_B2A_QW] + cp.int32(0), cp.float32)
                                b2a_px = thread.bitcast(bodySimPool[bodySimIndex, BS_B2A_PX] + cp.int32(0), cp.float32)
                                b2a_py = thread.bitcast(bodySimPool[bodySimIndex, BS_B2A_PY] + cp.int32(0), cp.float32)
                                b2a_pz = thread.bitcast(bodySimPool[bodySimIndex, BS_B2A_PZ] + cp.int32(0), cp.float32)

                                # getAbsPose: body2World.transform(body2Actor.transformInv(shape2Actor))
                                # Step 1: t0 = body2Actor.transformInv(shape2Actor)
                                #   inv.q = conjugate(b2a.q)
                                ti_qx = -b2a_qx
                                ti_qy = -b2a_qy
                                ti_qz = -b2a_qz
                                ti_qw = b2a_qw
                                #   inv.p = inv.q.rotate(-b2a.p)
                                neg_px = -b2a_px
                                neg_py = -b2a_py
                                neg_pz = -b2a_pz
                                cr1_x = ti_qy * neg_pz - ti_qz * neg_py
                                cr1_y = ti_qz * neg_px - ti_qx * neg_pz
                                cr1_z = ti_qx * neg_py - ti_qy * neg_px
                                ti_px = neg_px + cp.float32(2.0) * (ti_qw * cr1_x + ti_qy * cr1_z - ti_qz * cr1_y)
                                ti_py = neg_py + cp.float32(2.0) * (ti_qw * cr1_y + ti_qz * cr1_x - ti_qx * cr1_z)
                                ti_pz = neg_pz + cp.float32(2.0) * (ti_qw * cr1_z + ti_qx * cr1_y - ti_qy * cr1_x)
                                #   t0 = inv.transform(s2a)
                                #   t0.q = inv.q * s2a.q
                                t0_qx = ti_qw * s2a_qx + ti_qx * s2a_qw + ti_qy * s2a_qz - ti_qz * s2a_qy
                                t0_qy = ti_qw * s2a_qy - ti_qx * s2a_qz + ti_qy * s2a_qw + ti_qz * s2a_qx
                                t0_qz = ti_qw * s2a_qz + ti_qx * s2a_qy - ti_qy * s2a_qx + ti_qz * s2a_qw
                                t0_qw = ti_qw * s2a_qw - ti_qx * s2a_qx - ti_qy * s2a_qy - ti_qz * s2a_qz
                                #   t0.p = inv.q.rotate(s2a.p) + inv.p
                                cr2_x = ti_qy * s2a_pz - ti_qz * s2a_py
                                cr2_y = ti_qz * s2a_px - ti_qx * s2a_pz
                                cr2_z = ti_qx * s2a_py - ti_qy * s2a_px
                                t0_px = s2a_px + cp.float32(2.0) * (ti_qw * cr2_x + ti_qy * cr2_z - ti_qz * cr2_y) + ti_px
                                t0_py = s2a_py + cp.float32(2.0) * (ti_qw * cr2_y + ti_qz * cr2_x - ti_qx * cr2_z) + ti_py
                                t0_pz = s2a_pz + cp.float32(2.0) * (ti_qw * cr2_z + ti_qx * cr2_y - ti_qy * cr2_x) + ti_pz

                                # Step 2: absPos = body2World.transform(t0)
                                abs_qx = b2w_qw * t0_qx + b2w_qx * t0_qw + b2w_qy * t0_qz - b2w_qz * t0_qy
                                abs_qy = b2w_qw * t0_qy - b2w_qx * t0_qz + b2w_qy * t0_qw + b2w_qz * t0_qx
                                abs_qz = b2w_qw * t0_qz + b2w_qx * t0_qy - b2w_qy * t0_qx + b2w_qz * t0_qw
                                abs_qw = b2w_qw * t0_qw - b2w_qx * t0_qx - b2w_qy * t0_qy - b2w_qz * t0_qz
                                abs_cr_x = b2w_qy * t0_pz - b2w_qz * t0_py
                                abs_cr_y = b2w_qz * t0_px - b2w_qx * t0_pz
                                abs_cr_z = b2w_qx * t0_py - b2w_qy * t0_px
                                abs_px = t0_px + cp.float32(2.0) * (b2w_qw * abs_cr_x + b2w_qy * abs_cr_z - b2w_qz * abs_cr_y) + b2w_px
                                abs_py = t0_py + cp.float32(2.0) * (b2w_qw * abs_cr_y + b2w_qz * abs_cr_x - b2w_qx * abs_cr_z) + b2w_py
                                abs_pz = t0_pz + cp.float32(2.0) * (b2w_qw * abs_cr_z + b2w_qx * abs_cr_y - b2w_qy * abs_cr_x) + b2w_pz

                                # setTransformCache (flags=0)
                                transformCache[elementIndex, CT_QX] = thread.bitcast(abs_qx, cp.int32)
                                transformCache[elementIndex, CT_QY] = thread.bitcast(abs_qy, cp.int32)
                                transformCache[elementIndex, CT_QZ] = thread.bitcast(abs_qz, cp.int32)
                                transformCache[elementIndex, CT_QW] = thread.bitcast(abs_qw, cp.int32)
                                transformCache[elementIndex, CT_PX] = thread.bitcast(abs_px, cp.int32)
                                transformCache[elementIndex, CT_PY] = thread.bitcast(abs_py, cp.int32)
                                transformCache[elementIndex, CT_PZ] = thread.bitcast(abs_pz, cp.int32)
                                transformCache[elementIndex, CT_FLAGS] = cp.int32(0)

                                # updateBounds (only if isBPOrSq)
                                if isBPOrSq != cp.int32(0):
                                    # Read local bounds
                                    lb_min_x = thread.bitcast(shapeSimPool[i, SS_LBOUNDS_MIN_X] + cp.int32(0), cp.float32)
                                    lb_min_y = thread.bitcast(shapeSimPool[i, SS_LBOUNDS_MIN_Y] + cp.int32(0), cp.float32)
                                    lb_min_z = thread.bitcast(shapeSimPool[i, SS_LBOUNDS_MIN_Z] + cp.int32(0), cp.float32)
                                    lb_max_x = thread.bitcast(shapeSimPool[i, SS_LBOUNDS_MAX_X] + cp.int32(0), cp.float32)
                                    lb_max_y = thread.bitcast(shapeSimPool[i, SS_LBOUNDS_MAX_Y] + cp.int32(0), cp.float32)
                                    lb_max_z = thread.bitcast(shapeSimPool[i, SS_LBOUNDS_MAX_Z] + cp.int32(0), cp.float32)

                                    # Pre-declare output vars
                                    out_min_x = cp.float32(0.0)
                                    out_min_y = cp.float32(0.0)
                                    out_min_z = cp.float32(0.0)
                                    out_max_x = cp.float32(0.0)
                                    out_max_y = cp.float32(0.0)
                                    out_max_z = cp.float32(0.0)

                                    # Pre-declare rotation matrix variables used in capsule/box branches
                                    x2 = cp.float32(0.0)
                                    y2 = cp.float32(0.0)
                                    z2 = cp.float32(0.0)
                                    xx = cp.float32(0.0)
                                    xy = cp.float32(0.0)
                                    xz = cp.float32(0.0)
                                    yy = cp.float32(0.0)
                                    yz = cp.float32(0.0)
                                    zz = cp.float32(0.0)
                                    wx = cp.float32(0.0)
                                    wy = cp.float32(0.0)
                                    wz = cp.float32(0.0)
                                    m00 = cp.float32(0.0)
                                    m01 = cp.float32(0.0)
                                    m02 = cp.float32(0.0)
                                    m10 = cp.float32(0.0)
                                    m11 = cp.float32(0.0)
                                    m12 = cp.float32(0.0)
                                    m20 = cp.float32(0.0)
                                    m21 = cp.float32(0.0)
                                    m22 = cp.float32(0.0)
                                    w_x = cp.float32(0.0)
                                    w_y = cp.float32(0.0)
                                    w_z = cp.float32(0.0)

                                    if shapeType == cp.int32(GEO_SPHERE):
                                        # Sphere: center +/- radius (stored in localBounds)
                                        out_min_x = abs_px + lb_min_x
                                        out_min_y = abs_py + lb_min_y
                                        out_min_z = abs_pz + lb_min_z
                                        out_max_x = abs_px + lb_max_x
                                        out_max_y = abs_py + lb_max_y
                                        out_max_z = abs_pz + lb_max_z
                                    elif shapeType == cp.int32(GEO_CAPSULE):
                                        # Capsule: radius = lb_max_y, halfHeight = lb_max_x - radius
                                        radius = lb_max_y
                                        halfHeight = lb_max_x - radius
                                        # d = pose.q.getBasisVector0() = first column of rotation matrix
                                        # Quaternion to basis vector 0:
                                        #   d.x = 1 - 2*(qy^2 + qz^2)
                                        #   d.y = 2*(qx*qy + qw*qz)
                                        #   d.z = 2*(qx*qz - qw*qy)
                                        dx = cp.float32(1.0) - cp.float32(2.0) * (abs_qy * abs_qy + abs_qz * abs_qz)
                                        dy = cp.float32(2.0) * (abs_qx * abs_qy + abs_qw * abs_qz)
                                        dz = cp.float32(2.0) * (abs_qx * abs_qz - abs_qw * abs_qy)
                                        ext_x = abs_f(dx) * halfHeight + radius
                                        ext_y = abs_f(dy) * halfHeight + radius
                                        ext_z = abs_f(dz) * halfHeight + radius
                                        out_min_x = abs_px - ext_x
                                        out_min_y = abs_py - ext_y
                                        out_min_z = abs_pz - ext_z
                                        out_max_x = abs_px + ext_x
                                        out_max_y = abs_py + ext_y
                                        out_max_z = abs_pz + ext_z
                                    elif shapeType == cp.int32(GEO_BOX):
                                        # Box: halfExtents = lb_max, extents = basisExtent(PxMat33(q), halfExtents)
                                        hex = lb_max_x
                                        hey = lb_max_y
                                        hez = lb_max_z
                                        # Build rotation matrix from quaternion
                                        x2 = abs_qx + abs_qx
                                        y2 = abs_qy + abs_qy
                                        z2 = abs_qz + abs_qz
                                        xx = abs_qx * x2
                                        xy = abs_qx * y2
                                        xz = abs_qx * z2
                                        yy = abs_qy * y2
                                        yz = abs_qy * z2
                                        zz = abs_qz * z2
                                        wx = abs_qw * x2
                                        wy = abs_qw * y2
                                        wz = abs_qw * z2
                                        m00 = cp.float32(1.0) - (yy + zz)
                                        m10 = xy + wz
                                        m20 = xz - wy
                                        m01 = xy - wz
                                        m11 = cp.float32(1.0) - (xx + zz)
                                        m21 = yz + wx
                                        m02 = xz + wy
                                        m12 = yz - wx
                                        m22 = cp.float32(1.0) - (xx + yy)
                                        # basisExtent
                                        w_x = abs_f(m00) * hex + abs_f(m01) * hey + abs_f(m02) * hez
                                        w_y = abs_f(m10) * hex + abs_f(m11) * hey + abs_f(m12) * hez
                                        w_z = abs_f(m20) * hex + abs_f(m21) * hey + abs_f(m22) * hez
                                        out_min_x = abs_px - w_x
                                        out_min_y = abs_py - w_y
                                        out_min_z = abs_pz - w_z
                                        out_max_x = abs_px + w_x
                                        out_max_y = abs_py + w_y
                                        out_max_z = abs_pz + w_z
                                    else:
                                        # Default / convex mesh / trimesh / heightfield:
                                        # Use PxBounds3::transformFast(pose, localBound)
                                        # NOTE: Convex mesh vertex iteration requires GPU pointer
                                        # chasing which cannot be represented in Capybara. Using
                                        # transformFast as approximation (same as CUDA default case).
                                        cx = (lb_min_x + lb_max_x) * cp.float32(0.5)
                                        cy = (lb_min_y + lb_max_y) * cp.float32(0.5)
                                        cz = (lb_min_z + lb_max_z) * cp.float32(0.5)
                                        ex = (lb_max_x - lb_min_x) * cp.float32(0.5)
                                        ey = (lb_max_y - lb_min_y) * cp.float32(0.5)
                                        ez = (lb_max_z - lb_min_z) * cp.float32(0.5)
                                        # Rotate center
                                        rc_cr_x = abs_qy * cz - abs_qz * cy
                                        rc_cr_y = abs_qz * cx - abs_qx * cz
                                        rc_cr_z = abs_qx * cy - abs_qy * cx
                                        rc_x = cx + cp.float32(2.0) * (abs_qw * rc_cr_x + abs_qy * rc_cr_z - abs_qz * rc_cr_y) + abs_px
                                        rc_y = cy + cp.float32(2.0) * (abs_qw * rc_cr_y + abs_qz * rc_cr_x - abs_qx * rc_cr_z) + abs_py
                                        rc_z = cz + cp.float32(2.0) * (abs_qw * rc_cr_z + abs_qx * rc_cr_y - abs_qy * rc_cr_x) + abs_pz
                                        # Rotated extents via basisExtent
                                        x2 = abs_qx + abs_qx
                                        y2 = abs_qy + abs_qy
                                        z2 = abs_qz + abs_qz
                                        xx = abs_qx * x2
                                        xy = abs_qx * y2
                                        xz = abs_qx * z2
                                        yy = abs_qy * y2
                                        yz = abs_qy * z2
                                        zz = abs_qz * z2
                                        wx = abs_qw * x2
                                        wy = abs_qw * y2
                                        wz = abs_qw * z2
                                        m00 = cp.float32(1.0) - (yy + zz)
                                        m10 = xy + wz
                                        m20 = xz - wy
                                        m01 = xy - wz
                                        m11 = cp.float32(1.0) - (xx + zz)
                                        m21 = yz + wx
                                        m02 = xz + wy
                                        m12 = yz - wx
                                        m22 = cp.float32(1.0) - (xx + yy)
                                        w_x = abs_f(m00) * ex + abs_f(m01) * ey + abs_f(m02) * ez
                                        w_y = abs_f(m10) * ex + abs_f(m11) * ey + abs_f(m12) * ez
                                        w_z = abs_f(m20) * ex + abs_f(m21) * ey + abs_f(m22) * ez
                                        out_min_x = rc_x - w_x
                                        out_min_y = rc_y - w_y
                                        out_min_z = rc_z - w_z
                                        out_max_x = rc_x + w_x
                                        out_max_y = rc_y + w_y
                                        out_max_z = rc_z + w_z

                                    bounds[elementIndex, BD_MIN_X] = out_min_x
                                    bounds[elementIndex, BD_MIN_Y] = out_min_y
                                    bounds[elementIndex, BD_MIN_Z] = out_min_z
                                    bounds[elementIndex, BD_MAX_X] = out_max_x
                                    bounds[elementIndex, BD_MAX_Y] = out_max_y
                                    bounds[elementIndex, BD_MAX_Z] = out_max_z

                            # Activate / deactivate flags
                            isActivate = (internalFlags & cp.int32(eACTIVATE_THIS_FRAME)) != cp.int32(0)
                            isDeactivate = (internalFlags & cp.int32(eDEACTIVATE_THIS_FRAME)) != cp.int32(0)
                            if isActivate:
                                active[bodySimIndex] = cp.int32(1)
                            elif isDeactivate:
                                deactivate[bodySimIndex] = cp.int32(1)

                    else:
                        # --- Articulation path ---
                        articulationId = bodySimPool[bodySimIndex, BS_ARTIC_REMAP_ID] + cp.int32(0)

                        artiInternalFlags = artiSleepData[articulationId, SD_INTERNAL_FLAGS] + cp.int32(0)

                        linkId = nodeLinkID >> cp.int32(1)  # articulationLinkId = mLinkID >> 1

                        # Read body2World from articulation link poses
                        linkOff = linkId * cp.int32(7)
                        ab2w_qx = artiLinkBody2Worlds[articulationId, linkOff + cp.int32(0)] + cp.float32(0.0)
                        ab2w_qy = artiLinkBody2Worlds[articulationId, linkOff + cp.int32(1)] + cp.float32(0.0)
                        ab2w_qz = artiLinkBody2Worlds[articulationId, linkOff + cp.int32(2)] + cp.float32(0.0)
                        ab2w_qw = artiLinkBody2Worlds[articulationId, linkOff + cp.int32(3)] + cp.float32(0.0)
                        ab2w_px = artiLinkBody2Worlds[articulationId, linkOff + cp.int32(4)] + cp.float32(0.0)
                        ab2w_py = artiLinkBody2Worlds[articulationId, linkOff + cp.int32(5)] + cp.float32(0.0)
                        ab2w_pz = artiLinkBody2Worlds[articulationId, linkOff + cp.int32(6)] + cp.float32(0.0)

                        if isBP != cp.int32(0):
                            updated[elementIndex] = cp.int32(1)

                        # Read body2Actor from articulation
                        ab2a_qx = artiLinkBody2Actors[articulationId, linkOff + cp.int32(0)] + cp.float32(0.0)
                        ab2a_qy = artiLinkBody2Actors[articulationId, linkOff + cp.int32(1)] + cp.float32(0.0)
                        ab2a_qz = artiLinkBody2Actors[articulationId, linkOff + cp.int32(2)] + cp.float32(0.0)
                        ab2a_qw = artiLinkBody2Actors[articulationId, linkOff + cp.int32(3)] + cp.float32(0.0)
                        ab2a_px = artiLinkBody2Actors[articulationId, linkOff + cp.int32(4)] + cp.float32(0.0)
                        ab2a_py = artiLinkBody2Actors[articulationId, linkOff + cp.int32(5)] + cp.float32(0.0)
                        ab2a_pz = artiLinkBody2Actors[articulationId, linkOff + cp.int32(6)] + cp.float32(0.0)

                        # Read shape2Actor
                        as2a_qx = thread.bitcast(shapeSimPool[i, SS_XFORM_QX] + cp.int32(0), cp.float32)
                        as2a_qy = thread.bitcast(shapeSimPool[i, SS_XFORM_QY] + cp.int32(0), cp.float32)
                        as2a_qz = thread.bitcast(shapeSimPool[i, SS_XFORM_QZ] + cp.int32(0), cp.float32)
                        as2a_qw = thread.bitcast(shapeSimPool[i, SS_XFORM_QW] + cp.int32(0), cp.float32)
                        as2a_px = thread.bitcast(shapeSimPool[i, SS_XFORM_PX] + cp.int32(0), cp.float32)
                        as2a_py = thread.bitcast(shapeSimPool[i, SS_XFORM_PY] + cp.int32(0), cp.float32)
                        as2a_pz = thread.bitcast(shapeSimPool[i, SS_XFORM_PZ] + cp.int32(0), cp.float32)

                        # getAbsPose: ab2w.transform(ab2a.transformInv(as2a))
                        # Step 1: t0 = ab2a.transformInv(as2a)
                        ati_qx = -ab2a_qx
                        ati_qy = -ab2a_qy
                        ati_qz = -ab2a_qz
                        ati_qw = ab2a_qw
                        aneg_px = -ab2a_px
                        aneg_py = -ab2a_py
                        aneg_pz = -ab2a_pz
                        acr1_x = ati_qy * aneg_pz - ati_qz * aneg_py
                        acr1_y = ati_qz * aneg_px - ati_qx * aneg_pz
                        acr1_z = ati_qx * aneg_py - ati_qy * aneg_px
                        ati_px = aneg_px + cp.float32(2.0) * (ati_qw * acr1_x + ati_qy * acr1_z - ati_qz * acr1_y)
                        ati_py = aneg_py + cp.float32(2.0) * (ati_qw * acr1_y + ati_qz * acr1_x - ati_qx * acr1_z)
                        ati_pz = aneg_pz + cp.float32(2.0) * (ati_qw * acr1_z + ati_qx * acr1_y - ati_qy * acr1_x)
                        # t0.q = ati.q * as2a.q
                        at0_qx = ati_qw * as2a_qx + ati_qx * as2a_qw + ati_qy * as2a_qz - ati_qz * as2a_qy
                        at0_qy = ati_qw * as2a_qy - ati_qx * as2a_qz + ati_qy * as2a_qw + ati_qz * as2a_qx
                        at0_qz = ati_qw * as2a_qz + ati_qx * as2a_qy - ati_qy * as2a_qx + ati_qz * as2a_qw
                        at0_qw = ati_qw * as2a_qw - ati_qx * as2a_qx - ati_qy * as2a_qy - ati_qz * as2a_qz
                        # t0.p = ati.q.rotate(as2a.p) + ati.p
                        acr2_x = ati_qy * as2a_pz - ati_qz * as2a_py
                        acr2_y = ati_qz * as2a_px - ati_qx * as2a_pz
                        acr2_z = ati_qx * as2a_py - ati_qy * as2a_px
                        at0_px = as2a_px + cp.float32(2.0) * (ati_qw * acr2_x + ati_qy * acr2_z - ati_qz * acr2_y) + ati_px
                        at0_py = as2a_py + cp.float32(2.0) * (ati_qw * acr2_y + ati_qz * acr2_x - ati_qx * acr2_z) + ati_py
                        at0_pz = as2a_pz + cp.float32(2.0) * (ati_qw * acr2_z + ati_qx * acr2_y - ati_qy * acr2_x) + ati_pz

                        # Step 2: absPos = ab2w.transform(at0)
                        aabs_qx = ab2w_qw * at0_qx + ab2w_qx * at0_qw + ab2w_qy * at0_qz - ab2w_qz * at0_qy
                        aabs_qy = ab2w_qw * at0_qy - ab2w_qx * at0_qz + ab2w_qy * at0_qw + ab2w_qz * at0_qx
                        aabs_qz = ab2w_qw * at0_qz + ab2w_qx * at0_qy - ab2w_qy * at0_qx + ab2w_qz * at0_qw
                        aabs_qw = ab2w_qw * at0_qw - ab2w_qx * at0_qx - ab2w_qy * at0_qy - ab2w_qz * at0_qz
                        aabs_cr_x = ab2w_qy * at0_pz - ab2w_qz * at0_py
                        aabs_cr_y = ab2w_qz * at0_px - ab2w_qx * at0_pz
                        aabs_cr_z = ab2w_qx * at0_py - ab2w_qy * at0_px
                        aabs_px = at0_px + cp.float32(2.0) * (ab2w_qw * aabs_cr_x + ab2w_qy * aabs_cr_z - ab2w_qz * aabs_cr_y) + ab2w_px
                        aabs_py = at0_py + cp.float32(2.0) * (ab2w_qw * aabs_cr_y + ab2w_qz * aabs_cr_x - ab2w_qx * aabs_cr_z) + ab2w_py
                        aabs_pz = at0_pz + cp.float32(2.0) * (ab2w_qw * aabs_cr_z + ab2w_qx * aabs_cr_y - ab2w_qy * aabs_cr_x) + ab2w_pz

                        # setTransformCache (flags=0)
                        transformCache[elementIndex, CT_QX] = thread.bitcast(aabs_qx, cp.int32)
                        transformCache[elementIndex, CT_QY] = thread.bitcast(aabs_qy, cp.int32)
                        transformCache[elementIndex, CT_QZ] = thread.bitcast(aabs_qz, cp.int32)
                        transformCache[elementIndex, CT_QW] = thread.bitcast(aabs_qw, cp.int32)
                        transformCache[elementIndex, CT_PX] = thread.bitcast(aabs_px, cp.int32)
                        transformCache[elementIndex, CT_PY] = thread.bitcast(aabs_py, cp.int32)
                        transformCache[elementIndex, CT_PZ] = thread.bitcast(aabs_pz, cp.int32)
                        transformCache[elementIndex, CT_FLAGS] = cp.int32(0)

                        # updateBounds (only if isBPOrSq)
                        if isBPOrSq != cp.int32(0):
                            alb_min_x = thread.bitcast(shapeSimPool[i, SS_LBOUNDS_MIN_X] + cp.int32(0), cp.float32)
                            alb_min_y = thread.bitcast(shapeSimPool[i, SS_LBOUNDS_MIN_Y] + cp.int32(0), cp.float32)
                            alb_min_z = thread.bitcast(shapeSimPool[i, SS_LBOUNDS_MIN_Z] + cp.int32(0), cp.float32)
                            alb_max_x = thread.bitcast(shapeSimPool[i, SS_LBOUNDS_MAX_X] + cp.int32(0), cp.float32)
                            alb_max_y = thread.bitcast(shapeSimPool[i, SS_LBOUNDS_MAX_Y] + cp.int32(0), cp.float32)
                            alb_max_z = thread.bitcast(shapeSimPool[i, SS_LBOUNDS_MAX_Z] + cp.int32(0), cp.float32)

                            aout_min_x = cp.float32(0.0)
                            aout_min_y = cp.float32(0.0)
                            aout_min_z = cp.float32(0.0)
                            aout_max_x = cp.float32(0.0)
                            aout_max_y = cp.float32(0.0)
                            aout_max_z = cp.float32(0.0)

                            # Pre-declare rotation matrix vars for capsule/box branches
                            ax2 = cp.float32(0.0)
                            ay2 = cp.float32(0.0)
                            az2 = cp.float32(0.0)
                            axx = cp.float32(0.0)
                            axy = cp.float32(0.0)
                            axz = cp.float32(0.0)
                            ayy = cp.float32(0.0)
                            ayz = cp.float32(0.0)
                            azz = cp.float32(0.0)
                            awx = cp.float32(0.0)
                            awy = cp.float32(0.0)
                            awz = cp.float32(0.0)
                            am00 = cp.float32(0.0)
                            am01 = cp.float32(0.0)
                            am02 = cp.float32(0.0)
                            am10 = cp.float32(0.0)
                            am11 = cp.float32(0.0)
                            am12 = cp.float32(0.0)
                            am20 = cp.float32(0.0)
                            am21 = cp.float32(0.0)
                            am22 = cp.float32(0.0)
                            aw_x = cp.float32(0.0)
                            aw_y = cp.float32(0.0)
                            aw_z = cp.float32(0.0)

                            if shapeType == cp.int32(GEO_SPHERE):
                                aout_min_x = aabs_px + alb_min_x
                                aout_min_y = aabs_py + alb_min_y
                                aout_min_z = aabs_pz + alb_min_z
                                aout_max_x = aabs_px + alb_max_x
                                aout_max_y = aabs_py + alb_max_y
                                aout_max_z = aabs_pz + alb_max_z
                            elif shapeType == cp.int32(GEO_CAPSULE):
                                aradius = alb_max_y
                                ahalfHeight = alb_max_x - aradius
                                adx = cp.float32(1.0) - cp.float32(2.0) * (aabs_qy * aabs_qy + aabs_qz * aabs_qz)
                                ady = cp.float32(2.0) * (aabs_qx * aabs_qy + aabs_qw * aabs_qz)
                                adz = cp.float32(2.0) * (aabs_qx * aabs_qz - aabs_qw * aabs_qy)
                                aext_x = abs_f(adx) * ahalfHeight + aradius
                                aext_y = abs_f(ady) * ahalfHeight + aradius
                                aext_z = abs_f(adz) * ahalfHeight + aradius
                                aout_min_x = aabs_px - aext_x
                                aout_min_y = aabs_py - aext_y
                                aout_min_z = aabs_pz - aext_z
                                aout_max_x = aabs_px + aext_x
                                aout_max_y = aabs_py + aext_y
                                aout_max_z = aabs_pz + aext_z
                            elif shapeType == cp.int32(GEO_BOX):
                                ahex = alb_max_x
                                ahey = alb_max_y
                                ahez = alb_max_z
                                ax2 = aabs_qx + aabs_qx
                                ay2 = aabs_qy + aabs_qy
                                az2 = aabs_qz + aabs_qz
                                axx = aabs_qx * ax2
                                axy = aabs_qx * ay2
                                axz = aabs_qx * az2
                                ayy = aabs_qy * ay2
                                ayz = aabs_qy * az2
                                azz = aabs_qz * az2
                                awx = aabs_qw * ax2
                                awy = aabs_qw * ay2
                                awz = aabs_qw * az2
                                am00 = cp.float32(1.0) - (ayy + azz)
                                am10 = axy + awz
                                am20 = axz - awy
                                am01 = axy - awz
                                am11 = cp.float32(1.0) - (axx + azz)
                                am21 = ayz + awx
                                am02 = axz + awy
                                am12 = ayz - awx
                                am22 = cp.float32(1.0) - (axx + ayy)
                                aw_x = abs_f(am00) * ahex + abs_f(am01) * ahey + abs_f(am02) * ahez
                                aw_y = abs_f(am10) * ahex + abs_f(am11) * ahey + abs_f(am12) * ahez
                                aw_z = abs_f(am20) * ahex + abs_f(am21) * ahey + abs_f(am22) * ahez
                                aout_min_x = aabs_px - aw_x
                                aout_min_y = aabs_py - aw_y
                                aout_min_z = aabs_pz - aw_z
                                aout_max_x = aabs_px + aw_x
                                aout_max_y = aabs_py + aw_y
                                aout_max_z = aabs_pz + aw_z
                            else:
                                # transformFast fallback
                                acx = (alb_min_x + alb_max_x) * cp.float32(0.5)
                                acy = (alb_min_y + alb_max_y) * cp.float32(0.5)
                                acz = (alb_min_z + alb_max_z) * cp.float32(0.5)
                                aex = (alb_max_x - alb_min_x) * cp.float32(0.5)
                                aey = (alb_max_y - alb_min_y) * cp.float32(0.5)
                                aez = (alb_max_z - alb_min_z) * cp.float32(0.5)
                                arc_cr_x = aabs_qy * acz - aabs_qz * acy
                                arc_cr_y = aabs_qz * acx - aabs_qx * acz
                                arc_cr_z = aabs_qx * acy - aabs_qy * acx
                                arc_x = acx + cp.float32(2.0) * (aabs_qw * arc_cr_x + aabs_qy * arc_cr_z - aabs_qz * arc_cr_y) + aabs_px
                                arc_y = acy + cp.float32(2.0) * (aabs_qw * arc_cr_y + aabs_qz * arc_cr_x - aabs_qx * arc_cr_z) + aabs_py
                                arc_z = acz + cp.float32(2.0) * (aabs_qw * arc_cr_z + aabs_qx * arc_cr_y - aabs_qy * arc_cr_x) + aabs_pz
                                ax2 = aabs_qx + aabs_qx
                                ay2 = aabs_qy + aabs_qy
                                az2 = aabs_qz + aabs_qz
                                axx = aabs_qx * ax2
                                axy = aabs_qx * ay2
                                axz = aabs_qx * az2
                                ayy = aabs_qy * ay2
                                ayz = aabs_qy * az2
                                azz = aabs_qz * az2
                                awx = aabs_qw * ax2
                                awy = aabs_qw * ay2
                                awz = aabs_qw * az2
                                am00 = cp.float32(1.0) - (ayy + azz)
                                am10 = axy + awz
                                am20 = axz - awy
                                am01 = axy - awz
                                am11 = cp.float32(1.0) - (axx + azz)
                                am21 = ayz + awx
                                am02 = axz + awy
                                am12 = ayz - awx
                                am22 = cp.float32(1.0) - (axx + ayy)
                                aw_x = abs_f(am00) * aex + abs_f(am01) * aey + abs_f(am02) * aez
                                aw_y = abs_f(am10) * aex + abs_f(am11) * aey + abs_f(am12) * aez
                                aw_z = abs_f(am20) * aex + abs_f(am21) * aey + abs_f(am22) * aez
                                aout_min_x = arc_x - aw_x
                                aout_min_y = arc_y - aw_y
                                aout_min_z = arc_z - aw_z
                                aout_max_x = arc_x + aw_x
                                aout_max_y = arc_y + aw_y
                                aout_max_z = arc_z + aw_z

                            bounds[elementIndex, BD_MIN_X] = aout_min_x
                            bounds[elementIndex, BD_MIN_Y] = aout_min_y
                            bounds[elementIndex, BD_MIN_Z] = aout_min_z
                            bounds[elementIndex, BD_MAX_X] = aout_max_x
                            bounds[elementIndex, BD_MAX_Y] = aout_max_y
                            bounds[elementIndex, BD_MAX_Z] = aout_max_z

                        # Activate / deactivate flags
                        aIsActivate = (artiInternalFlags & cp.int32(eACTIVATE_THIS_FRAME)) != cp.int32(0)
                        aIsDeactivate = (artiInternalFlags & cp.int32(eDEACTIVATE_THIS_FRAME)) != cp.int32(0)
                        if aIsActivate:
                            active[bodySimIndex] = cp.int32(1)
                        elif aIsDeactivate:
                            deactivate[bodySimIndex] = cp.int32(1)

                i = i + totalThreads


# =====================================================================
# Kernel 3: updateChangedAABBMgrHandlesLaunch
# =====================================================================
@cp.kernel
def updateChangedAABBMgrHandlesLaunch(
    updated,                 # int32[numElements] -- per-element update flags
    changedAABBMgrHandles,   # int32[numElements / 32] -- output bitmask words
    numElements,             # = bitMapWordCounts * 32
    BLOCK_SIZE: cp.constexpr = 256,
    NUM_BLOCKS: cp.constexpr = 256
):
    """Compact per-element update flags into warp-wide ballot bitmask words.
    Uses single-pass: each thread processes exactly one element (no grid-stride loop).
    Grid must be launched with enough blocks to cover all elements.
    """
    with cp.Kernel(cp.ceildiv(numElements, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for warp_id, warp in block.warps():
            for lane, thread in warp.threads():
                tid = warp_id * cp.int32(WARP_SIZE) + lane
                i = bx * BLOCK_SIZE + tid
                updateBit = cp.int32(0)
                if i < numElements:
                    updateBit = updated[i] + cp.int32(0)
                word = thread.coll.ballot(updateBit != cp.int32(0))
                if lane == cp.int32(0):
                    if i < numElements:
                        changedAABBMgrHandles[i // cp.int32(WARP_SIZE)] = word


# =====================================================================
# Kernel 4: mergeChangedAABBMgrHandlesLaunch
# =====================================================================
@cp.kernel
def mergeChangedAABBMgrHandlesLaunch(
    updated,                 # int32[numElements] -- Direct API changed handles
    changedAABBMgrHandles,   # int32[numElements / 32] -- CPU API changed handles (OR'd in place)
    numElements,             # = bitMapWordCounts * 32
    BLOCK_SIZE: cp.constexpr = 256,
    NUM_BLOCKS: cp.constexpr = 256
):
    """Merge direct API update flags into existing CPU API changed handle bitmask."""
    with cp.Kernel(cp.ceildiv(numElements, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for warp_id, warp in block.warps():
            for lane, thread in warp.threads():
                tid = warp_id * cp.int32(WARP_SIZE) + lane
                i = bx * BLOCK_SIZE + tid
                updateBit = cp.int32(0)
                if i < numElements:
                    updateBit = updated[i] + cp.int32(0)
                word = thread.coll.ballot(updateBit != cp.int32(0))
                if lane == cp.int32(0):
                    if i < numElements:
                        wordIdx = i // cp.int32(WARP_SIZE)
                        existing = changedAABBMgrHandles[wordIdx] + cp.int32(0)
                        changedAABBMgrHandles[wordIdx] = existing | word


# =====================================================================
# Kernel 5: computeFrozenAndUnfrozenHistogramLaunch
# =====================================================================
@cp.kernel
def computeFrozenAndUnfrozenHistogramLaunch(
    gFrozen,         # int32[shapePoolSize] -- in/out: per-shape frozen, then exclusive prefix sum
    gUnfrozen,       # int32[shapePoolSize] -- in/out: per-shape unfrozen, then exclusive prefix sum
    gFrozenBlock,    # int32[GRID_SIZE] -- output: per-block frozen total
    gUnfrozenBlock,  # int32[GRID_SIZE] -- output: per-block unfrozen total
    nbTotalShapes,
    BLOCK_SIZE: cp.constexpr = 256,
    GRID_SIZE: cp.constexpr = 32,
    NUM_WARPS: cp.constexpr = 8,
    LOG2_NUM_WARPS: cp.constexpr = 3
):
    """Two-pass warp reduction for counting frozen/unfrozen shapes with intra-block prefix sum.

    The CUDA kernel runs a loop over iterations, maintaining a running block accumulator
    in shared memory across iterations. Each iteration has 3 phases separated by __syncthreads.
    In Capybara, variables cannot cross block.barrier() boundaries, so we store all
    intermediate state in shared memory. We also store per-thread ballot results in shared
    memory so phase 3 can reuse phase 1 results without re-computing the ballot.
    """
    with cp.Kernel(GRID_SIZE, threads=BLOCK_SIZE) as (bx, block):
        sFrozenWarpAccum = block.alloc((NUM_WARPS,), dtype=cp.int32)
        sUnfrozenWarpAccum = block.alloc((NUM_WARPS,), dtype=cp.int32)
        sFrozenBlockAccum = block.alloc((1,), dtype=cp.int32)
        sUnfrozenBlockAccum = block.alloc((1,), dtype=cp.int32)
        # Per-thread storage for ballot results from phase 1 (needed in phase 3)
        sFrozenPerThread = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        sUnfrozenPerThread = block.alloc((BLOCK_SIZE,), dtype=cp.int32)
        # Store prevBlockAccum snapshot (captured before barrier in phase 1)
        sPrevFrozenBlockAccum = block.alloc((1,), dtype=cp.int32)
        sPrevUnfrozenBlockAccum = block.alloc((1,), dtype=cp.int32)

        # Initialize block accumulators
        for tid, thread in block.threads():
            if tid == cp.int32(0):
                sFrozenBlockAccum[0] = cp.disjoint(cp.int32(0))
                sUnfrozenBlockAccum[0] = cp.disjoint(cp.int32(0))

        block.barrier()

        nbBlocksRequired = cp.ceildiv(nbTotalShapes, cp.int32(BLOCK_SIZE))
        nbIterationsPerBlock = cp.ceildiv(nbBlocksRequired, cp.int32(GRID_SIZE))

        for iteration in range(256):
            # Phase 1: Load, warp ballot, save per-warp totals and per-thread accum to smem
            for tid, thread in block.threads():
                if cp.assume_uniform(cp.int32(iteration) < nbIterationsPerBlock):
                    warpIndex = tid // cp.int32(WARP_SIZE)
                    threadIndexInWarp = tid % cp.int32(WARP_SIZE)
                    workIndex = cp.int32(iteration) * cp.int32(BLOCK_SIZE) + tid + nbIterationsPerBlock * bx * cp.int32(BLOCK_SIZE)

                    frozenVal = cp.int32(0)
                    unfrozenVal = cp.int32(0)
                    if workIndex < nbTotalShapes:
                        frozenVal = gFrozen[workIndex] + cp.int32(0)
                        unfrozenVal = gUnfrozen[workIndex] + cp.int32(0)

                    # Use __ballot_sync + __popc equivalent via bitwise
                    # Since we're in block.threads() scope, compute ballot manually:
                    # Each warp's ballot = bitmask of (frozenVal != 0) per lane
                    # In block scope, use shfl_xor-based warp reduction for popcount
                    # Actually, simplify: just count per warp using shfl pattern
                    hasFrozen = cp.int32(1) if frozenVal != cp.int32(0) else cp.int32(0)
                    hasUnfrozen = cp.int32(1) if unfrozenVal != cp.int32(0) else cp.int32(0)

                    # Inclusive warp scan via shfl_up for frozen
                    f_scan = hasFrozen + cp.int32(0)
                    n = thread.shfl_up(f_scan, 1)
                    if threadIndexInWarp >= cp.int32(1):
                        f_scan = f_scan + n
                    n = thread.shfl_up(f_scan, 2)
                    if threadIndexInWarp >= cp.int32(2):
                        f_scan = f_scan + n
                    n = thread.shfl_up(f_scan, 4)
                    if threadIndexInWarp >= cp.int32(4):
                        f_scan = f_scan + n
                    n = thread.shfl_up(f_scan, 8)
                    if threadIndexInWarp >= cp.int32(8):
                        f_scan = f_scan + n
                    n = thread.shfl_up(f_scan, 16)
                    if threadIndexInWarp >= cp.int32(16):
                        f_scan = f_scan + n
                    frozenAccumVal = f_scan - hasFrozen  # exclusive scan

                    # Same for unfrozen
                    u_scan = hasUnfrozen + cp.int32(0)
                    n2 = thread.shfl_up(u_scan, 1)
                    if threadIndexInWarp >= cp.int32(1):
                        u_scan = u_scan + n2
                    n2 = thread.shfl_up(u_scan, 2)
                    if threadIndexInWarp >= cp.int32(2):
                        u_scan = u_scan + n2
                    n2 = thread.shfl_up(u_scan, 4)
                    if threadIndexInWarp >= cp.int32(4):
                        u_scan = u_scan + n2
                    n2 = thread.shfl_up(u_scan, 8)
                    if threadIndexInWarp >= cp.int32(8):
                        u_scan = u_scan + n2
                    n2 = thread.shfl_up(u_scan, 16)
                    if threadIndexInWarp >= cp.int32(16):
                        u_scan = u_scan + n2
                    unfrozenAccumVal = u_scan - hasUnfrozen  # exclusive scan

                    # Save per-thread values for phase 3
                    sFrozenPerThread[tid] = cp.disjoint(frozenAccumVal)
                    sUnfrozenPerThread[tid] = cp.disjoint(unfrozenAccumVal)

                    # Last thread in warp writes warp total
                    if threadIndexInWarp == cp.int32(WARP_SIZE - 1):
                        sFrozenWarpAccum[warpIndex] = cp.disjoint(frozenAccumVal + hasFrozen)
                        sUnfrozenWarpAccum[warpIndex] = cp.disjoint(unfrozenAccumVal + hasUnfrozen)

                    # Snapshot prev block accum (BEFORE barrier, for use in phase 3)
                    if tid == cp.int32(0):
                        sPrevFrozenBlockAccum[0] = cp.disjoint(sFrozenBlockAccum[0] + cp.int32(0))
                        sPrevUnfrozenBlockAccum[0] = cp.disjoint(sUnfrozenBlockAccum[0] + cp.int32(0))

            block.barrier()

            # Phase 2: Scan warp totals (only first NUM_WARPS threads)
            for tid, thread in block.threads():
                if cp.assume_uniform(cp.int32(iteration) < nbIterationsPerBlock):
                    if cp.assume_uniform(tid < cp.int32(NUM_WARPS)):
                        fv = sFrozenWarpAccum[tid] + cp.int32(0)
                        uv = sUnfrozenWarpAccum[tid] + cp.int32(0)

                        # Inclusive scan with shfl_up (LOG2_NUM_WARPS=3 -> 3 rounds)
                        fn = thread.shfl_up(fv, 1)
                        un = thread.shfl_up(uv, 1)
                        if tid >= cp.int32(1):
                            fv = fv + fn
                            uv = uv + un
                        fn = thread.shfl_up(fv, 2)
                        un = thread.shfl_up(uv, 2)
                        if tid >= cp.int32(2):
                            fv = fv + fn
                            uv = uv + un
                        fn = thread.shfl_up(fv, 4)
                        un = thread.shfl_up(uv, 4)
                        if tid >= cp.int32(4):
                            fv = fv + fn
                            uv = uv + un

                        # Exclusive = inclusive - original value
                        frozenOrig = sFrozenWarpAccum[tid] + cp.int32(0)
                        unfrozenOrig = sUnfrozenWarpAccum[tid] + cp.int32(0)
                        frozenExcl = fv - frozenOrig
                        unfrozenExcl = uv - unfrozenOrig
                        sFrozenWarpAccum[tid] = cp.disjoint(frozenExcl)
                        sUnfrozenWarpAccum[tid] = cp.disjoint(unfrozenExcl)

                        if tid == cp.int32(NUM_WARPS - 1):
                            sFrozenBlockAccum[0] = cp.disjoint(sFrozenBlockAccum[0] + fv)
                            sUnfrozenBlockAccum[0] = cp.disjoint(sUnfrozenBlockAccum[0] + uv)

            block.barrier()

            # Phase 3: Write output using saved per-thread values
            for tid, thread in block.threads():
                if cp.assume_uniform(cp.int32(iteration) < nbIterationsPerBlock):
                    warpIndex = tid // cp.int32(WARP_SIZE)
                    workIndex = cp.int32(iteration) * cp.int32(BLOCK_SIZE) + tid + nbIterationsPerBlock * bx * cp.int32(BLOCK_SIZE)

                    if workIndex < nbTotalShapes:
                        frozenAccumVal = sFrozenPerThread[tid] + cp.int32(0)
                        unfrozenAccumVal = sUnfrozenPerThread[tid] + cp.int32(0)
                        prevFBA = sPrevFrozenBlockAccum[0] + cp.int32(0)
                        prevUBA = sPrevUnfrozenBlockAccum[0] + cp.int32(0)
                        gFrozen[workIndex] = frozenAccumVal + prevFBA + sFrozenWarpAccum[warpIndex]
                        gUnfrozen[workIndex] = unfrozenAccumVal + prevUBA + sUnfrozenWarpAccum[warpIndex]

            block.barrier()

        # Write per-block totals
        for tid, thread in block.threads():
            if tid == cp.int32(0):
                gFrozenBlock[bx] = sFrozenBlockAccum[0]
                gUnfrozenBlock[bx] = sUnfrozenBlockAccum[0]


# =====================================================================
# Kernel 6: outputFrozenAndUnfrozenHistogram
# =====================================================================
@cp.kernel
def outputFrozenAndUnfrozenHistogram(
    gFrozen,              # int32[shapePoolSize] -- in/out prefix sums
    gUnfrozen,            # int32[shapePoolSize] -- in/out prefix sums
    gFrozenBlock,         # int32[NB_BLOCKS] -- per-block frozen totals
    gUnfrozenBlock,       # int32[NB_BLOCKS] -- per-block unfrozen totals
    totalFrozenShapes,    # int32[1] -- output
    totalUnfrozenShapes,  # int32[1] -- output
    nbTotalShapes,
    BLOCK_SIZE: cp.constexpr = 256,
    NB_BLOCKS: cp.constexpr = 32
):
    """Cross-block prefix sum fix-up using warp scan on per-block totals."""
    with cp.Kernel(NB_BLOCKS, threads=BLOCK_SIZE) as (bx, block):
        sFrozenBlockAccum = block.alloc((NB_BLOCKS,), dtype=cp.int32)
        sUnfrozenBlockAccum = block.alloc((NB_BLOCKS,), dtype=cp.int32)

        # Phase 1: Warp scan on per-block totals (only first NB_BLOCKS threads)
        for tid, thread in block.threads():
            frozenVal = cp.int32(0)
            unfrozenVal = cp.int32(0)

            if cp.assume_uniform(tid < cp.int32(NB_BLOCKS)):
                frozenVal = gFrozenBlock[tid] + cp.int32(0)
                unfrozenVal = gUnfrozenBlock[tid] + cp.int32(0)

            # Inclusive scan (5 rounds for up to 32 elements)
            fn = thread.shfl_up(frozenVal, 1)
            un = thread.shfl_up(unfrozenVal, 1)
            if tid >= cp.int32(1):
                frozenVal = frozenVal + fn
                unfrozenVal = unfrozenVal + un
            fn = thread.shfl_up(frozenVal, 2)
            un = thread.shfl_up(unfrozenVal, 2)
            if tid >= cp.int32(2):
                frozenVal = frozenVal + fn
                unfrozenVal = unfrozenVal + un
            fn = thread.shfl_up(frozenVal, 4)
            un = thread.shfl_up(unfrozenVal, 4)
            if tid >= cp.int32(4):
                frozenVal = frozenVal + fn
                unfrozenVal = unfrozenVal + un
            fn = thread.shfl_up(frozenVal, 8)
            un = thread.shfl_up(unfrozenVal, 8)
            if tid >= cp.int32(8):
                frozenVal = frozenVal + fn
                unfrozenVal = unfrozenVal + un
            fn = thread.shfl_up(frozenVal, 16)
            un = thread.shfl_up(unfrozenVal, 16)
            if tid >= cp.int32(16):
                frozenVal = frozenVal + fn
                unfrozenVal = unfrozenVal + un

            if cp.assume_uniform(tid < cp.int32(NB_BLOCKS)):
                frozenOrig = gFrozenBlock[tid] + cp.int32(0)
                unfrozenOrig = gUnfrozenBlock[tid] + cp.int32(0)
                sFrozenBlockAccum[tid] = cp.disjoint(frozenVal - frozenOrig)
                sUnfrozenBlockAccum[tid] = cp.disjoint(unfrozenVal - unfrozenOrig)

            globalThreadIndex = tid + cp.int32(BLOCK_SIZE) * bx
            if globalThreadIndex == cp.int32(NB_BLOCKS - 1):
                totalFrozenShapes[0] = frozenVal
                totalUnfrozenShapes[0] = unfrozenVal

        block.barrier()

        # Phase 2: Fix up prefix sums across blocks
        totalBlockRequired = cp.ceildiv(nbTotalShapes, cp.int32(BLOCK_SIZE))
        numIterationPerBlock = cp.ceildiv(totalBlockRequired, cp.int32(NB_BLOCKS))

        for iteration in range(256):
            for tid, thread in block.threads():
                if cp.assume_uniform(cp.int32(iteration) < numIterationPerBlock):
                    workIndex = cp.int32(iteration) * cp.int32(BLOCK_SIZE) + tid + numIterationPerBlock * bx * cp.int32(BLOCK_SIZE)

                    frozenBlockAccum = sFrozenBlockAccum[bx] + cp.int32(0)
                    unfrozenBlockAccum = sUnfrozenBlockAccum[bx] + cp.int32(0)

                    if workIndex < nbTotalShapes:
                        gFrozen[workIndex] = gFrozen[workIndex] + frozenBlockAccum
                        gUnfrozen[workIndex] = gUnfrozen[workIndex] + unfrozenBlockAccum

            block.barrier()


# =====================================================================
# Kernel 7: createFrozenAndUnfrozenArray
# =====================================================================
@cp.kernel
def createFrozenAndUnfrozenArray(
    gFrozen,              # int32[shapePoolSize] -- exclusive prefix sums
    gUnfrozen,            # int32[shapePoolSize] -- exclusive prefix sums
    gFrozenRes,           # int32[totalFrozenShapes] -- output: frozen shape indices
    gUnfrozenRes,         # int32[totalUnfrozenShapes] -- output: unfrozen shape indices
    nbTotalShapes,
    nbFrozenTotalShapes,
    nbUnfrozenTotalShapes,
    BLOCK_SIZE: cp.constexpr = 256,
    NUM_BLOCKS: cp.constexpr = 256
):
    """Binary search to create frozen/unfrozen output arrays from prefix sums."""
    with cp.Kernel(NUM_BLOCKS, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gti = bx * BLOCK_SIZE + tid
            totalThreads = cp.int32(BLOCK_SIZE) * cp.int32(NUM_BLOCKS)

            # Frozen array
            fi = gti
            while fi < nbFrozenTotalShapes:
                # binarySearch: find largest index s.t. gFrozen[index] <= fi
                left = cp.int32(0)
                right = nbTotalShapes
                while left < right:
                    pos = (left + right) >> cp.int32(1)
                    element = gFrozen[pos] + cp.int32(0)
                    if element <= fi:
                        left = pos + cp.int32(1)
                    else:
                        right = pos
                result = left - cp.int32(1)
                if left == cp.int32(0):
                    result = cp.int32(0)
                gFrozenRes[fi] = result
                fi = fi + totalThreads

            # Unfrozen array
            ui = gti
            while ui < nbUnfrozenTotalShapes:
                left = cp.int32(0)
                right = nbTotalShapes
                while left < right:
                    pos = (left + right) >> cp.int32(1)
                    element = gUnfrozen[pos] + cp.int32(0)
                    if element <= ui:
                        left = pos + cp.int32(1)
                    else:
                        right = pos
                result = left - cp.int32(1)
                if left == cp.int32(0):
                    result = cp.int32(0)
                gUnfrozenRes[ui] = result
                ui = ui + totalThreads
