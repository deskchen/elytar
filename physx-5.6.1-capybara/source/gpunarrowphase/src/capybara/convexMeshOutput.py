"""Capybara DSL port of gpunarrowphase/CUDA/convexMeshOutput.cu -- convexTrimeshFinishContacts.

Ported kernel (matching CUDA name for PTX replacement):
  - convexTrimeshFinishContacts -- finalise convex-trimesh contact output

Each warp processes one pair.  Within a warp the 32 lanes are divided into
up to PXG_MULTIMANIFOLD_MAX_SUBMANIFOLDS=4 sub-manifolds of
PXG_SUBMANIFOLD_MAX_CONTACTS=6 lanes (4*6=24, lanes 24-31 idle).

ABI differences from CUDA (struct -> flat tensor decomposition):
  - ConvexMeshPair             -> int32[N, PAIR_ROW_SIZE]  (56 bytes = 14 int32)
      PAIR_MATIDX_X=12, PAIR_MATIDX_Y=13  (materialIndices.x/y as PxU32)
  - PxsCachedTransform         -> float32[N, CT_SIZE]  (32 bytes = 8 float32)
      CT_QX..CT_QW=0..3, CT_PX..CT_PZ=4..6, CT_FLAGS=7(int as float)
  - PxgShape                   -> int32[N, SHAPE_SIZE]  (48 bytes = 12 int32)
      Scale stored as int32 (bitcast to float32). SHAPE_TYPE=10.
  - PxgContactManagerInput     -> int32[N, CMI_SIZE]  (16 bytes = 4 int32)
  - PxsContactManagerOutput    -> int32[N, CMO_SIZE]  (48 bytes = 12 int32)
      CMO_ALLFLAGS=8 (packed: byte0=allflagsStart, byte1=nbPatches,
        byte2=statusFlag, byte3=prevPatches)
      CMO_NB_CONTACTS=9 (low 16 bits = nbContacts)
  - PxgPersistentContactMultiManifold -> int32[N, MM_SIZE]
      Flattened to int32 (1216 bytes = 304 int32).
      mContacts[4][6]: each PxgContact=48 bytes=12 int32, total 24 contacts,
        starting at int32 offset 0.  Contact [sub][ct] at offset (sub*6+ct)*12.
      mNbContacts[4]: int32 offsets 288,289,290,291.
      mNbManifolds: int32 offset 300.
  - PxgPatchAndContactCounters -> int32[4]
  - contactStream              -> float32[maxContacts, 4] (PxContact: x,y,z,pen)
  - patchStream                -> int32[maxPatches, PATCH_SIZE] (PxContactPatch 64B=16 int32)
  - forceAndIndiceStream       -> int32[maxForceWords]  (flat int32 buffer)
  - touchChangeFlags, patchChangeFlags -> int32[N]
  - startContactPatches/Points/Forces -> int64 scalars (base GPU ptrs)
  - patchBytesLimit, contactBytesLimit, forceBytesLimit -> int32 scalars
  - insertAveragePoint -> int32 scalar (bool as int)

Capybara structural notes:
  - All variables that appear in any if/elif/else branch are pre-declared.
  - No tuple-return @cp.inline in conditionals.
  - `+ cp.int32(0)` / `+ cp.float32(0.0)` force-loads from tensors before
    conditionals or bitcast.
  - Boolean flags use cp.int32(0/1), never Python bool.
  - No bare `if boolvar:` -- always `if var != cp.int32(0):`.
  - Python `/` on int32 is invalid -- use `>> 1` or `//`.
  - thread.bitcast(val + cp.int32(0), cp.float32) for int->float reinterpret.
"""

import capybara as cp

WARP_SIZE = 32


# ===== Helper: quaternion rotate vector (component-wise) =====
@cp.inline
def quat_rotate_x(qx, qy, qz, qw, vx, vy, vz):
    c1x = qy * vz - qz * vy + qw * vx
    c1y = qz * vx - qx * vz + qw * vy
    c1z = qx * vy - qy * vx + qw * vz
    c2x = qy * c1z - qz * c1y
    return vx + cp.float32(2.0) * c2x


@cp.inline
def quat_rotate_y(qx, qy, qz, qw, vx, vy, vz):
    c1x = qy * vz - qz * vy + qw * vx
    c1y = qz * vx - qx * vz + qw * vy
    c1z = qx * vy - qy * vx + qw * vz
    c2y = qz * c1x - qx * c1z
    return vy + cp.float32(2.0) * c2y


@cp.inline
def quat_rotate_z(qx, qy, qz, qw, vx, vy, vz):
    c1x = qy * vz - qz * vy + qw * vx
    c1y = qz * vx - qx * vz + qw * vy
    c1z = qx * vy - qy * vx + qw * vz
    c2z = qx * c1y - qy * c1x
    return vz + cp.float32(2.0) * c2z


# ===== Helper: combine scalars for material combining =====
@cp.inline
def combine_scalars(a, b, mode):
    """PxsCombinePxReal: mode 0=avg, 1=min, 2=multiply, 3=max."""
    result = cp.float32(0.0)
    if mode == cp.int32(0):
        result = cp.float32(0.5) * (a + b)
    elif mode == cp.int32(1):
        result = a
        if b < a:
            result = b
    elif mode == cp.int32(2):
        result = a * b
    elif mode == cp.int32(3):
        result = a
        if b > a:
            result = b
    return result


# ===== Helper: max of two float32 =====
@cp.inline
def max_f32(a, b):
    r = a
    if b > a:
        r = b
    return r


# ===== Kernel: convexTrimeshFinishContacts =====
@cp.kernel
def convexTrimeshFinishContacts(
    pairs,                      # int32[N, PAIR_ROW_SIZE]  (ConvexMeshPair)
    transformCache,             # float32[N, CT_SIZE]  (PxsCachedTransform)
    gpuShapes,                  # int32[N, SHAPE_SIZE]  (PxgShape)
    cmInputs,                   # int32[N, CMI_SIZE]  (PxgContactManagerInput)
    cmOutputs,                  # int32[N, CMO_SIZE]  (PxsContactManagerOutput)
    cmMultiManifold,            # int32[N, MM_SIZE]  (PxgPersistentContactMultiManifold)
    numPairs,                   # int32 scalar
    materials,                  # int32[N, MAT_SIZE]  (PxsMaterialData)
    contactStream,              # float32[maxContacts, 4]
    patchStream,                # int32[maxPatches, PATCH_SIZE]
    forceAndIndiceStream,       # int32[maxForceWords]
    insertAveragePoint,         # int32 scalar (bool)
    patchAndContactCounters,    # int32[4]
    touchChangeFlags,           # int32[N]
    patchChangeFlags,           # int32[N]
    startContactPatches,        # int64 scalar (base GPU ptr)
    startContactPoints,         # int64 scalar (base GPU ptr)
    startContactForces,         # int64 scalar (base GPU ptr)
    patchBytesLimit,            # int32 scalar
    contactBytesLimit,          # int32 scalar
    forceBytesLimit,            # int32 scalar
    # --- constexpr parameters ---
    WARPS_PER_BLOCK: cp.constexpr = 8,
    # Struct sizes
    PAIR_ROW_SIZE: cp.constexpr = 14,
    CT_SIZE: cp.constexpr = 8,
    SHAPE_SIZE: cp.constexpr = 12,
    CMI_SIZE: cp.constexpr = 4,
    CMO_SIZE: cp.constexpr = 12,
    MM_SIZE: cp.constexpr = 304,
    MAT_SIZE: cp.constexpr = 6,
    PATCH_SIZE: cp.constexpr = 16,
    # ConvexMeshPair field offsets
    PAIR_MATIDX_X: cp.constexpr = 12,
    PAIR_MATIDX_Y: cp.constexpr = 13,
    # CachedTransform field offsets
    CT_QX: cp.constexpr = 0,
    CT_QY: cp.constexpr = 1,
    CT_QZ: cp.constexpr = 2,
    CT_QW: cp.constexpr = 3,
    CT_PX: cp.constexpr = 4,
    CT_PY: cp.constexpr = 5,
    CT_PZ: cp.constexpr = 6,
    # Shape field offsets
    SHAPE_SCALE_Y: cp.constexpr = 1,
    SHAPE_TYPE: cp.constexpr = 10,
    # CMI field offsets
    CMI_SHAPE_REF0: cp.constexpr = 0,
    CMI_SHAPE_REF1: cp.constexpr = 1,
    CMI_TRANSFORM_REF0: cp.constexpr = 2,
    CMI_TRANSFORM_REF1: cp.constexpr = 3,
    # CMO field offsets
    CMO_CONTACT_PATCHES_LO: cp.constexpr = 0,
    CMO_CONTACT_PATCHES_HI: cp.constexpr = 1,
    CMO_CONTACT_POINTS_LO: cp.constexpr = 2,
    CMO_CONTACT_POINTS_HI: cp.constexpr = 3,
    CMO_CONTACT_FORCES_LO: cp.constexpr = 4,
    CMO_CONTACT_FORCES_HI: cp.constexpr = 5,
    CMO_ALLFLAGS: cp.constexpr = 8,
    CMO_NB_CONTACTS: cp.constexpr = 9,
    # Material field offsets
    MAT_DYN_FRICTION: cp.constexpr = 0,
    MAT_STA_FRICTION: cp.constexpr = 1,
    MAT_RESTITUTION: cp.constexpr = 2,
    MAT_DAMPING: cp.constexpr = 3,
    MAT_FLAGS_MODES: cp.constexpr = 4,
    MAT_DAMPING_MODE: cp.constexpr = 5,
    # MultiManifold offsets (int32 indices)
    # mContacts starts at int32 offset 0.  Contact[sub][ct] at (sub*6+ct)*12.
    # PxgContact fields within a contact (12 int32):
    #   pointA: float offsets 0,1,2; pointB: 3,4,5; normal: 6,7,8;
    #   penetration: float offset 9; triIndex: int offset 10; pad: 11
    MM_NB_CONTACTS_BASE: cp.constexpr = 288,   # mNbContacts[0]
    MM_NB_MANIFOLDS: cp.constexpr = 300,        # mNbManifolds
    CONTACT_FLOATS: cp.constexpr = 12,          # sizeof(PxgContact)/4
    SUBMANIFOLD_MAX_CONTACTS: cp.constexpr = 6,
    # PatchAndContactCounters offsets
    COUNTER_PATCHES_BYTES: cp.constexpr = 0,
    COUNTER_CONTACTS_BYTES: cp.constexpr = 1,
    COUNTER_FORCE_BYTES: cp.constexpr = 2,
    COUNTER_OVERFLOW: cp.constexpr = 3,
    # sizeof constants
    SIZEOF_PX_CONTACT: cp.constexpr = 16,       # sizeof(PxContact) = float4 = 16
    SIZEOF_PX_CONTACT_PATCH: cp.constexpr = 64,
    SIZEOF_PX_U32: cp.constexpr = 4,
    # Geometry type enums
    GEO_SPHERE: cp.constexpr = 0,
    GEO_CAPSULE: cp.constexpr = 2,
    GEO_TRIANGLE_MESH: cp.constexpr = 8,
    # Status flags (PxsContactManagerStatusFlag)
    STATUS_HAS_NO_TOUCH: cp.constexpr = 1,
    STATUS_HAS_TOUCH: cp.constexpr = 2,
    STATUS_TOUCH_KNOWN: cp.constexpr = 3,
    # Material flags
    MATFLAG_DISABLE_FRICTION: cp.constexpr = 4,
    MATFLAG_DISABLE_STRONG_FRICTION: cp.constexpr = 8,
    MATFLAG_COMPLIANT_ACC_SPRING: cp.constexpr = 64,
    # Patch stream field offsets (PxContactPatch as int32[16])
    PS_MASS_MOD_LINEAR0: cp.constexpr = 0,
    PS_MASS_MOD_ANGULAR0: cp.constexpr = 1,
    PS_MASS_MOD_LINEAR1: cp.constexpr = 2,
    PS_MASS_MOD_ANGULAR1: cp.constexpr = 3,
    PS_NORMAL_X: cp.constexpr = 4,
    PS_NORMAL_Y: cp.constexpr = 5,
    PS_NORMAL_Z: cp.constexpr = 6,
    PS_RESTITUTION: cp.constexpr = 7,
    PS_DYN_FRICTION: cp.constexpr = 8,
    PS_STA_FRICTION: cp.constexpr = 9,
    PS_DAMPING: cp.constexpr = 10,
    PS_START_NB_MATFLAGS: cp.constexpr = 11,
    PS_INTFLAGS_MATIDX0: cp.constexpr = 12,
    PS_MATIDX1_PAD: cp.constexpr = 13,
    # PxContactPatch::eHAS_FACE_INDICES
    PATCH_FLAG_HAS_FACE_INDICES: cp.constexpr = 1,
    # Overflow error bit flags (match PxgPatchAndContactCounters::OverflowError)
    PATCH_BUFFER_OVERFLOW: cp.constexpr = 4,
    CONTACT_BUFFER_OVERFLOW: cp.constexpr = 1,
    FORCE_BUFFER_OVERFLOW: cp.constexpr = 2,
):
    BLOCK_SIZE = WARPS_PER_BLOCK * WARP_SIZE
    numBlocks = cp.ceildiv(numPairs, WARPS_PER_BLOCK)

    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for warp_id, warp in block.warps():
            globalWarpIndex = bx * cp.int32(WARPS_PER_BLOCK) + warp_id
            for lane, thread in warp.threads():
                if globalWarpIndex < numPairs:
                    # ============================================================
                    # 1. Read multi-manifold header
                    # ============================================================
                    nbManifolds = cmMultiManifold[globalWarpIndex, MM_NB_MANIFOLDS] + cp.int32(0)

                    singleManifoldIndex = lane // cp.int32(SUBMANIFOLD_MAX_CONTACTS)
                    threadIndexInManifold = lane % cp.int32(SUBMANIFOLD_MAX_CONTACTS)

                    # Number of contacts in this sub-manifold
                    numContacts = cp.int32(0)
                    if singleManifoldIndex < nbManifolds:
                        numContacts = cmMultiManifold[globalWarpIndex, MM_NB_CONTACTS_BASE + singleManifoldIndex] + cp.int32(0)

                    hasContacts = cp.int32(0)
                    if threadIndexInManifold < numContacts:
                        hasContacts = cp.int32(1)

                    # Ballot: which lanes have contacts
                    contactMask = thread.coll.ballot(hasContacts != cp.int32(0))
                    totalNumContacts = thread.popcount(contactMask)

                    # Exclusive scan: writeIndex = popcount(contactMask & ((1 << lane) - 1))
                    laneMaskLt = (cp.int32(1) << lane) - cp.int32(1)
                    writeIndex = thread.popcount(contactMask & laneMaskLt)

                    # ============================================================
                    # 2. Read/compute CMO allflags
                    # ============================================================
                    allflags = cmOutputs[globalWarpIndex, CMO_ALLFLAGS] + cp.int32(0)
                    # oldStatusFlags = byte 2 = (allflags >> 16) & 0xFF
                    oldStatusFlags = (allflags >> cp.int32(16)) & cp.int32(0xFF)
                    statusFlags = oldStatusFlags & (~cp.int32(STATUS_TOUCH_KNOWN))
                    # prevPatches = byte 1 (current nbPatches becomes prev)
                    prevPatches = (allflags >> cp.int32(8)) & cp.int32(0xFF)

                    if totalNumContacts != cp.int32(0):
                        statusFlags = statusFlags | cp.int32(STATUS_HAS_TOUCH)
                    else:
                        statusFlags = statusFlags | cp.int32(STATUS_HAS_NO_TOUCH)

                    # ============================================================
                    # 3. Lane 0: atomic alloc patch/contact/force buffers
                    # ============================================================
                    # Pre-declare variables written by lane 0 and broadcast later
                    patchByteOffset = cp.int32(-1)
                    contactByteOffset = cp.int32(-1)
                    forceAndIndiceByteOffset = cp.int32(-1)
                    overflow = cp.int32(0)

                    # These will be updated by lane 0 logic
                    nbManifolds_l0 = nbManifolds
                    totalNumContacts_l0 = totalNumContacts
                    statusFlags_l0 = statusFlags

                    if lane == cp.int32(0):
                        nbInsertAveragePoint = cp.int32(0)
                        if insertAveragePoint != cp.int32(0):
                            nbInsertAveragePoint = nbManifolds
                        totalNumContacts_l0 = totalNumContacts + nbInsertAveragePoint

                        if totalNumContacts_l0 != cp.int32(0):
                            patchByteOffset = thread.atomic_add(patchAndContactCounters[COUNTER_PATCHES_BYTES], cp.int32(SIZEOF_PX_CONTACT_PATCH) * nbManifolds)
                            contactByteOffset = thread.atomic_add(patchAndContactCounters[COUNTER_CONTACTS_BYTES], cp.int32(SIZEOF_PX_CONTACT) * totalNumContacts_l0)
                            forceAndIndiceByteOffset = thread.atomic_add(patchAndContactCounters[COUNTER_FORCE_BYTES], cp.int32(SIZEOF_PX_U32) * totalNumContacts_l0 * cp.int32(2))

                            # Check overflow: patches
                            patchEnd = patchByteOffset + cp.int32(SIZEOF_PX_CONTACT_PATCH) * nbManifolds
                            if patchEnd > patchBytesLimit:
                                # Set overflow flag via atomic OR
                                patchAndContactCounters[COUNTER_OVERFLOW] = patchAndContactCounters[COUNTER_OVERFLOW] | cp.int32(PATCH_BUFFER_OVERFLOW)
                                patchByteOffset = cp.int32(-1)
                                overflow = cp.int32(1)

                            if overflow == cp.int32(0):
                                contactEnd = contactByteOffset + cp.int32(SIZEOF_PX_CONTACT) * totalNumContacts_l0
                                if contactEnd > contactBytesLimit:
                                    patchAndContactCounters[COUNTER_OVERFLOW] = patchAndContactCounters[COUNTER_OVERFLOW] | cp.int32(CONTACT_BUFFER_OVERFLOW)
                                    contactByteOffset = cp.int32(-1)
                                    overflow = cp.int32(1)

                            if overflow == cp.int32(0):
                                forceEnd = forceAndIndiceByteOffset + cp.int32(SIZEOF_PX_U32) * totalNumContacts_l0 * cp.int32(2)
                                if forceEnd > forceBytesLimit:
                                    patchAndContactCounters[COUNTER_OVERFLOW] = patchAndContactCounters[COUNTER_OVERFLOW] | cp.int32(FORCE_BUFFER_OVERFLOW)
                                    forceAndIndiceByteOffset = cp.int32(-1)
                                    overflow = cp.int32(1)

                            if overflow != cp.int32(0):
                                nbManifolds_l0 = cp.int32(0)
                                totalNumContacts_l0 = cp.int32(0)
                                statusFlags_l0 = statusFlags_l0 & (~cp.int32(STATUS_TOUCH_KNOWN))
                                statusFlags_l0 = statusFlags_l0 | cp.int32(STATUS_HAS_NO_TOUCH)

                        # Determine touch change
                        previouslyHadTouch = oldStatusFlags & cp.int32(STATUS_HAS_TOUCH)
                        prevTouchKnown = oldStatusFlags & cp.int32(STATUS_TOUCH_KNOWN)
                        currentlyHasTouch = cp.int32(0)
                        if nbManifolds_l0 != cp.int32(0):
                            currentlyHasTouch = cp.int32(1)

                        change = cp.int32(0)
                        xorTouch = previouslyHadTouch ^ currentlyHasTouch
                        if xorTouch != cp.int32(0):
                            change = cp.int32(1)
                        if prevTouchKnown == cp.int32(0):
                            change = cp.int32(1)

                        touchChangeFlags[globalWarpIndex] = change

                        patchChanged = cp.int32(0)
                        if prevPatches != nbManifolds_l0:
                            patchChanged = cp.int32(1)
                        patchChangeFlags[globalWarpIndex] = patchChanged

                        # Pack allflags: merge(merge(prevPatches, statusFlags), merge(nbManifolds, 0))
                        # byte3=prevPatches, byte2=statusFlags_l0, byte1=nbManifolds_l0, byte0=0
                        newAllflags = (prevPatches << cp.int32(24)) | (statusFlags_l0 << cp.int32(16)) | (nbManifolds_l0 << cp.int32(8))
                        cmOutputs[globalWarpIndex, CMO_ALLFLAGS] = newAllflags

                        # nbContacts: low 16 bits of CMO word 9
                        oldNbContactWord = cmOutputs[globalWarpIndex, CMO_NB_CONTACTS] + cp.int32(0)
                        newNbContactWord = (oldNbContactWord & cp.int32(0xFFFF0000)) | (totalNumContacts_l0 & cp.int32(0xFFFF))
                        cmOutputs[globalWarpIndex, CMO_NB_CONTACTS] = newNbContactWord

                        if overflow == cp.int32(0):
                            # Store byte offsets as pointer lo/hi in output
                            # (matching cudaSphere.py convention: offset in lo, 0 in hi)
                            cmOutputs[globalWarpIndex, CMO_CONTACT_FORCES_LO] = forceAndIndiceByteOffset
                            cmOutputs[globalWarpIndex, CMO_CONTACT_FORCES_HI] = cp.int32(0)
                            cmOutputs[globalWarpIndex, CMO_CONTACT_PATCHES_LO] = patchByteOffset
                            cmOutputs[globalWarpIndex, CMO_CONTACT_PATCHES_HI] = cp.int32(0)
                            cmOutputs[globalWarpIndex, CMO_CONTACT_POINTS_LO] = contactByteOffset
                            cmOutputs[globalWarpIndex, CMO_CONTACT_POINTS_HI] = cp.int32(0)
                        else:
                            cmOutputs[globalWarpIndex, CMO_CONTACT_FORCES_LO] = cp.int32(0)
                            cmOutputs[globalWarpIndex, CMO_CONTACT_FORCES_HI] = cp.int32(0)
                            cmOutputs[globalWarpIndex, CMO_CONTACT_PATCHES_LO] = cp.int32(0)
                            cmOutputs[globalWarpIndex, CMO_CONTACT_PATCHES_HI] = cp.int32(0)
                            cmOutputs[globalWarpIndex, CMO_CONTACT_POINTS_LO] = cp.int32(0)
                            cmOutputs[globalWarpIndex, CMO_CONTACT_POINTS_HI] = cp.int32(0)

                    # Broadcast overflow from lane 0
                    overflowBcast = thread.shfl_idx(overflow, 0)
                    nbManifoldsBcast = thread.shfl_idx(nbManifolds_l0, 0)

                    # Early exit if overflow or no manifolds
                    if overflowBcast == cp.int32(0):
                        if nbManifoldsBcast != cp.int32(0):
                            # ============================================================
                            # 4. Read pair inputs (warp-read broadcast)
                            # ============================================================
                            shapeRef0 = cmInputs[globalWarpIndex, CMI_SHAPE_REF0] + cp.int32(0)
                            shapeRef1 = cmInputs[globalWarpIndex, CMI_SHAPE_REF1] + cp.int32(0)
                            transformCacheRef0 = cmInputs[globalWarpIndex, CMI_TRANSFORM_REF0] + cp.int32(0)
                            transformCacheRef1 = cmInputs[globalWarpIndex, CMI_TRANSFORM_REF1] + cp.int32(0)

                            # Check if shape0 is triangle mesh -> flip
                            shape0Type = gpuShapes[shapeRef0, SHAPE_TYPE] + cp.int32(0)
                            flip = cp.int32(0)
                            if shape0Type == cp.int32(GEO_TRIANGLE_MESH):
                                flip = cp.int32(1)
                                # Swap shapeRef and transformCacheRef
                                tmpSR = shapeRef0
                                shapeRef0 = shapeRef1
                                shapeRef1 = tmpSR
                                tmpTR = transformCacheRef0
                                transformCacheRef0 = transformCacheRef1
                                transformCacheRef1 = tmpTR

                            # Read shape0 (the convex/sphere/capsule shape)
                            s0_scale_y_i = gpuShapes[shapeRef0, SHAPE_SCALE_Y] + cp.int32(0)
                            s0_scale_y = thread.bitcast(s0_scale_y_i + cp.int32(0), cp.float32)
                            s0_type = gpuShapes[shapeRef0, SHAPE_TYPE] + cp.int32(0)

                            # Read material indices from pairs
                            materialIndex0 = pairs[globalWarpIndex, PAIR_MATIDX_X] + cp.int32(0)
                            materialIndex1 = pairs[globalWarpIndex, PAIR_MATIDX_Y] + cp.int32(0)

                            # Broadcast byte offsets from lane 0
                            patchByteOffset_b = thread.shfl_idx(patchByteOffset, 0)
                            contactByteOffset_b = thread.shfl_idx(contactByteOffset, 0)
                            forceAndIndiceByteOffset_b = thread.shfl_idx(forceAndIndiceByteOffset, 0)
                            totalNumContacts_b = thread.shfl_idx(totalNumContacts_l0, 0)

                            # ============================================================
                            # 5. Read cached transforms (trimesh = ref1, convex = ref0)
                            # ============================================================
                            tri_qx = transformCache[transformCacheRef1, CT_QX] + cp.float32(0.0)
                            tri_qy = transformCache[transformCacheRef1, CT_QY] + cp.float32(0.0)
                            tri_qz = transformCache[transformCacheRef1, CT_QZ] + cp.float32(0.0)
                            tri_qw = transformCache[transformCacheRef1, CT_QW] + cp.float32(0.0)
                            tri_px = transformCache[transformCacheRef1, CT_PX] + cp.float32(0.0)
                            tri_py = transformCache[transformCacheRef1, CT_PY] + cp.float32(0.0)
                            tri_pz = transformCache[transformCacheRef1, CT_PZ] + cp.float32(0.0)

                            sph_qx = transformCache[transformCacheRef0, CT_QX] + cp.float32(0.0)
                            sph_qy = transformCache[transformCacheRef0, CT_QY] + cp.float32(0.0)
                            sph_qz = transformCache[transformCacheRef0, CT_QZ] + cp.float32(0.0)
                            sph_qw = transformCache[transformCacheRef0, CT_QW] + cp.float32(0.0)
                            sph_px = transformCache[transformCacheRef0, CT_PX] + cp.float32(0.0)
                            sph_py = transformCache[transformCacheRef0, CT_PY] + cp.float32(0.0)
                            sph_pz = transformCache[transformCacheRef0, CT_PZ] + cp.float32(0.0)

                            # ============================================================
                            # 6. Compute world normal (getWorldNormal)
                            #    Average normals from this sub-manifold's contacts,
                            #    rotate by mesh transform, normalize.
                            # ============================================================
                            # Each lane reads its contact's normal (or zero if inactive)
                            contactBaseOffset = (singleManifoldIndex * cp.int32(SUBMANIFOLD_MAX_CONTACTS) + threadIndexInManifold) * cp.int32(CONTACT_FLOATS)

                            # Read normal from mContacts (stored as int32, bitcast to float)
                            norm_x_i = cp.int32(0)
                            norm_y_i = cp.int32(0)
                            norm_z_i = cp.int32(0)
                            if hasContacts != cp.int32(0):
                                norm_x_i = cmMultiManifold[globalWarpIndex, contactBaseOffset + cp.int32(6)] + cp.int32(0)
                                norm_y_i = cmMultiManifold[globalWarpIndex, contactBaseOffset + cp.int32(7)] + cp.int32(0)
                                norm_z_i = cmMultiManifold[globalWarpIndex, contactBaseOffset + cp.int32(8)] + cp.int32(0)

                            local_nx = cp.float32(0.0)
                            local_ny = cp.float32(0.0)
                            local_nz = cp.float32(0.0)
                            if hasContacts != cp.int32(0):
                                local_nx = thread.bitcast(norm_x_i + cp.int32(0), cp.float32)
                                local_ny = thread.bitcast(norm_y_i + cp.int32(0), cp.float32)
                                local_nz = thread.bitcast(norm_z_i + cp.int32(0), cp.float32)

                            # In getWorldNormal, all contacts in the same sub-manifold
                            # share the same normal (contacts[0].normal), so the "average"
                            # is just the first contact's normal.  The CUDA code reads
                            # contacts[0].normal for all threads in the manifold group.
                            # For threads with threadIndexInManifold >= numContacts, normal = 0.
                            # The first lane of each sub-manifold group always has the normal.
                            # We broadcast from the base lane of each sub-manifold group.
                            baseLane = singleManifoldIndex * cp.int32(SUBMANIFOLD_MAX_CONTACTS)
                            avg_nx = thread.shfl_idx(local_nx, baseLane)
                            avg_ny = thread.shfl_idx(local_ny, baseLane)
                            avg_nz = thread.shfl_idx(local_nz, baseLane)

                            # Rotate by mesh transform (trimesh)
                            wn_x = quat_rotate_x(tri_qx, tri_qy, tri_qz, tri_qw, avg_nx, avg_ny, avg_nz)
                            wn_y = quat_rotate_y(tri_qx, tri_qy, tri_qz, tri_qw, avg_nx, avg_ny, avg_nz)
                            wn_z = quat_rotate_z(tri_qx, tri_qy, tri_qz, tri_qw, avg_nx, avg_ny, avg_nz)

                            # Normalize
                            lenSq = wn_x * wn_x + wn_y * wn_y + wn_z * wn_z
                            invLen = thread.rsqrt(lenSq)
                            wn_x = wn_x * invLen
                            wn_y = wn_y * invLen
                            wn_z = wn_z * invLen

                            # ============================================================
                            # 7. Write contacts (only lanes with hasContacts)
                            # ============================================================
                            if hasContacts != cp.int32(0):
                                # Read contact data: pointA, pointB, penetration, triIndex
                                ptA_x_i = cmMultiManifold[globalWarpIndex, contactBaseOffset + cp.int32(0)] + cp.int32(0)
                                ptA_y_i = cmMultiManifold[globalWarpIndex, contactBaseOffset + cp.int32(1)] + cp.int32(0)
                                ptA_z_i = cmMultiManifold[globalWarpIndex, contactBaseOffset + cp.int32(2)] + cp.int32(0)
                                ptB_x_i = cmMultiManifold[globalWarpIndex, contactBaseOffset + cp.int32(3)] + cp.int32(0)
                                ptB_y_i = cmMultiManifold[globalWarpIndex, contactBaseOffset + cp.int32(4)] + cp.int32(0)
                                ptB_z_i = cmMultiManifold[globalWarpIndex, contactBaseOffset + cp.int32(5)] + cp.int32(0)
                                pen_i = cmMultiManifold[globalWarpIndex, contactBaseOffset + cp.int32(9)] + cp.int32(0)
                                triIndex = cmMultiManifold[globalWarpIndex, contactBaseOffset + cp.int32(10)] + cp.int32(0)

                                ptA_x = thread.bitcast(ptA_x_i + cp.int32(0), cp.float32)
                                ptA_y = thread.bitcast(ptA_y_i + cp.int32(0), cp.float32)
                                ptA_z = thread.bitcast(ptA_z_i + cp.int32(0), cp.float32)
                                ptB_x = thread.bitcast(ptB_x_i + cp.int32(0), cp.float32)
                                ptB_y = thread.bitcast(ptB_y_i + cp.int32(0), cp.float32)
                                ptB_z = thread.bitcast(ptB_z_i + cp.int32(0), cp.float32)
                                pen = thread.bitcast(pen_i + cp.int32(0), cp.float32)

                                # --------------------------------------------------------
                                # Write patch header (lane 0 of each sub-manifold)
                                # --------------------------------------------------------
                                if threadIndexInManifold == cp.int32(0):
                                    # Flip normal if needed
                                    patch_nx = wn_x
                                    patch_ny = wn_y
                                    patch_nz = wn_z
                                    if flip != cp.int32(0):
                                        patch_nx = -wn_x
                                        patch_ny = -wn_y
                                        patch_nz = -wn_z

                                    # combineMaterials inlined
                                    mat0_dynFric_i = materials[materialIndex0, MAT_DYN_FRICTION] + cp.int32(0)
                                    mat0_staFric_i = materials[materialIndex0, MAT_STA_FRICTION] + cp.int32(0)
                                    mat0_rest_i = materials[materialIndex0, MAT_RESTITUTION] + cp.int32(0)
                                    mat0_damp_i = materials[materialIndex0, MAT_DAMPING] + cp.int32(0)
                                    mat0_flagsModes = materials[materialIndex0, MAT_FLAGS_MODES] + cp.int32(0)
                                    mat0_dampMode_raw = materials[materialIndex0, MAT_DAMPING_MODE] + cp.int32(0)

                                    mat1_dynFric_i = materials[materialIndex1, MAT_DYN_FRICTION] + cp.int32(0)
                                    mat1_staFric_i = materials[materialIndex1, MAT_STA_FRICTION] + cp.int32(0)
                                    mat1_rest_i = materials[materialIndex1, MAT_RESTITUTION] + cp.int32(0)
                                    mat1_damp_i = materials[materialIndex1, MAT_DAMPING] + cp.int32(0)
                                    mat1_flagsModes = materials[materialIndex1, MAT_FLAGS_MODES] + cp.int32(0)
                                    mat1_dampMode_raw = materials[materialIndex1, MAT_DAMPING_MODE] + cp.int32(0)

                                    mat0_dynFric = thread.bitcast(mat0_dynFric_i + cp.int32(0), cp.float32)
                                    mat0_staFric = thread.bitcast(mat0_staFric_i + cp.int32(0), cp.float32)
                                    mat0_rest = thread.bitcast(mat0_rest_i + cp.int32(0), cp.float32)
                                    mat0_damp = thread.bitcast(mat0_damp_i + cp.int32(0), cp.float32)
                                    mat1_dynFric = thread.bitcast(mat1_dynFric_i + cp.int32(0), cp.float32)
                                    mat1_staFric = thread.bitcast(mat1_staFric_i + cp.int32(0), cp.float32)
                                    mat1_rest = thread.bitcast(mat1_rest_i + cp.int32(0), cp.float32)
                                    mat1_damp = thread.bitcast(mat1_damp_i + cp.int32(0), cp.float32)

                                    # flags: low 16 bits
                                    mat0_flags = mat0_flagsModes & cp.int32(0xFFFF)
                                    mat1_flags = mat1_flagsModes & cp.int32(0xFFFF)
                                    # friction combine mode: bits 16-23
                                    mat0_fricMode = (mat0_flagsModes >> cp.int32(16)) & cp.int32(0xFF)
                                    mat1_fricMode = (mat1_flagsModes >> cp.int32(16)) & cp.int32(0xFF)
                                    # restitution combine mode: bits 24-31
                                    mat0_restMode = (mat0_flagsModes >> cp.int32(24)) & cp.int32(0xFF)
                                    mat1_restMode = (mat1_flagsModes >> cp.int32(24)) & cp.int32(0xFF)
                                    # damping combine mode: low byte of MAT_DAMPING_MODE
                                    mat0_dampMode = mat0_dampMode_raw & cp.int32(0xFF)
                                    mat1_dampMode = mat1_dampMode_raw & cp.int32(0xFF)

                                    # -- Combine restitution --
                                    compliant0 = cp.int32(0)
                                    if mat0_rest < cp.float32(0.0):
                                        compliant0 = cp.int32(1)
                                    compliant1 = cp.int32(0)
                                    if mat1_rest < cp.float32(0.0):
                                        compliant1 = cp.int32(1)
                                    bothCompliant = compliant0 & compliant1
                                    exactlyOneCompliant = compliant0 ^ compliant1

                                    compliantAcc0 = cp.int32(0)
                                    if (mat0_flags & cp.int32(MATFLAG_COMPLIANT_ACC_SPRING)) != cp.int32(0):
                                        compliantAcc0 = cp.int32(1)
                                    compliantAcc1 = cp.int32(0)
                                    if (mat1_flags & cp.int32(MATFLAG_COMPLIANT_ACC_SPRING)) != cp.int32(0):
                                        compliantAcc1 = cp.int32(1)
                                    exactlyOneAccCompliant = compliantAcc0 ^ compliantAcc1

                                    combinedRestitution = cp.float32(0.0)
                                    if (bothCompliant != cp.int32(0)) & (exactlyOneAccCompliant != cp.int32(0)):
                                        combinedRestitution = mat0_rest
                                        if compliantAcc0 == cp.int32(0):
                                            combinedRestitution = mat1_rest
                                    else:
                                        restCombineMode = mat0_restMode
                                        if mat1_restMode > mat0_restMode:
                                            restCombineMode = mat1_restMode
                                        if exactlyOneCompliant != cp.int32(0):
                                            restCombineMode = cp.int32(1)  # eMIN
                                        flipSign = cp.float32(1.0)
                                        if (bothCompliant != cp.int32(0)) & (restCombineMode == cp.int32(2)):
                                            flipSign = cp.float32(-1.0)
                                        combinedRestitution = flipSign * combine_scalars(mat0_rest, mat1_rest, restCombineMode)

                                    # -- Combine damping --
                                    combinedDamping = cp.float32(0.0)
                                    if (bothCompliant != cp.int32(0)) & (exactlyOneAccCompliant != cp.int32(0)):
                                        combinedDamping = mat0_damp
                                        if compliantAcc0 == cp.int32(0):
                                            combinedDamping = mat1_damp
                                    else:
                                        dampCombineMode = mat0_dampMode
                                        if mat1_dampMode > mat0_dampMode:
                                            dampCombineMode = mat1_dampMode
                                        if exactlyOneCompliant != cp.int32(0):
                                            dampCombineMode = cp.int32(3)  # eMAX
                                        combinedDamping = combine_scalars(mat0_damp, mat1_damp, dampCombineMode)

                                    # -- Combine friction --
                                    combineFlags = mat0_flags | mat1_flags
                                    combinedStaticFriction = cp.float32(0.0)
                                    combinedDynamicFriction = cp.float32(0.0)
                                    combinedMaterialFlags = combineFlags

                                    if (combineFlags & cp.int32(MATFLAG_DISABLE_FRICTION)) == cp.int32(0):
                                        fricCombineMode = mat0_fricMode
                                        if mat1_fricMode > mat0_fricMode:
                                            fricCombineMode = mat1_fricMode
                                        dynFric = combine_scalars(mat0_dynFric, mat1_dynFric, fricCombineMode)
                                        staFric = combine_scalars(mat0_staFric, mat1_staFric, fricCombineMode)
                                        fDynFriction = max_f32(dynFric, cp.float32(0.0))
                                        fStaFriction = fDynFriction
                                        if (staFric - fDynFriction) >= cp.float32(0.0):
                                            fStaFriction = staFric
                                        combinedDynamicFriction = fDynFriction
                                        combinedStaticFriction = fStaFriction
                                    else:
                                        combinedMaterialFlags = combineFlags | cp.int32(MATFLAG_DISABLE_STRONG_FRICTION)

                                    # Write PxContactPatch to patchStream
                                    # patchStream row index = patchByteOffset_b / SIZEOF_PX_CONTACT_PATCH + singleManifoldIndex
                                    patchRowBase = patchByteOffset_b // cp.int32(SIZEOF_PX_CONTACT_PATCH)
                                    patchRow = patchRowBase + singleManifoldIndex

                                    # mass modification = 1.0 for all 4 fields
                                    one_f_i = thread.bitcast(cp.float32(1.0), cp.int32)
                                    patchStream[patchRow, PS_MASS_MOD_LINEAR0] = one_f_i
                                    patchStream[patchRow, PS_MASS_MOD_ANGULAR0] = one_f_i
                                    patchStream[patchRow, PS_MASS_MOD_LINEAR1] = one_f_i
                                    patchStream[patchRow, PS_MASS_MOD_ANGULAR1] = one_f_i

                                    # normal
                                    patchStream[patchRow, PS_NORMAL_X] = thread.bitcast(patch_nx, cp.int32)
                                    patchStream[patchRow, PS_NORMAL_Y] = thread.bitcast(patch_ny, cp.int32)
                                    patchStream[patchRow, PS_NORMAL_Z] = thread.bitcast(patch_nz, cp.int32)

                                    # restitution, friction, damping
                                    patchStream[patchRow, PS_RESTITUTION] = thread.bitcast(combinedRestitution, cp.int32)
                                    patchStream[patchRow, PS_DYN_FRICTION] = thread.bitcast(combinedDynamicFriction, cp.int32)
                                    patchStream[patchRow, PS_STA_FRICTION] = thread.bitcast(combinedStaticFriction, cp.int32)
                                    patchStream[patchRow, PS_DAMPING] = thread.bitcast(combinedDamping, cp.int32)

                                    # PxContactPatch word 11 (PS_START_NB_MATFLAGS):
                                    #   startContactIndex(u16) | nbContacts(u8)<<16 | materialFlags(u8)<<24
                                    startNbMatflags = (writeIndex & cp.int32(0xFFFF)) | ((numContacts & cp.int32(0xFF)) << cp.int32(16)) | ((combinedMaterialFlags & cp.int32(0xFF)) << cp.int32(24))
                                    patchStream[patchRow, PS_START_NB_MATFLAGS] = startNbMatflags

                                    # PxContactPatch word 12 (PS_INTFLAGS_MATIDX0):
                                    #   internalFlags(u16) | materialIndex0(u16)<<16
                                    intflagsMatidx0 = cp.int32(PATCH_FLAG_HAS_FACE_INDICES) | ((materialIndex0 & cp.int32(0xFFFF)) << cp.int32(16))
                                    patchStream[patchRow, PS_INTFLAGS_MATIDX0] = intflagsMatidx0

                                    # PxContactPatch word 13 (PS_MATIDX1_PAD):
                                    #   materialIndex1(u16) | pad(u16)
                                    patchStream[patchRow, PS_MATIDX1_PAD] = materialIndex1 & cp.int32(0xFFFF)

                                # --------------------------------------------------------
                                # Compute world contact point and penetration
                                # --------------------------------------------------------
                                # Pre-declare for if/elif/else branches
                                worldPt_x = cp.float32(0.0)
                                worldPt_y = cp.float32(0.0)
                                worldPt_z = cp.float32(0.0)
                                finalPen = pen
                                radius = cp.float32(0.0)
                                tptA_x = cp.float32(0.0)
                                tptA_y = cp.float32(0.0)
                                tptA_z = cp.float32(0.0)

                                if s0_type == cp.int32(GEO_SPHERE):
                                    radius = s0_scale_y
                                    finalPen = pen - radius
                                    worldPt_x = sph_px - wn_x * radius
                                    worldPt_y = sph_py - wn_y * radius
                                    worldPt_z = sph_pz - wn_z * radius
                                elif s0_type == cp.int32(GEO_CAPSULE):
                                    radius = s0_scale_y
                                    finalPen = pen - radius
                                    # sphereTransform.transform(pointA) - worldNormal * radius
                                    tptA_x = quat_rotate_x(sph_qx, sph_qy, sph_qz, sph_qw, ptA_x, ptA_y, ptA_z) + sph_px
                                    tptA_y = quat_rotate_y(sph_qx, sph_qy, sph_qz, sph_qw, ptA_x, ptA_y, ptA_z) + sph_py
                                    tptA_z = quat_rotate_z(sph_qx, sph_qy, sph_qz, sph_qw, ptA_x, ptA_y, ptA_z) + sph_pz
                                    worldPt_x = tptA_x - wn_x * radius
                                    worldPt_y = tptA_y - wn_y * radius
                                    worldPt_z = tptA_z - wn_z * radius
                                else:
                                    # General convex: trimeshTransform.transform(pointB)
                                    worldPt_x = quat_rotate_x(tri_qx, tri_qy, tri_qz, tri_qw, ptB_x, ptB_y, ptB_z) + tri_px
                                    worldPt_y = quat_rotate_y(tri_qx, tri_qy, tri_qz, tri_qw, ptB_x, ptB_y, ptB_z) + tri_py
                                    worldPt_z = quat_rotate_z(tri_qx, tri_qy, tri_qz, tri_qw, ptB_x, ptB_y, ptB_z) + tri_pz

                                # Write to contactStream (float4: x,y,z,pen)
                                # contactStream row = contactByteOffset_b / SIZEOF_PX_CONTACT + writeIndex
                                if contactByteOffset_b != cp.int32(-1):
                                    contactRow = contactByteOffset_b // cp.int32(SIZEOF_PX_CONTACT) + writeIndex
                                    contactStream[contactRow, 0] = worldPt_x
                                    contactStream[contactRow, 1] = worldPt_y
                                    contactStream[contactRow, 2] = worldPt_z
                                    contactStream[contactRow, 3] = finalPen

                                # Write face index to forceAndIndice stream
                                # faceIndex array starts at forceAndIndiceByteOffset + totalNumContacts*sizeof(PxU32)
                                if forceAndIndiceByteOffset_b != cp.int32(-1):
                                    faceIndexBase = forceAndIndiceByteOffset_b // cp.int32(SIZEOF_PX_U32) + totalNumContacts_b
                                    forceAndIndiceStream[faceIndexBase + writeIndex] = triIndex
