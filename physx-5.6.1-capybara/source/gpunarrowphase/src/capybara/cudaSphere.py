"""Capybara DSL port of gpunarrowphase/CUDA/cudaSphere.cu -- sphereNphase_Kernel.

Ported kernel (matching CUDA name for PTX replacement):
  - sphereNphase_Kernel  -- narrowphase collision for sphere-based geometry pairs

Collision functions inlined (from sphereCollision.cuh):
  - spheresphere:     sphere vs sphere (1 contact max)
  - sphereplane:      sphere vs plane (1 contact max)
  - spherecapsule:    sphere vs capsule (1 contact max)
  - spherebox:        sphere vs box (1 contact max)
  - planeCapsule:     plane vs capsule (2 contacts max)
  - capsuleCapsule:   capsule vs capsule (2 contacts max)

Output helpers inlined (from contactPatchUtils.cuh):
  - setContactPointAndForcePointers: atomic alloc in contact/force byte streams
  - registerContactPatch: atomic alloc in patch stream, touch/patch change flags
  - insertIntoPatchStream: write PxContactPatch header (material combine + normal)

ABI differences from CUDA:
  - PxgContactManagerInput  -> int32[N, CMI_SIZE] flat tensor
      CMI_SHAPE_REF0=0, CMI_SHAPE_REF1=1, CMI_TRANSFORM_REF0=2, CMI_TRANSFORM_REF1=3
  - PxsContactManagerOutput -> int32[N, CMO_SIZE] flat tensor
      Pointer fields stored as pairs of int32 (lo,hi) for 64-bit addresses.
      CMO_CONTACT_PATCHES=0..1, CMO_CONTACT_POINTS=2..3, CMO_CONTACT_FORCES=4..5,
      CMO_FRICTION_PATCHES=6..7, CMO_ALLFLAGS=8 (packed u8x4), CMO_NB_CONTACTS=9,
      CMO_PAD=10..11
  - PxgShape -> int32[N, SHAPE_SIZE] flat tensor (48 bytes = 12 int32s on 64-bit)
      SHAPE_SCALE_X=0, SHAPE_SCALE_Y=1, SHAPE_SCALE_Z=2 (float as int32 bitcast)
      SHAPE_MATERIAL_INDEX=7 (uint32, materialIndex stored as PxU32)
      SHAPE_TYPE=10
  - PxsCachedTransform -> float32[N, CT_SIZE] flat tensor (32 bytes = 8 float32s)
      CT_QX=0, CT_QY=1, CT_QZ=2, CT_QW=3, CT_PX=4, CT_PY=5, CT_PZ=6, CT_FLAGS=7
  - contactDistance -> float32[N] tensor
  - PxsMaterialData -> int32[N, MAT_SIZE] flat tensor (24 bytes = 6 int32s)
      MAT_DYN_FRICTION=0, MAT_STA_FRICTION=1, MAT_RESTITUTION=2, MAT_DAMPING=3
      MAT_FLAGS_MODES=4 (u16 flags | u8 fricCombineMode<<16 | u8 restCombineMode<<24)
      MAT_DAMPING_MODE=5 (u8 dampingCombineMode in low byte)
  - contactStream -> float32[maxContacts, 4] (PxContact: x,y,z,separation)
  - patchStream -> int32[maxPatches, PATCH_SIZE] (PxContactPatch as int32 row, 64 bytes = 16 int32s)
  - PxgPatchAndContactCounters -> int32[4] (patchesBytes[0], contactsBytes[1],
      forceAndIndiceBytes[2], overflowError[3])
  - touchChangeFlags, patchChangeFlags -> int32[N]
  - startContactPatches, startContactPoints, startContactForces -> int64 scalars (base ptrs)
  - patchBytesLimit, contactBytesLimit, forceBytesLimit -> int32 scalars

Capybara structural notes:
  - All collision functions are inlined as scalar math (no @cp.inline with multi-return).
  - Variables assigned in if/elif/else must be pre-declared before the conditional.
  - No method chaining on Capybara structs; use intermediate variables.
  - `+ cp.float32(0.0)` / `+ cp.int32(0)` for force-loads from tensors.
  - `thread.bitcast()` for int32<->float32 reinterpretation.
  - `thread.sqrt()` for sqrt, `thread.rsqrt()` for reciprocal sqrt.
  - Boolean flags stored as cp.int32, not Python bool.
  - Geometry type enums: eSPHERE=0, ePLANE=1, eCAPSULE=2, eBOX=3.
"""

import capybara as cp


# ===== Helper: quaternion rotate vector =====
# q.rotate(v) = v + 2*cross(q.xyz, cross(q.xyz, v) + q.w*v)
@cp.inline
def quat_rotate_x(qx, qy, qz, qw, vx, vy, vz):
    """Returns x component of quaternion rotation of vector (vx,vy,vz)."""
    # cross1 = cross(q.xyz, v) + q.w * v
    c1x = qy * vz - qz * vy + qw * vx
    c1y = qz * vx - qx * vz + qw * vy
    c1z = qx * vy - qy * vx + qw * vz
    # cross2 = cross(q.xyz, c1)
    c2x = qy * c1z - qz * c1y
    return vx + cp.float32(2.0) * c2x


@cp.inline
def quat_rotate_y(qx, qy, qz, qw, vx, vy, vz):
    """Returns y component of quaternion rotation of vector (vx,vy,vz)."""
    c1x = qy * vz - qz * vy + qw * vx
    c1y = qz * vx - qx * vz + qw * vy
    c1z = qx * vy - qy * vx + qw * vz
    c2y = qz * c1x - qx * c1z
    return vy + cp.float32(2.0) * c2y


@cp.inline
def quat_rotate_z(qx, qy, qz, qw, vx, vy, vz):
    """Returns z component of quaternion rotation of vector (vx,vy,vz)."""
    c1x = qy * vz - qz * vy + qw * vx
    c1y = qz * vx - qx * vz + qw * vy
    c1z = qx * vy - qy * vx + qw * vz
    c2z = qx * c1y - qy * c1x
    return vz + cp.float32(2.0) * c2z


# ===== Helper: quaternion inverse rotate vector =====
# q.rotateInv(v) = v + 2*cross(-q.xyz, cross(-q.xyz, v) + q.w*v)
# equivalently: conjugate q then rotate
@cp.inline
def quat_rotate_inv_x(qx, qy, qz, qw, vx, vy, vz):
    """Returns x component of inverse quaternion rotation."""
    nqx = -qx
    nqy = -qy
    nqz = -qz
    c1x = nqy * vz - nqz * vy + qw * vx
    c1y = nqz * vx - nqx * vz + qw * vy
    c1z = nqx * vy - nqy * vx + qw * vz
    c2x = nqy * c1z - nqz * c1y
    return vx + cp.float32(2.0) * c2x


@cp.inline
def quat_rotate_inv_y(qx, qy, qz, qw, vx, vy, vz):
    """Returns y component of inverse quaternion rotation."""
    nqx = -qx
    nqy = -qy
    nqz = -qz
    c1x = nqy * vz - nqz * vy + qw * vx
    c1y = nqz * vx - nqx * vz + qw * vy
    c1z = nqx * vy - nqy * vx + qw * vz
    c2y = nqz * c1x - nqx * c1z
    return vy + cp.float32(2.0) * c2y


@cp.inline
def quat_rotate_inv_z(qx, qy, qz, qw, vx, vy, vz):
    """Returns z component of inverse quaternion rotation."""
    nqx = -qx
    nqy = -qy
    nqz = -qz
    c1x = nqy * vz - nqz * vy + qw * vx
    c1y = nqz * vx - nqx * vz + qw * vy
    c1z = nqx * vy - nqy * vx + qw * vz
    c2z = nqx * c1y - nqy * c1x
    return vz + cp.float32(2.0) * c2z


# ===== Helper: getBasisVector0 from quaternion =====
# Returns the x-axis (first column) of the rotation matrix built from quaternion.
# basisVector0 = rotate(q, (1,0,0))
# = (1 - 2*(qy^2+qz^2), 2*(qx*qy+qw*qz), 2*(qx*qz-qw*qy))
@cp.inline
def basis0_x(qx, qy, qz, qw):
    return cp.float32(1.0) - cp.float32(2.0) * (qy * qy + qz * qz)


@cp.inline
def basis0_y(qx, qy, qz, qw):
    return cp.float32(2.0) * (qx * qy + qw * qz)


@cp.inline
def basis0_z(qx, qy, qz, qw):
    return cp.float32(2.0) * (qx * qz - qw * qy)


# ===== Helper: clamp =====
@cp.inline
def clamp_f32(val, lo, hi):
    r = val
    if val < lo:
        r = lo
    if val > hi:
        r = hi
    return r


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


# ===== Helper: max of two int32 =====
@cp.inline
def max_i32(a, b):
    r = a
    if b > a:
        r = b
    return r


# ===== Helper: max of two float32 =====
@cp.inline
def max_f32(a, b):
    r = a
    if b > a:
        r = b
    return r


# ===== Helper: abs float =====
@cp.inline
def abs_f32(v):
    r = v
    if v < cp.float32(0.0):
        r = -v
    return r


# ===== Kernel: sphereNphase_Kernel =====
@cp.kernel
def sphereNphase_Kernel(
    numTests,               # int32 scalar
    toleranceLength,        # float32 scalar (unused in sphere kernel, kept for ABI)
    cmInputs,              # int32[N, CMI_SIZE]
    cmOutputs,             # int32[N, CMO_SIZE]
    shapes,                # int32[N, SHAPE_SIZE]
    transformCache,        # float32[N, CT_SIZE]
    contactDistance,        # float32[N]
    materials,             # int32[N, MAT_SIZE]
    contactStream,         # float32[maxContacts, 4]  (PxContact: x,y,z,pen)
    patchStream,           # int32[maxPatches, PATCH_SIZE]  (PxContactPatch as int32)
    patchAndContactCounters,  # int32[4]
    touchChangeFlags,      # int32[N]
    patchChangeFlags,      # int32[N]
    startContactPatches,   # int64 scalar (base GPU ptr for patches)
    startContactPoints,    # int64 scalar (base GPU ptr for contact points)
    startContactForces,    # int64 scalar (base GPU ptr for forces)
    patchBytesLimit,       # int32 scalar
    contactBytesLimit,     # int32 scalar
    forceBytesLimit,       # int32 scalar
    BLOCK_SIZE: cp.constexpr = 256,
    # --- struct sizes ---
    CMI_SIZE: cp.constexpr = 4,
    CMO_SIZE: cp.constexpr = 12,
    SHAPE_SIZE: cp.constexpr = 12,
    CT_SIZE: cp.constexpr = 8,
    MAT_SIZE: cp.constexpr = 6,
    PATCH_SIZE: cp.constexpr = 16,
    # --- CMI field offsets ---
    CMI_SHAPE_REF0: cp.constexpr = 0,
    CMI_SHAPE_REF1: cp.constexpr = 1,
    CMI_TRANSFORM_REF0: cp.constexpr = 2,
    CMI_TRANSFORM_REF1: cp.constexpr = 3,
    # --- CMO field offsets ---
    CMO_CONTACT_PATCHES_LO: cp.constexpr = 0,
    CMO_CONTACT_PATCHES_HI: cp.constexpr = 1,
    CMO_CONTACT_POINTS_LO: cp.constexpr = 2,
    CMO_CONTACT_POINTS_HI: cp.constexpr = 3,
    CMO_CONTACT_FORCES_LO: cp.constexpr = 4,
    CMO_CONTACT_FORCES_HI: cp.constexpr = 5,
    CMO_FRICTION_PATCHES_LO: cp.constexpr = 6,
    CMO_FRICTION_PATCHES_HI: cp.constexpr = 7,
    CMO_ALLFLAGS: cp.constexpr = 8,
    CMO_NB_CONTACTS: cp.constexpr = 9,
    # --- Shape field offsets ---
    SHAPE_SCALE_X: cp.constexpr = 0,
    SHAPE_SCALE_Y: cp.constexpr = 1,
    SHAPE_SCALE_Z: cp.constexpr = 2,
    SHAPE_MATERIAL_INDEX: cp.constexpr = 7,
    SHAPE_TYPE: cp.constexpr = 10,
    # --- CachedTransform field offsets ---
    CT_QX: cp.constexpr = 0,
    CT_QY: cp.constexpr = 1,
    CT_QZ: cp.constexpr = 2,
    CT_QW: cp.constexpr = 3,
    CT_PX: cp.constexpr = 4,
    CT_PY: cp.constexpr = 5,
    CT_PZ: cp.constexpr = 6,
    # --- Material field offsets ---
    MAT_DYN_FRICTION: cp.constexpr = 0,
    MAT_STA_FRICTION: cp.constexpr = 1,
    MAT_RESTITUTION: cp.constexpr = 2,
    MAT_DAMPING: cp.constexpr = 3,
    MAT_FLAGS_MODES: cp.constexpr = 4,
    MAT_DAMPING_MODE: cp.constexpr = 5,
    # --- PatchAndContactCounters offsets ---
    COUNTER_PATCHES_BYTES: cp.constexpr = 0,
    COUNTER_CONTACTS_BYTES: cp.constexpr = 1,
    COUNTER_FORCE_BYTES: cp.constexpr = 2,
    COUNTER_OVERFLOW: cp.constexpr = 3,
    # --- sizeof constants ---
    SIZEOF_PX_CONTACT: cp.constexpr = 16,
    SIZEOF_PX_CONTACT_PATCH: cp.constexpr = 64,
    SIZEOF_PX_U32: cp.constexpr = 4,
    # --- Geometry type enums ---
    GEO_SPHERE: cp.constexpr = 0,
    GEO_PLANE: cp.constexpr = 1,
    GEO_CAPSULE: cp.constexpr = 2,
    GEO_BOX: cp.constexpr = 3,
    # --- Status flags ---
    STATUS_HAS_NO_TOUCH: cp.constexpr = 1,
    STATUS_HAS_TOUCH: cp.constexpr = 2,
    STATUS_TOUCH_KNOWN: cp.constexpr = 3,
    # --- Material flags ---
    MATFLAG_DISABLE_FRICTION: cp.constexpr = 4,
    MATFLAG_DISABLE_STRONG_FRICTION: cp.constexpr = 8,
    MATFLAG_COMPLIANT_ACC_SPRING: cp.constexpr = 64,
    # --- Patch stream field offsets (PxContactPatch as int32[16]) ---
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
):
    numBlocks = cp.ceildiv(numTests, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            globalThreadIndex = bx * cp.int32(BLOCK_SIZE) + tid

            if globalThreadIndex < numTests:
                # ============================================================
                # 1. Read collision pair input
                # ============================================================
                shapeRef0 = cmInputs[globalThreadIndex, CMI_SHAPE_REF0] + cp.int32(0)
                shapeRef1 = cmInputs[globalThreadIndex, CMI_SHAPE_REF1] + cp.int32(0)
                transformCacheRef0 = cmInputs[globalThreadIndex, CMI_TRANSFORM_REF0] + cp.int32(0)
                transformCacheRef1 = cmInputs[globalThreadIndex, CMI_TRANSFORM_REF1] + cp.int32(0)

                # ============================================================
                # 2. Read transforms (as float32)
                # ============================================================
                t0_qx = transformCache[transformCacheRef0, CT_QX] + cp.float32(0.0)
                t0_qy = transformCache[transformCacheRef0, CT_QY] + cp.float32(0.0)
                t0_qz = transformCache[transformCacheRef0, CT_QZ] + cp.float32(0.0)
                t0_qw = transformCache[transformCacheRef0, CT_QW] + cp.float32(0.0)
                t0_px = transformCache[transformCacheRef0, CT_PX] + cp.float32(0.0)
                t0_py = transformCache[transformCacheRef0, CT_PY] + cp.float32(0.0)
                t0_pz = transformCache[transformCacheRef0, CT_PZ] + cp.float32(0.0)

                t1_qx = transformCache[transformCacheRef1, CT_QX] + cp.float32(0.0)
                t1_qy = transformCache[transformCacheRef1, CT_QY] + cp.float32(0.0)
                t1_qz = transformCache[transformCacheRef1, CT_QZ] + cp.float32(0.0)
                t1_qw = transformCache[transformCacheRef1, CT_QW] + cp.float32(0.0)
                t1_px = transformCache[transformCacheRef1, CT_PX] + cp.float32(0.0)
                t1_py = transformCache[transformCacheRef1, CT_PY] + cp.float32(0.0)
                t1_pz = transformCache[transformCacheRef1, CT_PZ] + cp.float32(0.0)

                # ============================================================
                # 3. Read contact distance
                # ============================================================
                cDist_raw0 = contactDistance[transformCacheRef0] + cp.float32(0.0)
                cDist_raw1 = contactDistance[transformCacheRef1] + cp.float32(0.0)
                cDistance = cDist_raw0 + cDist_raw1

                # ============================================================
                # 4. Read shape data (bitcast scale floats from int32 storage)
                # ============================================================
                s0_scale_x_i = shapes[shapeRef0, SHAPE_SCALE_X] + cp.int32(0)
                s0_scale_y_i = shapes[shapeRef0, SHAPE_SCALE_Y] + cp.int32(0)
                s0_scale_z_i = shapes[shapeRef0, SHAPE_SCALE_Z] + cp.int32(0)
                s0_scale_x = thread.bitcast(s0_scale_x_i, cp.float32)
                s0_scale_y = thread.bitcast(s0_scale_y_i, cp.float32)
                s0_scale_z = thread.bitcast(s0_scale_z_i, cp.float32)
                s0_type = shapes[shapeRef0, SHAPE_TYPE] + cp.int32(0)
                s0_matIdx = shapes[shapeRef0, SHAPE_MATERIAL_INDEX] + cp.int32(0)

                s1_scale_x_i = shapes[shapeRef1, SHAPE_SCALE_X] + cp.int32(0)
                s1_scale_y_i = shapes[shapeRef1, SHAPE_SCALE_Y] + cp.int32(0)
                s1_scale_z_i = shapes[shapeRef1, SHAPE_SCALE_Z] + cp.int32(0)
                s1_scale_x = thread.bitcast(s1_scale_x_i, cp.float32)
                s1_scale_y = thread.bitcast(s1_scale_y_i, cp.float32)
                s1_scale_z = thread.bitcast(s1_scale_z_i, cp.float32)
                s1_type = shapes[shapeRef1, SHAPE_TYPE] + cp.int32(0)
                s1_matIdx = shapes[shapeRef1, SHAPE_MATERIAL_INDEX] + cp.int32(0)

                # ============================================================
                # 5. Determine type pair; flip if needed (shape0 should be lower type)
                # ============================================================
                type0 = s0_type
                type1 = s1_type
                scale0_x = s0_scale_x
                scale0_y = s0_scale_y
                scale0_z = s0_scale_z
                scale1_x = s1_scale_x
                scale1_y = s1_scale_y
                scale1_z = s1_scale_z
                tr0_qx = t0_qx
                tr0_qy = t0_qy
                tr0_qz = t0_qz
                tr0_qw = t0_qw
                tr0_px = t0_px
                tr0_py = t0_py
                tr0_pz = t0_pz
                tr1_qx = t1_qx
                tr1_qy = t1_qy
                tr1_qz = t1_qz
                tr1_qw = t1_qw
                tr1_px = t1_px
                tr1_py = t1_py
                tr1_pz = t1_pz
                matIdx0 = s0_matIdx
                matIdx1 = s1_matIdx

                flip = cp.int32(0)
                if s1_type < s0_type:
                    flip = cp.int32(1)
                    type0 = s1_type
                    type1 = s0_type
                    scale0_x = s1_scale_x
                    scale0_y = s1_scale_y
                    scale0_z = s1_scale_z
                    scale1_x = s0_scale_x
                    scale1_y = s0_scale_y
                    scale1_z = s0_scale_z
                    tr0_qx = t1_qx
                    tr0_qy = t1_qy
                    tr0_qz = t1_qz
                    tr0_qw = t1_qw
                    tr0_px = t1_px
                    tr0_py = t1_py
                    tr0_pz = t1_pz
                    tr1_qx = t0_qx
                    tr1_qy = t0_qy
                    tr1_qz = t0_qz
                    tr1_qw = t0_qw
                    tr1_px = t0_px
                    tr1_py = t0_py
                    tr1_pz = t0_pz

                # ============================================================
                # 6. Collision detection -- all 6 functions inlined
                # ============================================================
                # Pre-declare outputs
                nb_contacts = cp.int32(0)
                pp0_x = cp.float32(0.0)
                pp0_y = cp.float32(0.0)
                pp0_z = cp.float32(0.0)
                pp0_w = cp.float32(0.0)
                pp1_x = cp.float32(0.0)
                pp1_y = cp.float32(0.0)
                pp1_z = cp.float32(0.0)
                pp1_w = cp.float32(0.0)
                normal_x = cp.float32(0.0)
                normal_y = cp.float32(0.0)
                normal_z = cp.float32(0.0)
                # Pre-declare vars used across if/elif collision branches
                nn_x = cp.float32(0.0)
                nn_y = cp.float32(0.0)
                nn_z = cp.float32(0.0)
                bv0_x = cp.float32(0.0)
                bv0_y = cp.float32(0.0)
                bv0_z = cp.float32(0.0)
                PX_EPS_REAL = cp.float32(0.0)
                aRecip = cp.float32(0.0)
                a_val = cp.float32(0.0)
                ab0_x = cp.float32(0.0)
                ab0_y = cp.float32(0.0)
                ab0_z = cp.float32(0.0)
                ab1_x = cp.float32(0.0)
                ab1_y = cp.float32(0.0)
                ab1_z = cp.float32(0.0)
                ab_x = cp.float32(0.0)
                ab_y = cp.float32(0.0)
                ab_z = cp.float32(0.0)
                abs_cl_x = cp.float32(0.0)
                abs_cl_y = cp.float32(0.0)
                abs_cl_z = cp.float32(0.0)
                abs_sc_x = cp.float32(0.0)
                abs_sc_y = cp.float32(0.0)
                abs_sc_z = cp.float32(0.0)
                allflags = cp.float32(0.0)
                ap00_x = cp.float32(0.0)
                ap00_y = cp.float32(0.0)
                ap00_z = cp.float32(0.0)
                ap01_x = cp.float32(0.0)
                ap01_y = cp.float32(0.0)
                ap01_z = cp.float32(0.0)
                ap10_x = cp.float32(0.0)
                ap10_y = cp.float32(0.0)
                ap10_z = cp.float32(0.0)
                ap11_x = cp.float32(0.0)
                ap11_y = cp.float32(0.0)
                ap11_z = cp.float32(0.0)
                ap_x = cp.float32(0.0)
                ap_y = cp.float32(0.0)
                ap_z = cp.float32(0.0)
                bInsideBox = cp.float32(0.0)
                bToA_px = cp.float32(0.0)
                bToA_py = cp.float32(0.0)
                bToA_pz = cp.float32(0.0)
                bToA_qw = cp.float32(0.0)
                bToA_qx = cp.float32(0.0)
                bToA_qy = cp.float32(0.0)
                bToA_qz = cp.float32(0.0)
                b_val = cp.float32(0.0)
                bothCompliant = cp.float32(0.0)
                boxHalfX = cp.float32(0.0)
                boxHalfY = cp.float32(0.0)
                boxHalfZ = cp.float32(0.0)
                bv1_x = cp.float32(0.0)
                bv1_y = cp.float32(0.0)
                bv1_z = cp.float32(0.0)
                bv_x = cp.float32(0.0)
                bv_y = cp.float32(0.0)
                bv_z = cp.float32(0.0)
                c_val = cp.float32(0.0)
                capsuleHalfHeight1 = cp.float32(0.0)
                capsuleRadius = cp.float32(0.0)
                capsuleRadius0 = cp.float32(0.0)
                capsuleRadius1 = cp.float32(0.0)
                change = cp.float32(0.0)
                cl_x = cp.float32(0.0)
                cl_y = cp.float32(0.0)
                cl_z = cp.float32(0.0)
                closA_x = cp.float32(0.0)
                closA_y = cp.float32(0.0)
                closA_z = cp.float32(0.0)
                closB_x = cp.float32(0.0)
                closB_y = cp.float32(0.0)
                closB_z = cp.float32(0.0)
                combineFlags = cp.float32(0.0)
                combinedDamp = cp.float32(0.0)
                combinedDynFric = cp.float32(0.0)
                combinedMatFlags = cp.float32(0.0)
                combinedRest = cp.float32(0.0)
                combinedStaFric = cp.float32(0.0)
                comp_val = cp.float32(0.0)
                compliant0 = cp.float32(0.0)
                compliant1 = cp.float32(0.0)
                compliantAcc0 = cp.float32(0.0)
                compliantAcc1 = cp.float32(0.0)
                contactAllocSize = cp.float32(0.0)
                contactByteOffset = cp.float32(0.0)
                contactIndex = cp.float32(0.0)
                contactPtrHi = cp.float32(0.0)
                contactPtrLo = cp.float32(0.0)
                cos_val = cp.float32(0.0)
                cp_x = cp.float32(0.0)
                cp_y = cp.float32(0.0)
                cp_z = cp.float32(0.0)
                cq0w = cp.float32(0.0)
                cq0x = cp.float32(0.0)
                cq0y = cp.float32(0.0)
                cq0z = cp.float32(0.0)
                currentlyHasTouch = cp.float32(0.0)
                d0_x = cp.float32(0.0)
                d0_y = cp.float32(0.0)
                d0_z = cp.float32(0.0)
                d1_x = cp.float32(0.0)
                d1_y = cp.float32(0.0)
                d1_z = cp.float32(0.0)
                d_p = cp.float32(0.0)
                dampCombMode = cp.float32(0.0)
                denom0 = cp.float32(0.0)
                denom1 = cp.float32(0.0)
                denom_ss = cp.float32(0.0)
                denom_val = cp.float32(0.0)
                diff_x = cp.float32(0.0)
                diff_y = cp.float32(0.0)
                diff_z = cp.float32(0.0)
                dir0_x = cp.float32(0.0)
                dir0_y = cp.float32(0.0)
                dir0_z = cp.float32(0.0)
                dir1_x = cp.float32(0.0)
                dir1_y = cp.float32(0.0)
                dir1_z = cp.float32(0.0)
                dir_x = cp.float32(0.0)
                dir_y = cp.float32(0.0)
                dir_z = cp.float32(0.0)
                dist = cp.float32(0.0)
                distSq = cp.float32(0.0)
                dist_val = cp.float32(0.0)
                dp_x = cp.float32(0.0)
                dp_y = cp.float32(0.0)
                dp_z = cp.float32(0.0)
                dts_x = cp.float32(0.0)
                dts_y = cp.float32(0.0)
                dts_z = cp.float32(0.0)
                dx = cp.float32(0.0)
                dy = cp.float32(0.0)
                dz = cp.float32(0.0)
                e0_x = cp.float32(0.0)
                e0_y = cp.float32(0.0)
                e0_z = cp.float32(0.0)
                e1_x = cp.float32(0.0)
                e1_y = cp.float32(0.0)
                e1_z = cp.float32(0.0)
                eRecip = cp.float32(0.0)
                e_val = cp.float32(0.0)
                eps = cp.float32(0.0)
                eps_cc = cp.float32(0.0)
                ex = cp.float32(0.0)
                exactlyOneAccCompliant = cp.float32(0.0)
                exactlyOneCompliant = cp.float32(0.0)
                ey = cp.float32(0.0)
                ez = cp.float32(0.0)
                f_val = cp.float32(0.0)
                flipSign = cp.float32(0.0)
                fn_inv = cp.float32(0.0)
                fn_len = cp.float32(0.0)
                fn_x = cp.float32(0.0)
                fn_y = cp.float32(0.0)
                fn_z = cp.float32(0.0)
                forceAllocSize = cp.float32(0.0)
                forceByteOffset = cp.float32(0.0)
                forcePtrHi = cp.float32(0.0)
                forcePtrLo = cp.float32(0.0)
                fricCombMode = cp.float32(0.0)
                halfHeight = cp.float32(0.0)
                halfHeight0 = cp.float32(0.0)
                inflRadius = cp.float32(0.0)
                inflSum = cp.float32(0.0)
                inflSum_b = cp.float32(0.0)
                inflSum_c = cp.float32(0.0)
                inflatedSumSq = cp.float32(0.0)
                inflatedSum_cc = cp.float32(0.0)
                intflagMatIdx0 = cp.float32(0.0)
                invDist = cp.float32(0.0)
                invLen = cp.float32(0.0)
                inv_a = cp.float32(0.0)
                inv_e = cp.float32(0.0)
                inv_vv = cp.float32(0.0)
                le_x = cp.float32(0.0)
                le_y = cp.float32(0.0)
                le_z = cp.float32(0.0)
                lenSq = cp.float32(0.0)
                length = cp.float32(0.0)
                lengthSq = cp.float32(0.0)
                length_b = cp.float32(0.0)
                ln_x = cp.float32(0.0)
                ln_y = cp.float32(0.0)
                ln_z = cp.float32(0.0)
                locN_x = cp.float32(0.0)
                locN_y = cp.float32(0.0)
                locN_z = cp.float32(0.0)
                ls_x = cp.float32(0.0)
                ls_y = cp.float32(0.0)
                ls_z = cp.float32(0.0)
                mat0_damp = cp.float32(0.0)
                mat0_dampMode = cp.float32(0.0)
                mat0_dampMode_word = cp.float32(0.0)
                mat0_damp_i = cp.float32(0.0)
                mat0_dynFric = cp.float32(0.0)
                mat0_dynFric_i = cp.float32(0.0)
                mat0_flags = cp.float32(0.0)
                mat0_flagsModes = cp.float32(0.0)
                mat0_fricMode = cp.float32(0.0)
                mat0_rest = cp.float32(0.0)
                mat0_restMode = cp.float32(0.0)
                mat0_rest_i = cp.float32(0.0)
                mat0_staFric = cp.float32(0.0)
                mat0_staFric_i = cp.float32(0.0)
                mat1_damp = cp.float32(0.0)
                mat1_dampMode = cp.float32(0.0)
                mat1_dampMode_word = cp.float32(0.0)
                mat1_damp_i = cp.float32(0.0)
                mat1_dynFric = cp.float32(0.0)
                mat1_dynFric_i = cp.float32(0.0)
                mat1_flags = cp.float32(0.0)
                mat1_flagsModes = cp.float32(0.0)
                mat1_fricMode = cp.float32(0.0)
                mat1_rest = cp.float32(0.0)
                mat1_restMode = cp.float32(0.0)
                mat1_rest_i = cp.float32(0.0)
                mat1_staFric = cp.float32(0.0)
                mat1_staFric_i = cp.float32(0.0)
                matIdx1Pad = cp.float32(0.0)
                n_px = cp.float32(0.0)
                n_py = cp.float32(0.0)
                n_pz = cp.float32(0.0)
                nbContactsWord = cp.float32(0.0)
                newAllflags = cp.float32(0.0)
                nom00 = cp.float32(0.0)
                nom01 = cp.float32(0.0)
                nom10 = cp.float32(0.0)
                nom11 = cp.float32(0.0)
                nom_val = cp.float32(0.0)
                numPatches = cp.float32(0.0)
                oldStatusFlags = cp.float32(0.0)
                p0_x = cp.float32(0.0)
                p0_y = cp.float32(0.0)
                p0_z = cp.float32(0.0)
                p1_x = cp.float32(0.0)
                p1_y = cp.float32(0.0)
                p1_z = cp.float32(0.0)
                parallelTol = cp.float32(0.0)
                patchDiff = cp.float32(0.0)
                patchIndex = cp.float32(0.0)
                patchRow = cp.float32(0.0)
                pen_f = cp.float32(0.0)
                pen_p = cp.float32(0.0)
                penetration = cp.float32(0.0)
                pn_inv = cp.float32(0.0)
                pn_len = cp.float32(0.0)
                pn_x = cp.float32(0.0)
                pn_y = cp.float32(0.0)
                pn_z = cp.float32(0.0)
                po_x = cp.float32(0.0)
                po_y = cp.float32(0.0)
                po_z = cp.float32(0.0)
                prevPatches = cp.float32(0.0)
                prevTouchKnown = cp.float32(0.0)
                previouslyHadTouch = cp.float32(0.0)
                proj_x = cp.float32(0.0)
                proj_y = cp.float32(0.0)
                proj_z = cp.float32(0.0)
                pt_x = cp.float32(0.0)
                pt_y = cp.float32(0.0)
                pt_z = cp.float32(0.0)
                r_x = cp.float32(0.0)
                r_y = cp.float32(0.0)
                r_z = cp.float32(0.0)
                radSum_c = cp.float32(0.0)
                radiusSum = cp.float32(0.0)
                recipLen = cp.float32(0.0)
                restCombMode = cp.float32(0.0)
                revertAllflags = cp.float32(0.0)
                revertPatchDiff = cp.float32(0.0)
                s0_raw = cp.float32(0.0)
                s0_x = cp.float32(0.0)
                s0_y = cp.float32(0.0)
                s0_z = cp.float32(0.0)
                s1_x = cp.float32(0.0)
                s1_y = cp.float32(0.0)
                s1_z = cp.float32(0.0)
                sTmp = cp.float32(0.0)
                sc_x = cp.float32(0.0)
                sc_y = cp.float32(0.0)
                sc_z = cp.float32(0.0)
                seg_s = cp.float32(0.0)
                seg_t = cp.float32(0.0)
                separation = cp.float32(0.0)
                signDist0 = cp.float32(0.0)
                signDist1 = cp.float32(0.0)
                sph_local_x = cp.float32(0.0)
                sphereRadius = cp.float32(0.0)
                sphereRadius1 = cp.float32(0.0)
                sqD = cp.float32(0.0)
                sqDist0 = cp.float32(0.0)
                sqInflSum_b = cp.float32(0.0)
                squareDist = cp.float32(0.0)
                startNbMatflags = cp.float32(0.0)
                statusFlags = cp.float32(0.0)
                sumRadius = cp.float32(0.0)
                sx = cp.float32(0.0)
                sy = cp.float32(0.0)
                sz = cp.float32(0.0)
                t0_x = cp.float32(0.0)
                t0_y = cp.float32(0.0)
                t0_z = cp.float32(0.0)
                t1_bx = cp.float32(0.0)
                t1_by = cp.float32(0.0)
                t1_bz = cp.float32(0.0)
                t2_val = cp.float32(0.0)
                tRaw = cp.float32(0.0)
                tTmp = cp.float32(0.0)
                tV0 = cp.float32(0.0)
                tV1 = cp.float32(0.0)
                tV2 = cp.float32(0.0)
                tV3 = cp.float32(0.0)
                tVal = cp.float32(0.0)
                temp_ss = cp.float32(0.0)
                tmp_bx = cp.float32(0.0)
                tmp_by = cp.float32(0.0)
                tmp_bz = cp.float32(0.0)
                tmp_x = cp.float32(0.0)
                tmp_y = cp.float32(0.0)
                tmp_z = cp.float32(0.0)
                touchXor = cp.float32(0.0)
                v_px = cp.float32(0.0)
                v_py = cp.float32(0.0)
                v_pz = cp.float32(0.0)
                v_x = cp.float32(0.0)
                v_y = cp.float32(0.0)
                v_z = cp.float32(0.0)
                vv_cx = cp.float32(0.0)
                vv_cy = cp.float32(0.0)
                vv_cz = cp.float32(0.0)
                vv_len = cp.float32(0.0)
                vv_x = cp.float32(0.0)
                vv_y = cp.float32(0.0)
                vv_z = cp.float32(0.0)
                we_x = cp.float32(0.0)
                we_y = cp.float32(0.0)
                we_z = cp.float32(0.0)
                wn_x = cp.float32(0.0)
                wn_y = cp.float32(0.0)
                wn_z = cp.float32(0.0)
                wp_x = cp.float32(0.0)
                wp_y = cp.float32(0.0)
                wp_z = cp.float32(0.0)
                ws_x = cp.float32(0.0)
                ws_y = cp.float32(0.0)
                ws_z = cp.float32(0.0)

                sphereRadius = scale0_x

                if type0 == cp.int32(GEO_SPHERE):
                    # ---- sphere vs sphere ----
                    if type1 == cp.int32(GEO_SPHERE):
                        sphereRadius1 = scale1_x
                        dx = tr0_px - tr1_px
                        dy = tr0_py - tr1_py
                        dz = tr0_pz - tr1_pz
                        distSq = dx * dx + dy * dy + dz * dz
                        radiusSum = sphereRadius + sphereRadius1
                        inflSum = radiusSum + cDistance
                        if inflSum * inflSum > distSq:
                            eps = cp.float32(1e-4)
                            dist = thread.sqrt(distSq)
                            # normal
                            nn_x = cp.float32(1.0)
                            nn_y = cp.float32(0.0)
                            nn_z = cp.float32(0.0)
                            if dist > eps:
                                invDist = cp.float32(1.0) / dist
                                nn_x = dx * invDist
                                nn_y = dy * invDist
                                nn_z = dz * invDist
                            # point = p1 + normal * r1
                            pp0_x = tr1_px + nn_x * sphereRadius1
                            pp0_y = tr1_py + nn_y * sphereRadius1
                            pp0_z = tr1_pz + nn_z * sphereRadius1
                            pp0_w = dist - radiusSum
                            normal_x = nn_x
                            normal_y = nn_y
                            normal_z = nn_z
                            nb_contacts = cp.int32(1)

                    # ---- sphere vs plane ----
                    elif type1 == cp.int32(GEO_PLANE):
                        # transformInv: sphereCenter in plane space
                        # diff = p0 - p1
                        diff_x = tr0_px - tr1_px
                        diff_y = tr0_py - tr1_py
                        diff_z = tr0_pz - tr1_pz
                        # rotate by inverse of q1
                        sph_local_x = quat_rotate_inv_x(tr1_qx, tr1_qy, tr1_qz, tr1_qw, diff_x, diff_y, diff_z)
                        separation = sph_local_x - sphereRadius
                        if cDistance >= separation:
                            # world normal = q1.getBasisVector0()
                            wn_x = basis0_x(tr1_qx, tr1_qy, tr1_qz, tr1_qw)
                            wn_y = basis0_y(tr1_qx, tr1_qy, tr1_qz, tr1_qw)
                            wn_z = basis0_z(tr1_qx, tr1_qy, tr1_qz, tr1_qw)
                            pp0_x = tr0_px - wn_x * sphereRadius
                            pp0_y = tr0_py - wn_y * sphereRadius
                            pp0_z = tr0_pz - wn_z * sphereRadius
                            pp0_w = separation
                            normal_x = wn_x
                            normal_y = wn_y
                            normal_z = wn_z
                            nb_contacts = cp.int32(1)

                    # ---- sphere vs capsule ----
                    elif type1 == cp.int32(GEO_CAPSULE):
                        capsuleRadius = scale1_y
                        halfHeight = scale1_x
                        # capsule segment endpoints
                        bv0_x = basis0_x(tr1_qx, tr1_qy, tr1_qz, tr1_qw)
                        bv0_y = basis0_y(tr1_qx, tr1_qy, tr1_qz, tr1_qw)
                        bv0_z = basis0_z(tr1_qx, tr1_qy, tr1_qz, tr1_qw)
                        tmp_x = bv0_x * halfHeight
                        tmp_y = bv0_y * halfHeight
                        tmp_z = bv0_z * halfHeight
                        sx = tr1_px + tmp_x
                        sy = tr1_py + tmp_y
                        sz = tr1_pz + tmp_z
                        ex = tr1_px - tmp_x
                        ey = tr1_py - tmp_y
                        ez = tr1_pz - tmp_z
                        # distancePointSegmentSquared(s, e, sphereCenter, t)
                        ap_x = tr0_px - sx
                        ap_y = tr0_py - sy
                        ap_z = tr0_pz - sz
                        ab_x = ex - sx
                        ab_y = ey - sy
                        ab_z = ez - sz
                        nom_val = ap_x * ab_x + ap_y * ab_y + ap_z * ab_z
                        denom_val = ab_x * ab_x + ab_y * ab_y + ab_z * ab_z
                        tVal = cp.float32(0.0)
                        if denom_val > cp.float32(0.0):
                            tRaw = nom_val / denom_val
                            tVal = clamp_f32(tRaw, cp.float32(0.0), cp.float32(1.0))
                        v_x = ap_x - ab_x * tVal
                        v_y = ap_y - ab_y * tVal
                        v_z = ap_z - ab_z * tVal
                        squareDist = v_x * v_x + v_y * v_y + v_z * v_z

                        radSum_c = sphereRadius + capsuleRadius
                        inflSum_c = radSum_c + cDistance
                        if inflSum_c * inflSum_c > squareDist:
                            # closest point on segment
                            cp_x = sx + (ex - sx) * tVal
                            cp_y = sy + (ey - sy) * tVal
                            cp_z = sz + (ez - sz) * tVal
                            dir_x = tr0_px - cp_x
                            dir_y = tr0_py - cp_y
                            dir_z = tr0_pz - cp_z
                            lenSq = dir_x * dir_x + dir_y * dir_y + dir_z * dir_z
                            length = thread.sqrt(lenSq)
                            PX_EPS_REAL = cp.float32(1e-7)
                            nn_x = cp.float32(1.0)
                            nn_y = cp.float32(0.0)
                            nn_z = cp.float32(0.0)
                            if length > PX_EPS_REAL:
                                invLen = cp.float32(1.0) / length
                                nn_x = dir_x * invLen
                                nn_y = dir_y * invLen
                                nn_z = dir_z * invLen
                            pp0_x = tr0_px - nn_x * sphereRadius
                            pp0_y = tr0_py - nn_y * sphereRadius
                            pp0_z = tr0_pz - nn_z * sphereRadius
                            pp0_w = thread.sqrt(squareDist) - radSum_c
                            normal_x = nn_x
                            normal_y = nn_y
                            normal_z = nn_z
                            nb_contacts = cp.int32(1)

                    # ---- sphere vs box ----
                    elif type1 == cp.int32(GEO_BOX):
                        boxHalfX = scale1_x
                        boxHalfY = scale1_y
                        boxHalfZ = scale1_z
                        # sphere center in box local space: transformInv
                        diff_x = tr0_px - tr1_px
                        diff_y = tr0_py - tr1_py
                        diff_z = tr0_pz - tr1_pz
                        sc_x = quat_rotate_inv_x(tr1_qx, tr1_qy, tr1_qz, tr1_qw, diff_x, diff_y, diff_z)
                        sc_y = quat_rotate_inv_y(tr1_qx, tr1_qy, tr1_qz, tr1_qw, diff_x, diff_y, diff_z)
                        sc_z = quat_rotate_inv_z(tr1_qx, tr1_qy, tr1_qz, tr1_qw, diff_x, diff_y, diff_z)

                        inflSum_b = sphereRadius + cDistance
                        sqInflSum_b = inflSum_b * inflSum_b

                        cl_x = clamp_f32(sc_x, -boxHalfX, boxHalfX)
                        cl_y = clamp_f32(sc_y, -boxHalfY, boxHalfY)
                        cl_z = clamp_f32(sc_z, -boxHalfZ, boxHalfZ)
                        vv_x = sc_x - cl_x
                        vv_y = sc_y - cl_y
                        vv_z = sc_z - cl_z
                        lengthSq = vv_x * vv_x + vv_y * vv_y + vv_z * vv_z

                        if sqInflSum_b > lengthSq:
                            # check if sphere center is inside box
                            bInsideBox = cp.int32(0)
                            abs_sc_x = abs_f32(sc_x)
                            abs_sc_y = abs_f32(sc_y)
                            abs_sc_z = abs_f32(sc_z)
                            if boxHalfX >= abs_sc_x:
                                if boxHalfY >= abs_sc_y:
                                    if boxHalfZ >= abs_sc_z:
                                        bInsideBox = cp.int32(1)

                            if bInsideBox == cp.int32(1):
                                # sphere center inside box
                                abs_cl_x = abs_f32(cl_x)
                                abs_cl_y = abs_f32(cl_y)
                                abs_cl_z = abs_f32(cl_z)
                                dts_x = boxHalfX - abs_cl_x
                                dts_y = boxHalfY - abs_cl_y
                                dts_z = boxHalfZ - abs_cl_z
                                # assume z is smallest
                                ln_x = cp.float32(0.0)
                                ln_y = cp.float32(0.0)
                                ln_z = cp.float32(1.0)
                                if cl_z < cp.float32(0.0):
                                    ln_z = cp.float32(-1.0)
                                dist_val = -dts_z
                                if dts_x <= dts_y:
                                    if dts_x <= dts_z:
                                        ln_x = cp.float32(1.0)
                                        if cl_x < cp.float32(0.0):
                                            ln_x = cp.float32(-1.0)
                                        ln_y = cp.float32(0.0)
                                        ln_z = cp.float32(0.0)
                                        dist_val = -dts_x
                                elif dts_y <= dts_z:
                                    ln_x = cp.float32(0.0)
                                    ln_y = cp.float32(1.0)
                                    if cl_y < cp.float32(0.0):
                                        ln_y = cp.float32(-1.0)
                                    ln_z = cp.float32(0.0)
                                    dist_val = -dts_y
                                # rotate local normal to world
                                wn_x = quat_rotate_x(tr1_qx, tr1_qy, tr1_qz, tr1_qw, ln_x, ln_y, ln_z)
                                wn_y = quat_rotate_y(tr1_qx, tr1_qy, tr1_qz, tr1_qw, ln_x, ln_y, ln_z)
                                wn_z = quat_rotate_z(tr1_qx, tr1_qy, tr1_qz, tr1_qw, ln_x, ln_y, ln_z)
                                penetration = dist_val - sphereRadius
                                pp0_x = tr0_px - wn_x * dist_val
                                pp0_y = tr0_py - wn_y * dist_val
                                pp0_z = tr0_pz - wn_z * dist_val
                                pp0_w = penetration
                                normal_x = wn_x
                                normal_y = wn_y
                                normal_z = wn_z
                            else:
                                # sphere center outside box
                                recipLen = thread.rsqrt(lengthSq)
                                length_b = cp.float32(1.0) / recipLen
                                locN_x = vv_x * recipLen
                                locN_y = vv_y * recipLen
                                locN_z = vv_z * recipLen
                                penetration = length_b - sphereRadius
                                wn_x = quat_rotate_x(tr1_qx, tr1_qy, tr1_qz, tr1_qw, locN_x, locN_y, locN_z)
                                wn_y = quat_rotate_y(tr1_qx, tr1_qy, tr1_qz, tr1_qw, locN_x, locN_y, locN_z)
                                wn_z = quat_rotate_z(tr1_qx, tr1_qy, tr1_qz, tr1_qw, locN_x, locN_y, locN_z)
                                # point = transform1.transform(p) = q1.rotate(p) + p1
                                wp_x = quat_rotate_x(tr1_qx, tr1_qy, tr1_qz, tr1_qw, cl_x, cl_y, cl_z) + tr1_px
                                wp_y = quat_rotate_y(tr1_qx, tr1_qy, tr1_qz, tr1_qw, cl_x, cl_y, cl_z) + tr1_py
                                wp_z = quat_rotate_z(tr1_qx, tr1_qy, tr1_qz, tr1_qw, cl_x, cl_y, cl_z) + tr1_pz
                                pp0_x = wp_x
                                pp0_y = wp_y
                                pp0_z = wp_z
                                pp0_w = penetration
                                normal_x = wn_x
                                normal_y = wn_y
                                normal_z = wn_z
                            nb_contacts = cp.int32(1)

                # ---- type0 is not sphere: must be plane-capsule or capsule-capsule ----
                else:
                    capsuleRadius1 = scale1_y
                    capsuleHalfHeight1 = scale1_x

                    # ---- plane vs capsule (type0=PLANE, type1=CAPSULE) ----
                    if type0 == cp.int32(GEO_PLANE):
                        # bToA = transform0.transformInv(transform1)
                        # bToA.p = rotateInv(q0, p1-p0)
                        # bToA.q = conj(q0) * q1
                        dp_x = tr1_px - tr0_px
                        dp_y = tr1_py - tr0_py
                        dp_z = tr1_pz - tr0_pz
                        bToA_px = quat_rotate_inv_x(tr0_qx, tr0_qy, tr0_qz, tr0_qw, dp_x, dp_y, dp_z)
                        bToA_py = quat_rotate_inv_y(tr0_qx, tr0_qy, tr0_qz, tr0_qw, dp_x, dp_y, dp_z)
                        bToA_pz = quat_rotate_inv_z(tr0_qx, tr0_qy, tr0_qz, tr0_qw, dp_x, dp_y, dp_z)
                        # bToA.q = conj(q0) * q1
                        # conj(q0) = (-q0x, -q0y, -q0z, q0w)
                        cq0x = -tr0_qx
                        cq0y = -tr0_qy
                        cq0z = -tr0_qz
                        cq0w = tr0_qw
                        # quaternion multiply: cq0 * q1
                        bToA_qx = cq0w * tr1_qx + cq0x * tr1_qw + cq0y * tr1_qz - cq0z * tr1_qy
                        bToA_qy = cq0w * tr1_qy - cq0x * tr1_qz + cq0y * tr1_qw + cq0z * tr1_qx
                        bToA_qz = cq0w * tr1_qz + cq0x * tr1_qy - cq0y * tr1_qx + cq0z * tr1_qw
                        bToA_qw = cq0w * tr1_qw - cq0x * tr1_qx - cq0y * tr1_qy - cq0z * tr1_qz

                        # plane normal in world space
                        pn_x = basis0_x(tr0_qx, tr0_qy, tr0_qz, tr0_qw)
                        pn_y = basis0_y(tr0_qx, tr0_qy, tr0_qz, tr0_qw)
                        pn_z = basis0_z(tr0_qx, tr0_qy, tr0_qz, tr0_qw)
                        # normalize (should already be unit but match CUDA)
                        pn_len = thread.sqrt(pn_x * pn_x + pn_y * pn_y + pn_z * pn_z)
                        pn_inv = cp.float32(1.0) / pn_len
                        pn_x = pn_x * pn_inv
                        pn_y = pn_y * pn_inv
                        pn_z = pn_z * pn_inv

                        # outNormal = -planeNormal
                        normal_x = -pn_x
                        normal_y = -pn_y
                        normal_z = -pn_z

                        # capsule endpoints in plane-local space
                        bv_x = basis0_x(bToA_qx, bToA_qy, bToA_qz, bToA_qw)
                        bv_y = basis0_y(bToA_qx, bToA_qy, bToA_qz, bToA_qw)
                        bv_z = basis0_z(bToA_qx, bToA_qy, bToA_qz, bToA_qw)
                        tmp_bx = bv_x * capsuleHalfHeight1
                        tmp_by = bv_y * capsuleHalfHeight1
                        tmp_bz = bv_z * capsuleHalfHeight1
                        ls_x = bToA_px + tmp_bx
                        ls_y = bToA_py + tmp_by
                        ls_z = bToA_pz + tmp_bz
                        le_x = bToA_px - tmp_bx
                        le_y = bToA_py - tmp_by
                        le_z = bToA_pz - tmp_bz

                        inflRadius = capsuleRadius1 + cDistance

                        # endpoint s
                        signDist0 = ls_x  # dot with (1,0,0)
                        if inflRadius >= signDist0:
                            # worldPoint = transform0.transform(s) - planeNormal * radius
                            ws_x = quat_rotate_x(tr0_qx, tr0_qy, tr0_qz, tr0_qw, ls_x, ls_y, ls_z) + tr0_px
                            ws_y = quat_rotate_y(tr0_qx, tr0_qy, tr0_qz, tr0_qw, ls_x, ls_y, ls_z) + tr0_py
                            ws_z = quat_rotate_z(tr0_qx, tr0_qy, tr0_qz, tr0_qw, ls_x, ls_y, ls_z) + tr0_pz
                            pp0_x = ws_x - pn_x * capsuleRadius1
                            pp0_y = ws_y - pn_y * capsuleRadius1
                            pp0_z = ws_z - pn_z * capsuleRadius1
                            pp0_w = signDist0 - capsuleRadius1
                            nb_contacts = nb_contacts + cp.int32(1)

                        # endpoint e
                        signDist1 = le_x
                        if inflRadius >= signDist1:
                            we_x = quat_rotate_x(tr0_qx, tr0_qy, tr0_qz, tr0_qw, le_x, le_y, le_z) + tr0_px
                            we_y = quat_rotate_y(tr0_qx, tr0_qy, tr0_qz, tr0_qw, le_x, le_y, le_z) + tr0_py
                            we_z = quat_rotate_z(tr0_qx, tr0_qy, tr0_qz, tr0_qw, le_x, le_y, le_z) + tr0_pz
                            if nb_contacts == cp.int32(0):
                                pp0_x = we_x - pn_x * capsuleRadius1
                                pp0_y = we_y - pn_y * capsuleRadius1
                                pp0_z = we_z - pn_z * capsuleRadius1
                                pp0_w = signDist1 - capsuleRadius1
                            else:
                                pp1_x = we_x - pn_x * capsuleRadius1
                                pp1_y = we_y - pn_y * capsuleRadius1
                                pp1_z = we_z - pn_z * capsuleRadius1
                                pp1_w = signDist1 - capsuleRadius1
                            nb_contacts = nb_contacts + cp.int32(1)

                    # ---- capsule vs capsule (type0=CAPSULE, type1=CAPSULE) ----
                    elif type0 == cp.int32(GEO_CAPSULE):
                        capsuleRadius0 = scale0_y
                        halfHeight0 = scale0_x

                        # position offset (centroid)
                        po_x = (tr0_px + tr1_px) * cp.float32(0.5)
                        po_y = (tr0_py + tr1_py) * cp.float32(0.5)
                        po_z = (tr0_pz + tr1_pz) * cp.float32(0.5)
                        p0_x = tr0_px - po_x
                        p0_y = tr0_py - po_y
                        p0_z = tr0_pz - po_z
                        p1_x = tr1_px - po_x
                        p1_y = tr1_py - po_y
                        p1_z = tr1_pz - po_z

                        bv0_x = basis0_x(tr0_qx, tr0_qy, tr0_qz, tr0_qw)
                        bv0_y = basis0_y(tr0_qx, tr0_qy, tr0_qz, tr0_qw)
                        bv0_z = basis0_z(tr0_qx, tr0_qy, tr0_qz, tr0_qw)
                        t0_x = bv0_x * halfHeight0
                        t0_y = bv0_y * halfHeight0
                        t0_z = bv0_z * halfHeight0
                        s0_x = p0_x + t0_x
                        s0_y = p0_y + t0_y
                        s0_z = p0_z + t0_z
                        e0_x = p0_x - t0_x
                        e0_y = p0_y - t0_y
                        e0_z = p0_z - t0_z
                        d0_x = e0_x - s0_x
                        d0_y = e0_y - s0_y
                        d0_z = e0_z - s0_z

                        bv1_x = basis0_x(tr1_qx, tr1_qy, tr1_qz, tr1_qw)
                        bv1_y = basis0_y(tr1_qx, tr1_qy, tr1_qz, tr1_qw)
                        bv1_z = basis0_z(tr1_qx, tr1_qy, tr1_qz, tr1_qw)
                        t1_bx = bv1_x * capsuleHalfHeight1
                        t1_by = bv1_y * capsuleHalfHeight1
                        t1_bz = bv1_z * capsuleHalfHeight1
                        s1_x = p1_x + t1_bx
                        s1_y = p1_y + t1_by
                        s1_z = p1_z + t1_bz
                        e1_x = p1_x - t1_bx
                        e1_y = p1_y - t1_by
                        e1_z = p1_z - t1_bz
                        d1_x = e1_x - s1_x
                        d1_y = e1_y - s1_y
                        d1_z = e1_z - s1_z

                        sumRadius = capsuleRadius0 + capsuleRadius1
                        inflatedSum_cc = sumRadius + cDistance
                        inflatedSumSq = inflatedSum_cc * inflatedSum_cc
                        a_val = d0_x * d0_x + d0_y * d0_y + d0_z * d0_z
                        e_val = d1_x * d1_x + d1_y * d1_y + d1_z * d1_z
                        eps_cc = cp.float32(1e-6)

                        # distanceSegmentSegmentSquared inline
                        r_x = s0_x - s1_x
                        r_y = s0_y - s1_y
                        r_z = s0_z - s1_z
                        b_val = d0_x * d1_x + d0_y * d1_y + d0_z * d1_z
                        c_val = d0_x * r_x + d0_y * r_y + d0_z * r_z
                        aRecip = cp.float32(0.0)
                        if a_val > eps_cc:
                            aRecip = cp.float32(1.0) / a_val
                        eRecip = cp.float32(0.0)
                        if e_val > eps_cc:
                            eRecip = cp.float32(1.0) / e_val
                        f_val = d1_x * r_x + d1_y * r_y + d1_z * r_z
                        denom_ss = a_val * e_val - b_val * b_val
                        temp_ss = b_val * f_val - c_val * e_val
                        s0_raw = clamp_f32(temp_ss / denom_ss, cp.float32(0.0), cp.float32(1.0))
                        sTmp = cp.float32(0.5)
                        if denom_ss > eps_cc:
                            sTmp = s0_raw
                        tTmp = (b_val * sTmp + f_val) * eRecip
                        t2_val = clamp_f32(tTmp, cp.float32(0.0), cp.float32(1.0))
                        comp_val = (b_val * t2_val - c_val) * aRecip
                        seg_s = clamp_f32(comp_val, cp.float32(0.0), cp.float32(1.0))
                        seg_t = t2_val

                        closA_x = s0_x + d0_x * seg_s
                        closA_y = s0_y + d0_y * seg_s
                        closA_z = s0_z + d0_z * seg_s
                        closB_x = s1_x + d1_x * seg_t
                        closB_y = s1_y + d1_y * seg_t
                        closB_z = s1_z + d1_z * seg_t
                        vv_cx = closA_x - closB_x
                        vv_cy = closA_y - closB_y
                        vv_cz = closA_z - closB_z
                        sqDist0 = vv_cx * vv_cx + vv_cy * vv_cy + vv_cz * vv_cz

                        if eps_cc > sqDist0:
                            # segments nearly coincident -- use fallback normal
                            # _normal = transform0.q.rotateInv((0,1,0))
                            fn_x = quat_rotate_inv_x(tr0_qx, tr0_qy, tr0_qz, tr0_qw, cp.float32(0.0), cp.float32(1.0), cp.float32(0.0))
                            fn_y = quat_rotate_inv_y(tr0_qx, tr0_qy, tr0_qz, tr0_qw, cp.float32(0.0), cp.float32(1.0), cp.float32(0.0))
                            fn_z = quat_rotate_inv_z(tr0_qx, tr0_qy, tr0_qz, tr0_qw, cp.float32(0.0), cp.float32(1.0), cp.float32(0.0))
                            fn_len = thread.sqrt(fn_x * fn_x + fn_y * fn_y + fn_z * fn_z)
                            fn_inv = cp.float32(1.0) / fn_len
                            fn_x = fn_x * fn_inv
                            fn_y = fn_y * fn_inv
                            fn_z = fn_z * fn_inv
                            pt_x = closA_x - fn_x * capsuleRadius0 + po_x
                            pt_y = closA_y - fn_y * capsuleRadius0 + po_y
                            pt_z = closA_z - fn_z * capsuleRadius0 + po_z
                            pp0_x = pt_x
                            pp0_y = pt_y
                            pp0_z = pt_z
                            pp0_w = -capsuleRadius0
                            normal_x = fn_x
                            normal_y = fn_y
                            normal_z = fn_z
                            nb_contacts = cp.int32(1)
                        elif inflatedSumSq >= sqDist0:
                            # check parallel
                            parallelTol = cp.float32(0.9998)
                            dir0_x = cp.float32(0.0)
                            dir0_y = cp.float32(0.0)
                            dir0_z = cp.float32(0.0)
                            if a_val > eps_cc:
                                inv_a = cp.float32(1.0) / thread.sqrt(a_val)
                                dir0_x = d0_x * inv_a
                                dir0_y = d0_y * inv_a
                                dir0_z = d0_z * inv_a
                            dir1_x = cp.float32(0.0)
                            dir1_y = cp.float32(0.0)
                            dir1_z = cp.float32(0.0)
                            if e_val > eps_cc:
                                inv_e = cp.float32(1.0) / thread.sqrt(e_val)
                                dir1_x = d1_x * inv_e
                                dir1_y = d1_y * inv_e
                                dir1_z = d1_z * inv_e
                            cos_val = abs_f32(dir0_x * dir1_x + dir0_y * dir1_y + dir0_z * dir1_z)

                            if cos_val > parallelTol:
                                # parallel case -- project endpoints
                                # pcmDistancePointSegmentTValue22 inline
                                # tV0: s1 -> s0e0, tV1: e1 -> s0e0, tV2: s0 -> s1e1, tV3: e0 -> s1e1
                                ab0_x = e0_x - s0_x
                                ab0_y = e0_y - s0_y
                                ab0_z = e0_z - s0_z
                                ab1_x = e1_x - s1_x
                                ab1_y = e1_y - s1_y
                                ab1_z = e1_z - s1_z

                                ap00_x = s1_x - s0_x
                                ap00_y = s1_y - s0_y
                                ap00_z = s1_z - s0_z
                                ap10_x = e1_x - s0_x
                                ap10_y = e1_y - s0_y
                                ap10_z = e1_z - s0_z
                                ap01_x = s0_x - s1_x
                                ap01_y = s0_y - s1_y
                                ap01_z = s0_z - s1_z
                                ap11_x = e0_x - s1_x
                                ap11_y = e0_y - s1_y
                                ap11_z = e0_z - s1_z

                                nom00 = ap00_x * ab0_x + ap00_y * ab0_y + ap00_z * ab0_z
                                nom10 = ap10_x * ab0_x + ap10_y * ab0_y + ap10_z * ab0_z
                                nom01 = ap01_x * ab1_x + ap01_y * ab1_y + ap01_z * ab1_z
                                nom11 = ap11_x * ab1_x + ap11_y * ab1_y + ap11_z * ab1_z

                                denom0 = ab0_x * ab0_x + ab0_y * ab0_y + ab0_z * ab0_z
                                denom1 = ab1_x * ab1_x + ab1_y * ab1_y + ab1_z * ab1_z

                                tV0 = cp.float32(0.0)
                                tV1 = cp.float32(0.0)
                                tV2 = cp.float32(0.0)
                                tV3 = cp.float32(0.0)
                                if abs_f32(denom0) >= eps_cc:
                                    tV0 = nom00 / denom0
                                    tV1 = nom10 / denom0
                                if abs_f32(denom1) >= eps_cc:
                                    tV2 = nom01 / denom1
                                    tV3 = nom11 / denom1

                                # Try tV0: s1 projected onto s0e0
                                if tV0 >= cp.float32(0.0):
                                    if tV0 <= cp.float32(1.0):
                                        proj_x = s0_x + d0_x * tV0
                                        proj_y = s0_y + d0_y * tV0
                                        proj_z = s0_z + d0_z * tV0
                                        v_px = proj_x - s1_x
                                        v_py = proj_y - s1_y
                                        v_pz = proj_z - s1_z
                                        sqD = v_px * v_px + v_py * v_py + v_pz * v_pz
                                        d_p = thread.sqrt(sqD)
                                        pen_p = d_p - sumRadius
                                        n_px = v_px / d_p
                                        n_py = v_py / d_p
                                        n_pz = v_pz / d_p
                                        pt_x = proj_x - n_px * capsuleRadius0 + po_x
                                        pt_y = proj_y - n_py * capsuleRadius0 + po_y
                                        pt_z = proj_z - n_pz * capsuleRadius0 + po_z
                                        pp0_x = pt_x
                                        pp0_y = pt_y
                                        pp0_z = pt_z
                                        pp0_w = pen_p
                                        normal_x = n_px
                                        normal_y = n_py
                                        normal_z = n_pz
                                        nb_contacts = nb_contacts + cp.int32(1)

                                # Try tV1: e1 projected onto s0e0
                                if tV1 >= cp.float32(0.0):
                                    if tV1 <= cp.float32(1.0):
                                        proj_x = s0_x + d0_x * tV1
                                        proj_y = s0_y + d0_y * tV1
                                        proj_z = s0_z + d0_z * tV1
                                        v_px = proj_x - e1_x
                                        v_py = proj_y - e1_y
                                        v_pz = proj_z - e1_z
                                        sqD = v_px * v_px + v_py * v_py + v_pz * v_pz
                                        d_p = thread.sqrt(sqD)
                                        pen_p = d_p - sumRadius
                                        n_px = v_px / d_p
                                        n_py = v_py / d_p
                                        n_pz = v_pz / d_p
                                        pt_x = proj_x - n_px * capsuleRadius0 + po_x
                                        pt_y = proj_y - n_py * capsuleRadius0 + po_y
                                        pt_z = proj_z - n_pz * capsuleRadius0 + po_z
                                        if nb_contacts == cp.int32(0):
                                            pp0_x = pt_x
                                            pp0_y = pt_y
                                            pp0_z = pt_z
                                            pp0_w = pen_p
                                        else:
                                            pp1_x = pt_x
                                            pp1_y = pt_y
                                            pp1_z = pt_z
                                            pp1_w = pen_p
                                        normal_x = n_px
                                        normal_y = n_py
                                        normal_z = n_pz
                                        nb_contacts = nb_contacts + cp.int32(1)

                                # Try tV2: s0 projected onto s1e1 (if numContacts < 2)
                                if nb_contacts < cp.int32(2):
                                    if tV2 >= cp.float32(0.0):
                                        if tV2 <= cp.float32(1.0):
                                            proj_x = s1_x + d1_x * tV2
                                            proj_y = s1_y + d1_y * tV2
                                            proj_z = s1_z + d1_z * tV2
                                            v_px = s0_x - proj_x
                                            v_py = s0_y - proj_y
                                            v_pz = s0_z - proj_z
                                            sqD = v_px * v_px + v_py * v_py + v_pz * v_pz
                                            d_p = thread.sqrt(sqD)
                                            pen_p = d_p - sumRadius
                                            n_px = v_px / d_p
                                            n_py = v_py / d_p
                                            n_pz = v_pz / d_p
                                            pt_x = s0_x - n_px * capsuleRadius0 + po_x
                                            pt_y = s0_y - n_py * capsuleRadius0 + po_y
                                            pt_z = s0_z - n_pz * capsuleRadius0 + po_z
                                            if nb_contacts == cp.int32(0):
                                                pp0_x = pt_x
                                                pp0_y = pt_y
                                                pp0_z = pt_z
                                                pp0_w = pen_p
                                            else:
                                                pp1_x = pt_x
                                                pp1_y = pt_y
                                                pp1_z = pt_z
                                                pp1_w = pen_p
                                            normal_x = n_px
                                            normal_y = n_py
                                            normal_z = n_pz
                                            nb_contacts = nb_contacts + cp.int32(1)

                                # Try tV3: e0 projected onto s1e1 (if numContacts < 2)
                                if nb_contacts < cp.int32(2):
                                    if tV3 >= cp.float32(0.0):
                                        if tV3 <= cp.float32(1.0):
                                            proj_x = s1_x + d1_x * tV3
                                            proj_y = s1_y + d1_y * tV3
                                            proj_z = s1_z + d1_z * tV3
                                            v_px = e0_x - proj_x
                                            v_py = e0_y - proj_y
                                            v_pz = e0_z - proj_z
                                            sqD = v_px * v_px + v_py * v_py + v_pz * v_pz
                                            d_p = thread.sqrt(sqD)
                                            pen_p = d_p - sumRadius
                                            n_px = v_px / d_p
                                            n_py = v_py / d_p
                                            n_pz = v_pz / d_p
                                            pt_x = e0_x - n_px * capsuleRadius0 + po_x
                                            pt_y = e0_y - n_py * capsuleRadius0 + po_y
                                            pt_z = e0_z - n_pz * capsuleRadius0 + po_z
                                            if nb_contacts == cp.int32(0):
                                                pp0_x = pt_x
                                                pp0_y = pt_y
                                                pp0_z = pt_z
                                                pp0_w = pen_p
                                            else:
                                                pp1_x = pt_x
                                                pp1_y = pt_y
                                                pp1_z = pt_z
                                                pp1_w = pen_p
                                            normal_x = n_px
                                            normal_y = n_py
                                            normal_z = n_pz
                                            nb_contacts = nb_contacts + cp.int32(1)

                            # fallback: non-parallel or no contacts from parallel case
                            if nb_contacts == cp.int32(0):
                                vv_len = thread.sqrt(sqDist0)
                                inv_vv = cp.float32(1.0) / vv_len
                                nn_x = vv_cx * inv_vv
                                nn_y = vv_cy * inv_vv
                                nn_z = vv_cz * inv_vv
                                pt_x = closA_x - nn_x * capsuleRadius0 + po_x
                                pt_y = closA_y - nn_y * capsuleRadius0 + po_y
                                pt_z = closA_z - nn_z * capsuleRadius0 + po_z
                                pen_f = vv_len - sumRadius
                                pp0_x = pt_x
                                pp0_y = pt_y
                                pp0_z = pt_z
                                pp0_w = pen_f
                                normal_x = nn_x
                                normal_y = nn_y
                                normal_z = nn_z
                                nb_contacts = cp.int32(1)

                # ============================================================
                # 7. Flip normal if shapes were swapped
                # ============================================================
                if nb_contacts > cp.int32(0):
                    if flip == cp.int32(1):
                        normal_x = -normal_x
                        normal_y = -normal_y
                        normal_z = -normal_z

                # ============================================================
                # 8. setContactPointAndForcePointers (inline)
                #    Atomic alloc in contact and force byte streams
                # ============================================================
                contactByteOffset = cp.int32(0x7FFFFFFF)  # use max positive as sentinel
                # We use -1 (0xFFFFFFFF) as sentinel in the CUDA code.
                # In Capybara int32, 0xFFFFFFFF = -1.
                contactByteOffset = cp.int32(-1)

                if nb_contacts > cp.int32(0):
                    contactAllocSize = cp.int32(SIZEOF_PX_CONTACT) * nb_contacts
                    forceAllocSize = cp.int32(SIZEOF_PX_U32) * nb_contacts
                    contactByteOffset = thread.atomic_add(patchAndContactCounters[COUNTER_CONTACTS_BYTES], contactAllocSize)
                    forceByteOffset = thread.atomic_add(patchAndContactCounters[COUNTER_FORCE_BYTES], forceAllocSize)

                    # Check overflow
                    if contactByteOffset + cp.int32(SIZEOF_PX_CONTACT) > contactBytesLimit:
                        # overflow -- set error flag
                        thread.atomic_add(patchAndContactCounters[COUNTER_OVERFLOW], cp.int32(1))
                        contactByteOffset = cp.int32(-1)
                    if forceByteOffset + cp.int32(SIZEOF_PX_U32) > forceBytesLimit:
                        thread.atomic_add(patchAndContactCounters[COUNTER_OVERFLOW], cp.int32(2))

                    if contactByteOffset != cp.int32(-1):
                        # Store byte offsets as contact point and force pointers in output
                        # contactPoints = startContactPoints + contactByteOffset
                        # contactForces = startContactForces + forceByteOffset
                        # These are stored as 64-bit pointers split into lo/hi int32
                        contactPtrLo = thread.bitcast(contactByteOffset, cp.int32)
                        contactPtrHi = cp.int32(0)
                        forcePtrLo = thread.bitcast(forceByteOffset, cp.int32)
                        forcePtrHi = cp.int32(0)
                        cmOutputs[globalThreadIndex, CMO_CONTACT_POINTS_LO] = contactPtrLo
                        cmOutputs[globalThreadIndex, CMO_CONTACT_POINTS_HI] = contactPtrHi
                        cmOutputs[globalThreadIndex, CMO_CONTACT_FORCES_LO] = forcePtrLo
                        cmOutputs[globalThreadIndex, CMO_CONTACT_FORCES_HI] = forcePtrHi
                    else:
                        cmOutputs[globalThreadIndex, CMO_CONTACT_POINTS_LO] = cp.int32(0)
                        cmOutputs[globalThreadIndex, CMO_CONTACT_POINTS_HI] = cp.int32(0)
                        cmOutputs[globalThreadIndex, CMO_CONTACT_FORCES_LO] = cp.int32(0)
                        cmOutputs[globalThreadIndex, CMO_CONTACT_FORCES_HI] = cp.int32(0)

                # ============================================================
                # 9. Write contact points to contact stream
                # ============================================================
                if contactByteOffset != cp.int32(-1):
                    # contactStream is float32[maxContacts, 4] where each row = PxContact (16 bytes)
                    # contactByteOffset / 16 = contact index
                    # But we need to handle this as a byte offset into a flat float32 stream.
                    # contactIndex = contactByteOffset / SIZEOF_PX_CONTACT
                    # NOTE: SIZEOF_PX_CONTACT = 16, and each contact = 4 float32s = row
                    contactIndex = contactByteOffset // cp.int32(SIZEOF_PX_CONTACT)
                    contactStream[contactIndex, 0] = pp0_x
                    contactStream[contactIndex, 1] = pp0_y
                    contactStream[contactIndex, 2] = pp0_z
                    contactStream[contactIndex, 3] = pp0_w
                    if nb_contacts > cp.int32(1):
                        contactStream[contactIndex + cp.int32(1), 0] = pp1_x
                        contactStream[contactIndex + cp.int32(1), 1] = pp1_y
                        contactStream[contactIndex + cp.int32(1), 2] = pp1_z
                        contactStream[contactIndex + cp.int32(1), 3] = pp1_w

                # ============================================================
                # 10. registerContactPatch (inline)
                #     Atomic alloc in patch stream, update touch/patch change flags
                # ============================================================
                # Read old allflags word
                allflags = cmOutputs[globalThreadIndex, CMO_ALLFLAGS] + cp.int32(0)
                # allflags layout (little-endian bytes in struct):
                #   byte0 = allflagsStart
                #   byte1 = nbPatches
                #   byte2 = statusFlag
                #   byte3 = prevPatches
                # As int32: allflags = byte0 | (byte1<<8) | (byte2<<16) | (byte3<<24)
                oldStatusFlags = (allflags >> cp.int32(16)) & cp.int32(0xFF)
                prevPatches = (allflags >> cp.int32(8)) & cp.int32(0xFF)

                statusFlags = oldStatusFlags & (~cp.int32(STATUS_TOUCH_KNOWN))
                if nb_contacts > cp.int32(0):
                    statusFlags = statusFlags | cp.int32(STATUS_HAS_TOUCH)
                else:
                    statusFlags = statusFlags | cp.int32(STATUS_HAS_NO_TOUCH)

                numPatches = cp.int32(0)
                if nb_contacts > cp.int32(0):
                    numPatches = cp.int32(1)

                previouslyHadTouch = cp.int32(0)
                if (oldStatusFlags & cp.int32(STATUS_HAS_TOUCH)) != cp.int32(0):
                    previouslyHadTouch = cp.int32(1)
                prevTouchKnown = cp.int32(0)
                if (oldStatusFlags & cp.int32(STATUS_TOUCH_KNOWN)) != cp.int32(0):
                    prevTouchKnown = cp.int32(1)

                currentlyHasTouch = cp.int32(0)
                if nb_contacts > cp.int32(0):
                    currentlyHasTouch = cp.int32(1)

                # change = (previouslyHadTouch ^ currentlyHasTouch) || (!prevTouchKnown)
                touchXor = previouslyHadTouch ^ currentlyHasTouch
                change = cp.int32(0)
                if touchXor != cp.int32(0):
                    change = cp.int32(1)
                if prevTouchKnown == cp.int32(0):
                    change = cp.int32(1)
                touchChangeFlags[globalThreadIndex] = change
                patchDiff = cp.int32(0)
                if prevPatches != numPatches:
                    patchDiff = cp.int32(1)
                patchChangeFlags[globalThreadIndex] = patchDiff

                # Write updated allflags:
                # CUDA: merge(merge(prevPatches, statusFlags), merge(numPatches, 0))
                # merge(uchar hi, uchar lo) -> ushort = (hi<<8)|lo
                # merge(ushort hi, ushort lo) -> uint = (hi<<16)|lo
                # inner1 = (prevPatches<<8)|statusFlags, inner2 = (numPatches<<8)|0
                # result = (inner1<<16)|inner2
                # byte0 = 0, byte1 = numPatches, byte2 = statusFlags, byte3 = prevPatches
                newAllflags = ((numPatches & cp.int32(0xFF)) << cp.int32(8)) | ((statusFlags & cp.int32(0xFF)) << cp.int32(16)) | ((prevPatches & cp.int32(0xFF)) << cp.int32(24))
                cmOutputs[globalThreadIndex, CMO_ALLFLAGS] = newAllflags

                # nbContacts in CMO (u16 in low 16 bits)
                nbContactsWord = cmOutputs[globalThreadIndex, CMO_NB_CONTACTS] + cp.int32(0)
                nbContactsWord = (nbContactsWord & cp.int32(0xFFFF0000)) | (nb_contacts & cp.int32(0xFFFF))
                cmOutputs[globalThreadIndex, CMO_NB_CONTACTS] = nbContactsWord

                # Allocate patch
                patchIndex = cp.int32(-1)
                if nb_contacts > cp.int32(0):
                    patchIndex = thread.atomic_add(patchAndContactCounters[COUNTER_PATCHES_BYTES], cp.int32(SIZEOF_PX_CONTACT_PATCH))
                    if patchIndex + cp.int32(SIZEOF_PX_CONTACT_PATCH) > patchBytesLimit:
                        # overflow
                        thread.atomic_add(patchAndContactCounters[COUNTER_OVERFLOW], cp.int32(4))
                        patchIndex = cp.int32(-1)

                        # Revert status
                        statusFlags = statusFlags & (~cp.int32(STATUS_TOUCH_KNOWN))
                        statusFlags = statusFlags | cp.int32(STATUS_HAS_NO_TOUCH)
                        # CUDA: merge(merge(prevPatches, statusFlags), 0)
                        # byte0=0, byte1=0, byte2=statusFlags, byte3=prevPatches
                        revertAllflags = ((statusFlags & cp.int32(0xFF)) << cp.int32(16)) | ((prevPatches & cp.int32(0xFF)) << cp.int32(24))
                        cmOutputs[globalThreadIndex, CMO_ALLFLAGS] = revertAllflags
                        cmOutputs[globalThreadIndex, CMO_NB_CONTACTS] = (nbContactsWord & cp.int32(0xFFFF0000))

                        touchChangeFlags[globalThreadIndex] = previouslyHadTouch
                        revertPatchDiff = cp.int32(0)
                        if prevPatches != cp.int32(0):
                            revertPatchDiff = cp.int32(1)
                        patchChangeFlags[globalThreadIndex] = revertPatchDiff
                    else:
                        # Store patch byte offset as contactPatches pointer
                        cmOutputs[globalThreadIndex, CMO_CONTACT_PATCHES_LO] = patchIndex
                        cmOutputs[globalThreadIndex, CMO_CONTACT_PATCHES_HI] = cp.int32(0)

                # ============================================================
                # 11. insertIntoPatchStream (inline)
                #     Write PxContactPatch header: material combine + normal
                # ============================================================
                if patchIndex != cp.int32(-1):
                    # patchRow = patchIndex / SIZEOF_PX_CONTACT_PATCH (=64) -> patch row index
                    patchRow = patchIndex // cp.int32(SIZEOF_PX_CONTACT_PATCH)

                    # Material combining (simplified inline of combineMaterials)
                    # matIdx0/matIdx1 are the original (un-flipped) material indices
                    # Read material data for both shapes
                    mat0_dynFric_i = materials[matIdx0, MAT_DYN_FRICTION] + cp.int32(0)
                    mat0_staFric_i = materials[matIdx0, MAT_STA_FRICTION] + cp.int32(0)
                    mat0_rest_i = materials[matIdx0, MAT_RESTITUTION] + cp.int32(0)
                    mat0_damp_i = materials[matIdx0, MAT_DAMPING] + cp.int32(0)
                    mat0_flagsModes = materials[matIdx0, MAT_FLAGS_MODES] + cp.int32(0)
                    mat0_dampMode_word = materials[matIdx0, MAT_DAMPING_MODE] + cp.int32(0)
                    mat0_dynFric = thread.bitcast(mat0_dynFric_i, cp.float32)
                    mat0_staFric = thread.bitcast(mat0_staFric_i, cp.float32)
                    mat0_rest = thread.bitcast(mat0_rest_i, cp.float32)
                    mat0_damp = thread.bitcast(mat0_damp_i, cp.float32)
                    mat0_flags = mat0_flagsModes & cp.int32(0xFFFF)
                    mat0_fricMode = (mat0_flagsModes >> cp.int32(16)) & cp.int32(0xFF)
                    mat0_restMode = (mat0_flagsModes >> cp.int32(24)) & cp.int32(0xFF)
                    mat0_dampMode = mat0_dampMode_word & cp.int32(0xFF)

                    mat1_dynFric_i = materials[matIdx1, MAT_DYN_FRICTION] + cp.int32(0)
                    mat1_staFric_i = materials[matIdx1, MAT_STA_FRICTION] + cp.int32(0)
                    mat1_rest_i = materials[matIdx1, MAT_RESTITUTION] + cp.int32(0)
                    mat1_damp_i = materials[matIdx1, MAT_DAMPING] + cp.int32(0)
                    mat1_flagsModes = materials[matIdx1, MAT_FLAGS_MODES] + cp.int32(0)
                    mat1_dampMode_word = materials[matIdx1, MAT_DAMPING_MODE] + cp.int32(0)
                    mat1_dynFric = thread.bitcast(mat1_dynFric_i, cp.float32)
                    mat1_staFric = thread.bitcast(mat1_staFric_i, cp.float32)
                    mat1_rest = thread.bitcast(mat1_rest_i, cp.float32)
                    mat1_damp = thread.bitcast(mat1_damp_i, cp.float32)
                    mat1_flags = mat1_flagsModes & cp.int32(0xFFFF)
                    mat1_fricMode = (mat1_flagsModes >> cp.int32(16)) & cp.int32(0xFF)
                    mat1_restMode = (mat1_flagsModes >> cp.int32(24)) & cp.int32(0xFF)
                    mat1_dampMode = mat1_dampMode_word & cp.int32(0xFF)

                    # ---- Combine restitution ----
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

                    combinedRest = cp.float32(0.0)
                    if (bothCompliant != cp.int32(0)) & (exactlyOneAccCompliant != cp.int32(0)):
                        combinedRest = mat0_rest
                        if compliantAcc0 == cp.int32(0):
                            combinedRest = mat1_rest
                    else:
                        restCombMode = max_i32(mat0_restMode, mat1_restMode)
                        if exactlyOneCompliant != cp.int32(0):
                            restCombMode = cp.int32(1)  # eMIN
                        flipSign = cp.float32(1.0)
                        if bothCompliant != cp.int32(0):
                            if restCombMode == cp.int32(2):  # eMULTIPLY
                                flipSign = cp.float32(-1.0)
                        combinedRest = flipSign * combine_scalars(mat0_rest, mat1_rest, restCombMode)

                    # ---- Combine damping ----
                    combinedDamp = cp.float32(0.0)
                    if (bothCompliant != cp.int32(0)) & (exactlyOneAccCompliant != cp.int32(0)):
                        combinedDamp = mat0_damp
                        if compliantAcc0 == cp.int32(0):
                            combinedDamp = mat1_damp
                    else:
                        dampCombMode = max_i32(mat0_dampMode, mat1_dampMode)
                        if exactlyOneCompliant != cp.int32(0):
                            dampCombMode = cp.int32(3)  # eMAX
                        combinedDamp = combine_scalars(mat0_damp, mat1_damp, dampCombMode)

                    # ---- Combine friction ----
                    combineFlags = mat0_flags | mat1_flags
                    combinedDynFric = cp.float32(0.0)
                    combinedStaFric = cp.float32(0.0)
                    combinedMatFlags = combineFlags
                    if (combineFlags & cp.int32(MATFLAG_DISABLE_FRICTION)) == cp.int32(0):
                        fricCombMode = max_i32(mat0_fricMode, mat1_fricMode)
                        combinedDynFric = combine_scalars(mat0_dynFric, mat1_dynFric, fricCombMode)
                        combinedStaFric = combine_scalars(mat0_staFric, mat1_staFric, fricCombMode)
                        # clamp dynamic friction >= 0
                        combinedDynFric = max_f32(combinedDynFric, cp.float32(0.0))
                        # static friction >= dynamic friction
                        if combinedStaFric - combinedDynFric < cp.float32(0.0):
                            combinedStaFric = combinedDynFric
                    else:
                        combinedMatFlags = combineFlags | cp.int32(MATFLAG_DISABLE_STRONG_FRICTION)

                    # ---- Write patch to stream ----
                    # Mass modification: all 1.0
                    patchStream[patchRow, PS_MASS_MOD_LINEAR0] = thread.bitcast(cp.float32(1.0), cp.int32)
                    patchStream[patchRow, PS_MASS_MOD_ANGULAR0] = thread.bitcast(cp.float32(1.0), cp.int32)
                    patchStream[patchRow, PS_MASS_MOD_LINEAR1] = thread.bitcast(cp.float32(1.0), cp.int32)
                    patchStream[patchRow, PS_MASS_MOD_ANGULAR1] = thread.bitcast(cp.float32(1.0), cp.int32)
                    # Normal
                    patchStream[patchRow, PS_NORMAL_X] = thread.bitcast(normal_x, cp.int32)
                    patchStream[patchRow, PS_NORMAL_Y] = thread.bitcast(normal_y, cp.int32)
                    patchStream[patchRow, PS_NORMAL_Z] = thread.bitcast(normal_z, cp.int32)
                    # Material properties
                    patchStream[patchRow, PS_RESTITUTION] = thread.bitcast(combinedRest, cp.int32)
                    patchStream[patchRow, PS_DYN_FRICTION] = thread.bitcast(combinedDynFric, cp.int32)
                    patchStream[patchRow, PS_STA_FRICTION] = thread.bitcast(combinedStaFric, cp.int32)
                    patchStream[patchRow, PS_DAMPING] = thread.bitcast(combinedDamp, cp.int32)
                    # startContactIndex(u16=0) | nbContacts(u8) | materialFlags(u8)
                    startNbMatflags = (nb_contacts & cp.int32(0xFF)) << cp.int32(16)
                    startNbMatflags = startNbMatflags | ((combinedMatFlags & cp.int32(0xFF)) << cp.int32(24))
                    patchStream[patchRow, PS_START_NB_MATFLAGS] = startNbMatflags
                    # internalFlags(u16=0) | materialIndex0(u16)
                    intflagMatIdx0 = (matIdx0 & cp.int32(0xFFFF)) << cp.int32(16)
                    patchStream[patchRow, PS_INTFLAGS_MATIDX0] = intflagMatIdx0
                    # materialIndex1(u16) | pad(u16=0)
                    matIdx1Pad = matIdx1 & cp.int32(0xFFFF)
                    patchStream[patchRow, PS_MATIDX1_PAD] = matIdx1Pad
                    # Remaining pad words = 0
                    patchStream[patchRow, cp.int32(14)] = cp.int32(0)
                    patchStream[patchRow, cp.int32(15)] = cp.int32(0)
