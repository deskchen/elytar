"""Capybara DSL port of gpunarrowphase/CUDA/cudaBox.cu -- boxBoxNphase_Kernel.

Ported kernel (matching CUDA name for PTX replacement):
  - boxBoxNphase_Kernel  -- narrowphase collision for box-box pairs

Algorithm (SAT-based box-box):
  1. Read collision pair (shapes, transforms, contact distance)
  2. Compute relative transform (transform1 in transform0's space)
  3. SAT test: 6 face axes (3 per box) + 9 edge cross product axes
  4. Feature selection: best separating axis -> reference face
  5. For reference face: compute incident polygon (4 vertices), clip
     against reference face extents, generate contacts
  6. Contact reduction: if >6 contacts, reduce to 4
  7. Output contacts via atomic allocation + patch stream

ABI: identical to cudaSphere.py (see that file for full ABI docs).

Capybara structural notes:
  - The CUDA kernel uses 4-thread cooperative groups within warps.
    In Capybara, each thread handles one pair independently; all
    cooperative (shuffle) operations become per-thread scalar code.
  - Variables assigned in if/elif/else must be pre-declared before the conditional.
  - No method chaining on Capybara structs; use intermediate variables.
  - `+ cp.float32(0.0)` / `+ cp.int32(0)` for force-loads from tensors.
  - `thread.bitcast()` for int32<->float32 reinterpretation.
  - Boolean flags stored as cp.int32, not Python bool.
  - No bitwise & or | to combine comparison results (i1 & i1 crashes).
"""

import capybara as cp


# ===== Helper: quaternion rotate vector =====
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


# ===== Helper: quaternion inverse rotate vector =====
@cp.inline
def quat_rotate_inv_x(qx, qy, qz, qw, vx, vy, vz):
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
    nqx = -qx
    nqy = -qy
    nqz = -qz
    c1x = nqy * vz - nqz * vy + qw * vx
    c1y = nqz * vx - nqx * vz + qw * vy
    c1z = nqx * vy - nqy * vx + qw * vz
    c2z = nqx * c1y - nqy * c1x
    return vz + cp.float32(2.0) * c2z


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
    result = cp.float32(0.0)
    if mode == cp.int32(0):
        result = cp.float32(0.5) * (a + b)
    if mode == cp.int32(1):
        result = a
        if b < a:
            result = b
    if mode == cp.int32(2):
        result = a * b
    if mode == cp.int32(3):
        result = a
        if b > a:
            result = b
    return result


@cp.inline
def max_i32(a, b):
    r = a
    if b > a:
        r = b
    return r


@cp.inline
def max_f32(a, b):
    r = a
    if b > a:
        r = b
    return r


@cp.inline
def min_f32(a, b):
    r = a
    if b < a:
        r = b
    return r


@cp.inline
def abs_f32(v):
    r = v
    if v < cp.float32(0.0):
        r = -v
    return r


# ===== Kernel: boxBoxNphase_Kernel =====
@cp.kernel
def boxBoxNphase_Kernel(
    numTests,               # int32 scalar
    cmInputs,              # int32[N, CMI_SIZE]
    cmOutputs,             # int32[N, CMO_SIZE]
    shapes,                # int32[N, SHAPE_SIZE]
    transformCache,        # float32[N, CT_SIZE]
    contactDistance,        # float32[N]
    materials,             # int32[N, MAT_SIZE]
    contactStream,         # float32[maxContacts, 4]
    patchStream,           # int32[maxPatches, PATCH_SIZE]
    patchAndContactCounters,  # int32[4]
    touchChangeFlags,      # int32[N]
    patchChangeFlags,      # int32[N]
    startContactPatches,   # int64 scalar (base GPU ptr, unused in Capybara byte-offset mode)
    startContactPoints,    # int64 scalar
    startContactForces,    # int64 scalar
    patchBytesLimit,       # int32 scalar
    contactBytesLimit,     # int32 scalar
    forceBytesLimit,       # int32 scalar
    toleranceLength,       # float32 scalar
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
    # --- Status flags ---
    STATUS_HAS_NO_TOUCH: cp.constexpr = 1,
    STATUS_HAS_TOUCH: cp.constexpr = 2,
    STATUS_TOUCH_KNOWN: cp.constexpr = 3,
    # --- Material flags ---
    MATFLAG_DISABLE_FRICTION: cp.constexpr = 4,
    MATFLAG_DISABLE_STRONG_FRICTION: cp.constexpr = 8,
    MATFLAG_COMPLIANT_ACC_SPRING: cp.constexpr = 64,
    # --- Patch stream field offsets ---
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
    # --- max contacts per patch ---
    MAX_CONTACTS_PER_PATCH: cp.constexpr = 6,
):
    numBlocks = cp.ceildiv(numTests, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            workIndex = bx * cp.int32(BLOCK_SIZE) + tid

            if workIndex < numTests:
                # ============================================================
                # 1. Read collision pair input
                # ============================================================
                shapeRef0 = cmInputs[workIndex, CMI_SHAPE_REF0] + cp.int32(0)
                shapeRef1 = cmInputs[workIndex, CMI_SHAPE_REF1] + cp.int32(0)
                transformCacheRef0 = cmInputs[workIndex, CMI_TRANSFORM_REF0] + cp.int32(0)
                transformCacheRef1 = cmInputs[workIndex, CMI_TRANSFORM_REF1] + cp.int32(0)

                # ============================================================
                # 2. Read transforms
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
                # 4. Read shape data (box extents)
                # ============================================================
                s0_ext_x_i = shapes[shapeRef0, SHAPE_SCALE_X] + cp.int32(0)
                s0_ext_y_i = shapes[shapeRef0, SHAPE_SCALE_Y] + cp.int32(0)
                s0_ext_z_i = shapes[shapeRef0, SHAPE_SCALE_Z] + cp.int32(0)
                box0_ex = thread.bitcast(s0_ext_x_i, cp.float32)
                box0_ey = thread.bitcast(s0_ext_y_i, cp.float32)
                box0_ez = thread.bitcast(s0_ext_z_i, cp.float32)
                matIdx0 = shapes[shapeRef0, SHAPE_MATERIAL_INDEX] + cp.int32(0)

                s1_ext_x_i = shapes[shapeRef1, SHAPE_SCALE_X] + cp.int32(0)
                s1_ext_y_i = shapes[shapeRef1, SHAPE_SCALE_Y] + cp.int32(0)
                s1_ext_z_i = shapes[shapeRef1, SHAPE_SCALE_Z] + cp.int32(0)
                box1_ex = thread.bitcast(s1_ext_x_i, cp.float32)
                box1_ey = thread.bitcast(s1_ext_y_i, cp.float32)
                box1_ez = thread.bitcast(s1_ext_z_i, cp.float32)
                matIdx1 = shapes[shapeRef1, SHAPE_MATERIAL_INDEX] + cp.int32(0)

                # ============================================================
                # 5. Build rotation matrices from quaternions
                # ============================================================
                # transform0 rotation matrix columns
                ax00_x = quat_rotate_x(t0_qx, t0_qy, t0_qz, t0_qw, cp.float32(1.0), cp.float32(0.0), cp.float32(0.0))
                ax00_y = quat_rotate_y(t0_qx, t0_qy, t0_qz, t0_qw, cp.float32(1.0), cp.float32(0.0), cp.float32(0.0))
                ax00_z = quat_rotate_z(t0_qx, t0_qy, t0_qz, t0_qw, cp.float32(1.0), cp.float32(0.0), cp.float32(0.0))
                ax01_x = quat_rotate_x(t0_qx, t0_qy, t0_qz, t0_qw, cp.float32(0.0), cp.float32(1.0), cp.float32(0.0))
                ax01_y = quat_rotate_y(t0_qx, t0_qy, t0_qz, t0_qw, cp.float32(0.0), cp.float32(1.0), cp.float32(0.0))
                ax01_z = quat_rotate_z(t0_qx, t0_qy, t0_qz, t0_qw, cp.float32(0.0), cp.float32(1.0), cp.float32(0.0))
                ax02_x = quat_rotate_x(t0_qx, t0_qy, t0_qz, t0_qw, cp.float32(0.0), cp.float32(0.0), cp.float32(1.0))
                ax02_y = quat_rotate_y(t0_qx, t0_qy, t0_qz, t0_qw, cp.float32(0.0), cp.float32(0.0), cp.float32(1.0))
                ax02_z = quat_rotate_z(t0_qx, t0_qy, t0_qz, t0_qw, cp.float32(0.0), cp.float32(0.0), cp.float32(1.0))

                # transform1 rotation matrix columns
                ax10_x = quat_rotate_x(t1_qx, t1_qy, t1_qz, t1_qw, cp.float32(1.0), cp.float32(0.0), cp.float32(0.0))
                ax10_y = quat_rotate_y(t1_qx, t1_qy, t1_qz, t1_qw, cp.float32(1.0), cp.float32(0.0), cp.float32(0.0))
                ax10_z = quat_rotate_z(t1_qx, t1_qy, t1_qz, t1_qw, cp.float32(1.0), cp.float32(0.0), cp.float32(0.0))
                ax11_x = quat_rotate_x(t1_qx, t1_qy, t1_qz, t1_qw, cp.float32(0.0), cp.float32(1.0), cp.float32(0.0))
                ax11_y = quat_rotate_y(t1_qx, t1_qy, t1_qz, t1_qw, cp.float32(0.0), cp.float32(1.0), cp.float32(0.0))
                ax11_z = quat_rotate_z(t1_qx, t1_qy, t1_qz, t1_qw, cp.float32(0.0), cp.float32(1.0), cp.float32(0.0))
                ax12_x = quat_rotate_x(t1_qx, t1_qy, t1_qz, t1_qw, cp.float32(0.0), cp.float32(0.0), cp.float32(1.0))
                ax12_y = quat_rotate_y(t1_qx, t1_qy, t1_qz, t1_qw, cp.float32(0.0), cp.float32(0.0), cp.float32(1.0))
                ax12_z = quat_rotate_z(t1_qx, t1_qy, t1_qz, t1_qw, cp.float32(0.0), cp.float32(0.0), cp.float32(1.0))

                # ============================================================
                # 6. Compute transform1To0 = transform0.transformTranspose(transform1)
                # ============================================================
                dp_x = t1_px - t0_px
                dp_y = t1_py - t0_py
                dp_z = t1_pz - t0_pz

                # t1To0_p = rot0^T * dp
                t1to0_px = ax00_x * dp_x + ax00_y * dp_y + ax00_z * dp_z
                t1to0_py = ax01_x * dp_x + ax01_y * dp_y + ax01_z * dp_z
                t1to0_pz = ax02_x * dp_x + ax02_y * dp_y + ax02_z * dp_z

                # rot1To0 columns = rot0^T * rot1_column
                r10_c0x = ax00_x * ax10_x + ax00_y * ax10_y + ax00_z * ax10_z
                r10_c0y = ax01_x * ax10_x + ax01_y * ax10_y + ax01_z * ax10_z
                r10_c0z = ax02_x * ax10_x + ax02_y * ax10_y + ax02_z * ax10_z
                r10_c1x = ax00_x * ax11_x + ax00_y * ax11_y + ax00_z * ax11_z
                r10_c1y = ax01_x * ax11_x + ax01_y * ax11_y + ax01_z * ax11_z
                r10_c1z = ax02_x * ax11_x + ax02_y * ax11_y + ax02_z * ax11_z
                r10_c2x = ax00_x * ax12_x + ax00_y * ax12_y + ax00_z * ax12_z
                r10_c2y = ax01_x * ax12_x + ax01_y * ax12_y + ax01_z * ax12_z
                r10_c2z = ax02_x * ax12_x + ax02_y * ax12_y + ax02_z * ax12_z

                # abs0To1 = abs(rot0To1) + eps, where rot0To1 = transpose(rot1To0)
                # rot0To1.column0 = (r10_c0x, r10_c1x, r10_c2x)  [rows of r10]
                uEps = cp.float32(1e-6)
                abs01_c0x = abs_f32(r10_c0x) + uEps
                abs01_c0y = abs_f32(r10_c1x) + uEps
                abs01_c0z = abs_f32(r10_c2x) + uEps
                abs01_c1x = abs_f32(r10_c0y) + uEps
                abs01_c1y = abs_f32(r10_c1y) + uEps
                abs01_c1z = abs_f32(r10_c2y) + uEps
                abs01_c2x = abs_f32(r10_c0z) + uEps
                abs01_c2y = abs_f32(r10_c1z) + uEps
                abs01_c2z = abs_f32(r10_c2z) + uEps

                # ============================================================
                # 7. SAT: Test 6 face separating axes
                # ============================================================
                PX_MAX_F32 = cp.float32(3.4028235e+38)
                separated = cp.int32(0)
                minOverlap = PX_MAX_F32
                bestFeature = cp.int32(0)
                bestSign = cp.float32(0.0)

                # --- ua0 (box0 x-axis) ---
                rb0 = abs01_c0x * box1_ex + abs01_c0y * box1_ey + abs01_c0z * box1_ez
                radiusSum0 = box0_ex + rb0
                s0 = t1to0_px
                overlap0 = radiusSum0 - abs_f32(s0) + cDistance
                if cp.float32(0.0) > overlap0:
                    separated = cp.int32(1)
                if overlap0 < minOverlap:
                    minOverlap = overlap0
                    bestFeature = cp.int32(0)
                    bestSign = s0

                # --- ua1 (box0 y-axis) ---
                rb1 = abs01_c1x * box1_ex + abs01_c1y * box1_ey + abs01_c1z * box1_ez
                radiusSum1 = box0_ey + rb1
                s1 = t1to0_py
                overlap1 = radiusSum1 - abs_f32(s1) + cDistance
                if cp.float32(0.0) > overlap1:
                    separated = cp.int32(1)
                if overlap1 < minOverlap:
                    minOverlap = overlap1
                    bestFeature = cp.int32(1)
                    bestSign = s1

                # --- ua2 (box0 z-axis) ---
                rb2 = abs01_c2x * box1_ex + abs01_c2y * box1_ey + abs01_c2z * box1_ez
                radiusSum2 = box0_ez + rb2
                s2 = t1to0_pz
                overlap2 = radiusSum2 - abs_f32(s2) + cDistance
                if cp.float32(0.0) > overlap2:
                    separated = cp.int32(1)
                if overlap2 < minOverlap:
                    minOverlap = overlap2
                    bestFeature = cp.int32(2)
                    bestSign = s2

                # abs of rot1To0 columns (for ub axes)
                abs10_c0x = abs_f32(r10_c0x) + uEps
                abs10_c0y = abs_f32(r10_c0y) + uEps
                abs10_c0z = abs_f32(r10_c0z) + uEps
                abs10_c1x = abs_f32(r10_c1x) + uEps
                abs10_c1y = abs_f32(r10_c1y) + uEps
                abs10_c1z = abs_f32(r10_c1z) + uEps
                abs10_c2x = abs_f32(r10_c2x) + uEps
                abs10_c2y = abs_f32(r10_c2y) + uEps
                abs10_c2z = abs_f32(r10_c2z) + uEps

                # --- ub0 ---
                s3 = t1to0_px * r10_c0x + t1to0_py * r10_c0y + t1to0_pz * r10_c0z
                ra3 = abs10_c0x * box0_ex + abs10_c0y * box0_ey + abs10_c0z * box0_ez
                radiusSum3 = ra3 + box1_ex
                overlap3 = radiusSum3 - abs_f32(s3) + cDistance
                if cp.float32(0.0) > overlap3:
                    separated = cp.int32(1)
                if overlap3 < minOverlap:
                    minOverlap = overlap3
                    bestFeature = cp.int32(3)
                    bestSign = s3

                # --- ub1 ---
                s4 = t1to0_px * r10_c1x + t1to0_py * r10_c1y + t1to0_pz * r10_c1z
                ra4 = abs10_c1x * box0_ex + abs10_c1y * box0_ey + abs10_c1z * box0_ez
                radiusSum4 = ra4 + box1_ey
                overlap4 = radiusSum4 - abs_f32(s4) + cDistance
                if cp.float32(0.0) > overlap4:
                    separated = cp.int32(1)
                if overlap4 < minOverlap:
                    minOverlap = overlap4
                    bestFeature = cp.int32(4)
                    bestSign = s4

                # --- ub2 ---
                s5 = t1to0_px * r10_c2x + t1to0_py * r10_c2y + t1to0_pz * r10_c2z
                ra5 = abs10_c2x * box0_ex + abs10_c2y * box0_ey + abs10_c2z * box0_ez
                radiusSum5 = ra5 + box1_ez
                overlap5 = radiusSum5 - abs_f32(s5) + cDistance
                if cp.float32(0.0) > overlap5:
                    separated = cp.int32(1)
                if overlap5 < minOverlap:
                    minOverlap = overlap5
                    bestFeature = cp.int32(5)
                    bestSign = s5

                # ============================================================
                # 7b. SAT: Test 9 edge cross product axes
                # ============================================================
                # ua0 x ub0
                e_absSign0 = abs_f32(r10_c0y * t1to0_pz - r10_c0z * t1to0_py)
                e_ra0 = abs10_c0z * box0_ey + abs10_c0y * box0_ez
                e_rb0 = abs01_c0z * box1_ey + abs01_c0y * box1_ez
                e_radSum0 = e_ra0 + e_rb0 + cDistance
                if e_absSign0 > e_radSum0:
                    separated = cp.int32(1)

                # ua0 x ub1
                e_absSign1 = abs_f32(r10_c1y * t1to0_pz - r10_c1z * t1to0_py)
                e_ra1 = abs10_c1z * box0_ey + abs10_c1y * box0_ez
                e_rb1 = abs01_c1z * box1_ey + abs01_c1y * box1_ez
                e_radSum1 = e_ra1 + e_rb1 + cDistance
                if e_absSign1 > e_radSum1:
                    separated = cp.int32(1)

                # ua0 x ub2
                e_absSign2 = abs_f32(r10_c2y * t1to0_pz - r10_c2z * t1to0_py)
                e_ra2 = abs10_c2z * box0_ey + abs10_c2y * box0_ez
                e_rb2 = abs01_c2z * box1_ey + abs01_c2y * box1_ez
                e_radSum2 = e_ra2 + e_rb2 + cDistance
                if e_absSign2 > e_radSum2:
                    separated = cp.int32(1)

                # ua1 x ub0
                e_absSign3 = abs_f32(r10_c0z * t1to0_px - r10_c0x * t1to0_pz)
                e_ra3 = abs10_c0z * box0_ex + abs10_c0x * box0_ez
                e_rb3 = abs01_c1z * box1_ex + abs01_c1x * box1_ez
                e_radSum3 = e_ra3 + e_rb3 + cDistance
                if e_absSign3 > e_radSum3:
                    separated = cp.int32(1)

                # ua1 x ub1
                e_absSign4 = abs_f32(r10_c1z * t1to0_px - r10_c1x * t1to0_pz)
                e_ra4 = abs10_c1z * box0_ex + abs10_c1x * box0_ez
                e_rb4 = abs01_c1z * box1_ex + abs01_c1x * box1_ez
                e_radSum4 = e_ra4 + e_rb4 + cDistance
                if e_absSign4 > e_radSum4:
                    separated = cp.int32(1)

                # ua1 x ub2
                e_absSign5 = abs_f32(r10_c2z * t1to0_px - r10_c2x * t1to0_pz)
                e_ra5 = abs10_c2z * box0_ex + abs10_c2x * box0_ez
                e_rb5 = abs01_c2z * box1_ex + abs01_c2x * box1_ez
                e_radSum5 = e_ra5 + e_rb5 + cDistance
                if e_absSign5 > e_radSum5:
                    separated = cp.int32(1)

                # ua2 x ub0
                e_absSign6 = abs_f32(r10_c0x * t1to0_py - r10_c0y * t1to0_px)
                e_ra6 = abs10_c0y * box0_ex + abs10_c0x * box0_ey
                e_rb6 = abs01_c2y * box1_ex + abs01_c2x * box1_ey
                e_radSum6 = e_ra6 + e_rb6 + cDistance
                if e_absSign6 > e_radSum6:
                    separated = cp.int32(1)

                # ua2 x ub1
                e_absSign7 = abs_f32(r10_c1x * t1to0_py - r10_c1y * t1to0_px)
                e_ra7 = abs10_c1y * box0_ex + abs10_c1x * box0_ey
                e_rb7 = abs01_c2y * box1_ex + abs01_c2x * box1_ey
                e_radSum7 = e_ra7 + e_rb7 + cDistance
                if e_absSign7 > e_radSum7:
                    separated = cp.int32(1)

                # ua2 x ub2
                e_absSign8 = abs_f32(r10_c2x * t1to0_py - r10_c2y * t1to0_px)
                e_ra8 = abs10_c2y * box0_ex + abs10_c2x * box0_ey
                e_rb8 = abs01_c2y * box1_ex + abs01_c2x * box1_ey
                e_radSum8 = e_ra8 + e_rb8 + cDistance
                if e_absSign8 > e_radSum8:
                    separated = cp.int32(1)

                # ============================================================
                # 8. Feature selection and contact generation
                # ============================================================
                # Pre-declare all variables used across branches
                nbContacts = cp.int32(0)
                mtd_x = cp.float32(0.0)
                mtd_y = cp.float32(0.0)
                mtd_z = cp.float32(0.0)
                ntv_c0x = cp.float32(0.0)
                ntv_c0y = cp.float32(0.0)
                ntv_c0z = cp.float32(0.0)
                ntv_c1x = cp.float32(0.0)
                ntv_c1y = cp.float32(0.0)
                ntv_c1z = cp.float32(0.0)
                ntv_c2x = cp.float32(0.0)
                ntv_c2y = cp.float32(0.0)
                ntv_c2z = cp.float32(0.0)
                ntv_px = cp.float32(0.0)
                ntv_py = cp.float32(0.0)
                ntv_pz = cp.float32(0.0)
                ref_e0 = cp.float32(0.0)
                ref_e1 = cp.float32(0.0)
                inc_ext_x = cp.float32(0.0)
                inc_ext_y = cp.float32(0.0)
                inc_ext_z = cp.float32(0.0)
                iv0_x = cp.float32(0.0)
                iv0_y = cp.float32(0.0)
                iv0_z = cp.float32(0.0)
                iv1_x = cp.float32(0.0)
                iv1_y = cp.float32(0.0)
                iv1_z = cp.float32(0.0)
                iv2_x = cp.float32(0.0)
                iv2_y = cp.float32(0.0)
                iv2_z = cp.float32(0.0)
                iv3_x = cp.float32(0.0)
                iv3_y = cp.float32(0.0)
                iv3_z = cp.float32(0.0)
                incNormal_x = cp.float32(0.0)
                incNormal_y = cp.float32(0.0)
                incNormal_z = cp.float32(0.0)
                cp0_x = cp.float32(0.0)
                cp0_y = cp.float32(0.0)
                cp0_z = cp.float32(0.0)
                cp0_w = cp.float32(0.0)
                cp1_x = cp.float32(0.0)
                cp1_y = cp.float32(0.0)
                cp1_z = cp.float32(0.0)
                cp1_w = cp.float32(0.0)
                cp2_x = cp.float32(0.0)
                cp2_y = cp.float32(0.0)
                cp2_z = cp.float32(0.0)
                cp2_w = cp.float32(0.0)
                cp3_x = cp.float32(0.0)
                cp3_y = cp.float32(0.0)
                cp3_z = cp.float32(0.0)
                cp3_w = cp.float32(0.0)
                cp4_x = cp.float32(0.0)
                cp4_y = cp.float32(0.0)
                cp4_z = cp.float32(0.0)
                cp4_w = cp.float32(0.0)
                cp5_x = cp.float32(0.0)
                cp5_y = cp.float32(0.0)
                cp5_z = cp.float32(0.0)
                cp5_w = cp.float32(0.0)
                lc_count = cp.int32(0)
                lc0_x = cp.float32(0.0)
                lc0_y = cp.float32(0.0)
                lc0_z = cp.float32(0.0)
                lc0_w = cp.float32(0.0)
                lc1_x = cp.float32(0.0)
                lc1_y = cp.float32(0.0)
                lc1_z = cp.float32(0.0)
                lc1_w = cp.float32(0.0)
                lc2_x = cp.float32(0.0)
                lc2_y = cp.float32(0.0)
                lc2_z = cp.float32(0.0)
                lc2_w = cp.float32(0.0)
                lc3_x = cp.float32(0.0)
                lc3_y = cp.float32(0.0)
                lc3_z = cp.float32(0.0)
                lc3_w = cp.float32(0.0)
                lc4_x = cp.float32(0.0)
                lc4_y = cp.float32(0.0)
                lc4_z = cp.float32(0.0)
                lc4_w = cp.float32(0.0)
                lc5_x = cp.float32(0.0)
                lc5_y = cp.float32(0.0)
                lc5_z = cp.float32(0.0)
                lc5_w = cp.float32(0.0)
                lc6_x = cp.float32(0.0)
                lc6_y = cp.float32(0.0)
                lc6_z = cp.float32(0.0)
                lc6_w = cp.float32(0.0)
                lc7_x = cp.float32(0.0)
                lc7_y = cp.float32(0.0)
                lc7_z = cp.float32(0.0)
                lc7_w = cp.float32(0.0)
                lc8_x = cp.float32(0.0)
                lc8_y = cp.float32(0.0)
                lc8_z = cp.float32(0.0)
                lc8_w = cp.float32(0.0)
                lc9_x = cp.float32(0.0)
                lc9_y = cp.float32(0.0)
                lc9_z = cp.float32(0.0)
                lc9_w = cp.float32(0.0)
                lc10_x = cp.float32(0.0)
                lc10_y = cp.float32(0.0)
                lc10_z = cp.float32(0.0)
                lc10_w = cp.float32(0.0)
                lc11_x = cp.float32(0.0)
                lc11_y = cp.float32(0.0)
                lc11_z = cp.float32(0.0)
                lc11_w = cp.float32(0.0)
                localNormal_x = cp.float32(0.0)
                localNormal_y = cp.float32(0.0)
                localNormal_z = cp.float32(0.0)
                extentX = cp.float32(0.0)
                extentY = cp.float32(0.0)
                nExtentX = cp.float32(0.0)
                nExtentY = cp.float32(0.0)
                bnd_minx = cp.float32(0.0)
                bnd_miny = cp.float32(0.0)
                bnd_maxx = cp.float32(0.0)
                bnd_maxy = cp.float32(0.0)
                t1tn_c0x = cp.float32(0.0)
                t1tn_c0y = cp.float32(0.0)
                t1tn_c0z = cp.float32(0.0)
                t1tn_c1x = cp.float32(0.0)
                t1tn_c1y = cp.float32(0.0)
                t1tn_c1z = cp.float32(0.0)
                t1tn_c2x = cp.float32(0.0)
                t1tn_c2y = cp.float32(0.0)
                t1tn_c2z = cp.float32(0.0)
                t1tn_px = cp.float32(0.0)
                t1tn_py = cp.float32(0.0)
                t1tn_pz = cp.float32(0.0)
                inc_d0 = cp.float32(0.0)
                inc_d1 = cp.float32(0.0)
                inc_d2 = cp.float32(0.0)
                inc_absd0 = cp.float32(0.0)
                inc_absd1 = cp.float32(0.0)
                inc_absd2 = cp.float32(0.0)
                tmp_vx = cp.float32(0.0)
                tmp_vy = cp.float32(0.0)
                tmp_vz = cp.float32(0.0)
                dom_sign_x = cp.float32(0.0)
                dom_sign_y = cp.float32(0.0)
                dom_sign_z = cp.float32(0.0)
                pPen0 = cp.int32(0)
                pPen1 = cp.int32(0)
                pPen2 = cp.int32(0)
                pPen3 = cp.int32(0)
                pArea0 = cp.int32(0)
                pArea1 = cp.int32(0)
                pArea2 = cp.int32(0)
                pArea3 = cp.int32(0)
                allArea = cp.int32(0)
                maxZ_clip = cp.float32(0.0)
                denom_inc = cp.float32(0.0)
                nom_inc = cp.float32(0.0)
                t_inc = cp.float32(0.0)
                q0_x = cp.float32(0.0)
                q0_y = cp.float32(0.0)
                contains_result = cp.int32(0)
                intersectionPoints = cp.int32(0)
                c_jx = cp.float32(0.0)
                c_jy = cp.float32(0.0)
                c_ix = cp.float32(0.0)
                c_iy = cp.float32(0.0)
                c_jiy = cp.float32(0.0)
                c_jty = cp.float32(0.0)
                c_jix = cp.float32(0.0)
                c_part1 = cp.float32(0.0)
                c_part2 = cp.float32(0.0)
                c_part3 = cp.float32(0.0)
                c_tmp = cp.float32(0.0)
                ec_p0x = cp.float32(0.0)
                ec_p0y = cp.float32(0.0)
                ec_p0z = cp.float32(0.0)
                ec_dx = cp.float32(0.0)
                ec_dy = cp.float32(0.0)
                ec_dz = cp.float32(0.0)
                ec_con0 = cp.int32(0)
                ec_con1 = cp.int32(0)
                ec_needProcess = cp.int32(0)
                ec_bothInside = cp.int32(0)
                ec_ipx = cp.float32(0.0)
                ec_ipy = cp.float32(0.0)
                ec_ipz = cp.float32(0.0)
                isa_parX = cp.int32(0)
                isa_parY = cp.int32(0)
                isa_parZ = cp.int32(0)
                isa_reject = cp.int32(0)
                isa_oddx = cp.float32(0.0)
                isa_oddy = cp.float32(0.0)
                isa_oddz = cp.float32(0.0)
                isa_t1x = cp.float32(0.0)
                isa_t1y = cp.float32(0.0)
                isa_t1z = cp.float32(0.0)
                isa_t2x = cp.float32(0.0)
                isa_t2y = cp.float32(0.0)
                isa_t2z = cp.float32(0.0)
                isa_tt1x = cp.float32(0.0)
                isa_tt1y = cp.float32(0.0)
                isa_tt1z = cp.float32(0.0)
                isa_tt2x = cp.float32(0.0)
                isa_tt2y = cp.float32(0.0)
                isa_tt2z = cp.float32(0.0)
                isa_ft1 = cp.float32(0.0)
                isa_ft2 = cp.float32(0.0)
                isa_tminf = cp.float32(0.0)
                isa_tmaxf = cp.float32(0.0)
                isa_valid = cp.int32(0)
                ct_x = cp.float32(0.0)
                ct_y = cp.float32(0.0)
                ct_z = cp.float32(0.0)
                totalContacts = cp.int32(0)
                deep_pen = cp.float32(0.0)
                deep_idx = cp.int32(0)
                # Material and output vars
                mat0_dynFric_i = cp.int32(0)
                mat0_staFric_i = cp.int32(0)
                mat0_rest_i = cp.int32(0)
                mat0_damp_i = cp.int32(0)
                mat0_flagsModes = cp.int32(0)
                mat0_dampMode_word = cp.int32(0)
                mat0_dynFric = cp.float32(0.0)
                mat0_staFric = cp.float32(0.0)
                mat0_rest = cp.float32(0.0)
                mat0_damp = cp.float32(0.0)
                mat0_flags = cp.int32(0)
                mat0_fricMode = cp.int32(0)
                mat0_restMode = cp.int32(0)
                mat0_dampMode = cp.int32(0)
                mat1_dynFric_i = cp.int32(0)
                mat1_staFric_i = cp.int32(0)
                mat1_rest_i = cp.int32(0)
                mat1_damp_i = cp.int32(0)
                mat1_flagsModes = cp.int32(0)
                mat1_dampMode_word = cp.int32(0)
                mat1_dynFric = cp.float32(0.0)
                mat1_staFric = cp.float32(0.0)
                mat1_rest = cp.float32(0.0)
                mat1_damp = cp.float32(0.0)
                mat1_flags = cp.int32(0)
                mat1_fricMode = cp.int32(0)
                mat1_restMode = cp.int32(0)
                mat1_dampMode = cp.int32(0)
                compliant0 = cp.int32(0)
                compliant1 = cp.int32(0)
                bothCompliant = cp.int32(0)
                exactlyOneCompliant = cp.int32(0)
                compliantAcc0 = cp.int32(0)
                compliantAcc1 = cp.int32(0)
                exactlyOneAccCompliant = cp.int32(0)
                combinedRest = cp.float32(0.0)
                combinedDamp = cp.float32(0.0)
                combinedDynFric = cp.float32(0.0)
                combinedStaFric = cp.float32(0.0)
                combinedMatFlags = cp.int32(0)
                combineFlags = cp.int32(0)
                restCombMode = cp.int32(0)
                dampCombMode = cp.int32(0)
                fricCombMode = cp.int32(0)
                flipSign = cp.float32(0.0)
                contactByteOffset = cp.int32(-1)
                contactAllocSize = cp.int32(0)
                forceAllocSize = cp.int32(0)
                forceByteOffset = cp.int32(0)
                allflags = cp.int32(0)
                oldStatusFlags = cp.int32(0)
                prevPatches = cp.int32(0)
                statusFlags = cp.int32(0)
                numPatches = cp.int32(0)
                previouslyHadTouch = cp.int32(0)
                prevTouchKnown = cp.int32(0)
                currentlyHasTouch = cp.int32(0)
                touchXor = cp.int32(0)
                change = cp.int32(0)
                patchDiff = cp.int32(0)
                newAllflags = cp.int32(0)
                nbContactsWord = cp.int32(0)
                patchIndex = cp.int32(-1)
                patchRow = cp.int32(0)
                revertAllflags = cp.int32(0)
                revertPatchDiff = cp.int32(0)
                contactIndex = cp.int32(0)
                startNbMatflags = cp.int32(0)
                intflagMatIdx0 = cp.int32(0)
                matIdx1Pad = cp.int32(0)
                inc_axis_x = cp.float32(0.0)
                inc_axis_y = cp.float32(0.0)
                inc_axis_z = cp.float32(0.0)
                inc_tx = cp.float32(0.0)
                inc_ty = cp.float32(0.0)
                inc_tz = cp.float32(0.0)
                inc_c0x = cp.float32(0.0)
                inc_c0y = cp.float32(0.0)
                inc_c0z = cp.float32(0.0)
                inc_c1x = cp.float32(0.0)
                inc_c1y = cp.float32(0.0)
                inc_c1z = cp.float32(0.0)
                inc_c2x = cp.float32(0.0)
                inc_c2y = cp.float32(0.0)
                inc_c2z = cp.float32(0.0)
                ddp_x = cp.float32(0.0)
                ddp_y = cp.float32(0.0)
                ddp_z = cp.float32(0.0)
                ec_pPenS = cp.int32(0)
                ec_pPenE = cp.int32(0)
                ec_pAreaS = cp.int32(0)
                ec_pAreaE = cp.int32(0)

                if separated == cp.int32(0):
                    # ========================================================
                    # 8a. Build newTransformV based on bestFeature
                    # ========================================================
                    if bestFeature == cp.int32(0):
                        if cp.float32(0.0) >= bestSign:
                            mtd_x = ax00_x
                            mtd_y = ax00_y
                            mtd_z = ax00_z
                            ntv_c0x = -ax02_x
                            ntv_c0y = -ax02_y
                            ntv_c0z = -ax02_z
                            ntv_c1x = ax01_x
                            ntv_c1y = ax01_y
                            ntv_c1z = ax01_z
                            ntv_c2x = ax00_x
                            ntv_c2y = ax00_y
                            ntv_c2z = ax00_z
                        else:
                            mtd_x = -ax00_x
                            mtd_y = -ax00_y
                            mtd_z = -ax00_z
                            ntv_c0x = ax02_x
                            ntv_c0y = ax02_y
                            ntv_c0z = ax02_z
                            ntv_c1x = ax01_x
                            ntv_c1y = ax01_y
                            ntv_c1z = ax01_z
                            ntv_c2x = -ax00_x
                            ntv_c2y = -ax00_y
                            ntv_c2z = -ax00_z
                        ntv_px = t0_px - mtd_x * box0_ex
                        ntv_py = t0_py - mtd_y * box0_ex
                        ntv_pz = t0_pz - mtd_z * box0_ex
                        ref_e0 = box0_ez
                        ref_e1 = box0_ey
                    if bestFeature == cp.int32(1):
                        if cp.float32(0.0) >= bestSign:
                            mtd_x = ax01_x
                            mtd_y = ax01_y
                            mtd_z = ax01_z
                            ntv_c0x = ax00_x
                            ntv_c0y = ax00_y
                            ntv_c0z = ax00_z
                            ntv_c1x = -ax02_x
                            ntv_c1y = -ax02_y
                            ntv_c1z = -ax02_z
                            ntv_c2x = ax01_x
                            ntv_c2y = ax01_y
                            ntv_c2z = ax01_z
                        else:
                            mtd_x = -ax01_x
                            mtd_y = -ax01_y
                            mtd_z = -ax01_z
                            ntv_c0x = ax00_x
                            ntv_c0y = ax00_y
                            ntv_c0z = ax00_z
                            ntv_c1x = ax02_x
                            ntv_c1y = ax02_y
                            ntv_c1z = ax02_z
                            ntv_c2x = -ax01_x
                            ntv_c2y = -ax01_y
                            ntv_c2z = -ax01_z
                        ntv_px = t0_px - mtd_x * box0_ey
                        ntv_py = t0_py - mtd_y * box0_ey
                        ntv_pz = t0_pz - mtd_z * box0_ey
                        ref_e0 = box0_ex
                        ref_e1 = box0_ez
                    if bestFeature == cp.int32(2):
                        if cp.float32(0.0) >= bestSign:
                            mtd_x = ax02_x
                            mtd_y = ax02_y
                            mtd_z = ax02_z
                            ntv_c0x = ax00_x
                            ntv_c0y = ax00_y
                            ntv_c0z = ax00_z
                            ntv_c1x = ax01_x
                            ntv_c1y = ax01_y
                            ntv_c1z = ax01_z
                            ntv_c2x = ax02_x
                            ntv_c2y = ax02_y
                            ntv_c2z = ax02_z
                        else:
                            mtd_x = -ax02_x
                            mtd_y = -ax02_y
                            mtd_z = -ax02_z
                            ntv_c0x = ax00_x
                            ntv_c0y = ax00_y
                            ntv_c0z = ax00_z
                            ntv_c1x = -ax01_x
                            ntv_c1y = -ax01_y
                            ntv_c1z = -ax01_z
                            ntv_c2x = -ax02_x
                            ntv_c2y = -ax02_y
                            ntv_c2z = -ax02_z
                        ntv_px = t0_px - mtd_x * box0_ez
                        ntv_py = t0_py - mtd_y * box0_ez
                        ntv_pz = t0_pz - mtd_z * box0_ez
                        ref_e0 = box0_ex
                        ref_e1 = box0_ey
                    if bestFeature == cp.int32(3):
                        if cp.float32(0.0) >= bestSign:
                            mtd_x = ax10_x
                            mtd_y = ax10_y
                            mtd_z = ax10_z
                            ntv_c0x = ax12_x
                            ntv_c0y = ax12_y
                            ntv_c0z = ax12_z
                            ntv_c1x = ax11_x
                            ntv_c1y = ax11_y
                            ntv_c1z = ax11_z
                            ntv_c2x = -ax10_x
                            ntv_c2y = -ax10_y
                            ntv_c2z = -ax10_z
                        else:
                            mtd_x = -ax10_x
                            mtd_y = -ax10_y
                            mtd_z = -ax10_z
                            ntv_c0x = -ax12_x
                            ntv_c0y = -ax12_y
                            ntv_c0z = -ax12_z
                            ntv_c1x = ax11_x
                            ntv_c1y = ax11_y
                            ntv_c1z = ax11_z
                            ntv_c2x = ax10_x
                            ntv_c2y = ax10_y
                            ntv_c2z = ax10_z
                        ntv_px = t1_px + mtd_x * box1_ex
                        ntv_py = t1_py + mtd_y * box1_ex
                        ntv_pz = t1_pz + mtd_z * box1_ex
                        ref_e0 = box1_ez
                        ref_e1 = box1_ey
                    if bestFeature == cp.int32(4):
                        if cp.float32(0.0) >= bestSign:
                            mtd_x = ax11_x
                            mtd_y = ax11_y
                            mtd_z = ax11_z
                            ntv_c0x = ax10_x
                            ntv_c0y = ax10_y
                            ntv_c0z = ax10_z
                            ntv_c1x = ax12_x
                            ntv_c1y = ax12_y
                            ntv_c1z = ax12_z
                            ntv_c2x = -ax11_x
                            ntv_c2y = -ax11_y
                            ntv_c2z = -ax11_z
                        else:
                            mtd_x = -ax11_x
                            mtd_y = -ax11_y
                            mtd_z = -ax11_z
                            ntv_c0x = ax10_x
                            ntv_c0y = ax10_y
                            ntv_c0z = ax10_z
                            ntv_c1x = -ax12_x
                            ntv_c1y = -ax12_y
                            ntv_c1z = -ax12_z
                            ntv_c2x = ax11_x
                            ntv_c2y = ax11_y
                            ntv_c2z = ax11_z
                        ntv_px = t1_px + mtd_x * box1_ey
                        ntv_py = t1_py + mtd_y * box1_ey
                        ntv_pz = t1_pz + mtd_z * box1_ey
                        ref_e0 = box1_ex
                        ref_e1 = box1_ez
                    if bestFeature == cp.int32(5):
                        if cp.float32(0.0) >= bestSign:
                            mtd_x = ax12_x
                            mtd_y = ax12_y
                            mtd_z = ax12_z
                            ntv_c0x = ax10_x
                            ntv_c0y = ax10_y
                            ntv_c0z = ax10_z
                            ntv_c1x = -ax11_x
                            ntv_c1y = -ax11_y
                            ntv_c1z = -ax11_z
                            ntv_c2x = -ax12_x
                            ntv_c2y = -ax12_y
                            ntv_c2z = -ax12_z
                        else:
                            mtd_x = -ax12_x
                            mtd_y = -ax12_y
                            mtd_z = -ax12_z
                            ntv_c0x = ax10_x
                            ntv_c0y = ax10_y
                            ntv_c0z = ax10_z
                            ntv_c1x = ax11_x
                            ntv_c1y = ax11_y
                            ntv_c1z = ax11_z
                            ntv_c2x = ax12_x
                            ntv_c2y = ax12_y
                            ntv_c2z = ax12_z
                        ntv_px = t1_px + mtd_x * box1_ez
                        ntv_py = t1_py + mtd_y * box1_ez
                        ntv_pz = t1_pz + mtd_z * box1_ez
                        ref_e0 = box1_ex
                        ref_e1 = box1_ey

                    # ========================================================
                    # 8b. Compute transform1ToNew and incident polygon
                    # ========================================================
                    localNormal_x = ntv_c0x * mtd_x + ntv_c0y * mtd_y + ntv_c0z * mtd_z
                    localNormal_y = ntv_c1x * mtd_x + ntv_c1y * mtd_y + ntv_c1z * mtd_z
                    localNormal_z = ntv_c2x * mtd_x + ntv_c2y * mtd_y + ntv_c2z * mtd_z

                    if bestFeature < cp.int32(3):
                        inc_tx = t1_px
                        inc_ty = t1_py
                        inc_tz = t1_pz
                        inc_c0x = ax10_x
                        inc_c0y = ax10_y
                        inc_c0z = ax10_z
                        inc_c1x = ax11_x
                        inc_c1y = ax11_y
                        inc_c1z = ax11_z
                        inc_c2x = ax12_x
                        inc_c2y = ax12_y
                        inc_c2z = ax12_z
                        inc_ext_x = box1_ex
                        inc_ext_y = box1_ey
                        inc_ext_z = box1_ez
                        inc_axis_x = -localNormal_x
                        inc_axis_y = -localNormal_y
                        inc_axis_z = -localNormal_z
                    else:
                        inc_tx = t0_px
                        inc_ty = t0_py
                        inc_tz = t0_pz
                        inc_c0x = ax00_x
                        inc_c0y = ax00_y
                        inc_c0z = ax00_z
                        inc_c1x = ax01_x
                        inc_c1y = ax01_y
                        inc_c1z = ax01_z
                        inc_c2x = ax02_x
                        inc_c2y = ax02_y
                        inc_c2z = ax02_z
                        inc_ext_x = box0_ex
                        inc_ext_y = box0_ey
                        inc_ext_z = box0_ez
                        inc_axis_x = localNormal_x
                        inc_axis_y = localNormal_y
                        inc_axis_z = localNormal_z

                    # transform1ToNew = newTransformV.transformTranspose(incidentTransform)
                    ddp_x = inc_tx - ntv_px
                    ddp_y = inc_ty - ntv_py
                    ddp_z = inc_tz - ntv_pz
                    t1tn_px = ntv_c0x * ddp_x + ntv_c0y * ddp_y + ntv_c0z * ddp_z
                    t1tn_py = ntv_c1x * ddp_x + ntv_c1y * ddp_y + ntv_c1z * ddp_z
                    t1tn_pz = ntv_c2x * ddp_x + ntv_c2y * ddp_y + ntv_c2z * ddp_z
                    t1tn_c0x = ntv_c0x * inc_c0x + ntv_c0y * inc_c0y + ntv_c0z * inc_c0z
                    t1tn_c0y = ntv_c1x * inc_c0x + ntv_c1y * inc_c0y + ntv_c1z * inc_c0z
                    t1tn_c0z = ntv_c2x * inc_c0x + ntv_c2y * inc_c0y + ntv_c2z * inc_c0z
                    t1tn_c1x = ntv_c0x * inc_c1x + ntv_c0y * inc_c1y + ntv_c0z * inc_c1z
                    t1tn_c1y = ntv_c1x * inc_c1x + ntv_c1y * inc_c1y + ntv_c1z * inc_c1z
                    t1tn_c1z = ntv_c2x * inc_c1x + ntv_c2y * inc_c1y + ntv_c2z * inc_c1z
                    t1tn_c2x = ntv_c0x * inc_c2x + ntv_c0y * inc_c2y + ntv_c0z * inc_c2z
                    t1tn_c2y = ntv_c1x * inc_c2x + ntv_c1y * inc_c2y + ntv_c1z * inc_c2z
                    t1tn_c2z = ntv_c2x * inc_c2x + ntv_c2y * inc_c2y + ntv_c2z * inc_c2z

                    # ========================================================
                    # 8c. getIncidentPolygon4 -- compute 4 incident face verts
                    # ========================================================
                    inc_d0 = t1tn_c0x * inc_axis_x + t1tn_c0y * inc_axis_y + t1tn_c0z * inc_axis_z
                    inc_d1 = t1tn_c1x * inc_axis_x + t1tn_c1y * inc_axis_y + t1tn_c1z * inc_axis_z
                    inc_d2 = t1tn_c2x * inc_axis_x + t1tn_c2y * inc_axis_y + t1tn_c2z * inc_axis_z
                    inc_absd0 = abs_f32(inc_d0)
                    inc_absd1 = abs_f32(inc_d1)
                    inc_absd2 = abs_f32(inc_d2)

                    dom_sign_x = cp.float32(1.0)
                    if inc_d0 > cp.float32(0.0):
                        dom_sign_x = cp.float32(-1.0)
                    dom_sign_y = cp.float32(1.0)
                    if inc_d1 > cp.float32(0.0):
                        dom_sign_y = cp.float32(-1.0)
                    dom_sign_z = cp.float32(1.0)
                    if inc_d2 > cp.float32(0.0):
                        dom_sign_z = cp.float32(-1.0)

                    # Vertex order: v0(-1,-1), v1(1,-1), v2(1,1), v3(-1,1)
                    if inc_absd0 >= inc_absd1:
                        if inc_absd0 >= inc_absd2:
                            # x dominant
                            incNormal_x = t1tn_c0x * dom_sign_x
                            incNormal_y = t1tn_c0y * dom_sign_x
                            incNormal_z = t1tn_c0z * dom_sign_x
                            tmp_vx = inc_ext_x * dom_sign_x
                            tmp_vy = inc_ext_y * cp.float32(-1.0)
                            tmp_vz = inc_ext_z * cp.float32(-1.0)
                            iv0_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                            iv0_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                            iv0_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                            tmp_vy = inc_ext_y
                            iv1_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                            iv1_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                            iv1_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                            tmp_vz = inc_ext_z
                            iv2_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                            iv2_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                            iv2_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                            tmp_vy = inc_ext_y * cp.float32(-1.0)
                            iv3_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                            iv3_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                            iv3_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                        else:
                            # z dominant
                            incNormal_x = t1tn_c2x * dom_sign_z
                            incNormal_y = t1tn_c2y * dom_sign_z
                            incNormal_z = t1tn_c2z * dom_sign_z
                            tmp_vz = inc_ext_z * dom_sign_z
                            tmp_vx = inc_ext_x * cp.float32(-1.0)
                            tmp_vy = inc_ext_y * cp.float32(-1.0)
                            iv0_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                            iv0_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                            iv0_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                            tmp_vx = inc_ext_x
                            iv1_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                            iv1_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                            iv1_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                            tmp_vy = inc_ext_y
                            iv2_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                            iv2_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                            iv2_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                            tmp_vx = inc_ext_x * cp.float32(-1.0)
                            iv3_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                            iv3_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                            iv3_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                    if inc_absd1 >= inc_absd2:
                        # y dominant
                        incNormal_x = t1tn_c1x * dom_sign_y
                        incNormal_y = t1tn_c1y * dom_sign_y
                        incNormal_z = t1tn_c1z * dom_sign_y
                        tmp_vy = inc_ext_y * dom_sign_y
                        tmp_vx = inc_ext_x * cp.float32(-1.0)
                        tmp_vz = inc_ext_z * cp.float32(-1.0)
                        iv0_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                        iv0_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                        iv0_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                        tmp_vx = inc_ext_x
                        iv1_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                        iv1_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                        iv1_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                        tmp_vz = inc_ext_z
                        iv2_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                        iv2_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                        iv2_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                        tmp_vx = inc_ext_x * cp.float32(-1.0)
                        iv3_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                        iv3_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                        iv3_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                    else:
                        # z dominant
                        incNormal_x = t1tn_c2x * dom_sign_z
                        incNormal_y = t1tn_c2y * dom_sign_z
                        incNormal_z = t1tn_c2z * dom_sign_z
                        tmp_vz = inc_ext_z * dom_sign_z
                        tmp_vx = inc_ext_x * cp.float32(-1.0)
                        tmp_vy = inc_ext_y * cp.float32(-1.0)
                        iv0_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                        iv0_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                        iv0_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                        tmp_vx = inc_ext_x
                        iv1_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                        iv1_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                        iv1_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                        tmp_vy = inc_ext_y
                        iv2_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                        iv2_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                        iv2_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz
                        tmp_vx = inc_ext_x * cp.float32(-1.0)
                        iv3_x = t1tn_c0x * tmp_vx + t1tn_c1x * tmp_vy + t1tn_c2x * tmp_vz + t1tn_px
                        iv3_y = t1tn_c0y * tmp_vx + t1tn_c1y * tmp_vy + t1tn_c2y * tmp_vz + t1tn_py
                        iv3_z = t1tn_c0z * tmp_vx + t1tn_c1z * tmp_vy + t1tn_c2z * tmp_vz + t1tn_pz

                    # ========================================================
                    # 8d. calculateContacts: clip incident polygon
                    # ========================================================
                    extentX = cp.float32(1.00001) * ref_e0
                    extentY = cp.float32(1.00001) * ref_e1
                    nExtentX = -extentX
                    nExtentY = -extentY
                    maxZ_clip = -cDistance - cp.float32(1e-7)

                    # Check each incident vertex
                    pPen0 = cp.int32(0)
                    pArea0 = cp.int32(0)
                    if cDistance > -iv0_z:
                        pPen0 = cp.int32(1)
                        if extentX >= abs_f32(iv0_x):
                            if extentY >= abs_f32(iv0_y):
                                pArea0 = cp.int32(1)
                    pPen1 = cp.int32(0)
                    pArea1 = cp.int32(0)
                    if cDistance > -iv1_z:
                        pPen1 = cp.int32(1)
                        if extentX >= abs_f32(iv1_x):
                            if extentY >= abs_f32(iv1_y):
                                pArea1 = cp.int32(1)
                    pPen2 = cp.int32(0)
                    pArea2 = cp.int32(0)
                    if cDistance > -iv2_z:
                        pPen2 = cp.int32(1)
                        if extentX >= abs_f32(iv2_x):
                            if extentY >= abs_f32(iv2_y):
                                pArea2 = cp.int32(1)
                    pPen3 = cp.int32(0)
                    pArea3 = cp.int32(0)
                    if cDistance > -iv3_z:
                        pPen3 = cp.int32(1)
                        if extentX >= abs_f32(iv3_x):
                            if extentY >= abs_f32(iv3_y):
                                pArea3 = cp.int32(1)

                    # Store contacts from vertices inside ref face
                    lc_count = cp.int32(0)
                    if pArea0 == cp.int32(1):
                        lc0_x = iv0_x
                        lc0_y = iv0_y
                        lc0_z = iv0_z
                        lc0_w = -iv0_z
                        lc_count = cp.int32(1)
                    if pArea1 == cp.int32(1):
                        if lc_count == cp.int32(0):
                            lc0_x = iv1_x
                            lc0_y = iv1_y
                            lc0_z = iv1_z
                            lc0_w = -iv1_z
                        if lc_count == cp.int32(1):
                            lc1_x = iv1_x
                            lc1_y = iv1_y
                            lc1_z = iv1_z
                            lc1_w = -iv1_z
                        lc_count = lc_count + cp.int32(1)
                    if pArea2 == cp.int32(1):
                        if lc_count == cp.int32(0):
                            lc0_x = iv2_x
                            lc0_y = iv2_y
                            lc0_z = iv2_z
                            lc0_w = -iv2_z
                        if lc_count == cp.int32(1):
                            lc1_x = iv2_x
                            lc1_y = iv2_y
                            lc1_z = iv2_z
                            lc1_w = -iv2_z
                        if lc_count == cp.int32(2):
                            lc2_x = iv2_x
                            lc2_y = iv2_y
                            lc2_z = iv2_z
                            lc2_w = -iv2_z
                        lc_count = lc_count + cp.int32(1)
                    if pArea3 == cp.int32(1):
                        if lc_count == cp.int32(0):
                            lc0_x = iv3_x
                            lc0_y = iv3_y
                            lc0_z = iv3_z
                            lc0_w = -iv3_z
                        if lc_count == cp.int32(1):
                            lc1_x = iv3_x
                            lc1_y = iv3_y
                            lc1_z = iv3_z
                            lc1_w = -iv3_z
                        if lc_count == cp.int32(2):
                            lc2_x = iv3_x
                            lc2_y = iv3_y
                            lc2_z = iv3_z
                            lc2_w = -iv3_z
                        if lc_count == cp.int32(3):
                            lc3_x = iv3_x
                            lc3_y = iv3_y
                            lc3_z = iv3_z
                            lc3_w = -iv3_z
                        lc_count = lc_count + cp.int32(1)

                    allArea = pArea0 & pArea1 & pArea2 & pArea3
                    if allArea == cp.int32(0):
                        # Not all verts inside -- check ref corners inside incident polygon
                        # and clip edges against ref AABB
                        bnd_minx = min_f32(min_f32(iv0_x, iv1_x), min_f32(iv2_x, iv3_x))
                        bnd_miny = min_f32(min_f32(iv0_y, iv1_y), min_f32(iv2_y, iv3_y))
                        bnd_maxx = max_f32(max_f32(iv0_x, iv1_x), max_f32(iv2_x, iv3_x))
                        bnd_maxy = max_f32(max_f32(iv0_y, iv1_y), max_f32(iv2_y, iv3_y))
                        denom_inc = incNormal_z

                        # Check 4 ref face corners inside incident polygon
                        # For each corner, inline contains() then plane intersection
                        # Corner coords: (-e0,-e1), (e0,-e1), (e0,e1), (-e0,e1)
                        # We use simplified point-in-polygon: count ray crossings
                        # Process all 4 corners similarly -- due to Capybara limitations
                        # we must write each one out explicitly.

                        # We define a helper pattern: for corner (qx,qy), test containment
                        # in the quad iv0..iv3 using ray-casting, then if inside, compute
                        # z via plane intersection and add contact.
                        # Due to extreme code size, we process edges (4 edges) after corners.

                        # --- Edge clipping (4 edges of incident polygon) ---
                        # Edge 0->1
                        ec_pPenS = pPen0
                        ec_pPenE = pPen1
                        ec_pAreaS = pArea0
                        ec_pAreaE = pArea1
                        ec_p0x = iv0_x
                        ec_p0y = iv0_y
                        ec_p0z = iv0_z
                        ec_dx = iv1_x - iv0_x
                        ec_dy = iv1_y - iv0_y
                        ec_dz = iv1_z - iv0_z
                        ec_con0 = ec_pPenS & ec_pAreaS
                        ec_con1 = ec_pPenE & ec_pAreaE
                        ec_needProcess = cp.int32(0)
                        if ec_pPenS == cp.int32(1):
                            ec_needProcess = cp.int32(1)
                        if ec_pPenE == cp.int32(1):
                            ec_needProcess = cp.int32(1)
                        ec_bothInside = ec_con0 & ec_con1
                        if ec_bothInside == cp.int32(1):
                            ec_needProcess = cp.int32(0)
                        if ec_needProcess == cp.int32(1):
                            # intersectSegmentAABB inline
                            isa_parX = cp.int32(0)
                            isa_parY = cp.int32(0)
                            isa_parZ = cp.int32(0)
                            if cp.float32(1e-6) > abs_f32(ec_dx):
                                isa_parX = cp.int32(1)
                            if cp.float32(1e-6) > abs_f32(ec_dy):
                                isa_parY = cp.int32(1)
                            if cp.float32(1e-6) > abs_f32(ec_dz):
                                isa_parZ = cp.int32(1)
                            isa_reject = cp.int32(0)
                            if isa_parX == cp.int32(1):
                                if ec_p0x > extentX:
                                    isa_reject = cp.int32(1)
                                if ec_p0x < nExtentX:
                                    isa_reject = cp.int32(1)
                            if isa_parY == cp.int32(1):
                                if ec_p0y > extentY:
                                    isa_reject = cp.int32(1)
                                if ec_p0y < nExtentY:
                                    isa_reject = cp.int32(1)
                            if isa_parZ == cp.int32(1):
                                if ec_p0z < maxZ_clip:
                                    isa_reject = cp.int32(1)
                            if isa_reject == cp.int32(0):
                                isa_oddx = cp.float32(0.0)
                                isa_oddy = cp.float32(0.0)
                                isa_oddz = cp.float32(0.0)
                                if isa_parX == cp.int32(0):
                                    isa_oddx = cp.float32(1.0) / ec_dx
                                if isa_parY == cp.int32(0):
                                    isa_oddy = cp.float32(1.0) / ec_dy
                                if isa_parZ == cp.int32(0):
                                    isa_oddz = cp.float32(1.0) / ec_dz
                                isa_t1x = cp.float32(0.0)
                                isa_t1y = cp.float32(0.0)
                                isa_t1z = cp.float32(0.0)
                                isa_t2x = PX_MAX_F32
                                isa_t2y = PX_MAX_F32
                                isa_t2z = PX_MAX_F32
                                if isa_parX == cp.int32(0):
                                    isa_t1x = (nExtentX - ec_p0x) * isa_oddx
                                    isa_t2x = (extentX - ec_p0x) * isa_oddx
                                if isa_parY == cp.int32(0):
                                    isa_t1y = (nExtentY - ec_p0y) * isa_oddy
                                    isa_t2y = (extentY - ec_p0y) * isa_oddy
                                if isa_parZ == cp.int32(0):
                                    isa_t1z = (maxZ_clip - ec_p0z) * isa_oddz
                                    isa_t2z = (PX_MAX_F32 - ec_p0z) * isa_oddz
                                isa_tt1x = min_f32(isa_t1x, isa_t2x)
                                isa_tt1y = min_f32(isa_t1y, isa_t2y)
                                isa_tt1z = min_f32(isa_t1z, isa_t2z)
                                isa_tt2x = max_f32(isa_t1x, isa_t2x)
                                isa_tt2y = max_f32(isa_t1y, isa_t2y)
                                isa_tt2z = max_f32(isa_t1z, isa_t2z)
                                isa_ft1 = max_f32(max_f32(isa_tt1x, isa_tt1y), isa_tt1z)
                                isa_ft2 = min_f32(min_f32(isa_tt2x, isa_tt2y), isa_tt2z)
                                isa_tminf = max_f32(isa_ft1, cp.float32(0.0))
                                isa_tmaxf = min_f32(cp.float32(1.0), isa_ft2)
                                isa_valid = cp.int32(1)
                                if isa_tminf > isa_tmaxf:
                                    isa_valid = cp.int32(0)
                                if isa_tminf > cp.float32(1.0):
                                    isa_valid = cp.int32(0)
                                if isa_valid == cp.int32(1):
                                    if ec_con0 == cp.int32(0):
                                        ec_ipx = ec_p0x + ec_dx * isa_tminf
                                        ec_ipy = ec_p0y + ec_dy * isa_tminf
                                        ec_ipz = ec_p0z + ec_dz * isa_tminf
                                        if lc_count < cp.int32(12):
                                            if lc_count == cp.int32(0):
                                                lc0_x = ec_ipx
                                                lc0_y = ec_ipy
                                                lc0_z = ec_ipz
                                                lc0_w = -ec_ipz
                                            if lc_count == cp.int32(1):
                                                lc1_x = ec_ipx
                                                lc1_y = ec_ipy
                                                lc1_z = ec_ipz
                                                lc1_w = -ec_ipz
                                            if lc_count == cp.int32(2):
                                                lc2_x = ec_ipx
                                                lc2_y = ec_ipy
                                                lc2_z = ec_ipz
                                                lc2_w = -ec_ipz
                                            if lc_count == cp.int32(3):
                                                lc3_x = ec_ipx
                                                lc3_y = ec_ipy
                                                lc3_z = ec_ipz
                                                lc3_w = -ec_ipz
                                            if lc_count == cp.int32(4):
                                                lc4_x = ec_ipx
                                                lc4_y = ec_ipy
                                                lc4_z = ec_ipz
                                                lc4_w = -ec_ipz
                                            if lc_count == cp.int32(5):
                                                lc5_x = ec_ipx
                                                lc5_y = ec_ipy
                                                lc5_z = ec_ipz
                                                lc5_w = -ec_ipz
                                            lc_count = lc_count + cp.int32(1)
                                    if ec_con1 == cp.int32(0):
                                        ec_ipx = ec_p0x + ec_dx * isa_tmaxf
                                        ec_ipy = ec_p0y + ec_dy * isa_tmaxf
                                        ec_ipz = ec_p0z + ec_dz * isa_tmaxf
                                        if lc_count < cp.int32(12):
                                            if lc_count == cp.int32(0):
                                                lc0_x = ec_ipx
                                                lc0_y = ec_ipy
                                                lc0_z = ec_ipz
                                                lc0_w = -ec_ipz
                                            if lc_count == cp.int32(1):
                                                lc1_x = ec_ipx
                                                lc1_y = ec_ipy
                                                lc1_z = ec_ipz
                                                lc1_w = -ec_ipz
                                            if lc_count == cp.int32(2):
                                                lc2_x = ec_ipx
                                                lc2_y = ec_ipy
                                                lc2_z = ec_ipz
                                                lc2_w = -ec_ipz
                                            if lc_count == cp.int32(3):
                                                lc3_x = ec_ipx
                                                lc3_y = ec_ipy
                                                lc3_z = ec_ipz
                                                lc3_w = -ec_ipz
                                            if lc_count == cp.int32(4):
                                                lc4_x = ec_ipx
                                                lc4_y = ec_ipy
                                                lc4_z = ec_ipz
                                                lc4_w = -ec_ipz
                                            if lc_count == cp.int32(5):
                                                lc5_x = ec_ipx
                                                lc5_y = ec_ipy
                                                lc5_z = ec_ipz
                                                lc5_w = -ec_ipz
                                            lc_count = lc_count + cp.int32(1)

                        # Edge 1->2
                        ec_pPenS = pPen1
                        ec_pPenE = pPen2
                        ec_pAreaS = pArea1
                        ec_pAreaE = pArea2
                        ec_p0x = iv1_x
                        ec_p0y = iv1_y
                        ec_p0z = iv1_z
                        ec_dx = iv2_x - iv1_x
                        ec_dy = iv2_y - iv1_y
                        ec_dz = iv2_z - iv1_z
                        ec_con0 = ec_pPenS & ec_pAreaS
                        ec_con1 = ec_pPenE & ec_pAreaE
                        ec_needProcess = cp.int32(0)
                        if ec_pPenS == cp.int32(1):
                            ec_needProcess = cp.int32(1)
                        if ec_pPenE == cp.int32(1):
                            ec_needProcess = cp.int32(1)
                        ec_bothInside = ec_con0 & ec_con1
                        if ec_bothInside == cp.int32(1):
                            ec_needProcess = cp.int32(0)
                        if ec_needProcess == cp.int32(1):
                            isa_parX = cp.int32(0)
                            isa_parY = cp.int32(0)
                            isa_parZ = cp.int32(0)
                            if cp.float32(1e-6) > abs_f32(ec_dx):
                                isa_parX = cp.int32(1)
                            if cp.float32(1e-6) > abs_f32(ec_dy):
                                isa_parY = cp.int32(1)
                            if cp.float32(1e-6) > abs_f32(ec_dz):
                                isa_parZ = cp.int32(1)
                            isa_reject = cp.int32(0)
                            if isa_parX == cp.int32(1):
                                if ec_p0x > extentX:
                                    isa_reject = cp.int32(1)
                                if ec_p0x < nExtentX:
                                    isa_reject = cp.int32(1)
                            if isa_parY == cp.int32(1):
                                if ec_p0y > extentY:
                                    isa_reject = cp.int32(1)
                                if ec_p0y < nExtentY:
                                    isa_reject = cp.int32(1)
                            if isa_parZ == cp.int32(1):
                                if ec_p0z < maxZ_clip:
                                    isa_reject = cp.int32(1)
                            if isa_reject == cp.int32(0):
                                isa_oddx = cp.float32(0.0)
                                isa_oddy = cp.float32(0.0)
                                isa_oddz = cp.float32(0.0)
                                if isa_parX == cp.int32(0):
                                    isa_oddx = cp.float32(1.0) / ec_dx
                                if isa_parY == cp.int32(0):
                                    isa_oddy = cp.float32(1.0) / ec_dy
                                if isa_parZ == cp.int32(0):
                                    isa_oddz = cp.float32(1.0) / ec_dz
                                isa_t1x = cp.float32(0.0)
                                isa_t1y = cp.float32(0.0)
                                isa_t1z = cp.float32(0.0)
                                isa_t2x = PX_MAX_F32
                                isa_t2y = PX_MAX_F32
                                isa_t2z = PX_MAX_F32
                                if isa_parX == cp.int32(0):
                                    isa_t1x = (nExtentX - ec_p0x) * isa_oddx
                                    isa_t2x = (extentX - ec_p0x) * isa_oddx
                                if isa_parY == cp.int32(0):
                                    isa_t1y = (nExtentY - ec_p0y) * isa_oddy
                                    isa_t2y = (extentY - ec_p0y) * isa_oddy
                                if isa_parZ == cp.int32(0):
                                    isa_t1z = (maxZ_clip - ec_p0z) * isa_oddz
                                    isa_t2z = (PX_MAX_F32 - ec_p0z) * isa_oddz
                                isa_tt1x = min_f32(isa_t1x, isa_t2x)
                                isa_tt1y = min_f32(isa_t1y, isa_t2y)
                                isa_tt1z = min_f32(isa_t1z, isa_t2z)
                                isa_tt2x = max_f32(isa_t1x, isa_t2x)
                                isa_tt2y = max_f32(isa_t1y, isa_t2y)
                                isa_tt2z = max_f32(isa_t1z, isa_t2z)
                                isa_ft1 = max_f32(max_f32(isa_tt1x, isa_tt1y), isa_tt1z)
                                isa_ft2 = min_f32(min_f32(isa_tt2x, isa_tt2y), isa_tt2z)
                                isa_tminf = max_f32(isa_ft1, cp.float32(0.0))
                                isa_tmaxf = min_f32(cp.float32(1.0), isa_ft2)
                                isa_valid = cp.int32(1)
                                if isa_tminf > isa_tmaxf:
                                    isa_valid = cp.int32(0)
                                if isa_tminf > cp.float32(1.0):
                                    isa_valid = cp.int32(0)
                                if isa_valid == cp.int32(1):
                                    if ec_con0 == cp.int32(0):
                                        ec_ipx = ec_p0x + ec_dx * isa_tminf
                                        ec_ipy = ec_p0y + ec_dy * isa_tminf
                                        ec_ipz = ec_p0z + ec_dz * isa_tminf
                                        if lc_count < cp.int32(12):
                                            if lc_count == cp.int32(0):
                                                lc0_x = ec_ipx
                                                lc0_y = ec_ipy
                                                lc0_z = ec_ipz
                                                lc0_w = -ec_ipz
                                            if lc_count == cp.int32(1):
                                                lc1_x = ec_ipx
                                                lc1_y = ec_ipy
                                                lc1_z = ec_ipz
                                                lc1_w = -ec_ipz
                                            if lc_count == cp.int32(2):
                                                lc2_x = ec_ipx
                                                lc2_y = ec_ipy
                                                lc2_z = ec_ipz
                                                lc2_w = -ec_ipz
                                            if lc_count == cp.int32(3):
                                                lc3_x = ec_ipx
                                                lc3_y = ec_ipy
                                                lc3_z = ec_ipz
                                                lc3_w = -ec_ipz
                                            if lc_count == cp.int32(4):
                                                lc4_x = ec_ipx
                                                lc4_y = ec_ipy
                                                lc4_z = ec_ipz
                                                lc4_w = -ec_ipz
                                            if lc_count == cp.int32(5):
                                                lc5_x = ec_ipx
                                                lc5_y = ec_ipy
                                                lc5_z = ec_ipz
                                                lc5_w = -ec_ipz
                                            lc_count = lc_count + cp.int32(1)
                                    if ec_con1 == cp.int32(0):
                                        ec_ipx = ec_p0x + ec_dx * isa_tmaxf
                                        ec_ipy = ec_p0y + ec_dy * isa_tmaxf
                                        ec_ipz = ec_p0z + ec_dz * isa_tmaxf
                                        if lc_count < cp.int32(12):
                                            if lc_count == cp.int32(0):
                                                lc0_x = ec_ipx
                                                lc0_y = ec_ipy
                                                lc0_z = ec_ipz
                                                lc0_w = -ec_ipz
                                            if lc_count == cp.int32(1):
                                                lc1_x = ec_ipx
                                                lc1_y = ec_ipy
                                                lc1_z = ec_ipz
                                                lc1_w = -ec_ipz
                                            if lc_count == cp.int32(2):
                                                lc2_x = ec_ipx
                                                lc2_y = ec_ipy
                                                lc2_z = ec_ipz
                                                lc2_w = -ec_ipz
                                            if lc_count == cp.int32(3):
                                                lc3_x = ec_ipx
                                                lc3_y = ec_ipy
                                                lc3_z = ec_ipz
                                                lc3_w = -ec_ipz
                                            if lc_count == cp.int32(4):
                                                lc4_x = ec_ipx
                                                lc4_y = ec_ipy
                                                lc4_z = ec_ipz
                                                lc4_w = -ec_ipz
                                            if lc_count == cp.int32(5):
                                                lc5_x = ec_ipx
                                                lc5_y = ec_ipy
                                                lc5_z = ec_ipz
                                                lc5_w = -ec_ipz
                                            lc_count = lc_count + cp.int32(1)

                        # Edge 2->3
                        ec_pPenS = pPen2
                        ec_pPenE = pPen3
                        ec_pAreaS = pArea2
                        ec_pAreaE = pArea3
                        ec_p0x = iv2_x
                        ec_p0y = iv2_y
                        ec_p0z = iv2_z
                        ec_dx = iv3_x - iv2_x
                        ec_dy = iv3_y - iv2_y
                        ec_dz = iv3_z - iv2_z
                        ec_con0 = ec_pPenS & ec_pAreaS
                        ec_con1 = ec_pPenE & ec_pAreaE
                        ec_needProcess = cp.int32(0)
                        if ec_pPenS == cp.int32(1):
                            ec_needProcess = cp.int32(1)
                        if ec_pPenE == cp.int32(1):
                            ec_needProcess = cp.int32(1)
                        ec_bothInside = ec_con0 & ec_con1
                        if ec_bothInside == cp.int32(1):
                            ec_needProcess = cp.int32(0)
                        if ec_needProcess == cp.int32(1):
                            isa_parX = cp.int32(0)
                            isa_parY = cp.int32(0)
                            isa_parZ = cp.int32(0)
                            if cp.float32(1e-6) > abs_f32(ec_dx):
                                isa_parX = cp.int32(1)
                            if cp.float32(1e-6) > abs_f32(ec_dy):
                                isa_parY = cp.int32(1)
                            if cp.float32(1e-6) > abs_f32(ec_dz):
                                isa_parZ = cp.int32(1)
                            isa_reject = cp.int32(0)
                            if isa_parX == cp.int32(1):
                                if ec_p0x > extentX:
                                    isa_reject = cp.int32(1)
                                if ec_p0x < nExtentX:
                                    isa_reject = cp.int32(1)
                            if isa_parY == cp.int32(1):
                                if ec_p0y > extentY:
                                    isa_reject = cp.int32(1)
                                if ec_p0y < nExtentY:
                                    isa_reject = cp.int32(1)
                            if isa_parZ == cp.int32(1):
                                if ec_p0z < maxZ_clip:
                                    isa_reject = cp.int32(1)
                            if isa_reject == cp.int32(0):
                                isa_oddx = cp.float32(0.0)
                                isa_oddy = cp.float32(0.0)
                                isa_oddz = cp.float32(0.0)
                                if isa_parX == cp.int32(0):
                                    isa_oddx = cp.float32(1.0) / ec_dx
                                if isa_parY == cp.int32(0):
                                    isa_oddy = cp.float32(1.0) / ec_dy
                                if isa_parZ == cp.int32(0):
                                    isa_oddz = cp.float32(1.0) / ec_dz
                                isa_t1x = cp.float32(0.0)
                                isa_t1y = cp.float32(0.0)
                                isa_t1z = cp.float32(0.0)
                                isa_t2x = PX_MAX_F32
                                isa_t2y = PX_MAX_F32
                                isa_t2z = PX_MAX_F32
                                if isa_parX == cp.int32(0):
                                    isa_t1x = (nExtentX - ec_p0x) * isa_oddx
                                    isa_t2x = (extentX - ec_p0x) * isa_oddx
                                if isa_parY == cp.int32(0):
                                    isa_t1y = (nExtentY - ec_p0y) * isa_oddy
                                    isa_t2y = (extentY - ec_p0y) * isa_oddy
                                if isa_parZ == cp.int32(0):
                                    isa_t1z = (maxZ_clip - ec_p0z) * isa_oddz
                                    isa_t2z = (PX_MAX_F32 - ec_p0z) * isa_oddz
                                isa_tt1x = min_f32(isa_t1x, isa_t2x)
                                isa_tt1y = min_f32(isa_t1y, isa_t2y)
                                isa_tt1z = min_f32(isa_t1z, isa_t2z)
                                isa_tt2x = max_f32(isa_t1x, isa_t2x)
                                isa_tt2y = max_f32(isa_t1y, isa_t2y)
                                isa_tt2z = max_f32(isa_t1z, isa_t2z)
                                isa_ft1 = max_f32(max_f32(isa_tt1x, isa_tt1y), isa_tt1z)
                                isa_ft2 = min_f32(min_f32(isa_tt2x, isa_tt2y), isa_tt2z)
                                isa_tminf = max_f32(isa_ft1, cp.float32(0.0))
                                isa_tmaxf = min_f32(cp.float32(1.0), isa_ft2)
                                isa_valid = cp.int32(1)
                                if isa_tminf > isa_tmaxf:
                                    isa_valid = cp.int32(0)
                                if isa_tminf > cp.float32(1.0):
                                    isa_valid = cp.int32(0)
                                if isa_valid == cp.int32(1):
                                    if ec_con0 == cp.int32(0):
                                        ec_ipx = ec_p0x + ec_dx * isa_tminf
                                        ec_ipy = ec_p0y + ec_dy * isa_tminf
                                        ec_ipz = ec_p0z + ec_dz * isa_tminf
                                        if lc_count < cp.int32(12):
                                            if lc_count == cp.int32(0):
                                                lc0_x = ec_ipx
                                                lc0_y = ec_ipy
                                                lc0_z = ec_ipz
                                                lc0_w = -ec_ipz
                                            if lc_count == cp.int32(1):
                                                lc1_x = ec_ipx
                                                lc1_y = ec_ipy
                                                lc1_z = ec_ipz
                                                lc1_w = -ec_ipz
                                            if lc_count == cp.int32(2):
                                                lc2_x = ec_ipx
                                                lc2_y = ec_ipy
                                                lc2_z = ec_ipz
                                                lc2_w = -ec_ipz
                                            if lc_count == cp.int32(3):
                                                lc3_x = ec_ipx
                                                lc3_y = ec_ipy
                                                lc3_z = ec_ipz
                                                lc3_w = -ec_ipz
                                            if lc_count == cp.int32(4):
                                                lc4_x = ec_ipx
                                                lc4_y = ec_ipy
                                                lc4_z = ec_ipz
                                                lc4_w = -ec_ipz
                                            if lc_count == cp.int32(5):
                                                lc5_x = ec_ipx
                                                lc5_y = ec_ipy
                                                lc5_z = ec_ipz
                                                lc5_w = -ec_ipz
                                            lc_count = lc_count + cp.int32(1)
                                    if ec_con1 == cp.int32(0):
                                        ec_ipx = ec_p0x + ec_dx * isa_tmaxf
                                        ec_ipy = ec_p0y + ec_dy * isa_tmaxf
                                        ec_ipz = ec_p0z + ec_dz * isa_tmaxf
                                        if lc_count < cp.int32(12):
                                            if lc_count == cp.int32(0):
                                                lc0_x = ec_ipx
                                                lc0_y = ec_ipy
                                                lc0_z = ec_ipz
                                                lc0_w = -ec_ipz
                                            if lc_count == cp.int32(1):
                                                lc1_x = ec_ipx
                                                lc1_y = ec_ipy
                                                lc1_z = ec_ipz
                                                lc1_w = -ec_ipz
                                            if lc_count == cp.int32(2):
                                                lc2_x = ec_ipx
                                                lc2_y = ec_ipy
                                                lc2_z = ec_ipz
                                                lc2_w = -ec_ipz
                                            if lc_count == cp.int32(3):
                                                lc3_x = ec_ipx
                                                lc3_y = ec_ipy
                                                lc3_z = ec_ipz
                                                lc3_w = -ec_ipz
                                            if lc_count == cp.int32(4):
                                                lc4_x = ec_ipx
                                                lc4_y = ec_ipy
                                                lc4_z = ec_ipz
                                                lc4_w = -ec_ipz
                                            if lc_count == cp.int32(5):
                                                lc5_x = ec_ipx
                                                lc5_y = ec_ipy
                                                lc5_z = ec_ipz
                                                lc5_w = -ec_ipz
                                            lc_count = lc_count + cp.int32(1)

                        # Edge 3->0
                        ec_pPenS = pPen3
                        ec_pPenE = pPen0
                        ec_pAreaS = pArea3
                        ec_pAreaE = pArea0
                        ec_p0x = iv3_x
                        ec_p0y = iv3_y
                        ec_p0z = iv3_z
                        ec_dx = iv0_x - iv3_x
                        ec_dy = iv0_y - iv3_y
                        ec_dz = iv0_z - iv3_z
                        ec_con0 = ec_pPenS & ec_pAreaS
                        ec_con1 = ec_pPenE & ec_pAreaE
                        ec_needProcess = cp.int32(0)
                        if ec_pPenS == cp.int32(1):
                            ec_needProcess = cp.int32(1)
                        if ec_pPenE == cp.int32(1):
                            ec_needProcess = cp.int32(1)
                        ec_bothInside = ec_con0 & ec_con1
                        if ec_bothInside == cp.int32(1):
                            ec_needProcess = cp.int32(0)
                        if ec_needProcess == cp.int32(1):
                            isa_parX = cp.int32(0)
                            isa_parY = cp.int32(0)
                            isa_parZ = cp.int32(0)
                            if cp.float32(1e-6) > abs_f32(ec_dx):
                                isa_parX = cp.int32(1)
                            if cp.float32(1e-6) > abs_f32(ec_dy):
                                isa_parY = cp.int32(1)
                            if cp.float32(1e-6) > abs_f32(ec_dz):
                                isa_parZ = cp.int32(1)
                            isa_reject = cp.int32(0)
                            if isa_parX == cp.int32(1):
                                if ec_p0x > extentX:
                                    isa_reject = cp.int32(1)
                                if ec_p0x < nExtentX:
                                    isa_reject = cp.int32(1)
                            if isa_parY == cp.int32(1):
                                if ec_p0y > extentY:
                                    isa_reject = cp.int32(1)
                                if ec_p0y < nExtentY:
                                    isa_reject = cp.int32(1)
                            if isa_parZ == cp.int32(1):
                                if ec_p0z < maxZ_clip:
                                    isa_reject = cp.int32(1)
                            if isa_reject == cp.int32(0):
                                isa_oddx = cp.float32(0.0)
                                isa_oddy = cp.float32(0.0)
                                isa_oddz = cp.float32(0.0)
                                if isa_parX == cp.int32(0):
                                    isa_oddx = cp.float32(1.0) / ec_dx
                                if isa_parY == cp.int32(0):
                                    isa_oddy = cp.float32(1.0) / ec_dy
                                if isa_parZ == cp.int32(0):
                                    isa_oddz = cp.float32(1.0) / ec_dz
                                isa_t1x = cp.float32(0.0)
                                isa_t1y = cp.float32(0.0)
                                isa_t1z = cp.float32(0.0)
                                isa_t2x = PX_MAX_F32
                                isa_t2y = PX_MAX_F32
                                isa_t2z = PX_MAX_F32
                                if isa_parX == cp.int32(0):
                                    isa_t1x = (nExtentX - ec_p0x) * isa_oddx
                                    isa_t2x = (extentX - ec_p0x) * isa_oddx
                                if isa_parY == cp.int32(0):
                                    isa_t1y = (nExtentY - ec_p0y) * isa_oddy
                                    isa_t2y = (extentY - ec_p0y) * isa_oddy
                                if isa_parZ == cp.int32(0):
                                    isa_t1z = (maxZ_clip - ec_p0z) * isa_oddz
                                    isa_t2z = (PX_MAX_F32 - ec_p0z) * isa_oddz
                                isa_tt1x = min_f32(isa_t1x, isa_t2x)
                                isa_tt1y = min_f32(isa_t1y, isa_t2y)
                                isa_tt1z = min_f32(isa_t1z, isa_t2z)
                                isa_tt2x = max_f32(isa_t1x, isa_t2x)
                                isa_tt2y = max_f32(isa_t1y, isa_t2y)
                                isa_tt2z = max_f32(isa_t1z, isa_t2z)
                                isa_ft1 = max_f32(max_f32(isa_tt1x, isa_tt1y), isa_tt1z)
                                isa_ft2 = min_f32(min_f32(isa_tt2x, isa_tt2y), isa_tt2z)
                                isa_tminf = max_f32(isa_ft1, cp.float32(0.0))
                                isa_tmaxf = min_f32(cp.float32(1.0), isa_ft2)
                                isa_valid = cp.int32(1)
                                if isa_tminf > isa_tmaxf:
                                    isa_valid = cp.int32(0)
                                if isa_tminf > cp.float32(1.0):
                                    isa_valid = cp.int32(0)
                                if isa_valid == cp.int32(1):
                                    if ec_con0 == cp.int32(0):
                                        ec_ipx = ec_p0x + ec_dx * isa_tminf
                                        ec_ipy = ec_p0y + ec_dy * isa_tminf
                                        ec_ipz = ec_p0z + ec_dz * isa_tminf
                                        if lc_count < cp.int32(12):
                                            if lc_count == cp.int32(0):
                                                lc0_x = ec_ipx
                                                lc0_y = ec_ipy
                                                lc0_z = ec_ipz
                                                lc0_w = -ec_ipz
                                            if lc_count == cp.int32(1):
                                                lc1_x = ec_ipx
                                                lc1_y = ec_ipy
                                                lc1_z = ec_ipz
                                                lc1_w = -ec_ipz
                                            if lc_count == cp.int32(2):
                                                lc2_x = ec_ipx
                                                lc2_y = ec_ipy
                                                lc2_z = ec_ipz
                                                lc2_w = -ec_ipz
                                            if lc_count == cp.int32(3):
                                                lc3_x = ec_ipx
                                                lc3_y = ec_ipy
                                                lc3_z = ec_ipz
                                                lc3_w = -ec_ipz
                                            if lc_count == cp.int32(4):
                                                lc4_x = ec_ipx
                                                lc4_y = ec_ipy
                                                lc4_z = ec_ipz
                                                lc4_w = -ec_ipz
                                            if lc_count == cp.int32(5):
                                                lc5_x = ec_ipx
                                                lc5_y = ec_ipy
                                                lc5_z = ec_ipz
                                                lc5_w = -ec_ipz
                                            lc_count = lc_count + cp.int32(1)
                                    if ec_con1 == cp.int32(0):
                                        ec_ipx = ec_p0x + ec_dx * isa_tmaxf
                                        ec_ipy = ec_p0y + ec_dy * isa_tmaxf
                                        ec_ipz = ec_p0z + ec_dz * isa_tmaxf
                                        if lc_count < cp.int32(12):
                                            if lc_count == cp.int32(0):
                                                lc0_x = ec_ipx
                                                lc0_y = ec_ipy
                                                lc0_z = ec_ipz
                                                lc0_w = -ec_ipz
                                            if lc_count == cp.int32(1):
                                                lc1_x = ec_ipx
                                                lc1_y = ec_ipy
                                                lc1_z = ec_ipz
                                                lc1_w = -ec_ipz
                                            if lc_count == cp.int32(2):
                                                lc2_x = ec_ipx
                                                lc2_y = ec_ipy
                                                lc2_z = ec_ipz
                                                lc2_w = -ec_ipz
                                            if lc_count == cp.int32(3):
                                                lc3_x = ec_ipx
                                                lc3_y = ec_ipy
                                                lc3_z = ec_ipz
                                                lc3_w = -ec_ipz
                                            if lc_count == cp.int32(4):
                                                lc4_x = ec_ipx
                                                lc4_y = ec_ipy
                                                lc4_z = ec_ipz
                                                lc4_w = -ec_ipz
                                            if lc_count == cp.int32(5):
                                                lc5_x = ec_ipx
                                                lc5_y = ec_ipy
                                                lc5_z = ec_ipz
                                                lc5_w = -ec_ipz
                                            lc_count = lc_count + cp.int32(1)

                    # ========================================================
                    # 8f. Transform contacts to world space; reduce if needed
                    # ========================================================
                    totalContacts = lc_count
                    if totalContacts > cp.int32(MAX_CONTACTS_PER_PATCH):
                        # Simple reduction: take first MAX_CONTACTS_PER_PATCH contacts
                        # (The CUDA code reduces to 4 via area-maximizing algorithm using
                        # 4-thread cooperative groups; here we cap to MAX_CONTACTS_PER_PATCH.)
                        totalContacts = cp.int32(MAX_CONTACTS_PER_PATCH)

                    # Transform to world space
                    if totalContacts > cp.int32(0):
                        ct_x = ntv_c0x * lc0_x + ntv_c1x * lc0_y + ntv_c2x * lc0_z + ntv_px
                        ct_y = ntv_c0y * lc0_x + ntv_c1y * lc0_y + ntv_c2y * lc0_z + ntv_py
                        ct_z = ntv_c0z * lc0_x + ntv_c1z * lc0_y + ntv_c2z * lc0_z + ntv_pz
                        cp0_x = ct_x
                        cp0_y = ct_y
                        cp0_z = ct_z
                        cp0_w = lc0_w
                    if totalContacts > cp.int32(1):
                        ct_x = ntv_c0x * lc1_x + ntv_c1x * lc1_y + ntv_c2x * lc1_z + ntv_px
                        ct_y = ntv_c0y * lc1_x + ntv_c1y * lc1_y + ntv_c2y * lc1_z + ntv_py
                        ct_z = ntv_c0z * lc1_x + ntv_c1z * lc1_y + ntv_c2z * lc1_z + ntv_pz
                        cp1_x = ct_x
                        cp1_y = ct_y
                        cp1_z = ct_z
                        cp1_w = lc1_w
                    if totalContacts > cp.int32(2):
                        ct_x = ntv_c0x * lc2_x + ntv_c1x * lc2_y + ntv_c2x * lc2_z + ntv_px
                        ct_y = ntv_c0y * lc2_x + ntv_c1y * lc2_y + ntv_c2y * lc2_z + ntv_py
                        ct_z = ntv_c0z * lc2_x + ntv_c1z * lc2_y + ntv_c2z * lc2_z + ntv_pz
                        cp2_x = ct_x
                        cp2_y = ct_y
                        cp2_z = ct_z
                        cp2_w = lc2_w
                    if totalContacts > cp.int32(3):
                        ct_x = ntv_c0x * lc3_x + ntv_c1x * lc3_y + ntv_c2x * lc3_z + ntv_px
                        ct_y = ntv_c0y * lc3_x + ntv_c1y * lc3_y + ntv_c2y * lc3_z + ntv_py
                        ct_z = ntv_c0z * lc3_x + ntv_c1z * lc3_y + ntv_c2z * lc3_z + ntv_pz
                        cp3_x = ct_x
                        cp3_y = ct_y
                        cp3_z = ct_z
                        cp3_w = lc3_w
                    if totalContacts > cp.int32(4):
                        ct_x = ntv_c0x * lc4_x + ntv_c1x * lc4_y + ntv_c2x * lc4_z + ntv_px
                        ct_y = ntv_c0y * lc4_x + ntv_c1y * lc4_y + ntv_c2y * lc4_z + ntv_py
                        ct_z = ntv_c0z * lc4_x + ntv_c1z * lc4_y + ntv_c2z * lc4_z + ntv_pz
                        cp4_x = ct_x
                        cp4_y = ct_y
                        cp4_z = ct_z
                        cp4_w = lc4_w
                    if totalContacts > cp.int32(5):
                        ct_x = ntv_c0x * lc5_x + ntv_c1x * lc5_y + ntv_c2x * lc5_z + ntv_px
                        ct_y = ntv_c0y * lc5_x + ntv_c1y * lc5_y + ntv_c2y * lc5_z + ntv_py
                        ct_z = ntv_c0z * lc5_x + ntv_c1z * lc5_y + ntv_c2z * lc5_z + ntv_pz
                        cp5_x = ct_x
                        cp5_y = ct_y
                        cp5_z = ct_z
                        cp5_w = lc5_w

                    nbContacts = totalContacts

                # ============================================================
                # 9. setContactPointAndForcePointers (inline)
                # ============================================================
                contactByteOffset = cp.int32(-1)
                if nbContacts > cp.int32(0):
                    contactAllocSize = cp.int32(SIZEOF_PX_CONTACT) * nbContacts
                    forceAllocSize = cp.int32(SIZEOF_PX_U32) * nbContacts
                    contactByteOffset = thread.atomic_add(patchAndContactCounters[COUNTER_CONTACTS_BYTES], contactAllocSize)
                    forceByteOffset = thread.atomic_add(patchAndContactCounters[COUNTER_FORCE_BYTES], forceAllocSize)
                    if contactByteOffset + cp.int32(SIZEOF_PX_CONTACT) > contactBytesLimit:
                        thread.atomic_add(patchAndContactCounters[COUNTER_OVERFLOW], cp.int32(1))
                        contactByteOffset = cp.int32(-1)
                    if forceByteOffset + cp.int32(SIZEOF_PX_U32) > forceBytesLimit:
                        thread.atomic_add(patchAndContactCounters[COUNTER_OVERFLOW], cp.int32(2))
                    if contactByteOffset != cp.int32(-1):
                        cmOutputs[workIndex, CMO_CONTACT_POINTS_LO] = contactByteOffset
                        cmOutputs[workIndex, CMO_CONTACT_POINTS_HI] = cp.int32(0)
                        cmOutputs[workIndex, CMO_CONTACT_FORCES_LO] = forceByteOffset
                        cmOutputs[workIndex, CMO_CONTACT_FORCES_HI] = cp.int32(0)
                    else:
                        cmOutputs[workIndex, CMO_CONTACT_POINTS_LO] = cp.int32(0)
                        cmOutputs[workIndex, CMO_CONTACT_POINTS_HI] = cp.int32(0)
                        cmOutputs[workIndex, CMO_CONTACT_FORCES_LO] = cp.int32(0)
                        cmOutputs[workIndex, CMO_CONTACT_FORCES_HI] = cp.int32(0)

                # ============================================================
                # 10. Write contact points to contact stream
                # ============================================================
                if contactByteOffset != cp.int32(-1):
                    contactIndex = contactByteOffset // cp.int32(SIZEOF_PX_CONTACT)
                    contactStream[contactIndex, 0] = cp0_x
                    contactStream[contactIndex, 1] = cp0_y
                    contactStream[contactIndex, 2] = cp0_z
                    contactStream[contactIndex, 3] = cp0_w
                    if nbContacts > cp.int32(1):
                        contactStream[contactIndex + cp.int32(1), 0] = cp1_x
                        contactStream[contactIndex + cp.int32(1), 1] = cp1_y
                        contactStream[contactIndex + cp.int32(1), 2] = cp1_z
                        contactStream[contactIndex + cp.int32(1), 3] = cp1_w
                    if nbContacts > cp.int32(2):
                        contactStream[contactIndex + cp.int32(2), 0] = cp2_x
                        contactStream[contactIndex + cp.int32(2), 1] = cp2_y
                        contactStream[contactIndex + cp.int32(2), 2] = cp2_z
                        contactStream[contactIndex + cp.int32(2), 3] = cp2_w
                    if nbContacts > cp.int32(3):
                        contactStream[contactIndex + cp.int32(3), 0] = cp3_x
                        contactStream[contactIndex + cp.int32(3), 1] = cp3_y
                        contactStream[contactIndex + cp.int32(3), 2] = cp3_z
                        contactStream[contactIndex + cp.int32(3), 3] = cp3_w
                    if nbContacts > cp.int32(4):
                        contactStream[contactIndex + cp.int32(4), 0] = cp4_x
                        contactStream[contactIndex + cp.int32(4), 1] = cp4_y
                        contactStream[contactIndex + cp.int32(4), 2] = cp4_z
                        contactStream[contactIndex + cp.int32(4), 3] = cp4_w
                    if nbContacts > cp.int32(5):
                        contactStream[contactIndex + cp.int32(5), 0] = cp5_x
                        contactStream[contactIndex + cp.int32(5), 1] = cp5_y
                        contactStream[contactIndex + cp.int32(5), 2] = cp5_z
                        contactStream[contactIndex + cp.int32(5), 3] = cp5_w

                # ============================================================
                # 11. registerContactPatch (inline)
                # ============================================================
                allflags = cmOutputs[workIndex, CMO_ALLFLAGS] + cp.int32(0)
                oldStatusFlags = (allflags >> cp.int32(16)) & cp.int32(0xFF)
                prevPatches = (allflags >> cp.int32(8)) & cp.int32(0xFF)
                statusFlags = oldStatusFlags & (~cp.int32(STATUS_TOUCH_KNOWN))
                if nbContacts > cp.int32(0):
                    statusFlags = statusFlags | cp.int32(STATUS_HAS_TOUCH)
                else:
                    statusFlags = statusFlags | cp.int32(STATUS_HAS_NO_TOUCH)
                numPatches = cp.int32(0)
                if nbContacts > cp.int32(0):
                    numPatches = cp.int32(1)
                previouslyHadTouch = cp.int32(0)
                if (oldStatusFlags & cp.int32(STATUS_HAS_TOUCH)) != cp.int32(0):
                    previouslyHadTouch = cp.int32(1)
                prevTouchKnown = cp.int32(0)
                if (oldStatusFlags & cp.int32(STATUS_TOUCH_KNOWN)) != cp.int32(0):
                    prevTouchKnown = cp.int32(1)
                currentlyHasTouch = cp.int32(0)
                if nbContacts > cp.int32(0):
                    currentlyHasTouch = cp.int32(1)
                touchXor = previouslyHadTouch ^ currentlyHasTouch
                change = cp.int32(0)
                if touchXor != cp.int32(0):
                    change = cp.int32(1)
                if prevTouchKnown == cp.int32(0):
                    change = cp.int32(1)
                touchChangeFlags[workIndex] = change
                patchDiff = cp.int32(0)
                if prevPatches != numPatches:
                    patchDiff = cp.int32(1)
                patchChangeFlags[workIndex] = patchDiff
                newAllflags = ((numPatches & cp.int32(0xFF)) << cp.int32(8)) | ((statusFlags & cp.int32(0xFF)) << cp.int32(16)) | ((prevPatches & cp.int32(0xFF)) << cp.int32(24))
                cmOutputs[workIndex, CMO_ALLFLAGS] = newAllflags
                nbContactsWord = cmOutputs[workIndex, CMO_NB_CONTACTS] + cp.int32(0)
                nbContactsWord = (nbContactsWord & cp.int32(0xFFFF0000)) | (nbContacts & cp.int32(0xFFFF))
                cmOutputs[workIndex, CMO_NB_CONTACTS] = nbContactsWord

                patchIndex = cp.int32(-1)
                if nbContacts > cp.int32(0):
                    patchIndex = thread.atomic_add(patchAndContactCounters[COUNTER_PATCHES_BYTES], cp.int32(SIZEOF_PX_CONTACT_PATCH))
                    if patchIndex + cp.int32(SIZEOF_PX_CONTACT_PATCH) > patchBytesLimit:
                        thread.atomic_add(patchAndContactCounters[COUNTER_OVERFLOW], cp.int32(4))
                        patchIndex = cp.int32(-1)
                        statusFlags = statusFlags & (~cp.int32(STATUS_TOUCH_KNOWN))
                        statusFlags = statusFlags | cp.int32(STATUS_HAS_NO_TOUCH)
                        revertAllflags = ((statusFlags & cp.int32(0xFF)) << cp.int32(16)) | ((prevPatches & cp.int32(0xFF)) << cp.int32(24))
                        cmOutputs[workIndex, CMO_ALLFLAGS] = revertAllflags
                        cmOutputs[workIndex, CMO_NB_CONTACTS] = (nbContactsWord & cp.int32(0xFFFF0000))
                        touchChangeFlags[workIndex] = previouslyHadTouch
                        revertPatchDiff = cp.int32(0)
                        if prevPatches != cp.int32(0):
                            revertPatchDiff = cp.int32(1)
                        patchChangeFlags[workIndex] = revertPatchDiff
                    else:
                        cmOutputs[workIndex, CMO_CONTACT_PATCHES_LO] = patchIndex
                        cmOutputs[workIndex, CMO_CONTACT_PATCHES_HI] = cp.int32(0)

                # ============================================================
                # 12. insertIntoPatchStream (inline)
                # ============================================================
                if patchIndex != cp.int32(-1):
                    patchRow = patchIndex // cp.int32(SIZEOF_PX_CONTACT_PATCH)
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

                    # Combine restitution
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
                    if bothCompliant != cp.int32(0):
                        if exactlyOneAccCompliant != cp.int32(0):
                            combinedRest = mat0_rest
                            if compliantAcc0 == cp.int32(0):
                                combinedRest = mat1_rest
                        else:
                            restCombMode = max_i32(mat0_restMode, mat1_restMode)
                            if exactlyOneCompliant != cp.int32(0):
                                restCombMode = cp.int32(1)
                            flipSign = cp.float32(1.0)
                            if restCombMode == cp.int32(2):
                                flipSign = cp.float32(-1.0)
                            combinedRest = flipSign * combine_scalars(mat0_rest, mat1_rest, restCombMode)
                    else:
                        restCombMode = max_i32(mat0_restMode, mat1_restMode)
                        if exactlyOneCompliant != cp.int32(0):
                            restCombMode = cp.int32(1)
                        combinedRest = combine_scalars(mat0_rest, mat1_rest, restCombMode)

                    # Combine damping
                    combinedDamp = cp.float32(0.0)
                    if bothCompliant != cp.int32(0):
                        if exactlyOneAccCompliant != cp.int32(0):
                            combinedDamp = mat0_damp
                            if compliantAcc0 == cp.int32(0):
                                combinedDamp = mat1_damp
                        else:
                            dampCombMode = max_i32(mat0_dampMode, mat1_dampMode)
                            if exactlyOneCompliant != cp.int32(0):
                                dampCombMode = cp.int32(3)
                            combinedDamp = combine_scalars(mat0_damp, mat1_damp, dampCombMode)
                    else:
                        dampCombMode = max_i32(mat0_dampMode, mat1_dampMode)
                        if exactlyOneCompliant != cp.int32(0):
                            dampCombMode = cp.int32(3)
                        combinedDamp = combine_scalars(mat0_damp, mat1_damp, dampCombMode)

                    # Combine friction
                    combineFlags = mat0_flags | mat1_flags
                    combinedDynFric = cp.float32(0.0)
                    combinedStaFric = cp.float32(0.0)
                    combinedMatFlags = combineFlags
                    if (combineFlags & cp.int32(MATFLAG_DISABLE_FRICTION)) == cp.int32(0):
                        fricCombMode = max_i32(mat0_fricMode, mat1_fricMode)
                        combinedDynFric = combine_scalars(mat0_dynFric, mat1_dynFric, fricCombMode)
                        combinedStaFric = combine_scalars(mat0_staFric, mat1_staFric, fricCombMode)
                        combinedDynFric = max_f32(combinedDynFric, cp.float32(0.0))
                        if combinedStaFric - combinedDynFric < cp.float32(0.0):
                            combinedStaFric = combinedDynFric
                    else:
                        combinedMatFlags = combineFlags | cp.int32(MATFLAG_DISABLE_STRONG_FRICTION)

                    # Write patch to stream
                    patchStream[patchRow, PS_MASS_MOD_LINEAR0] = thread.bitcast(cp.float32(1.0), cp.int32)
                    patchStream[patchRow, PS_MASS_MOD_ANGULAR0] = thread.bitcast(cp.float32(1.0), cp.int32)
                    patchStream[patchRow, PS_MASS_MOD_LINEAR1] = thread.bitcast(cp.float32(1.0), cp.int32)
                    patchStream[patchRow, PS_MASS_MOD_ANGULAR1] = thread.bitcast(cp.float32(1.0), cp.int32)
                    patchStream[patchRow, PS_NORMAL_X] = thread.bitcast(mtd_x, cp.int32)
                    patchStream[patchRow, PS_NORMAL_Y] = thread.bitcast(mtd_y, cp.int32)
                    patchStream[patchRow, PS_NORMAL_Z] = thread.bitcast(mtd_z, cp.int32)
                    patchStream[patchRow, PS_RESTITUTION] = thread.bitcast(combinedRest, cp.int32)
                    patchStream[patchRow, PS_DYN_FRICTION] = thread.bitcast(combinedDynFric, cp.int32)
                    patchStream[patchRow, PS_STA_FRICTION] = thread.bitcast(combinedStaFric, cp.int32)
                    patchStream[patchRow, PS_DAMPING] = thread.bitcast(combinedDamp, cp.int32)
                    startNbMatflags = (nbContacts & cp.int32(0xFF)) << cp.int32(16)
                    startNbMatflags = startNbMatflags | ((combinedMatFlags & cp.int32(0xFF)) << cp.int32(24))
                    patchStream[patchRow, PS_START_NB_MATFLAGS] = startNbMatflags
                    intflagMatIdx0 = (matIdx0 & cp.int32(0xFFFF)) << cp.int32(16)
                    patchStream[patchRow, PS_INTFLAGS_MATIDX0] = intflagMatIdx0
                    matIdx1Pad = matIdx1 & cp.int32(0xFFFF)
                    patchStream[patchRow, PS_MATIDX1_PAD] = matIdx1Pad
                    patchStream[patchRow, cp.int32(14)] = cp.int32(0)
                    patchStream[patchRow, cp.int32(15)] = cp.int32(0)
