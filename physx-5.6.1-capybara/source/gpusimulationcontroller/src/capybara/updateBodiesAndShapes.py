"""Capybara DSL port of updateBodiesAndShapes.cu -- all 20 kernels.

Ported kernels:
  1.  updateBodiesLaunch
  2.  updateBodiesLaunchDirectAPI
  3.  updateShapesLaunch
  4.  newArticulationsLaunch
  5.  updateArticulationsLaunch
  6.  updateBodyExternalVelocitiesLaunch
  7.  updateJointsLaunch
  8.  getRigidDynamicGlobalPose
  9.  getRigidDynamicLinearVelocity
  10. getRigidDynamicAngularVelocity
  11. getRigidDynamicLinearAcceleration
  12. getRigidDynamicAngularAcceleration
  13. setRigidDynamicGlobalPose
  14. setRigidDynamicLinearVelocity
  15. setRigidDynamicAngularVelocity
  16. setRigidDynamicForce
  17. setRigidDynamicTorque
  18. copyUserData
  19. getD6JointForces
  20. getD6JointTorques

ABI differences from CUDA:
  - All descriptor structs (PxgNewBodiesDesc, PxgUpdateActorDataDesc, etc.)
    decomposed: host resolves pointer members into flat tensor/scalar args.
  - PxgBodySim represented as int32[N, 60] tensor (240 bytes / 4 = 60 slots).
    Float fields loaded via thread.bitcast(val, cp.float32).
  - PxgShapeSim, PxgNewShapeSim decomposed to flat tensor/scalar fields.
  - PxsCachedTransform as float32[N, 8] (PxTransform=7 floats + flags=1 int).
  - PxBounds3 as float32[N, 6] (min xyz, max xyz).
  - PxgConstraintWriteback as float32[N, 8] (linearImpulse_broken float4 + angularImpulse_residual float4).
  - PxgConstraintIdMapEntry as int32[N] (mJointDataId, 0xFFFFFFFF = invalid).
  - Articulation kernels: all sub-arrays (links, joints, tendons, etc.) passed
    as flat int32 tensors with constexpr size parameters. warpCopy replaced by
    single-thread for/while copy loops.
  - PxgBodySimVelocities as float32[N, 8] (linearVelocity float4 + angularVelocity float4).
  - PxgBodySimVelocityUpdate as float32[N, 16] (4 float4s).
  - PxgPtrPair decomposed: src/dst as int32 tensors, size as scalar.
  - For setRigidDynamicGlobalPose: binary search + shape iteration decomposed
    to flat tensor operations on rigidNodeIndices, shapeIndices, shapeSimPool,
    transformCache, bounds, convexShapes, updated arrays.
"""

import capybara as cp

# ---------------------------------------------------------------------------
# PxgBodySim layout -- 240 bytes = 60 int32 slots (reused from integration.py)
# ---------------------------------------------------------------------------
BS_LIN_VEL_X, BS_LIN_VEL_Y, BS_LIN_VEL_Z, BS_INV_MASS_W = 0, 1, 2, 3
BS_ANG_VEL_X, BS_ANG_VEL_Y, BS_ANG_VEL_Z, BS_MAX_PEN_BIAS_W = 4, 5, 6, 7
BS_MAX_LIN_VEL_SQ, BS_MAX_ANG_VEL_SQ, BS_LIN_DAMP, BS_ANG_DAMP = 8, 9, 10, 11
BS_INV_INERTIA_X, BS_INV_INERTIA_Y, BS_INV_INERTIA_Z, BS_CONTACT_REPORT_THRESH = 12, 13, 14, 15
BS_SLEEP_LIN_X, BS_SLEEP_LIN_Y, BS_SLEEP_LIN_Z, BS_FREEZE_COUNT = 16, 17, 18, 19
BS_SLEEP_ANG_X, BS_SLEEP_ANG_Y, BS_SLEEP_ANG_Z, BS_ACCEL_SCALE = 20, 21, 22, 23
BS_FREEZE_THRESH, BS_WAKE_COUNTER, BS_SLEEP_THRESH, BS_BODY_SIM_IDX = 24, 25, 26, 27
BS_B2W_QX, BS_B2W_QY, BS_B2W_QZ, BS_B2W_QW = 28, 29, 30, 31
BS_B2W_PX, BS_B2W_PY, BS_B2W_PZ, BS_B2W_PW = 32, 33, 34, 35
BS_B2A_QX, BS_B2A_QY, BS_B2A_QZ, BS_B2A_QW = 36, 37, 38, 39
BS_B2A_PX, BS_B2A_PY, BS_B2A_PZ, BS_B2A_MAX_IMPULSE = 40, 41, 42, 43
BS_ARTIC_REMAP_ID = 44
BS_INTERNAL_FLAGS = 45
BS_LOCK_DISABLE_GRAVITY = 46
BS_OFFSET_SLOP = 47
# 48..51 padding
BS_EXT_LIN_ACC_X, BS_EXT_LIN_ACC_Y, BS_EXT_LIN_ACC_Z, BS_EXT_LIN_ACC_W = 52, 53, 54, 55
BS_EXT_ANG_ACC_X, BS_EXT_ANG_ACC_Y, BS_EXT_ANG_ACC_Z, BS_EXT_ANG_ACC_W = 56, 57, 58, 59

# PxgBodySim as uint4 has 15 elements (240/16=15); indices matching the C++ code
PXG_BODY_SIM_UINT4_SIZE = 15  # sizeof(PxgBodySim) / sizeof(uint4) = 240/16 = 15
PXG_BODY_SIM_MAX_LIN_VEL_IND = 2  # offset 32 / 16 = 2
PXG_BODY_SIM_BODYSIM_INDEX_IND = 6  # offset 96 / 16 = 6 (freezeThresh..bodySimIdx at byte 96)
PXG_BODY_SIM_BODY2WORLD_IND = 7  # offset 112 / 16 = 7
PXG_BODY_SIM_BODY2ACTOR_IND = 9  # offset 144 / 16 = 9
PXG_BODY_SIM_FLAGS_IND = 11  # offset 176 / 16 = 11
PXG_BODY_SIM_SIZE_WITHOUT_ACCELERATION = 13  # offset 208 / 16 = 13

# PxsRigidBody internal flags
eFIRST_BODY_COPY_GPU = 1 << 9
eVELOCITY_COPY_GPU = 1 << 10

# PxgShapeSim layout -- 60 bytes = 15 int32 slots
# PxTransform mTransform: q(xyzw) + p(xyz) = 7 floats [0..6]
# PxBounds3 mLocalBounds: min(xyz) + max(xyz) = 6 floats [7..12]
# PxNodeIndex mBodySimIndex: 2 uint32 [13] (mID) -- actually PxNodeIndex is 8 bytes but we use flat
# PxU32 mHullDataIndex [14 or depends on alignment]
# PxU16 mShapeFlags + PxU16 mShapeType packed
# For Capybara we use flat int32 tensor.
SS_XFORM_QX, SS_XFORM_QY, SS_XFORM_QZ, SS_XFORM_QW = 0, 1, 2, 3
SS_XFORM_PX, SS_XFORM_PY, SS_XFORM_PZ = 4, 5, 6
SS_LBOUNDS_MIN_X, SS_LBOUNDS_MIN_Y, SS_LBOUNDS_MIN_Z = 7, 8, 9
SS_LBOUNDS_MAX_X, SS_LBOUNDS_MAX_Y, SS_LBOUNDS_MAX_Z = 10, 11, 12
SS_BODY_SIM_INDEX = 13  # PxNodeIndex.mID
SS_HULL_DATA_INDEX = 14
SS_SHAPE_FLAGS_TYPE = 15  # packed: lower 16 = shapeFlags, upper 16 = shapeType

# PxgNewShapeSim = PxgShapeSim + mElementIndex at offset 16
NSS_ELEMENT_INDEX = 16

# PxsCachedTransform layout -- 32 bytes = 8 int32 slots (PxTransform=28 bytes + flags=4)
CT_QX, CT_QY, CT_QZ, CT_QW = 0, 1, 2, 3
CT_PX, CT_PY, CT_PZ = 4, 5, 6
CT_FLAGS = 7

# PxBounds3 layout -- 24 bytes = 6 float32 slots
BD_MIN_X, BD_MIN_Y, BD_MIN_Z = 0, 1, 2
BD_MAX_X, BD_MAX_Y, BD_MAX_Z = 3, 4, 5

# PxgConstraintWriteback layout -- 32 bytes = 8 float32 slots
CW_LIN_X, CW_LIN_Y, CW_LIN_Z, CW_LIN_W = 0, 1, 2, 3
CW_ANG_X, CW_ANG_Y, CW_ANG_Z, CW_ANG_W = 4, 5, 6, 7

# PxgBodySimVelocities layout -- 32 bytes = 8 float32 slots
BV_LIN_X, BV_LIN_Y, BV_LIN_Z, BV_LIN_W = 0, 1, 2, 3
BV_ANG_X, BV_ANG_Y, BV_ANG_Z, BV_ANG_W = 4, 5, 6, 7

# PxgBodySimVelocityUpdate layout -- 64 bytes = 16 float32 slots
VU_LIN_VEL_X, VU_LIN_VEL_Y, VU_LIN_VEL_Z, VU_BODY_INDEX_W = 0, 1, 2, 3
VU_ANG_VEL_X, VU_ANG_VEL_Y, VU_ANG_VEL_Z, VU_MAX_PEN_BIAS_W = 4, 5, 6, 7
VU_EXT_LIN_ACC_X, VU_EXT_LIN_ACC_Y, VU_EXT_LIN_ACC_Z, VU_EXT_LIN_ACC_W = 8, 9, 10, 11
VU_EXT_ANG_ACC_X, VU_EXT_ANG_ACC_Y, VU_EXT_ANG_ACC_Z, VU_EXT_ANG_ACC_W = 12, 13, 14, 15

# Articulation dirty flags (Dy::ArticulationDirtyFlag)
ARTI_DIRTY_JOINTS = 1 << 0
ARTI_DIRTY_POSITIONS = 1 << 1
ARTI_DIRTY_VELOCITIES = 1 << 2
ARTI_DIRTY_FORCES = 1 << 3
ARTI_DIRTY_ROOT_VELOCITIES = 1 << 5
ARTI_DIRTY_LINKS = 1 << 6
ARTI_IN_DIRTY_LIST = 1 << 7
ARTI_DIRTY_WAKECOUNTER = 1 << 8
ARTI_DIRTY_EXT_ACCEL = 1 << 9
ARTI_DIRTY_JOINT_TARGET_VEL = 1 << 12
ARTI_DIRTY_JOINT_TARGET_POS = 1 << 13
ARTI_DIRTY_SPATIAL_TENDON = 1 << 14
ARTI_DIRTY_SPATIAL_TENDON_ATTACHMENT = 1 << 15
ARTI_DIRTY_FIXED_TENDON = 1 << 16
ARTI_DIRTY_FIXED_TENDON_JOINT = 1 << 17
ARTI_DIRTY_MIMIC_JOINT = 1 << 18
ARTI_DIRTY_USER_FLAGS = 1 << 19

INVALID_ID = 0xFFFFFFFF

WARP_SIZE = 32

# Geometry types for updateBounds
GEO_SPHERE = 0
GEO_CAPSULE = 2
GEO_BOX = 3
GEO_CONVEXMESH = 5


# =====================================================================
# Inline helpers
# =====================================================================

@cp.inline
def quat_multiply(ax, ay, az, aw, bx, by, bz, bw):
    """Quaternion multiplication: a * b."""
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    rw = aw * bw - ax * bx - ay * by - az * bz
    return rx, ry, rz, rw


@cp.inline
def quat_rotate(qx, qy, qz, qw, vx, vy, vz):
    """Rotate vector v by quaternion q: q.rotate(v)."""
    # Cross product: c = q.xyz x v
    cx = qy * vz - qz * vy
    cy = qz * vx - qx * vz
    cz = qx * vy - qy * vx
    # result = v + 2*(qw*c + q.xyz x c)
    # q.xyz x c:
    c2x = qy * cz - qz * cy
    c2y = qz * cx - qx * cz
    c2z = qx * cy - qy * cx
    two = cp.float32(2.0)
    rx = vx + two * (qw * cx + c2x)
    ry = vy + two * (qw * cy + c2y)
    rz = vz + two * (qw * cz + c2z)
    return rx, ry, rz


@cp.inline
def transform_multiply_q(a_qx, a_qy, a_qz, a_qw, a_px, a_py, a_pz,
                          b_qx, b_qy, b_qz, b_qw, b_px, b_py, b_pz):
    """PxTransform multiply: a.transform(b) -> (q, p).
    result.q = a.q * b.q
    result.p = a.q.rotate(b.p) + a.p
    """
    rqx, rqy, rqz, rqw = quat_multiply(a_qx, a_qy, a_qz, a_qw,
                                         b_qx, b_qy, b_qz, b_qw)
    rpx, rpy, rpz = quat_rotate(a_qx, a_qy, a_qz, a_qw, b_px, b_py, b_pz)
    rpx = rpx + a_px
    rpy = rpy + a_py
    rpz = rpz + a_pz
    return rqx, rqy, rqz, rqw, rpx, rpy, rpz


@cp.inline
def transform_transform_inv(a_qx, a_qy, a_qz, a_qw, a_px, a_py, a_pz,
                             b_qx, b_qy, b_qz, b_qw, b_px, b_py, b_pz):
    """PxTransform a.transformInv(b): a^-1 * b.
    a_inv.q = conjugate(a.q)
    a_inv.p = a_inv.q.rotate(-a.p)
    result = a_inv.transform(b)
    """
    # a_inv.q = conjugate(a.q)
    inv_qx = -a_qx
    inv_qy = -a_qy
    inv_qz = -a_qz
    inv_qw = a_qw
    # a_inv.p = inv_q.rotate(-a.p)
    neg_px = -a_px
    neg_py = -a_py
    neg_pz = -a_pz
    inv_px, inv_py, inv_pz = quat_rotate(inv_qx, inv_qy, inv_qz, inv_qw,
                                          neg_px, neg_py, neg_pz)
    # result = a_inv.transform(b)
    rqx, rqy, rqz, rqw, rpx, rpy, rpz = transform_multiply_q(
        inv_qx, inv_qy, inv_qz, inv_qw, inv_px, inv_py, inv_pz,
        b_qx, b_qy, b_qz, b_qw, b_px, b_py, b_pz)
    return rqx, rqy, rqz, rqw, rpx, rpy, rpz


@cp.inline
def abs_f(x):
    r = x
    if r < cp.float32(0.0):
        r = -r
    return r


# =====================================================================
# Kernel 1: updateBodiesLaunch
# =====================================================================
# Host decomposes PxgNewBodiesDesc:
#   gBodySim: int32[totalNbBodies, BS_UINT4_SIZE*4] -- new body data as int32
#   gBodySimPool: int32[poolSize, BS_UINT4_SIZE*4] -- persistent pool as int32
#   totalNbBodies: int32
# We use uint4 indexing: each uint4 = 4 int32s.
# The cooperative 16-thread copy is replaced by single-thread full copy with
# shfl replaced by direct reads.

@cp.kernel
def updateBodiesLaunch(
    gBodySim,       # int32[totalNbBodies * UINT4_SIZE, 4] -- flat as uint4
    gBodySimPool,   # int32[poolSize * UINT4_SIZE, 4] -- flat as uint4
    totalNbBodies,
    BLOCK_SIZE: cp.constexpr = 256,
    UINT4_SIZE: cp.constexpr = 15
):
    """Copy new bodies to persistent pool. Each body is UINT4_SIZE uint4s.
    gBodySim: int32[totalNbBodies, UINT4_SIZE * 4] -- new body data.
    gBodySimPool: int32[poolSize, UINT4_SIZE * 4] -- persistent body pool.
    """
    numGroups = cp.ceildiv(totalNbBodies, BLOCK_SIZE)
    with cp.Kernel(numGroups, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            i = bx * BLOCK_SIZE + tid
            if i < totalNbBodies:
                # Read the bodySimIndex from slot BS_BODY_SIM_IDX (offset 27 in int32 terms)
                bodyIndex_bits = gBodySim[i, BS_BODY_SIM_IDX] + cp.int32(0)
                bodyIndex = thread.bitcast(bodyIndex_bits, cp.float32)
                bodyIndex_i = thread.bitcast(bodyIndex, cp.int32)
                # Actually bodySimIndex is stored as float reinterpret of uint32
                # In the C++ code: data.w at BODYSIM_INDEX_IND uint4 => field index 27
                # which is freezeThresh...bodySimIndex.w, stored as reinterpret_cast float of nodeIndex
                # We need the raw bits as int:
                bodyIdx = gBodySim[i, BS_BODY_SIM_IDX] + cp.int32(0)

                for _c in range(60):
                    gBodySimPool[bodyIdx, _c] = gBodySim[i, _c]


# =====================================================================
# Kernel 2: updateBodiesLaunchDirectAPI
# Simplified: always copies all 60 slots (full body).
# On non-first transfer, restores velocities afterwards if !copyVel,
# and recalculates body2World if body2Actor changed.
# =====================================================================
@cp.kernel
def updateBodiesLaunchDirectAPI(
    gBodySim,       # int32[totalNbBodies, 60]
    gBodySimPool,   # int32[poolSize, 60]
    totalNbBodies,
    BLOCK_SIZE: cp.constexpr = 256
):
    numGroups = cp.ceildiv(totalNbBodies, BLOCK_SIZE)
    with cp.Kernel(numGroups, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            i = bx * BLOCK_SIZE + tid
            if i < totalNbBodies:
                bodyIdx = gBodySim[i, BS_BODY_SIM_IDX] + cp.int32(0)
                internalFlags = gBodySim[i, BS_INTERNAL_FLAGS] + cp.int32(0)
                firstTransfer = internalFlags & cp.int32(eFIRST_BODY_COPY_GPU)
                copyVel = internalFlags & cp.int32(eVELOCITY_COPY_GPU)

                # Save old velocities before overwrite (in case !copyVel)
                old_lv0 = gBodySimPool[bodyIdx, 0] + cp.int32(0)
                old_lv1 = gBodySimPool[bodyIdx, 1] + cp.int32(0)
                old_lv2 = gBodySimPool[bodyIdx, 2] + cp.int32(0)
                old_av0 = gBodySimPool[bodyIdx, 4] + cp.int32(0)
                old_av1 = gBodySimPool[bodyIdx, 5] + cp.int32(0)
                old_av2 = gBodySimPool[bodyIdx, 6] + cp.int32(0)

                # Save old body2World and body2Actor
                ob2w_qx = thread.bitcast(gBodySimPool[bodyIdx, BS_B2W_QX] + cp.int32(0), cp.float32)
                ob2w_qy = thread.bitcast(gBodySimPool[bodyIdx, BS_B2W_QY] + cp.int32(0), cp.float32)
                ob2w_qz = thread.bitcast(gBodySimPool[bodyIdx, BS_B2W_QZ] + cp.int32(0), cp.float32)
                ob2w_qw = thread.bitcast(gBodySimPool[bodyIdx, BS_B2W_QW] + cp.int32(0), cp.float32)
                ob2w_px = thread.bitcast(gBodySimPool[bodyIdx, BS_B2W_PX] + cp.int32(0), cp.float32)
                ob2w_py = thread.bitcast(gBodySimPool[bodyIdx, BS_B2W_PY] + cp.int32(0), cp.float32)
                ob2w_pz = thread.bitcast(gBodySimPool[bodyIdx, BS_B2W_PZ] + cp.int32(0), cp.float32)
                oa_qx = thread.bitcast(gBodySimPool[bodyIdx, BS_B2A_QX] + cp.int32(0), cp.float32)
                oa_qy = thread.bitcast(gBodySimPool[bodyIdx, BS_B2A_QY] + cp.int32(0), cp.float32)
                oa_qz = thread.bitcast(gBodySimPool[bodyIdx, BS_B2A_QZ] + cp.int32(0), cp.float32)
                oa_qw = thread.bitcast(gBodySimPool[bodyIdx, BS_B2A_QW] + cp.int32(0), cp.float32)
                oa_px = thread.bitcast(gBodySimPool[bodyIdx, BS_B2A_PX] + cp.int32(0), cp.float32)
                oa_py = thread.bitcast(gBodySimPool[bodyIdx, BS_B2A_PY] + cp.int32(0), cp.float32)
                oa_pz = thread.bitcast(gBodySimPool[bodyIdx, BS_B2A_PZ] + cp.int32(0), cp.float32)

                # Always do full copy (all 60 int32 slots)
                for _c in range(60):
                    gBodySimPool[bodyIdx, _c] = gBodySim[i, _c]

                # Non-first, non-copyVel: restore old velocities
                if firstTransfer == cp.int32(0):
                    if copyVel == cp.int32(0):
                        gBodySimPool[bodyIdx, 0] = old_lv0
                        gBodySimPool[bodyIdx, 1] = old_lv1
                        gBodySimPool[bodyIdx, 2] = old_lv2
                        gBodySimPool[bodyIdx, 4] = old_av0
                        gBodySimPool[bodyIdx, 5] = old_av1
                        gBodySimPool[bodyIdx, 6] = old_av2

                    # Non-first: clear external accelerations (not copied from CPU)
                    for _ac in range(8):
                        gBodySimPool[bodyIdx, cp.int32(52) + _ac] = cp.int32(0)

                    # Check if body2Actor changed, recalculate body2World
                    na_qx = thread.bitcast(gBodySimPool[bodyIdx, BS_B2A_QX] + cp.int32(0), cp.float32)
                    na_qy = thread.bitcast(gBodySimPool[bodyIdx, BS_B2A_QY] + cp.int32(0), cp.float32)
                    na_qz = thread.bitcast(gBodySimPool[bodyIdx, BS_B2A_QZ] + cp.int32(0), cp.float32)
                    na_qw = thread.bitcast(gBodySimPool[bodyIdx, BS_B2A_QW] + cp.int32(0), cp.float32)
                    na_px = thread.bitcast(gBodySimPool[bodyIdx, BS_B2A_PX] + cp.int32(0), cp.float32)
                    na_py = thread.bitcast(gBodySimPool[bodyIdx, BS_B2A_PY] + cp.int32(0), cp.float32)
                    na_pz = thread.bitcast(gBodySimPool[bodyIdx, BS_B2A_PZ] + cp.int32(0), cp.float32)

                    changed = cp.int32(0)
                    if na_qx != oa_qx:
                        changed = cp.int32(1)
                    if na_qy != oa_qy:
                        changed = cp.int32(1)
                    if na_qz != oa_qz:
                        changed = cp.int32(1)
                    if na_qw != oa_qw:
                        changed = cp.int32(1)
                    if na_px != oa_px:
                        changed = cp.int32(1)
                    if na_py != oa_py:
                        changed = cp.int32(1)
                    if na_pz != oa_pz:
                        changed = cp.int32(1)

                    if changed != cp.int32(0):
                        # actor2World = body2World * oldBody2Actor.getInverse()
                        inv_qx = -oa_qx
                        inv_qy = -oa_qy
                        inv_qz = -oa_qz
                        inv_qw = oa_qw
                        # inv_p = inv_q.rotate(-old_p)
                        neg_px = -oa_px
                        neg_py = -oa_py
                        neg_pz = -oa_pz
                        icx = inv_qy * neg_pz - inv_qz * neg_py
                        icy = inv_qz * neg_px - inv_qx * neg_pz
                        icz = inv_qx * neg_py - inv_qy * neg_px
                        ip_x = neg_px + cp.float32(2.0) * (inv_qw * icx + inv_qy * icz - inv_qz * icy)
                        ip_y = neg_py + cp.float32(2.0) * (inv_qw * icy + inv_qz * icx - inv_qx * icz)
                        ip_z = neg_pz + cp.float32(2.0) * (inv_qw * icz + inv_qx * icy - inv_qy * icx)

                        # a2w = body2World * body2Actor_inv
                        # a2w.q = b2w.q * inv.q
                        aq_x = ob2w_qw * inv_qx + ob2w_qx * inv_qw + ob2w_qy * inv_qz - ob2w_qz * inv_qy
                        aq_y = ob2w_qw * inv_qy - ob2w_qx * inv_qz + ob2w_qy * inv_qw + ob2w_qz * inv_qx
                        aq_z = ob2w_qw * inv_qz + ob2w_qx * inv_qy - ob2w_qy * inv_qx + ob2w_qz * inv_qw
                        aq_w = ob2w_qw * inv_qw - ob2w_qx * inv_qx - ob2w_qy * inv_qy - ob2w_qz * inv_qz
                        # a2w.p = b2w.q.rotate(inv.p) + b2w.p
                        rcx = ob2w_qy * ip_z - ob2w_qz * ip_y
                        rcy = ob2w_qz * ip_x - ob2w_qx * ip_z
                        rcz = ob2w_qx * ip_y - ob2w_qy * ip_x
                        ap_x = ip_x + cp.float32(2.0) * (ob2w_qw * rcx + ob2w_qy * rcz - ob2w_qz * rcy) + ob2w_px
                        ap_y = ip_y + cp.float32(2.0) * (ob2w_qw * rcy + ob2w_qz * rcx - ob2w_qx * rcz) + ob2w_py
                        ap_z = ip_z + cp.float32(2.0) * (ob2w_qw * rcz + ob2w_qx * rcy - ob2w_qy * rcx) + ob2w_pz

                        # new_b2w = actor2World * newBody2Actor
                        # new_b2w.q = a2w.q * na.q
                        nq_x = aq_w * na_qx + aq_x * na_qw + aq_y * na_qz - aq_z * na_qy
                        nq_y = aq_w * na_qy - aq_x * na_qz + aq_y * na_qw + aq_z * na_qx
                        nq_z = aq_w * na_qz + aq_x * na_qy - aq_y * na_qx + aq_z * na_qw
                        nq_w = aq_w * na_qw - aq_x * na_qx - aq_y * na_qy - aq_z * na_qz
                        # new_b2w.p = a2w.q.rotate(na.p) + a2w.p
                        ncx = aq_y * na_pz - aq_z * na_py
                        ncy = aq_z * na_px - aq_x * na_pz
                        ncz = aq_x * na_py - aq_y * na_px
                        np_x = na_px + cp.float32(2.0) * (aq_w * ncx + aq_y * ncz - aq_z * ncy) + ap_x
                        np_y = na_py + cp.float32(2.0) * (aq_w * ncy + aq_z * ncx - aq_x * ncz) + ap_y
                        np_z = na_pz + cp.float32(2.0) * (aq_w * ncz + aq_x * ncy - aq_y * ncx) + ap_z

                        gBodySimPool[bodyIdx, BS_B2W_QX] = thread.bitcast(nq_x, cp.int32)
                        gBodySimPool[bodyIdx, BS_B2W_QY] = thread.bitcast(nq_y, cp.int32)
                        gBodySimPool[bodyIdx, BS_B2W_QZ] = thread.bitcast(nq_z, cp.int32)
                        gBodySimPool[bodyIdx, BS_B2W_QW] = thread.bitcast(nq_w, cp.int32)
                        gBodySimPool[bodyIdx, BS_B2W_PX] = thread.bitcast(np_x, cp.int32)
                        gBodySimPool[bodyIdx, BS_B2W_PY] = thread.bitcast(np_y, cp.int32)
                        gBodySimPool[bodyIdx, BS_B2W_PZ] = thread.bitcast(np_z, cp.int32)
                        gBodySimPool[bodyIdx, BS_B2W_PW] = thread.bitcast(cp.float32(0.0), cp.int32)


# NOTE: Old pre-declarations and nested if/else body removed — replaced by simplified
# copy-all-then-restore approach above.
# =====================================================================
# Kernel 3: updateShapesLaunch
# =====================================================================
@cp.kernel
def updateShapesLaunch(
    newShapeSimsData,   # int32[nbNewShapes, NSS_SIZE] -- PxgNewShapeSim flattened
    shapeSimsData,      # int32[shapePoolSize, SS_SIZE] -- PxgShapeSim pool
    nbNewShapes,
    BLOCK_SIZE: cp.constexpr = 256,
    NSS_SIZE: cp.constexpr = 17,  # PxgNewShapeSim int32 slots
    SS_SIZE: cp.constexpr = 16    # PxgShapeSim int32 slots
):
    """Copy new shapes into the persistent shape pool."""
    numBlocks = cp.ceildiv(nbNewShapes, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            idx = bx * BLOCK_SIZE + tid
            if idx < nbNewShapes:
                elementIndex = newShapeSimsData[idx, NSS_ELEMENT_INDEX] + cp.int32(0)
                # Copy shape fields (first SS_SIZE slots) to destination
                for _c in range(16):
                    shapeSimsData[elementIndex, _c] = newShapeSimsData[idx, _c]


# =====================================================================
# Kernel 4: newArticulationsLaunch
# =====================================================================
# This kernel copies entire articulation data structures.
# In Capybara, all sub-arrays are passed as flat int32 tensors.
# The warpCopy pattern becomes single-thread for/while loops.
@cp.kernel
def newArticulationsLaunch(
    # Articulation pool data (flat int32 tensors)
    newArticulationData,       # int32[nbArticulations, ARTI_INT_SIZE] -- new articulations
    articulationPoolData,      # int32[maxArticulations, ARTI_INT_SIZE] -- persistent pool
    articulationSleepData,     # int32[maxArticulations, SLEEP_INT_SIZE] -- sleep data pool
    # Link data
    newLinksData,              # int32[totalLinks, LINK_INT_SIZE]
    newLinkWakeCounters,       # float32[totalLinks]
    newLinkPropsData,          # int32[totalLinks, LINKPROP_INT_SIZE]
    newLinkParents,            # int32[totalLinks]
    newLinkChildren,           # int32[totalLinks, CHILDREN_INT_SIZE] -- ArticulationBitField
    newLinkBody2Worlds,        # int32[totalLinks, 7] -- PxTransform
    newLinkBody2Actors,        # int32[totalLinks, 7] -- PxTransform
    newLinkExtAccels,          # int32[totalLinks, ACCEL_INT_SIZE] -- UnAlignedSpatialVector
    # Joint data
    newJointCores,             # int32[totalLinks, JOINTCORE_INT_SIZE]
    newJointData,              # int32[totalLinks, JOINTDATA_INT_SIZE]
    # Indices
    indicesOffset,             # int32[nbArticulations, INDEX_INT_SIZE] -- PxgArticulationSimUpdate
    # DOF data
    dofData,                   # float32[totalDofs]
    # Spatial tendon data
    newSpatialTendonParamsData, # int32[totalSpatialTendons, STENDON_PARAM_INT_SIZE]
    newSpatialTendonData,       # int32[totalSpatialTendons, STENDON_INT_SIZE]
    newAttachmentFixedData,     # int32[totalAttachments, ATTACH_FIXED_INT_SIZE]
    newAttachmentModData,       # int32[totalAttachments, ATTACH_MOD_INT_SIZE]
    newTendonAttachmentRemap,   # int32[totalSpatialTendons]
    # Fixed tendon data
    newFixedTendonParamsData,   # int32[totalFixedTendons, FTENDON_PARAM_INT_SIZE]
    newFixedTendonData,         # int32[totalFixedTendons, FTENDON_INT_SIZE]
    newTendonJointFixedData,    # int32[totalTendonJoints, TJOINT_FIXED_INT_SIZE]
    newTendonTendonJointRemap,  # int32[totalFixedTendons]
    newTendonJointCoeffData,    # int32[totalTendonJoints, TJOINT_COEFF_INT_SIZE]
    # Mimic joints
    newMimicJointData,         # int32[totalMimicJoints, MIMIC_INT_SIZE]
    # Path to root
    newPathToRootData,         # int32[totalPathToRoot]
    # Articulation sub-array output pointers (flattened to pool-level)
    artiLinksPool,             # int32[maxArticulations, maxLinksPerArti * LINK_INT_SIZE]
    artiLinkWakeCountersPool,  # float32[maxArticulations, maxLinksPerArti]
    artiLinkPropsPool,         # int32[maxArticulations, maxLinksPerArti * LINKPROP_INT_SIZE]
    artiParentsPool,           # int32[maxArticulations, maxLinksPerArti]
    artiChildrenPool,          # int32[maxArticulations, maxLinksPerArti * CHILDREN_INT_SIZE]
    artiLinkBody2WorldsPool,   # int32[maxArticulations, maxLinksPerArti * 7]
    artiLinkBody2ActorsPool,   # int32[maxArticulations, maxLinksPerArti * 7]
    artiJointCoresPool,        # int32[maxArticulations, maxLinksPerArti * JOINTCORE_INT_SIZE]
    artiJointDataPool,         # int32[maxArticulations, maxLinksPerArti * JOINTDATA_INT_SIZE]
    artiMotionVelocitiesPool,  # int32[maxArticulations, maxLinksPerArti * SPATIAL_VEC_INT_SIZE]
    artiMotionAccelsPool,      # int32[maxArticulations, maxLinksPerArti * SPATIAL_VEC_INT_SIZE]
    artiCoriolisePool,         # int32[maxArticulations, maxLinksPerArti * SPATIAL_VEC_INT_SIZE]
    artiLinkSleepDataPool,     # int32[maxArticulations, maxLinksPerArti * LINKSLEEP_INT_SIZE]
    artiJointVelocitiesPool,   # float32[maxArticulations, maxDofs]
    artiJointPositionsPool,    # float32[maxArticulations, maxDofs]
    artiJointForcePool,        # float32[maxArticulations, maxDofs]
    artiJointTargetPosPool,    # float32[maxArticulations, maxDofs]
    artiJointTargetVelPool,    # float32[maxArticulations, maxDofs]
    artiExternalAccelsPool,    # int32[maxArticulations, maxLinksPerArti * SPATIAL_VEC_INT_SIZE]
    artiSpatialTendonsPool,    # int32[maxArticulations, maxTendonsPerArti * STENDON_INT_SIZE]
    artiSpatialTendonParamsPool, # int32[maxArticulations, maxTendonsPerArti * STENDON_PARAM_INT_SIZE]
    artiFixedTendonsPool,      # int32[maxArticulations, maxTendonsPerArti * FTENDON_INT_SIZE]
    artiFixedTendonParamsPool, # int32[maxArticulations, maxTendonsPerArti * FTENDON_PARAM_INT_SIZE]
    artiMimicJointCoresPool,   # int32[maxArticulations, maxMimicPerArti * MIMIC_INT_SIZE]
    artiPathToRootPool,        # int32[maxArticulations, maxPathToRoot]
    nbArticulations,
    # Index field offsets within PxgArticulationSimUpdate
    IDX_DIRTY_FLAGS: cp.constexpr = 0,
    IDX_ARTI_INDEX: cp.constexpr = 1,
    IDX_LINK_START: cp.constexpr = 2,
    IDX_DOF_START: cp.constexpr = 3,
    IDX_SPATIAL_TENDON_START: cp.constexpr = 4,
    IDX_SPATIAL_TENDON_ATTACH_START: cp.constexpr = 5,
    IDX_FIXED_TENDON_START: cp.constexpr = 6,
    IDX_FIXED_TENDON_JOINT_START: cp.constexpr = 7,
    IDX_MIMIC_JOINT_START: cp.constexpr = 8,
    IDX_PATH_TO_ROOT: cp.constexpr = 9,
    # Articulation data field offsets
    AD_NUM_LINKS: cp.constexpr = 0,
    AD_NUM_JOINT_DOFS: cp.constexpr = 1,
    AD_NUM_SPATIAL_TENDONS: cp.constexpr = 2,
    AD_NUM_FIXED_TENDONS: cp.constexpr = 3,
    AD_NUM_MIMIC_JOINTS: cp.constexpr = 4,
    AD_NUM_PATH_TO_ROOTS: cp.constexpr = 5,
    AD_INDEX: cp.constexpr = 6,
    # Struct sizes in int32
    ARTI_INT_SIZE: cp.constexpr = 64,
    SLEEP_INT_SIZE: cp.constexpr = 2,
    LINK_INT_SIZE: cp.constexpr = 20,
    LINKPROP_INT_SIZE: cp.constexpr = 4,
    CHILDREN_INT_SIZE: cp.constexpr = 2,
    JOINTCORE_INT_SIZE: cp.constexpr = 68,
    JOINTDATA_INT_SIZE: cp.constexpr = 5,
    SPATIAL_VEC_INT_SIZE: cp.constexpr = 6,
    LINKSLEEP_INT_SIZE: cp.constexpr = 4,
    INDEX_INT_SIZE: cp.constexpr = 12,
    STENDON_INT_SIZE: cp.constexpr = 8,
    STENDON_PARAM_INT_SIZE: cp.constexpr = 4,
    ATTACH_FIXED_INT_SIZE: cp.constexpr = 4,
    ATTACH_MOD_INT_SIZE: cp.constexpr = 8,
    FTENDON_INT_SIZE: cp.constexpr = 8,
    FTENDON_PARAM_INT_SIZE: cp.constexpr = 4,
    TJOINT_FIXED_INT_SIZE: cp.constexpr = 4,
    TJOINT_COEFF_INT_SIZE: cp.constexpr = 2,
    MIMIC_INT_SIZE: cp.constexpr = 8,
    BLOCK_SIZE: cp.constexpr = 128
):
    """Copy new articulations into persistent pool."""
    numBlocks = cp.ceildiv(nbArticulations, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gi = bx * BLOCK_SIZE + tid
            if gi < nbArticulations:
                # Read articulation index from the new articulation data
                articulationIndex = newArticulationData[gi, AD_INDEX] + cp.int32(0)

                # Copy articulation to persistent pool
                j = cp.int32(0)
                while j < ARTI_INT_SIZE:
                    articulationPoolData[articulationIndex, j] = newArticulationData[gi, j]
                    j = j + cp.int32(1)

                # Zero sleep data
                j2 = cp.int32(0)
                while j2 < SLEEP_INT_SIZE:
                    articulationSleepData[articulationIndex, j2] = cp.int32(0)
                    j2 = j2 + cp.int32(1)

                nbLinks = newArticulationData[gi, AD_NUM_LINKS] + cp.int32(0)
                numDofs_val = newArticulationData[gi, AD_NUM_JOINT_DOFS] + cp.int32(0)
                nbSpatialTendons = newArticulationData[gi, AD_NUM_SPATIAL_TENDONS] + cp.int32(0)
                nbFixedTendons = newArticulationData[gi, AD_NUM_FIXED_TENDONS] + cp.int32(0)
                nbMimicJoints = newArticulationData[gi, AD_NUM_MIMIC_JOINTS] + cp.int32(0)
                nbPathToRoots = newArticulationData[gi, AD_NUM_PATH_TO_ROOTS] + cp.int32(0)

                startIndex = indicesOffset[gi, IDX_LINK_START] + cp.int32(0)
                dofStartIndex = indicesOffset[gi, IDX_DOF_START] + cp.int32(0)
                spatialTendonStartIndex = indicesOffset[gi, IDX_SPATIAL_TENDON_START] + cp.int32(0)
                fixedTendonStartIndex = indicesOffset[gi, IDX_FIXED_TENDON_START] + cp.int32(0)
                mimicJointStartIndex = indicesOffset[gi, IDX_MIMIC_JOINT_START] + cp.int32(0)
                pathToRootIndex = indicesOffset[gi, IDX_PATH_TO_ROOT] + cp.int32(0)
                dirtyFlags = indicesOffset[gi, IDX_DIRTY_FLAGS] + cp.int32(0)

                # Copy links
                linkIdx = cp.int32(0)
                while linkIdx < nbLinks:
                    srcLink = startIndex + linkIdx
                    jj = cp.int32(0)
                    while jj < LINK_INT_SIZE:
                        artiLinksPool[articulationIndex, linkIdx * LINK_INT_SIZE + jj] = newLinksData[srcLink, jj]
                        jj = jj + cp.int32(1)
                    linkIdx = linkIdx + cp.int32(1)

                # Copy link wake counters
                linkIdx3 = cp.int32(0)
                while linkIdx3 < nbLinks:
                    artiLinkWakeCountersPool[articulationIndex, linkIdx3] = newLinkWakeCounters[startIndex + linkIdx3]
                    linkIdx3 = linkIdx3 + cp.int32(1)

                # Copy link properties
                linkIdx4 = cp.int32(0)
                while linkIdx4 < nbLinks:
                    srcLp = startIndex + linkIdx4
                    jj2 = cp.int32(0)
                    while jj2 < LINKPROP_INT_SIZE:
                        artiLinkPropsPool[articulationIndex, linkIdx4 * LINKPROP_INT_SIZE + jj2] = newLinkPropsData[srcLp, jj2]
                        jj2 = jj2 + cp.int32(1)
                    linkIdx4 = linkIdx4 + cp.int32(1)

                # Copy parents
                linkIdx5 = cp.int32(0)
                while linkIdx5 < nbLinks:
                    artiParentsPool[articulationIndex, linkIdx5] = newLinkParents[startIndex + linkIdx5]
                    linkIdx5 = linkIdx5 + cp.int32(1)

                # Copy children
                linkIdx6 = cp.int32(0)
                while linkIdx6 < nbLinks:
                    srcCh = startIndex + linkIdx6
                    jj3 = cp.int32(0)
                    while jj3 < CHILDREN_INT_SIZE:
                        artiChildrenPool[articulationIndex, linkIdx6 * CHILDREN_INT_SIZE + jj3] = newLinkChildren[srcCh, jj3]
                        jj3 = jj3 + cp.int32(1)
                    linkIdx6 = linkIdx6 + cp.int32(1)

                # Copy body2Worlds (PxTransform = 7 int32)
                linkIdx7 = cp.int32(0)
                while linkIdx7 < nbLinks:
                    srcB2W = startIndex + linkIdx7
                    jj4 = cp.int32(0)
                    while jj4 < cp.int32(7):
                        artiLinkBody2WorldsPool[articulationIndex, linkIdx7 * cp.int32(7) + jj4] = newLinkBody2Worlds[srcB2W, jj4]
                        jj4 = jj4 + cp.int32(1)
                    linkIdx7 = linkIdx7 + cp.int32(1)

                # Copy body2Actors (PxTransform = 7 int32)
                linkIdx8 = cp.int32(0)
                while linkIdx8 < nbLinks:
                    srcB2A = startIndex + linkIdx8
                    jj5 = cp.int32(0)
                    while jj5 < cp.int32(7):
                        artiLinkBody2ActorsPool[articulationIndex, linkIdx8 * cp.int32(7) + jj5] = newLinkBody2Actors[srcB2A, jj5]
                        jj5 = jj5 + cp.int32(1)
                    linkIdx8 = linkIdx8 + cp.int32(1)

                # Copy joint cores
                linkIdx9 = cp.int32(0)
                while linkIdx9 < nbLinks:
                    srcJC = startIndex + linkIdx9
                    jj6 = cp.int32(0)
                    while jj6 < JOINTCORE_INT_SIZE:
                        artiJointCoresPool[articulationIndex, linkIdx9 * JOINTCORE_INT_SIZE + jj6] = newJointCores[srcJC, jj6]
                        jj6 = jj6 + cp.int32(1)
                    linkIdx9 = linkIdx9 + cp.int32(1)

                # Copy joint data
                linkIdx10 = cp.int32(0)
                while linkIdx10 < nbLinks:
                    srcJD = startIndex + linkIdx10
                    jj7 = cp.int32(0)
                    while jj7 < JOINTDATA_INT_SIZE:
                        artiJointDataPool[articulationIndex, linkIdx10 * JOINTDATA_INT_SIZE + jj7] = newJointData[srcJD, jj7]
                        jj7 = jj7 + cp.int32(1)
                    linkIdx10 = linkIdx10 + cp.int32(1)

                # Zero motion velocities, motion accelerations, coriolise vectors
                totalSpatialInts = nbLinks * SPATIAL_VEC_INT_SIZE
                jz = cp.int32(0)
                while jz < totalSpatialInts:
                    artiMotionVelocitiesPool[articulationIndex, jz] = cp.int32(0)
                    artiMotionAccelsPool[articulationIndex, jz] = cp.int32(0)
                    artiCoriolisePool[articulationIndex, jz] = cp.int32(0)
                    jz = jz + cp.int32(1)

                # Zero link sleep data
                totalSleepInts = nbLinks * LINKSLEEP_INT_SIZE
                jz2 = cp.int32(0)
                while jz2 < totalSleepInts:
                    artiLinkSleepDataPool[articulationIndex, jz2] = cp.int32(0)
                    jz2 = jz2 + cp.int32(1)

                # Copy spatial tendon params and data
                stIdx = cp.int32(0)
                while stIdx < nbSpatialTendons:
                    srcST = spatialTendonStartIndex + stIdx
                    jj8 = cp.int32(0)
                    while jj8 < STENDON_PARAM_INT_SIZE:
                        artiSpatialTendonParamsPool[articulationIndex, stIdx * STENDON_PARAM_INT_SIZE + jj8] = newSpatialTendonParamsData[srcST, jj8]
                        jj8 = jj8 + cp.int32(1)
                    jj9 = cp.int32(0)
                    while jj9 < STENDON_INT_SIZE:
                        artiSpatialTendonsPool[articulationIndex, stIdx * STENDON_INT_SIZE + jj9] = newSpatialTendonData[srcST, jj9]
                        jj9 = jj9 + cp.int32(1)
                    stIdx = stIdx + cp.int32(1)

                # Copy fixed tendon params and data
                ftIdx = cp.int32(0)
                while ftIdx < nbFixedTendons:
                    srcFT = fixedTendonStartIndex + ftIdx
                    jj10 = cp.int32(0)
                    while jj10 < FTENDON_PARAM_INT_SIZE:
                        artiFixedTendonParamsPool[articulationIndex, ftIdx * FTENDON_PARAM_INT_SIZE + jj10] = newFixedTendonParamsData[srcFT, jj10]
                        jj10 = jj10 + cp.int32(1)
                    jj11 = cp.int32(0)
                    while jj11 < FTENDON_INT_SIZE:
                        artiFixedTendonsPool[articulationIndex, ftIdx * FTENDON_INT_SIZE + jj11] = newFixedTendonData[srcFT, jj11]
                        jj11 = jj11 + cp.int32(1)
                    ftIdx = ftIdx + cp.int32(1)

                # Copy mimic joints
                mjIdx = cp.int32(0)
                while mjIdx < nbMimicJoints:
                    srcMJ = mimicJointStartIndex + mjIdx
                    jj12 = cp.int32(0)
                    while jj12 < MIMIC_INT_SIZE:
                        artiMimicJointCoresPool[articulationIndex, mjIdx * MIMIC_INT_SIZE + jj12] = newMimicJointData[srcMJ, jj12]
                        jj12 = jj12 + cp.int32(1)
                    mjIdx = mjIdx + cp.int32(1)

                # Copy path to root
                prIdx = cp.int32(0)
                while prIdx < nbPathToRoots:
                    artiPathToRootPool[articulationIndex, prIdx] = newPathToRootData[pathToRootIndex + prIdx]
                    prIdx = prIdx + cp.int32(1)

                # Handle DOF data based on dirty flags
                offset = dofStartIndex

                # Joint positions
                if (dirtyFlags & cp.int32(ARTI_DIRTY_POSITIONS)) != cp.int32(0):
                    dIdx = cp.int32(0)
                    while dIdx < numDofs_val:
                        artiJointPositionsPool[articulationIndex, dIdx] = dofData[offset + dIdx]
                        dIdx = dIdx + cp.int32(1)
                    offset = offset + numDofs_val
                else:
                    dIdx2 = cp.int32(0)
                    while dIdx2 < numDofs_val:
                        artiJointPositionsPool[articulationIndex, dIdx2] = cp.float32(0.0)
                        dIdx2 = dIdx2 + cp.int32(1)

                # Joint velocities
                if (dirtyFlags & cp.int32(ARTI_DIRTY_VELOCITIES)) != cp.int32(0):
                    dIdx3 = cp.int32(0)
                    while dIdx3 < numDofs_val:
                        artiJointVelocitiesPool[articulationIndex, dIdx3] = dofData[offset + dIdx3]
                        dIdx3 = dIdx3 + cp.int32(1)
                    offset = offset + numDofs_val
                else:
                    dIdx4 = cp.int32(0)
                    while dIdx4 < numDofs_val:
                        artiJointVelocitiesPool[articulationIndex, dIdx4] = cp.float32(0.0)
                        dIdx4 = dIdx4 + cp.int32(1)

                # Joint forces
                if (dirtyFlags & cp.int32(ARTI_DIRTY_FORCES)) != cp.int32(0):
                    dIdx5 = cp.int32(0)
                    while dIdx5 < numDofs_val:
                        artiJointForcePool[articulationIndex, dIdx5] = dofData[offset + dIdx5]
                        dIdx5 = dIdx5 + cp.int32(1)
                    offset = offset + numDofs_val
                else:
                    dIdx6 = cp.int32(0)
                    while dIdx6 < numDofs_val:
                        artiJointForcePool[articulationIndex, dIdx6] = cp.float32(0.0)
                        dIdx6 = dIdx6 + cp.int32(1)

                # Joint target positions
                if (dirtyFlags & cp.int32(ARTI_DIRTY_JOINT_TARGET_POS)) != cp.int32(0):
                    dIdx7 = cp.int32(0)
                    while dIdx7 < numDofs_val:
                        artiJointTargetPosPool[articulationIndex, dIdx7] = dofData[offset + dIdx7]
                        dIdx7 = dIdx7 + cp.int32(1)
                    offset = offset + numDofs_val
                else:
                    dIdx8 = cp.int32(0)
                    while dIdx8 < numDofs_val:
                        artiJointTargetPosPool[articulationIndex, dIdx8] = cp.float32(0.0)
                        dIdx8 = dIdx8 + cp.int32(1)

                # Joint target velocities
                if (dirtyFlags & cp.int32(ARTI_DIRTY_JOINT_TARGET_VEL)) != cp.int32(0):
                    dIdx9 = cp.int32(0)
                    while dIdx9 < numDofs_val:
                        artiJointTargetVelPool[articulationIndex, dIdx9] = dofData[offset + dIdx9]
                        dIdx9 = dIdx9 + cp.int32(1)
                    offset = offset + numDofs_val
                else:
                    dIdx10 = cp.int32(0)
                    while dIdx10 < numDofs_val:
                        artiJointTargetVelPool[articulationIndex, dIdx10] = cp.float32(0.0)
                        dIdx10 = dIdx10 + cp.int32(1)

                # External accelerations
                if (dirtyFlags & cp.int32(ARTI_DIRTY_EXT_ACCEL)) != cp.int32(0):
                    totalAccelInts = nbLinks * SPATIAL_VEC_INT_SIZE
                    eIdx = cp.int32(0)
                    while eIdx < totalAccelInts:
                        linkOff = eIdx / SPATIAL_VEC_INT_SIZE
                        fieldOff = eIdx - linkOff * SPATIAL_VEC_INT_SIZE
                        artiExternalAccelsPool[articulationIndex, eIdx] = newLinkExtAccels[startIndex + linkOff, fieldOff]
                        eIdx = eIdx + cp.int32(1)
                else:
                    totalAccelInts2 = nbLinks * SPATIAL_VEC_INT_SIZE
                    eIdx2 = cp.int32(0)
                    while eIdx2 < totalAccelInts2:
                        artiExternalAccelsPool[articulationIndex, eIdx2] = cp.int32(0)
                        eIdx2 = eIdx2 + cp.int32(1)


# =====================================================================
# Kernel 5: updateArticulationsLaunch
# =====================================================================
@cp.kernel
def updateArticulationsLaunch(
    # Articulation pool
    articulationPoolData,      # int32[maxArticulations, ARTI_INT_SIZE]
    articulationSleepData,     # int32[maxArticulations, SLEEP_INT_SIZE]
    # New data arrays (same as newArticulations kernel)
    newLinksData,              # int32[totalLinks, LINK_INT_SIZE]
    newLinkWakeCounters,       # float32[totalLinks]
    newLinkPropsData,          # int32[totalLinks, LINKPROP_INT_SIZE]
    newLinkParents,            # int32[totalLinks]
    newLinkChildren,           # int32[totalLinks, CHILDREN_INT_SIZE]
    newLinkBody2Worlds,        # int32[totalLinks, 7]
    newLinkBody2Actors,        # int32[totalLinks, 7]
    newLinkExtAccels,          # int32[totalLinks, ACCEL_INT_SIZE]
    newJointCores,             # int32[totalLinks, JOINTCORE_INT_SIZE]
    newJointData,              # int32[totalLinks, JOINTDATA_INT_SIZE]
    # Sim updates (host-resolved from mapped memory)
    simUpdates,                # int32[nbSimUpdates, INDEX_INT_SIZE]
    nbSimUpdates,
    # DOF data
    dofData,                   # float32[totalDofs]
    directAPI,                 # int32 (bool)
    # Spatial tendon update data
    newSpatialTendonParamsData, # int32[totalSpatialTendons, STENDON_PARAM_INT_SIZE]
    newAttachmentModData,       # int32[totalAttachments, ATTACH_MOD_INT_SIZE]
    # Fixed tendon update data
    newFixedTendonParamsData,   # int32[totalFixedTendons, FTENDON_PARAM_INT_SIZE]
    newTendonJointCoeffData,    # int32[totalTendonJoints, TJOINT_COEFF_INT_SIZE]
    # Mimic joint update data
    newMimicJointData,         # int32[totalMimicJoints, MIMIC_INT_SIZE]
    # Articulation sub-array pools (same as newArticulations)
    artiLinksPool,
    artiLinkWakeCountersPool,
    artiLinkPropsPool,
    artiParentsPool,
    artiChildrenPool,
    artiLinkBody2WorldsPool,
    artiLinkBody2ActorsPool,
    artiJointCoresPool,
    artiJointDataPool,
    artiJointVelocitiesPool,
    artiJointPositionsPool,
    artiJointForcePool,
    artiJointTargetPosPool,
    artiJointTargetVelPool,
    artiExternalAccelsPool,
    artiSpatialTendonParamsPool,
    artiFixedTendonParamsPool,
    artiMimicJointCoresPool,
    # Tendon element data in persistent pool (for attachment/coefficient updates)
    artiSpatialTendonAttachModPool,  # int32[maxArticulations, maxTendonAttach * ATTACH_MOD_INT_SIZE]
    artiFixedTendonJointCoeffPool,   # int32[maxArticulations, maxTendonJoints * TJOINT_COEFF_INT_SIZE]
    # Tendon metadata for per-tendon element counts
    artiSpatialTendonNbElements,     # int32[maxArticulations, maxTendonsPerArti]
    artiFixedTendonNbElements,       # int32[maxArticulations, maxTendonsPerArti]
    # GPU dirty flag output field offset in articulationPoolData
    AD_GPU_DIRTY_FLAG: cp.constexpr = 10,
    AD_CONFI_DIRTY: cp.constexpr = 11,
    AD_UPDATE_DIRTY: cp.constexpr = 12,
    AD_USER_FLAGS: cp.constexpr = 13,
    # Index field offsets within simUpdates
    IDX_DIRTY_FLAGS: cp.constexpr = 0,
    IDX_ARTI_INDEX: cp.constexpr = 1,
    IDX_LINK_START: cp.constexpr = 2,
    IDX_DOF_START: cp.constexpr = 3,
    IDX_SPATIAL_TENDON_START: cp.constexpr = 4,
    IDX_SPATIAL_TENDON_ATTACH_START: cp.constexpr = 5,
    IDX_FIXED_TENDON_START: cp.constexpr = 6,
    IDX_FIXED_TENDON_JOINT_START: cp.constexpr = 7,
    IDX_MIMIC_JOINT_START: cp.constexpr = 8,
    IDX_USER_FLAGS: cp.constexpr = 10,
    # Articulation data field offsets
    AD_NUM_LINKS: cp.constexpr = 0,
    AD_NUM_JOINT_DOFS: cp.constexpr = 1,
    AD_NUM_SPATIAL_TENDONS: cp.constexpr = 2,
    AD_NUM_FIXED_TENDONS: cp.constexpr = 3,
    AD_NUM_MIMIC_JOINTS: cp.constexpr = 4,
    # Struct sizes
    ARTI_INT_SIZE: cp.constexpr = 64,
    SLEEP_INT_SIZE: cp.constexpr = 2,
    LINK_INT_SIZE: cp.constexpr = 20,
    LINKPROP_INT_SIZE: cp.constexpr = 4,
    CHILDREN_INT_SIZE: cp.constexpr = 2,
    JOINTCORE_INT_SIZE: cp.constexpr = 68,
    JOINTDATA_INT_SIZE: cp.constexpr = 5,
    SPATIAL_VEC_INT_SIZE: cp.constexpr = 6,
    INDEX_INT_SIZE: cp.constexpr = 12,
    STENDON_PARAM_INT_SIZE: cp.constexpr = 4,
    ATTACH_MOD_INT_SIZE: cp.constexpr = 8,
    FTENDON_PARAM_INT_SIZE: cp.constexpr = 4,
    TJOINT_COEFF_INT_SIZE: cp.constexpr = 2,
    MIMIC_INT_SIZE: cp.constexpr = 8,
    BLOCK_SIZE: cp.constexpr = 128
):
    """Update existing articulations based on dirty flags."""
    numBlocks = cp.ceildiv(nbSimUpdates, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gi = bx * BLOCK_SIZE + tid
            if gi < nbSimUpdates:
                flags = simUpdates[gi, IDX_DIRTY_FLAGS] + cp.int32(0)
                articIndex = simUpdates[gi, IDX_ARTI_INDEX] + cp.int32(0)
                linkStartIndex = simUpdates[gi, IDX_LINK_START] + cp.int32(0)
                dofStartIndex_val = simUpdates[gi, IDX_DOF_START] + cp.int32(0)

                nbLinks = articulationPoolData[articIndex, AD_NUM_LINKS] + cp.int32(0)
                numDofs_val = articulationPoolData[articIndex, AD_NUM_JOINT_DOFS] + cp.int32(0)

                # DIRTY_JOINTS
                if (flags & cp.int32(ARTI_DIRTY_JOINTS)) != cp.int32(0):
                    linkIdx = cp.int32(0)
                    while linkIdx < nbLinks:
                        srcJC = linkStartIndex + linkIdx
                        jj = cp.int32(0)
                        while jj < JOINTCORE_INT_SIZE:
                            artiJointCoresPool[articIndex, linkIdx * JOINTCORE_INT_SIZE + jj] = newJointCores[srcJC, jj]
                            jj = jj + cp.int32(1)
                        jj2 = cp.int32(0)
                        while jj2 < JOINTDATA_INT_SIZE:
                            artiJointDataPool[articIndex, linkIdx * JOINTDATA_INT_SIZE + jj2] = newJointData[srcJC, jj2]
                            jj2 = jj2 + cp.int32(1)
                        linkIdx = linkIdx + cp.int32(1)
                    articulationPoolData[articIndex, AD_CONFI_DIRTY] = cp.int32(1)

                # DIRTY_MIMIC_JOINT
                if (flags & cp.int32(ARTI_DIRTY_MIMIC_JOINT)) != cp.int32(0):
                    mimicStart = simUpdates[gi, IDX_MIMIC_JOINT_START] + cp.int32(0)
                    nbMimic = articulationPoolData[articIndex, AD_NUM_MIMIC_JOINTS] + cp.int32(0)
                    mIdx = cp.int32(0)
                    while mIdx < nbMimic:
                        srcMJ = mimicStart + mIdx
                        jj3 = cp.int32(0)
                        while jj3 < MIMIC_INT_SIZE:
                            artiMimicJointCoresPool[articIndex, mIdx * MIMIC_INT_SIZE + jj3] = newMimicJointData[srcMJ, jj3]
                            jj3 = jj3 + cp.int32(1)
                        mIdx = mIdx + cp.int32(1)

                # Tendon updates (only if not directAPI)
                if directAPI == cp.int32(0):
                    # DIRTY_SPATIAL_TENDON
                    if (flags & cp.int32(ARTI_DIRTY_SPATIAL_TENDON)) != cp.int32(0):
                        stStart = simUpdates[gi, IDX_SPATIAL_TENDON_START] + cp.int32(0)
                        nbST = articulationPoolData[articIndex, AD_NUM_SPATIAL_TENDONS] + cp.int32(0)
                        stIdx = cp.int32(0)
                        while stIdx < nbST:
                            srcST = stStart + stIdx
                            jj4 = cp.int32(0)
                            while jj4 < STENDON_PARAM_INT_SIZE:
                                artiSpatialTendonParamsPool[articIndex, stIdx * STENDON_PARAM_INT_SIZE + jj4] = newSpatialTendonParamsData[srcST, jj4]
                                jj4 = jj4 + cp.int32(1)
                            stIdx = stIdx + cp.int32(1)

                    # DIRTY_SPATIAL_TENDON_ATTACHMENT
                    if (flags & cp.int32(ARTI_DIRTY_SPATIAL_TENDON_ATTACHMENT)) != cp.int32(0):
                        stAttachStart = simUpdates[gi, IDX_SPATIAL_TENDON_ATTACH_START] + cp.int32(0)
                        nbST2 = articulationPoolData[articIndex, AD_NUM_SPATIAL_TENDONS] + cp.int32(0)
                        attachOffset = cp.int32(0)
                        stIdx2 = cp.int32(0)
                        while stIdx2 < nbST2:
                            nbElements = artiSpatialTendonNbElements[articIndex, stIdx2] + cp.int32(0)
                            eIdx = cp.int32(0)
                            while eIdx < nbElements:
                                srcA = stAttachStart + attachOffset + eIdx
                                jj5 = cp.int32(0)
                                while jj5 < ATTACH_MOD_INT_SIZE:
                                    artiSpatialTendonAttachModPool[articIndex, (attachOffset + eIdx) * ATTACH_MOD_INT_SIZE + jj5] = newAttachmentModData[srcA, jj5]
                                    jj5 = jj5 + cp.int32(1)
                                eIdx = eIdx + cp.int32(1)
                            attachOffset = attachOffset + nbElements
                            stIdx2 = stIdx2 + cp.int32(1)

                    # DIRTY_FIXED_TENDON
                    if (flags & cp.int32(ARTI_DIRTY_FIXED_TENDON)) != cp.int32(0):
                        ftStart = simUpdates[gi, IDX_FIXED_TENDON_START] + cp.int32(0)
                        nbFT = articulationPoolData[articIndex, AD_NUM_FIXED_TENDONS] + cp.int32(0)
                        ftIdx = cp.int32(0)
                        while ftIdx < nbFT:
                            srcFT = ftStart + ftIdx
                            jj6 = cp.int32(0)
                            while jj6 < FTENDON_PARAM_INT_SIZE:
                                artiFixedTendonParamsPool[articIndex, ftIdx * FTENDON_PARAM_INT_SIZE + jj6] = newFixedTendonParamsData[srcFT, jj6]
                                jj6 = jj6 + cp.int32(1)
                            ftIdx = ftIdx + cp.int32(1)

                    # DIRTY_FIXED_TENDON_JOINT
                    if (flags & cp.int32(ARTI_DIRTY_FIXED_TENDON_JOINT)) != cp.int32(0):
                        ftJointStart = simUpdates[gi, IDX_FIXED_TENDON_JOINT_START] + cp.int32(0)
                        nbFT2 = articulationPoolData[articIndex, AD_NUM_FIXED_TENDONS] + cp.int32(0)
                        jointOffset = cp.int32(0)
                        ftIdx2 = cp.int32(0)
                        while ftIdx2 < nbFT2:
                            nbElements2 = artiFixedTendonNbElements[articIndex, ftIdx2] + cp.int32(0)
                            eIdx2 = cp.int32(0)
                            while eIdx2 < nbElements2:
                                srcC = ftJointStart + jointOffset + eIdx2
                                jj7 = cp.int32(0)
                                while jj7 < TJOINT_COEFF_INT_SIZE:
                                    artiFixedTendonJointCoeffPool[articIndex, (jointOffset + eIdx2) * TJOINT_COEFF_INT_SIZE + jj7] = newTendonJointCoeffData[srcC, jj7]
                                    jj7 = jj7 + cp.int32(1)
                                eIdx2 = eIdx2 + cp.int32(1)
                            jointOffset = jointOffset + nbElements2
                            ftIdx2 = ftIdx2 + cp.int32(1)

                # DIRTY_LINKS or DIRTY_ROOT_VELOCITIES
                linksOrRootVel = (flags & cp.int32(ARTI_DIRTY_LINKS)) | (flags & cp.int32(ARTI_DIRTY_ROOT_VELOCITIES))
                if linksOrRootVel != cp.int32(0):
                    # Copy links
                    lIdx = cp.int32(0)
                    while lIdx < nbLinks:
                        srcL = linkStartIndex + lIdx
                        jj8 = cp.int32(0)
                        while jj8 < LINK_INT_SIZE:
                            artiLinksPool[articIndex, lIdx * LINK_INT_SIZE + jj8] = newLinksData[srcL, jj8]
                            jj8 = jj8 + cp.int32(1)
                        lIdx = lIdx + cp.int32(1)

                    if (flags & cp.int32(ARTI_DIRTY_LINKS)) != cp.int32(0):
                        # Copy link props
                        lIdx2 = cp.int32(0)
                        while lIdx2 < nbLinks:
                            srcLp = linkStartIndex + lIdx2
                            jj9 = cp.int32(0)
                            while jj9 < LINKPROP_INT_SIZE:
                                artiLinkPropsPool[articIndex, lIdx2 * LINKPROP_INT_SIZE + jj9] = newLinkPropsData[srcLp, jj9]
                                jj9 = jj9 + cp.int32(1)
                            lIdx2 = lIdx2 + cp.int32(1)

                        # Copy parents
                        lIdx3 = cp.int32(0)
                        while lIdx3 < nbLinks:
                            artiParentsPool[articIndex, lIdx3] = newLinkParents[linkStartIndex + lIdx3]
                            lIdx3 = lIdx3 + cp.int32(1)

                        # Copy children
                        lIdx4 = cp.int32(0)
                        while lIdx4 < nbLinks:
                            srcCh = linkStartIndex + lIdx4
                            jj10 = cp.int32(0)
                            while jj10 < CHILDREN_INT_SIZE:
                                artiChildrenPool[articIndex, lIdx4 * CHILDREN_INT_SIZE + jj10] = newLinkChildren[srcCh, jj10]
                                jj10 = jj10 + cp.int32(1)
                            lIdx4 = lIdx4 + cp.int32(1)

                        # Body2Worlds: if directAPI, recalculate from old+new body2Actor
                        if directAPI != cp.int32(0):
                            lIdx5 = cp.int32(0)
                            while lIdx5 < nbLinks:
                                # Read old body2Actor
                                ob2a_qx_b = artiLinkBody2ActorsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(0)] + cp.int32(0)
                                ob2a_qy_b = artiLinkBody2ActorsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(1)] + cp.int32(0)
                                ob2a_qz_b = artiLinkBody2ActorsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(2)] + cp.int32(0)
                                ob2a_qw_b = artiLinkBody2ActorsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(3)] + cp.int32(0)
                                ob2a_px_b = artiLinkBody2ActorsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(4)] + cp.int32(0)
                                ob2a_py_b = artiLinkBody2ActorsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(5)] + cp.int32(0)
                                ob2a_pz_b = artiLinkBody2ActorsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(6)] + cp.int32(0)

                                # Read new body2Actor
                                srcB2A = linkStartIndex + lIdx5
                                nb2a_qx_b = newLinkBody2Actors[srcB2A, 0] + cp.int32(0)
                                nb2a_qy_b = newLinkBody2Actors[srcB2A, 1] + cp.int32(0)
                                nb2a_qz_b = newLinkBody2Actors[srcB2A, 2] + cp.int32(0)
                                nb2a_qw_b = newLinkBody2Actors[srcB2A, 3] + cp.int32(0)
                                nb2a_px_b = newLinkBody2Actors[srcB2A, 4] + cp.int32(0)
                                nb2a_py_b = newLinkBody2Actors[srcB2A, 5] + cp.int32(0)
                                nb2a_pz_b = newLinkBody2Actors[srcB2A, 6] + cp.int32(0)

                                b2a_changed = cp.int32(0)
                                if ob2a_qx_b != nb2a_qx_b:
                                    b2a_changed = cp.int32(1)
                                if ob2a_qy_b != nb2a_qy_b:
                                    b2a_changed = cp.int32(1)
                                if ob2a_qz_b != nb2a_qz_b:
                                    b2a_changed = cp.int32(1)
                                if ob2a_qw_b != nb2a_qw_b:
                                    b2a_changed = cp.int32(1)
                                if ob2a_px_b != nb2a_px_b:
                                    b2a_changed = cp.int32(1)
                                if ob2a_py_b != nb2a_py_b:
                                    b2a_changed = cp.int32(1)
                                if ob2a_pz_b != nb2a_pz_b:
                                    b2a_changed = cp.int32(1)

                                if b2a_changed != cp.int32(0):
                                    # Read old body2World
                                    ob2w_qx = thread.bitcast(artiLinkBody2WorldsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(0)] + cp.int32(0), cp.float32)
                                    ob2w_qy = thread.bitcast(artiLinkBody2WorldsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(1)] + cp.int32(0), cp.float32)
                                    ob2w_qz = thread.bitcast(artiLinkBody2WorldsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(2)] + cp.int32(0), cp.float32)
                                    ob2w_qw = thread.bitcast(artiLinkBody2WorldsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(3)] + cp.int32(0), cp.float32)
                                    ob2w_px = thread.bitcast(artiLinkBody2WorldsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(4)] + cp.int32(0), cp.float32)
                                    ob2w_py = thread.bitcast(artiLinkBody2WorldsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(5)] + cp.int32(0), cp.float32)
                                    ob2w_pz = thread.bitcast(artiLinkBody2WorldsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(6)] + cp.int32(0), cp.float32)

                                    ob2a_qx = thread.bitcast(ob2a_qx_b, cp.float32)
                                    ob2a_qy = thread.bitcast(ob2a_qy_b, cp.float32)
                                    ob2a_qz = thread.bitcast(ob2a_qz_b, cp.float32)
                                    ob2a_qw = thread.bitcast(ob2a_qw_b, cp.float32)
                                    ob2a_px = thread.bitcast(ob2a_px_b, cp.float32)
                                    ob2a_py = thread.bitcast(ob2a_py_b, cp.float32)
                                    ob2a_pz = thread.bitcast(ob2a_pz_b, cp.float32)

                                    nb2a_qx = thread.bitcast(nb2a_qx_b, cp.float32)
                                    nb2a_qy = thread.bitcast(nb2a_qy_b, cp.float32)
                                    nb2a_qz = thread.bitcast(nb2a_qz_b, cp.float32)
                                    nb2a_qw = thread.bitcast(nb2a_qw_b, cp.float32)
                                    nb2a_px = thread.bitcast(nb2a_px_b, cp.float32)
                                    nb2a_py = thread.bitcast(nb2a_py_b, cp.float32)
                                    nb2a_pz = thread.bitcast(nb2a_pz_b, cp.float32)

                                    # actor2World = body2World * oldBody2Actor.getInverse()
                                    inv_qx = -ob2a_qx
                                    inv_qy = -ob2a_qy
                                    inv_qz = -ob2a_qz
                                    inv_qw = ob2a_qw
                                    # NOTE: inline quat_rotate to avoid MLIR domination error in conditional
                                    neg_ob2a_px = -ob2a_px
                                    neg_ob2a_py = -ob2a_py
                                    neg_ob2a_pz = -ob2a_pz
                                    iqr_x = inv_qy * neg_ob2a_pz - inv_qz * neg_ob2a_py
                                    iqr_y = inv_qz * neg_ob2a_px - inv_qx * neg_ob2a_pz
                                    iqr_z = inv_qx * neg_ob2a_py - inv_qy * neg_ob2a_px
                                    inv_px2 = neg_ob2a_px + cp.float32(2.0) * (inv_qw * iqr_x + inv_qy * iqr_z - inv_qz * iqr_y)
                                    inv_py2 = neg_ob2a_py + cp.float32(2.0) * (inv_qw * iqr_y + inv_qz * iqr_x - inv_qx * iqr_z)
                                    inv_pz2 = neg_ob2a_pz + cp.float32(2.0) * (inv_qw * iqr_z + inv_qx * iqr_y - inv_qy * iqr_x)

                                    # NOTE: inline transform_multiply_q to avoid MLIR domination error in conditional
                                    # quat_multiply(ob2w_q, inv_q)
                                    a2w_qx = ob2w_qw * inv_qx + ob2w_qx * inv_qw + ob2w_qy * inv_qz - ob2w_qz * inv_qy
                                    a2w_qy = ob2w_qw * inv_qy - ob2w_qx * inv_qz + ob2w_qy * inv_qw + ob2w_qz * inv_qx
                                    a2w_qz = ob2w_qw * inv_qz + ob2w_qx * inv_qy - ob2w_qy * inv_qx + ob2w_qz * inv_qw
                                    a2w_qw = ob2w_qw * inv_qw - ob2w_qx * inv_qx - ob2w_qy * inv_qy - ob2w_qz * inv_qz
                                    # quat_rotate(ob2w_q, inv_p2) + ob2w_p
                                    tmcr_x = ob2w_qy * inv_pz2 - ob2w_qz * inv_py2
                                    tmcr_y = ob2w_qz * inv_px2 - ob2w_qx * inv_pz2
                                    tmcr_z = ob2w_qx * inv_py2 - ob2w_qy * inv_px2
                                    a2w_px = inv_px2 + cp.float32(2.0) * (ob2w_qw * tmcr_x + ob2w_qy * tmcr_z - ob2w_qz * tmcr_y) + ob2w_px
                                    a2w_py = inv_py2 + cp.float32(2.0) * (ob2w_qw * tmcr_y + ob2w_qz * tmcr_x - ob2w_qx * tmcr_z) + ob2w_py
                                    a2w_pz = inv_pz2 + cp.float32(2.0) * (ob2w_qw * tmcr_z + ob2w_qx * tmcr_y - ob2w_qy * tmcr_x) + ob2w_pz

                                    # new body2World = actor2World * newBody2Actor
                                    # NOTE: inline transform_multiply_q to avoid MLIR domination error in conditional
                                    # quat_multiply(a2w_q, nb2a_q)
                                    r_qx = a2w_qw * nb2a_qx + a2w_qx * nb2a_qw + a2w_qy * nb2a_qz - a2w_qz * nb2a_qy
                                    r_qy = a2w_qw * nb2a_qy - a2w_qx * nb2a_qz + a2w_qy * nb2a_qw + a2w_qz * nb2a_qx
                                    r_qz = a2w_qw * nb2a_qz + a2w_qx * nb2a_qy - a2w_qy * nb2a_qx + a2w_qz * nb2a_qw
                                    r_qw = a2w_qw * nb2a_qw - a2w_qx * nb2a_qx - a2w_qy * nb2a_qy - a2w_qz * nb2a_qz
                                    # quat_rotate(a2w_q, nb2a_p) + a2w_p
                                    tmcr2_x = a2w_qy * nb2a_pz - a2w_qz * nb2a_py
                                    tmcr2_y = a2w_qz * nb2a_px - a2w_qx * nb2a_pz
                                    tmcr2_z = a2w_qx * nb2a_py - a2w_qy * nb2a_px
                                    r_px = nb2a_px + cp.float32(2.0) * (a2w_qw * tmcr2_x + a2w_qy * tmcr2_z - a2w_qz * tmcr2_y) + a2w_px
                                    r_py = nb2a_py + cp.float32(2.0) * (a2w_qw * tmcr2_y + a2w_qz * tmcr2_x - a2w_qx * tmcr2_z) + a2w_py
                                    r_pz = nb2a_pz + cp.float32(2.0) * (a2w_qw * tmcr2_z + a2w_qx * tmcr2_y - a2w_qy * tmcr2_x) + a2w_pz

                                    artiLinkBody2WorldsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(0)] = thread.bitcast(r_qx, cp.int32)
                                    artiLinkBody2WorldsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(1)] = thread.bitcast(r_qy, cp.int32)
                                    artiLinkBody2WorldsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(2)] = thread.bitcast(r_qz, cp.int32)
                                    artiLinkBody2WorldsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(3)] = thread.bitcast(r_qw, cp.int32)
                                    artiLinkBody2WorldsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(4)] = thread.bitcast(r_px, cp.int32)
                                    artiLinkBody2WorldsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(5)] = thread.bitcast(r_py, cp.int32)
                                    artiLinkBody2WorldsPool[articIndex, lIdx5 * cp.int32(7) + cp.int32(6)] = thread.bitcast(r_pz, cp.int32)

                                lIdx5 = lIdx5 + cp.int32(1)
                        else:
                            # Not directAPI: copy body2Worlds from host
                            lIdx6 = cp.int32(0)
                            while lIdx6 < nbLinks:
                                srcB2W = linkStartIndex + lIdx6
                                jj11 = cp.int32(0)
                                while jj11 < cp.int32(7):
                                    artiLinkBody2WorldsPool[articIndex, lIdx6 * cp.int32(7) + jj11] = newLinkBody2Worlds[srcB2W, jj11]
                                    jj11 = jj11 + cp.int32(1)
                                lIdx6 = lIdx6 + cp.int32(1)

                        # Copy body2Actors
                        lIdx7 = cp.int32(0)
                        while lIdx7 < nbLinks:
                            srcB2A2 = linkStartIndex + lIdx7
                            jj12 = cp.int32(0)
                            while jj12 < cp.int32(7):
                                artiLinkBody2ActorsPool[articIndex, lIdx7 * cp.int32(7) + jj12] = newLinkBody2Actors[srcB2A2, jj12]
                                jj12 = jj12 + cp.int32(1)
                            lIdx7 = lIdx7 + cp.int32(1)

                elif (flags & cp.int32(ARTI_DIRTY_POSITIONS)) != cp.int32(0):
                    if directAPI == cp.int32(0):
                        # Copy body2Worlds only
                        lIdx8 = cp.int32(0)
                        while lIdx8 < nbLinks:
                            srcB2W2 = linkStartIndex + lIdx8
                            jj13 = cp.int32(0)
                            while jj13 < cp.int32(7):
                                artiLinkBody2WorldsPool[articIndex, lIdx8 * cp.int32(7) + jj13] = newLinkBody2Worlds[srcB2W2, jj13]
                                jj13 = jj13 + cp.int32(1)
                            lIdx8 = lIdx8 + cp.int32(1)

                # DOF data (only if not directAPI)
                offset = dofStartIndex_val
                if directAPI == cp.int32(0):
                    if (flags & cp.int32(ARTI_DIRTY_POSITIONS)) != cp.int32(0):
                        dIdx = cp.int32(0)
                        while dIdx < numDofs_val:
                            artiJointPositionsPool[articIndex, dIdx] = dofData[offset + dIdx]
                            dIdx = dIdx + cp.int32(1)
                        offset = offset + numDofs_val
                        gpuDirty = articulationPoolData[articIndex, AD_GPU_DIRTY_FLAG] + cp.int32(0)
                        articulationPoolData[articIndex, AD_GPU_DIRTY_FLAG] = gpuDirty | cp.int32(ARTI_DIRTY_POSITIONS)

                    if (flags & cp.int32(ARTI_DIRTY_VELOCITIES)) != cp.int32(0):
                        dIdx2 = cp.int32(0)
                        while dIdx2 < numDofs_val:
                            artiJointVelocitiesPool[articIndex, dIdx2] = dofData[offset + dIdx2]
                            dIdx2 = dIdx2 + cp.int32(1)
                        offset = offset + numDofs_val
                        gpuDirty2 = articulationPoolData[articIndex, AD_GPU_DIRTY_FLAG] + cp.int32(0)
                        articulationPoolData[articIndex, AD_GPU_DIRTY_FLAG] = gpuDirty2 | cp.int32(ARTI_DIRTY_VELOCITIES)

                    if (flags & cp.int32(ARTI_DIRTY_FORCES)) != cp.int32(0):
                        dIdx3 = cp.int32(0)
                        while dIdx3 < numDofs_val:
                            artiJointForcePool[articIndex, dIdx3] = dofData[offset + dIdx3]
                            dIdx3 = dIdx3 + cp.int32(1)
                        offset = offset + numDofs_val
                        gpuDirty3 = articulationPoolData[articIndex, AD_GPU_DIRTY_FLAG] + cp.int32(0)
                        articulationPoolData[articIndex, AD_GPU_DIRTY_FLAG] = gpuDirty3 | cp.int32(ARTI_DIRTY_FORCES)

                    if (flags & cp.int32(ARTI_DIRTY_JOINT_TARGET_POS)) != cp.int32(0):
                        dIdx4 = cp.int32(0)
                        while dIdx4 < numDofs_val:
                            artiJointTargetPosPool[articIndex, dIdx4] = dofData[offset + dIdx4]
                            dIdx4 = dIdx4 + cp.int32(1)
                        offset = offset + numDofs_val
                        gpuDirty4 = articulationPoolData[articIndex, AD_GPU_DIRTY_FLAG] + cp.int32(0)
                        articulationPoolData[articIndex, AD_GPU_DIRTY_FLAG] = gpuDirty4 | cp.int32(ARTI_DIRTY_JOINT_TARGET_POS)

                    if (flags & cp.int32(ARTI_DIRTY_JOINT_TARGET_VEL)) != cp.int32(0):
                        dIdx5 = cp.int32(0)
                        while dIdx5 < numDofs_val:
                            artiJointTargetVelPool[articIndex, dIdx5] = dofData[offset + dIdx5]
                            dIdx5 = dIdx5 + cp.int32(1)
                        offset = offset + numDofs_val
                        gpuDirty5 = articulationPoolData[articIndex, AD_GPU_DIRTY_FLAG] + cp.int32(0)
                        articulationPoolData[articIndex, AD_GPU_DIRTY_FLAG] = gpuDirty5 | cp.int32(ARTI_DIRTY_JOINT_TARGET_VEL)

                # DIRTY_WAKECOUNTER
                if (flags & cp.int32(ARTI_DIRTY_WAKECOUNTER)) != cp.int32(0):
                    lIdx9 = cp.int32(0)
                    while lIdx9 < nbLinks:
                        artiLinkWakeCountersPool[articIndex, lIdx9] = newLinkWakeCounters[linkStartIndex + lIdx9]
                        lIdx9 = lIdx9 + cp.int32(1)

                # DIRTY_USER_FLAGS
                if (flags & cp.int32(ARTI_DIRTY_USER_FLAGS)) != cp.int32(0):
                    userFlags = simUpdates[gi, IDX_USER_FLAGS] + cp.int32(0)
                    articulationPoolData[articIndex, AD_USER_FLAGS] = userFlags
                    articulationPoolData[articIndex, AD_CONFI_DIRTY] = cp.int32(1)

                # External accelerations (only if not directAPI)
                if directAPI == cp.int32(0):
                    if (flags & cp.int32(ARTI_DIRTY_EXT_ACCEL)) != cp.int32(0):
                        totalAccelInts = nbLinks * SPATIAL_VEC_INT_SIZE
                        eIdx = cp.int32(0)
                        while eIdx < totalAccelInts:
                            linkOff = eIdx / SPATIAL_VEC_INT_SIZE
                            fieldOff = eIdx - linkOff * SPATIAL_VEC_INT_SIZE
                            artiExternalAccelsPool[articIndex, eIdx] = newLinkExtAccels[linkStartIndex + linkOff, fieldOff]
                            eIdx = eIdx + cp.int32(1)
                        gpuDirty6 = articulationPoolData[articIndex, AD_GPU_DIRTY_FLAG] + cp.int32(0)
                        articulationPoolData[articIndex, AD_GPU_DIRTY_FLAG] = gpuDirty6 | cp.int32(ARTI_DIRTY_EXT_ACCEL)

                # Update dirty flag: updateDirty = flags & ~(FORCES | IN_DIRTY_LIST)
                updateDirtyMask = ~(cp.int32(ARTI_DIRTY_FORCES) | cp.int32(ARTI_IN_DIRTY_LIST))
                articulationPoolData[articIndex, AD_UPDATE_DIRTY] = flags & updateDirtyMask


# =====================================================================
# Kernel 6: updateBodyExternalVelocitiesLaunch
# =====================================================================
@cp.kernel
def updateBodyExternalVelocitiesLaunch(
    bodySimPool,       # int32[poolSize, 60] -- PxgBodySim
    updatedBodies,     # float32[nbBodies, 16] -- PxgBodySimVelocityUpdate
    nbBodies,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Update body velocities and external accelerations from updated body descriptors."""
    numBlocks = cp.ceildiv(nbBodies, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            idx = bx * BLOCK_SIZE + tid
            if idx < nbBodies:
                # Read linear velocity + bodyIndex
                linVelX = updatedBodies[idx, VU_LIN_VEL_X] + cp.float32(0.0)
                linVelY = updatedBodies[idx, VU_LIN_VEL_Y] + cp.float32(0.0)
                linVelZ = updatedBodies[idx, VU_LIN_VEL_Z] + cp.float32(0.0)
                bodyIndexF = updatedBodies[idx, VU_BODY_INDEX_W] + cp.float32(0.0)
                bodyIndex = thread.bitcast(bodyIndexF, cp.int32)

                # Preserve invMass from existing data
                origInvMass_bits = bodySimPool[bodyIndex, BS_INV_MASS_W] + cp.int32(0)

                # Write linear velocity, keeping invMass
                bodySimPool[bodyIndex, BS_LIN_VEL_X] = thread.bitcast(linVelX, cp.int32)
                bodySimPool[bodyIndex, BS_LIN_VEL_Y] = thread.bitcast(linVelY, cp.int32)
                bodySimPool[bodyIndex, BS_LIN_VEL_Z] = thread.bitcast(linVelZ, cp.int32)
                # invMass stays as-is (already in pool)

                # Write angular velocity + maxPenBias
                angVelX = updatedBodies[idx, VU_ANG_VEL_X] + cp.float32(0.0)
                angVelY = updatedBodies[idx, VU_ANG_VEL_Y] + cp.float32(0.0)
                angVelZ = updatedBodies[idx, VU_ANG_VEL_Z] + cp.float32(0.0)
                maxPenBias = updatedBodies[idx, VU_MAX_PEN_BIAS_W] + cp.float32(0.0)
                bodySimPool[bodyIndex, BS_ANG_VEL_X] = thread.bitcast(angVelX, cp.int32)
                bodySimPool[bodyIndex, BS_ANG_VEL_Y] = thread.bitcast(angVelY, cp.int32)
                bodySimPool[bodyIndex, BS_ANG_VEL_Z] = thread.bitcast(angVelZ, cp.int32)
                bodySimPool[bodyIndex, BS_MAX_PEN_BIAS_W] = thread.bitcast(maxPenBias, cp.int32)

                # Write external accelerations
                extLinX = updatedBodies[idx, VU_EXT_LIN_ACC_X] + cp.float32(0.0)
                extLinY = updatedBodies[idx, VU_EXT_LIN_ACC_Y] + cp.float32(0.0)
                extLinZ = updatedBodies[idx, VU_EXT_LIN_ACC_Z] + cp.float32(0.0)
                extLinW = updatedBodies[idx, VU_EXT_LIN_ACC_W] + cp.float32(0.0)
                bodySimPool[bodyIndex, BS_EXT_LIN_ACC_X] = thread.bitcast(extLinX, cp.int32)
                bodySimPool[bodyIndex, BS_EXT_LIN_ACC_Y] = thread.bitcast(extLinY, cp.int32)
                bodySimPool[bodyIndex, BS_EXT_LIN_ACC_Z] = thread.bitcast(extLinZ, cp.int32)
                bodySimPool[bodyIndex, BS_EXT_LIN_ACC_W] = thread.bitcast(extLinW, cp.int32)

                extAngX = updatedBodies[idx, VU_EXT_ANG_ACC_X] + cp.float32(0.0)
                extAngY = updatedBodies[idx, VU_EXT_ANG_ACC_Y] + cp.float32(0.0)
                extAngZ = updatedBodies[idx, VU_EXT_ANG_ACC_Z] + cp.float32(0.0)
                extAngW = updatedBodies[idx, VU_EXT_ANG_ACC_W] + cp.float32(0.0)
                bodySimPool[bodyIndex, BS_EXT_ANG_ACC_X] = thread.bitcast(extAngX, cp.int32)
                bodySimPool[bodyIndex, BS_EXT_ANG_ACC_Y] = thread.bitcast(extAngY, cp.int32)
                bodySimPool[bodyIndex, BS_EXT_ANG_ACC_Z] = thread.bitcast(extAngZ, cp.int32)
                bodySimPool[bodyIndex, BS_EXT_ANG_ACC_W] = thread.bitcast(extAngW, cp.int32)


# =====================================================================
# Kernel 7: updateJointsLaunch
# =====================================================================
@cp.kernel
def updateJointsLaunch(
    # Rigid joints
    cpuRigidJointData,       # int32[maxJoints, JOINT_INT_SIZE]
    gpuRigidJointData,       # int32[maxJoints, JOINT_INT_SIZE]
    cpuRigidJointPrePrep,    # int32[maxJoints, PREPREP_INT_SIZE]
    gpuRigidJointPrePrep,    # int32[maxJoints, PREPREP_INT_SIZE]
    updatedRigidJointIndices, # int32[nbUpdatedRigidJoints]
    nbUpdatedRigidJoints,
    # Articulation joints
    cpuArtiJointData,        # int32[maxJoints, JOINT_INT_SIZE]
    gpuArtiJointData,        # int32[maxJoints, JOINT_INT_SIZE]
    cpuArtiJointPrePrep,     # int32[maxJoints, PREPREP_INT_SIZE]
    gpuArtiJointPrePrep,     # int32[maxJoints, PREPREP_INT_SIZE]
    updatedArtiJointIndices, # int32[nbUpdatedArtiJoints]
    nbUpdatedArtiJoints,
    # Joint type selector: blockIdx.y == 0 -> rigid, == 1 -> articulation
    # In Capybara we split into two sub-kernels dispatched via 2D grid
    JOINT_INT_SIZE: cp.constexpr = 124,   # sizeof(PxgD6JointData)/4 = 496/4
    PREPREP_INT_SIZE: cp.constexpr = 5,   # sizeof(PxgConstraintPrePrep)/4 = 20/4
    BLOCK_SIZE: cp.constexpr = 128
):
    """Copy updated joints (rigid + articulation) to GPU pool."""
    maxJoints = nbUpdatedRigidJoints
    if nbUpdatedArtiJoints > maxJoints:
        maxJoints = nbUpdatedArtiJoints
    numBlocks = cp.ceildiv(maxJoints, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gi = bx * BLOCK_SIZE + tid

            # Rigid joints
            if gi < nbUpdatedRigidJoints:
                jointIndex = updatedRigidJointIndices[gi] + cp.int32(0)
                j = cp.int32(0)
                while j < JOINT_INT_SIZE:
                    gpuRigidJointData[jointIndex, j] = cpuRigidJointData[jointIndex, j]
                    j = j + cp.int32(1)
                j2 = cp.int32(0)
                while j2 < PREPREP_INT_SIZE:
                    gpuRigidJointPrePrep[jointIndex, j2] = cpuRigidJointPrePrep[jointIndex, j2]
                    j2 = j2 + cp.int32(1)

            # Articulation joints
            if gi < nbUpdatedArtiJoints:
                artiJointIndex = updatedArtiJointIndices[gi] + cp.int32(0)
                j3 = cp.int32(0)
                while j3 < JOINT_INT_SIZE:
                    gpuArtiJointData[artiJointIndex, j3] = cpuArtiJointData[artiJointIndex, j3]
                    j3 = j3 + cp.int32(1)
                j4 = cp.int32(0)
                while j4 < PREPREP_INT_SIZE:
                    gpuArtiJointPrePrep[artiJointIndex, j4] = cpuArtiJointPrePrep[artiJointIndex, j4]
                    j4 = j4 + cp.int32(1)


# =====================================================================
# Kernel 8: getRigidDynamicGlobalPose
# =====================================================================
@cp.kernel
def getRigidDynamicGlobalPose(
    data,                  # float32[nbElements, 7] -- output PxTransform (q.xyzw, p.xyz)
    gpuIndices,            # int32[nbElements]
    bodySimPool,           # int32[poolSize, 60]
    nbElements,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Get global pose = body2World * body2Actor.getInverse()."""
    numBlocks = cp.ceildiv(nbElements, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gti = bx * BLOCK_SIZE + tid
            if gti < nbElements:
                index = gpuIndices[gti] + cp.int32(0)

                # Read body2World
                b2w_qx = thread.bitcast(bodySimPool[index, BS_B2W_QX] + cp.int32(0), cp.float32)
                b2w_qy = thread.bitcast(bodySimPool[index, BS_B2W_QY] + cp.int32(0), cp.float32)
                b2w_qz = thread.bitcast(bodySimPool[index, BS_B2W_QZ] + cp.int32(0), cp.float32)
                b2w_qw = thread.bitcast(bodySimPool[index, BS_B2W_QW] + cp.int32(0), cp.float32)
                b2w_px = thread.bitcast(bodySimPool[index, BS_B2W_PX] + cp.int32(0), cp.float32)
                b2w_py = thread.bitcast(bodySimPool[index, BS_B2W_PY] + cp.int32(0), cp.float32)
                b2w_pz = thread.bitcast(bodySimPool[index, BS_B2W_PZ] + cp.int32(0), cp.float32)

                # Read body2Actor
                b2a_qx = thread.bitcast(bodySimPool[index, BS_B2A_QX] + cp.int32(0), cp.float32)
                b2a_qy = thread.bitcast(bodySimPool[index, BS_B2A_QY] + cp.int32(0), cp.float32)
                b2a_qz = thread.bitcast(bodySimPool[index, BS_B2A_QZ] + cp.int32(0), cp.float32)
                b2a_qw = thread.bitcast(bodySimPool[index, BS_B2A_QW] + cp.int32(0), cp.float32)
                b2a_px = thread.bitcast(bodySimPool[index, BS_B2A_PX] + cp.int32(0), cp.float32)
                b2a_py = thread.bitcast(bodySimPool[index, BS_B2A_PY] + cp.int32(0), cp.float32)
                b2a_pz = thread.bitcast(bodySimPool[index, BS_B2A_PZ] + cp.int32(0), cp.float32)

                # body2Actor.getInverse(): q_inv = (-qx,-qy,-qz,qw), p_inv = q_inv.rotate(-p)
                inv_qx = -b2a_qx
                inv_qy = -b2a_qy
                inv_qz = -b2a_qz
                inv_qw = b2a_qw
                # NOTE: inline quat_rotate to avoid MLIR domination error in conditional
                neg_b2a_px = -b2a_px
                neg_b2a_py = -b2a_py
                neg_b2a_pz = -b2a_pz
                gp_cr_x = inv_qy * neg_b2a_pz - inv_qz * neg_b2a_py
                gp_cr_y = inv_qz * neg_b2a_px - inv_qx * neg_b2a_pz
                gp_cr_z = inv_qx * neg_b2a_py - inv_qy * neg_b2a_px
                inv_px = neg_b2a_px + cp.float32(2.0) * (inv_qw * gp_cr_x + inv_qy * gp_cr_z - inv_qz * gp_cr_y)
                inv_py = neg_b2a_py + cp.float32(2.0) * (inv_qw * gp_cr_y + inv_qz * gp_cr_x - inv_qx * gp_cr_z)
                inv_pz = neg_b2a_pz + cp.float32(2.0) * (inv_qw * gp_cr_z + inv_qx * gp_cr_y - inv_qy * gp_cr_x)

                # pose = body2World * body2Actor_inv
                # NOTE: inline transform_multiply_q to avoid MLIR domination error in conditional
                # quat_multiply(b2w_q, inv_q)
                r_qx = b2w_qw * inv_qx + b2w_qx * inv_qw + b2w_qy * inv_qz - b2w_qz * inv_qy
                r_qy = b2w_qw * inv_qy - b2w_qx * inv_qz + b2w_qy * inv_qw + b2w_qz * inv_qx
                r_qz = b2w_qw * inv_qz + b2w_qx * inv_qy - b2w_qy * inv_qx + b2w_qz * inv_qw
                r_qw = b2w_qw * inv_qw - b2w_qx * inv_qx - b2w_qy * inv_qy - b2w_qz * inv_qz
                # quat_rotate(b2w_q, inv_p) + b2w_p
                gp_cr2_x = b2w_qy * inv_pz - b2w_qz * inv_py
                gp_cr2_y = b2w_qz * inv_px - b2w_qx * inv_pz
                gp_cr2_z = b2w_qx * inv_py - b2w_qy * inv_px
                r_px = inv_px + cp.float32(2.0) * (b2w_qw * gp_cr2_x + b2w_qy * gp_cr2_z - b2w_qz * gp_cr2_y) + b2w_px
                r_py = inv_py + cp.float32(2.0) * (b2w_qw * gp_cr2_y + b2w_qz * gp_cr2_x - b2w_qx * gp_cr2_z) + b2w_py
                r_pz = inv_pz + cp.float32(2.0) * (b2w_qw * gp_cr2_z + b2w_qx * gp_cr2_y - b2w_qy * gp_cr2_x) + b2w_pz

                data[gti, 0] = r_qx
                data[gti, 1] = r_qy
                data[gti, 2] = r_qz
                data[gti, 3] = r_qw
                data[gti, 4] = r_px
                data[gti, 5] = r_py
                data[gti, 6] = r_pz


# =====================================================================
# Kernel 9: getRigidDynamicLinearVelocity
# =====================================================================
@cp.kernel
def getRigidDynamicLinearVelocity(
    data,                  # float32[nbElements, 3]
    gpuIndices,            # int32[nbElements]
    bodySimPool,           # int32[poolSize, 60]
    nbElements,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Get linear velocity (xyz) from bodySimPool."""
    numBlocks = cp.ceildiv(nbElements, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gti = bx * BLOCK_SIZE + tid
            if gti < nbElements:
                index = gpuIndices[gti] + cp.int32(0)
                data[gti, 0] = thread.bitcast(bodySimPool[index, BS_LIN_VEL_X] + cp.int32(0), cp.float32)
                data[gti, 1] = thread.bitcast(bodySimPool[index, BS_LIN_VEL_Y] + cp.int32(0), cp.float32)
                data[gti, 2] = thread.bitcast(bodySimPool[index, BS_LIN_VEL_Z] + cp.int32(0), cp.float32)


# =====================================================================
# Kernel 10: getRigidDynamicAngularVelocity
# =====================================================================
@cp.kernel
def getRigidDynamicAngularVelocity(
    data,                  # float32[nbElements, 3]
    gpuIndices,            # int32[nbElements]
    bodySimPool,           # int32[poolSize, 60]
    nbElements,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Get angular velocity (xyz) from bodySimPool."""
    numBlocks = cp.ceildiv(nbElements, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gti = bx * BLOCK_SIZE + tid
            if gti < nbElements:
                index = gpuIndices[gti] + cp.int32(0)
                data[gti, 0] = thread.bitcast(bodySimPool[index, BS_ANG_VEL_X] + cp.int32(0), cp.float32)
                data[gti, 1] = thread.bitcast(bodySimPool[index, BS_ANG_VEL_Y] + cp.int32(0), cp.float32)
                data[gti, 2] = thread.bitcast(bodySimPool[index, BS_ANG_VEL_Z] + cp.int32(0), cp.float32)


# =====================================================================
# Kernel 11: getRigidDynamicLinearAcceleration
# =====================================================================
@cp.kernel
def getRigidDynamicLinearAcceleration(
    data,                  # float32[nbElements, 3]
    gpuIndices,            # int32[nbElements]
    bodySimPool,           # int32[poolSize, 60]
    prevVelocities,        # float32[poolSize, 8] -- PxgBodySimVelocities
    nbElements,
    oneOverDt,             # float32
    BLOCK_SIZE: cp.constexpr = 256
):
    """Get linear acceleration = (currentVel - prevVel) * oneOverDt."""
    numBlocks = cp.ceildiv(nbElements, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gti = bx * BLOCK_SIZE + tid
            if gti < nbElements:
                index = gpuIndices[gti] + cp.int32(0)

                curX = thread.bitcast(bodySimPool[index, BS_LIN_VEL_X] + cp.int32(0), cp.float32)
                curY = thread.bitcast(bodySimPool[index, BS_LIN_VEL_Y] + cp.int32(0), cp.float32)
                curZ = thread.bitcast(bodySimPool[index, BS_LIN_VEL_Z] + cp.int32(0), cp.float32)

                prevX = prevVelocities[index, BV_LIN_X] + cp.float32(0.0)
                prevY = prevVelocities[index, BV_LIN_Y] + cp.float32(0.0)
                prevZ = prevVelocities[index, BV_LIN_Z] + cp.float32(0.0)

                data[gti, 0] = (curX - prevX) * oneOverDt
                data[gti, 1] = (curY - prevY) * oneOverDt
                data[gti, 2] = (curZ - prevZ) * oneOverDt


# =====================================================================
# Kernel 12: getRigidDynamicAngularAcceleration
# =====================================================================
@cp.kernel
def getRigidDynamicAngularAcceleration(
    data,                  # float32[nbElements, 3]
    gpuIndices,            # int32[nbElements]
    bodySimPool,           # int32[poolSize, 60]
    prevVelocities,        # float32[poolSize, 8] -- PxgBodySimVelocities
    nbElements,
    oneOverDt,             # float32
    BLOCK_SIZE: cp.constexpr = 256
):
    """Get angular acceleration = (currentAngVel - prevAngVel) * oneOverDt."""
    numBlocks = cp.ceildiv(nbElements, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gti = bx * BLOCK_SIZE + tid
            if gti < nbElements:
                index = gpuIndices[gti] + cp.int32(0)

                curX = thread.bitcast(bodySimPool[index, BS_ANG_VEL_X] + cp.int32(0), cp.float32)
                curY = thread.bitcast(bodySimPool[index, BS_ANG_VEL_Y] + cp.int32(0), cp.float32)
                curZ = thread.bitcast(bodySimPool[index, BS_ANG_VEL_Z] + cp.int32(0), cp.float32)

                prevX = prevVelocities[index, BV_ANG_X] + cp.float32(0.0)
                prevY = prevVelocities[index, BV_ANG_Y] + cp.float32(0.0)
                prevZ = prevVelocities[index, BV_ANG_Z] + cp.float32(0.0)

                data[gti, 0] = (curX - prevX) * oneOverDt
                data[gti, 1] = (curY - prevY) * oneOverDt
                data[gti, 2] = (curZ - prevZ) * oneOverDt


# =====================================================================
# Kernel 13: setRigidDynamicGlobalPose
# =====================================================================
@cp.kernel
def setRigidDynamicGlobalPose(
    data,                  # float32[nbElements, 7] -- input PxTransform
    gpuIndices,            # int32[nbElements]
    bodySimPool,           # int32[poolSize, 60]
    # Shape update arrays (from PxgUpdateActorDataDesc)
    rigidNodeIndices,      # int32[totalNumShapes] -- sorted by nodeIndex
    shapeIndices,          # int32[totalNumShapes]
    transformCache,        # int32[cacheSize, 8] -- PxsCachedTransform as int32
    bounds,                # float32[boundsSize, 6] -- PxBounds3
    shapeSimPool,          # int32[shapePoolSize, SS_SIZE] -- PxgShapeSim
    updated,               # int32[shapePoolSize] -- boolean flag per shape
    nbElements,
    totalNumShapes,
    SS_SIZE: cp.constexpr = 16,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Set global pose: compute body2World, update shapes via binary search."""
    numBlocks = cp.ceildiv(nbElements, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gti = bx * BLOCK_SIZE + tid
            if gti < nbElements:
                index = gpuIndices[gti] + cp.int32(0)

                # Read input transform (actor pose)
                in_qx = data[gti, 0] + cp.float32(0.0)
                in_qy = data[gti, 1] + cp.float32(0.0)
                in_qz = data[gti, 2] + cp.float32(0.0)
                in_qw = data[gti, 3] + cp.float32(0.0)
                in_px = data[gti, 4] + cp.float32(0.0)
                in_py = data[gti, 5] + cp.float32(0.0)
                in_pz = data[gti, 6] + cp.float32(0.0)

                # Read body2Actor from bodySimPool
                b2a_qx = thread.bitcast(bodySimPool[index, BS_B2A_QX] + cp.int32(0), cp.float32)
                b2a_qy = thread.bitcast(bodySimPool[index, BS_B2A_QY] + cp.int32(0), cp.float32)
                b2a_qz = thread.bitcast(bodySimPool[index, BS_B2A_QZ] + cp.int32(0), cp.float32)
                b2a_qw = thread.bitcast(bodySimPool[index, BS_B2A_QW] + cp.int32(0), cp.float32)
                b2a_px = thread.bitcast(bodySimPool[index, BS_B2A_PX] + cp.int32(0), cp.float32)
                b2a_py = thread.bitcast(bodySimPool[index, BS_B2A_PY] + cp.int32(0), cp.float32)
                b2a_pz = thread.bitcast(bodySimPool[index, BS_B2A_PZ] + cp.int32(0), cp.float32)

                # body2World = inputTransform * body2Actor
                # NOTE: inline transform_multiply_q to avoid MLIR domination error in conditional
                # quat_multiply(in_q, b2a_q)
                b2w_qx = in_qw * b2a_qx + in_qx * b2a_qw + in_qy * b2a_qz - in_qz * b2a_qy
                b2w_qy = in_qw * b2a_qy - in_qx * b2a_qz + in_qy * b2a_qw + in_qz * b2a_qx
                b2w_qz = in_qw * b2a_qz + in_qx * b2a_qy - in_qy * b2a_qx + in_qz * b2a_qw
                b2w_qw = in_qw * b2a_qw - in_qx * b2a_qx - in_qy * b2a_qy - in_qz * b2a_qz
                # quat_rotate(in_q, b2a_p) + in_p
                sp_cr_x = in_qy * b2a_pz - in_qz * b2a_py
                sp_cr_y = in_qz * b2a_px - in_qx * b2a_pz
                sp_cr_z = in_qx * b2a_py - in_qy * b2a_px
                b2w_px = b2a_px + cp.float32(2.0) * (in_qw * sp_cr_x + in_qy * sp_cr_z - in_qz * sp_cr_y) + in_px
                b2w_py = b2a_py + cp.float32(2.0) * (in_qw * sp_cr_y + in_qz * sp_cr_x - in_qx * sp_cr_z) + in_py
                b2w_pz = b2a_pz + cp.float32(2.0) * (in_qw * sp_cr_z + in_qx * sp_cr_y - in_qy * sp_cr_x) + in_pz

                # Write body2World
                bodySimPool[index, BS_B2W_QX] = thread.bitcast(b2w_qx, cp.int32)
                bodySimPool[index, BS_B2W_QY] = thread.bitcast(b2w_qy, cp.int32)
                bodySimPool[index, BS_B2W_QZ] = thread.bitcast(b2w_qz, cp.int32)
                bodySimPool[index, BS_B2W_QW] = thread.bitcast(b2w_qw, cp.int32)
                bodySimPool[index, BS_B2W_PX] = thread.bitcast(b2w_px, cp.int32)
                bodySimPool[index, BS_B2W_PY] = thread.bitcast(b2w_py, cp.int32)
                bodySimPool[index, BS_B2W_PZ] = thread.bitcast(b2w_pz, cp.int32)
                bodySimPool[index, BS_B2W_PW] = thread.bitcast(cp.float32(0.0), cp.int32)

                if totalNumShapes > cp.int32(0):
                    nodeIndex = index  # PxNodeIndex(index)

                    # Binary search for the first position matching nodeIndex
                    lo = cp.int32(0)
                    hi = totalNumShapes - cp.int32(1)
                    pos = cp.int32(-1)  # 0xFFFFFFFF equivalent sentinel

                    while lo <= hi:
                        mid = (lo + hi) >> cp.int32(1)
                        midVal = rigidNodeIndices[mid] + cp.int32(0)
                        if midVal < nodeIndex:
                            lo = mid + cp.int32(1)
                        elif midVal > nodeIndex:
                            hi = mid - cp.int32(1)
                        else:
                            pos = mid
                            hi = mid - cp.int32(1)

                    # Walk backward through matching entries
                    while pos >= cp.int32(0):
                        rni = rigidNodeIndices[pos] + cp.int32(0)
                        if rni != nodeIndex:
                            pos = cp.int32(-1)
                        else:
                            shapeIdx = shapeIndices[pos] + cp.int32(0)
                            if shapeIdx != cp.int32(INVALID_ID):
                                # Read shape transform (shape2Actor)
                                s2a_qx = thread.bitcast(shapeSimPool[shapeIdx, SS_XFORM_QX] + cp.int32(0), cp.float32)
                                s2a_qy = thread.bitcast(shapeSimPool[shapeIdx, SS_XFORM_QY] + cp.int32(0), cp.float32)
                                s2a_qz = thread.bitcast(shapeSimPool[shapeIdx, SS_XFORM_QZ] + cp.int32(0), cp.float32)
                                s2a_qw = thread.bitcast(shapeSimPool[shapeIdx, SS_XFORM_QW] + cp.int32(0), cp.float32)
                                s2a_px = thread.bitcast(shapeSimPool[shapeIdx, SS_XFORM_PX] + cp.int32(0), cp.float32)
                                s2a_py = thread.bitcast(shapeSimPool[shapeIdx, SS_XFORM_PY] + cp.int32(0), cp.float32)
                                s2a_pz = thread.bitcast(shapeSimPool[shapeIdx, SS_XFORM_PZ] + cp.int32(0), cp.float32)

                                # absPos = getAbsPose(body2World, shape2Actor, body2Actor)
                                # = body2World.transform(body2Actor.transformInv(shape2Actor))
                                # NOTE: inline transform_transform_inv to avoid MLIR domination error in conditional
                                # Step 1: b2a_inv.q = conjugate(b2a_q)
                                ti_inv_qx = -b2a_qx
                                ti_inv_qy = -b2a_qy
                                ti_inv_qz = -b2a_qz
                                ti_inv_qw = b2a_qw
                                # Step 2: b2a_inv.p = conjugate(b2a_q).rotate(-b2a_p)
                                ti_neg_px = -b2a_px
                                ti_neg_py = -b2a_py
                                ti_neg_pz = -b2a_pz
                                ti_cr1_x = ti_inv_qy * ti_neg_pz - ti_inv_qz * ti_neg_py
                                ti_cr1_y = ti_inv_qz * ti_neg_px - ti_inv_qx * ti_neg_pz
                                ti_cr1_z = ti_inv_qx * ti_neg_py - ti_inv_qy * ti_neg_px
                                ti_inv_px = ti_neg_px + cp.float32(2.0) * (ti_inv_qw * ti_cr1_x + ti_inv_qy * ti_cr1_z - ti_inv_qz * ti_cr1_y)
                                ti_inv_py = ti_neg_py + cp.float32(2.0) * (ti_inv_qw * ti_cr1_y + ti_inv_qz * ti_cr1_x - ti_inv_qx * ti_cr1_z)
                                ti_inv_pz = ti_neg_pz + cp.float32(2.0) * (ti_inv_qw * ti_cr1_z + ti_inv_qx * ti_cr1_y - ti_inv_qy * ti_cr1_x)
                                # Step 3: t0 = b2a_inv.transform(s2a) = transform_multiply_q(b2a_inv, s2a)
                                # quat_multiply(ti_inv_q, s2a_q)
                                t0_qx = ti_inv_qw * s2a_qx + ti_inv_qx * s2a_qw + ti_inv_qy * s2a_qz - ti_inv_qz * s2a_qy
                                t0_qy = ti_inv_qw * s2a_qy - ti_inv_qx * s2a_qz + ti_inv_qy * s2a_qw + ti_inv_qz * s2a_qx
                                t0_qz = ti_inv_qw * s2a_qz + ti_inv_qx * s2a_qy - ti_inv_qy * s2a_qx + ti_inv_qz * s2a_qw
                                t0_qw = ti_inv_qw * s2a_qw - ti_inv_qx * s2a_qx - ti_inv_qy * s2a_qy - ti_inv_qz * s2a_qz
                                # quat_rotate(ti_inv_q, s2a_p) + ti_inv_p
                                ti_cr2_x = ti_inv_qy * s2a_pz - ti_inv_qz * s2a_py
                                ti_cr2_y = ti_inv_qz * s2a_px - ti_inv_qx * s2a_pz
                                ti_cr2_z = ti_inv_qx * s2a_py - ti_inv_qy * s2a_px
                                t0_px = s2a_px + cp.float32(2.0) * (ti_inv_qw * ti_cr2_x + ti_inv_qy * ti_cr2_z - ti_inv_qz * ti_cr2_y) + ti_inv_px
                                t0_py = s2a_py + cp.float32(2.0) * (ti_inv_qw * ti_cr2_y + ti_inv_qz * ti_cr2_x - ti_inv_qx * ti_cr2_z) + ti_inv_py
                                t0_pz = s2a_pz + cp.float32(2.0) * (ti_inv_qw * ti_cr2_z + ti_inv_qx * ti_cr2_y - ti_inv_qy * ti_cr2_x) + ti_inv_pz

                                # NOTE: inline transform_multiply_q to avoid MLIR domination error in conditional
                                # abs = b2w.transform(t0)
                                # quat_multiply(b2w_q, t0_q)
                                abs_qx = b2w_qw * t0_qx + b2w_qx * t0_qw + b2w_qy * t0_qz - b2w_qz * t0_qy
                                abs_qy = b2w_qw * t0_qy - b2w_qx * t0_qz + b2w_qy * t0_qw + b2w_qz * t0_qx
                                abs_qz = b2w_qw * t0_qz + b2w_qx * t0_qy - b2w_qy * t0_qx + b2w_qz * t0_qw
                                abs_qw = b2w_qw * t0_qw - b2w_qx * t0_qx - b2w_qy * t0_qy - b2w_qz * t0_qz
                                # quat_rotate(b2w_q, t0_p) + b2w_p
                                abs_cr_x = b2w_qy * t0_pz - b2w_qz * t0_py
                                abs_cr_y = b2w_qz * t0_px - b2w_qx * t0_pz
                                abs_cr_z = b2w_qx * t0_py - b2w_qy * t0_px
                                abs_px = t0_px + cp.float32(2.0) * (b2w_qw * abs_cr_x + b2w_qy * abs_cr_z - b2w_qz * abs_cr_y) + b2w_px
                                abs_py = t0_py + cp.float32(2.0) * (b2w_qw * abs_cr_y + b2w_qz * abs_cr_x - b2w_qx * abs_cr_z) + b2w_py
                                abs_pz = t0_pz + cp.float32(2.0) * (b2w_qw * abs_cr_z + b2w_qx * abs_cr_y - b2w_qy * abs_cr_x) + b2w_pz

                                # setTransformCache
                                transformCache[shapeIdx, CT_QX] = thread.bitcast(abs_qx, cp.int32)
                                transformCache[shapeIdx, CT_QY] = thread.bitcast(abs_qy, cp.int32)
                                transformCache[shapeIdx, CT_QZ] = thread.bitcast(abs_qz, cp.int32)
                                transformCache[shapeIdx, CT_QW] = thread.bitcast(abs_qw, cp.int32)
                                transformCache[shapeIdx, CT_PX] = thread.bitcast(abs_px, cp.int32)
                                transformCache[shapeIdx, CT_PY] = thread.bitcast(abs_py, cp.int32)
                                transformCache[shapeIdx, CT_PZ] = thread.bitcast(abs_pz, cp.int32)
                                transformCache[shapeIdx, CT_FLAGS] = cp.int32(0)

                                # updateBounds (simplified: use transformFast for all types)
                                lb_min_x = thread.bitcast(shapeSimPool[shapeIdx, SS_LBOUNDS_MIN_X] + cp.int32(0), cp.float32)
                                lb_min_y = thread.bitcast(shapeSimPool[shapeIdx, SS_LBOUNDS_MIN_Y] + cp.int32(0), cp.float32)
                                lb_min_z = thread.bitcast(shapeSimPool[shapeIdx, SS_LBOUNDS_MIN_Z] + cp.int32(0), cp.float32)
                                lb_max_x = thread.bitcast(shapeSimPool[shapeIdx, SS_LBOUNDS_MAX_X] + cp.int32(0), cp.float32)
                                lb_max_y = thread.bitcast(shapeSimPool[shapeIdx, SS_LBOUNDS_MAX_Y] + cp.int32(0), cp.float32)
                                lb_max_z = thread.bitcast(shapeSimPool[shapeIdx, SS_LBOUNDS_MAX_Z] + cp.int32(0), cp.float32)

                                # PxBounds3::transformFast(pose, localBounds)
                                # center = (min+max)/2, extents = (max-min)/2
                                cx = (lb_min_x + lb_max_x) * cp.float32(0.5)
                                cy = (lb_min_y + lb_max_y) * cp.float32(0.5)
                                cz = (lb_min_z + lb_max_z) * cp.float32(0.5)
                                ex = (lb_max_x - lb_min_x) * cp.float32(0.5)
                                ey = (lb_max_y - lb_min_y) * cp.float32(0.5)
                                ez = (lb_max_z - lb_min_z) * cp.float32(0.5)

                                # Rotate center
                                # NOTE: inline quat_rotate to avoid MLIR domination error in conditional
                                rc_cr_x = abs_qy * cz - abs_qz * cy
                                rc_cr_y = abs_qz * cx - abs_qx * cz
                                rc_cr_z = abs_qx * cy - abs_qy * cx
                                rc_x = cx + cp.float32(2.0) * (abs_qw * rc_cr_x + abs_qy * rc_cr_z - abs_qz * rc_cr_y)
                                rc_y = cy + cp.float32(2.0) * (abs_qw * rc_cr_y + abs_qz * rc_cr_x - abs_qx * rc_cr_z)
                                rc_z = cz + cp.float32(2.0) * (abs_qw * rc_cr_z + abs_qx * rc_cr_y - abs_qy * rc_cr_x)
                                rc_x = rc_x + abs_px
                                rc_y = rc_y + abs_py
                                rc_z = rc_z + abs_pz

                                # Compute rotated extents via basisExtent
                                # rotation matrix columns from quaternion
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

                                # Column 0
                                m00 = cp.float32(1.0) - (yy + zz)
                                m10 = xy + wz
                                m20 = xz - wy
                                # Column 1
                                m01 = xy - wz
                                m11 = cp.float32(1.0) - (xx + zz)
                                m21 = yz + wx
                                # Column 2
                                m02 = xz + wy
                                m12 = yz - wx
                                m22 = cp.float32(1.0) - (xx + yy)

                                # basisExtent: w.x = |c0.x|*ex + |c1.x|*ey + |c2.x|*ez
                                w_x = abs_f(m00) * ex + abs_f(m01) * ey + abs_f(m02) * ez
                                w_y = abs_f(m10) * ex + abs_f(m11) * ey + abs_f(m12) * ez
                                w_z = abs_f(m20) * ex + abs_f(m21) * ey + abs_f(m22) * ez

                                bounds[shapeIdx, BD_MIN_X] = rc_x - w_x
                                bounds[shapeIdx, BD_MIN_Y] = rc_y - w_y
                                bounds[shapeIdx, BD_MIN_Z] = rc_z - w_z
                                bounds[shapeIdx, BD_MAX_X] = rc_x + w_x
                                bounds[shapeIdx, BD_MAX_Y] = rc_y + w_y
                                bounds[shapeIdx, BD_MAX_Z] = rc_z + w_z

                                updated[shapeIdx] = cp.int32(1)

                            pos = pos - cp.int32(1)


# =====================================================================
# Kernel 14: setRigidDynamicLinearVelocity
# =====================================================================
@cp.kernel
def setRigidDynamicLinearVelocity(
    data,                  # float32[nbElements, 3] -- input PxVec3
    gpuIndices,            # int32[nbElements]
    bodySimPool,           # int32[poolSize, 60]
    prevVelocities,        # float32[poolSize, 8] or dummy (has_prevVelocities flag)
    nbElements,
    has_prevVelocities,    # int32 (0 or 1)
    BLOCK_SIZE: cp.constexpr = 256
):
    """Set linear velocity, preserving invMass. Optionally update prevVelocities."""
    numBlocks = cp.ceildiv(nbElements, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gti = bx * BLOCK_SIZE + tid
            if gti < nbElements:
                index = gpuIndices[gti] + cp.int32(0)

                vx = data[gti, 0] + cp.float32(0.0)
                vy = data[gti, 1] + cp.float32(0.0)
                vz = data[gti, 2] + cp.float32(0.0)

                # Preserve invMass.w
                # (already in pool, we only overwrite xyz)
                bodySimPool[index, BS_LIN_VEL_X] = thread.bitcast(vx, cp.int32)
                bodySimPool[index, BS_LIN_VEL_Y] = thread.bitcast(vy, cp.int32)
                bodySimPool[index, BS_LIN_VEL_Z] = thread.bitcast(vz, cp.int32)

                if has_prevVelocities != cp.int32(0):
                    invMassF = thread.bitcast(bodySimPool[index, BS_INV_MASS_W] + cp.int32(0), cp.float32)
                    prevVelocities[index, BV_LIN_X] = vx
                    prevVelocities[index, BV_LIN_Y] = vy
                    prevVelocities[index, BV_LIN_Z] = vz
                    prevVelocities[index, BV_LIN_W] = invMassF


# =====================================================================
# Kernel 15: setRigidDynamicAngularVelocity
# =====================================================================
@cp.kernel
def setRigidDynamicAngularVelocity(
    data,                  # float32[nbElements, 3] -- input PxVec3
    gpuIndices,            # int32[nbElements]
    bodySimPool,           # int32[poolSize, 60]
    prevVelocities,        # float32[poolSize, 8] or dummy
    nbElements,
    has_prevVelocities,    # int32 (0 or 1)
    BLOCK_SIZE: cp.constexpr = 256
):
    """Set angular velocity, preserving maxPenBias.w. Optionally update prevVelocities."""
    numBlocks = cp.ceildiv(nbElements, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gti = bx * BLOCK_SIZE + tid
            if gti < nbElements:
                index = gpuIndices[gti] + cp.int32(0)

                vx = data[gti, 0] + cp.float32(0.0)
                vy = data[gti, 1] + cp.float32(0.0)
                vz = data[gti, 2] + cp.float32(0.0)

                bodySimPool[index, BS_ANG_VEL_X] = thread.bitcast(vx, cp.int32)
                bodySimPool[index, BS_ANG_VEL_Y] = thread.bitcast(vy, cp.int32)
                bodySimPool[index, BS_ANG_VEL_Z] = thread.bitcast(vz, cp.int32)

                if has_prevVelocities != cp.int32(0):
                    maxPenBiasF = thread.bitcast(bodySimPool[index, BS_MAX_PEN_BIAS_W] + cp.int32(0), cp.float32)
                    prevVelocities[index, BV_ANG_X] = vx
                    prevVelocities[index, BV_ANG_Y] = vy
                    prevVelocities[index, BV_ANG_Z] = vz
                    prevVelocities[index, BV_ANG_W] = maxPenBiasF


# =====================================================================
# Kernel 16: setRigidDynamicForce
# =====================================================================
@cp.kernel
def setRigidDynamicForce(
    data,                  # float32[nbElements, 3] -- input PxVec3 force
    gpuIndices,            # int32[nbElements]
    bodySimPool,           # int32[poolSize, 60]
    nbElements,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Set force: externalLinearAcceleration = force * invMass."""
    numBlocks = cp.ceildiv(nbElements, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gti = bx * BLOCK_SIZE + tid
            if gti < nbElements:
                index = gpuIndices[gti] + cp.int32(0)

                fx = data[gti, 0] + cp.float32(0.0)
                fy = data[gti, 1] + cp.float32(0.0)
                fz = data[gti, 2] + cp.float32(0.0)

                invMass = thread.bitcast(bodySimPool[index, BS_INV_MASS_W] + cp.int32(0), cp.float32)

                bodySimPool[index, BS_EXT_LIN_ACC_X] = thread.bitcast(fx * invMass, cp.int32)
                bodySimPool[index, BS_EXT_LIN_ACC_Y] = thread.bitcast(fy * invMass, cp.int32)
                bodySimPool[index, BS_EXT_LIN_ACC_Z] = thread.bitcast(fz * invMass, cp.int32)
                bodySimPool[index, BS_EXT_LIN_ACC_W] = thread.bitcast(cp.float32(0.0), cp.int32)


# =====================================================================
# Kernel 17: setRigidDynamicTorque
# =====================================================================
@cp.kernel
def setRigidDynamicTorque(
    data,                  # float32[nbElements, 3] -- input PxVec3 torque
    gpuIndices,            # int32[nbElements]
    bodySimPool,           # int32[poolSize, 60]
    nbElements,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Set torque: externalAngularAcceleration = inverseInertiaWorldSpace * torque.
    Cm::transformInertiaTensor(invInertia, PxMat33(body2World.q)) * torque.
    """
    numBlocks = cp.ceildiv(nbElements, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gti = bx * BLOCK_SIZE + tid
            if gti < nbElements:
                index = gpuIndices[gti] + cp.int32(0)

                tx = data[gti, 0] + cp.float32(0.0)
                ty = data[gti, 1] + cp.float32(0.0)
                tz = data[gti, 2] + cp.float32(0.0)

                # Read inverse inertia diagonal
                inv_ix = thread.bitcast(bodySimPool[index, BS_INV_INERTIA_X] + cp.int32(0), cp.float32)
                inv_iy = thread.bitcast(bodySimPool[index, BS_INV_INERTIA_Y] + cp.int32(0), cp.float32)
                inv_iz = thread.bitcast(bodySimPool[index, BS_INV_INERTIA_Z] + cp.int32(0), cp.float32)

                # Read body2World quaternion for rotation matrix
                qx = thread.bitcast(bodySimPool[index, BS_B2W_QX] + cp.int32(0), cp.float32)
                qy = thread.bitcast(bodySimPool[index, BS_B2W_QY] + cp.int32(0), cp.float32)
                qz = thread.bitcast(bodySimPool[index, BS_B2W_QZ] + cp.int32(0), cp.float32)
                qw = thread.bitcast(bodySimPool[index, BS_B2W_QW] + cp.int32(0), cp.float32)

                # Build rotation matrix from quaternion
                x2 = qx + qx
                y2 = qy + qy
                z2 = qz + qz
                xx = qx * x2
                xy = qx * y2
                xz = qx * z2
                yy = qy * y2
                yz = qy * z2
                zz = qz * z2
                wx = qw * x2
                wy = qw * y2
                wz = qw * z2

                # Column 0
                m00 = cp.float32(1.0) - (yy + zz)
                m10 = xy + wz
                m20 = xz - wy
                # Column 1
                m01 = xy - wz
                m11 = cp.float32(1.0) - (xx + zz)
                m21 = yz + wx
                # Column 2
                m02 = xz + wy
                m12 = yz - wx
                m22 = cp.float32(1.0) - (xx + yy)

                # Cm::transformInertiaTensor(invInertia, rot) computes:
                # result = rot * diag(invInertia) * rot^T
                # Then result * torque
                # Equivalent: rot * (invInertia .* (rot^T * torque))

                # rot^T * torque
                rt_x = m00 * tx + m10 * ty + m20 * tz
                rt_y = m01 * tx + m11 * ty + m21 * tz
                rt_z = m02 * tx + m12 * ty + m22 * tz

                # Scale by invInertia
                s_x = inv_ix * rt_x
                s_y = inv_iy * rt_y
                s_z = inv_iz * rt_z

                # rot * scaled
                deltaX = m00 * s_x + m01 * s_y + m02 * s_z
                deltaY = m10 * s_x + m11 * s_y + m12 * s_z
                deltaZ = m20 * s_x + m21 * s_y + m22 * s_z

                bodySimPool[index, BS_EXT_ANG_ACC_X] = thread.bitcast(deltaX, cp.int32)
                bodySimPool[index, BS_EXT_ANG_ACC_Y] = thread.bitcast(deltaY, cp.int32)
                bodySimPool[index, BS_EXT_ANG_ACC_Z] = thread.bitcast(deltaZ, cp.int32)
                bodySimPool[index, BS_EXT_ANG_ACC_W] = thread.bitcast(cp.float32(0.0), cp.int32)


# =====================================================================
# Kernel 18: copyUserData
# =====================================================================
@cp.kernel
def copyUserData(
    src,           # int32[numToProcess, maxSize] -- source data per pair
    dst,           # int32[numToProcess, maxSize] -- destination data per pair
    sizes,         # int32[numToProcess] -- size in bytes for each pair
    numToProcess,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Copy user data: each pair is a src->dst memcpy of sizes[pairIdx]/4 int32s.
    Host flattens: each pair gets maxSize int32 elements.
    """
    with cp.Kernel(cp.ceildiv(numToProcess, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            pairIdx = bx * BLOCK_SIZE + tid
            if pairIdx < numToProcess:
                size = sizes[pairIdx] + cp.int32(0)
                sizeInInts = size >> cp.int32(2)
                j = cp.int32(0)
                while j < sizeInInts:
                    dst[pairIdx, j] = src[pairIdx, j]
                    j = j + cp.int32(1)


# =====================================================================
# Kernel 19: getD6JointForces
# =====================================================================
@cp.kernel
def getD6JointForces(
    data,                          # float32[nbElements, 3] -- output PxVec3
    gpuIndices,                    # int32[nbElements]
    nbElements,
    constraintWriteBackBuffer,     # float32[bufSize, 8] -- PxgConstraintWriteback
    oneOverDt,                     # float32
    constraintIdMap,               # int32[constraintIdMapSize] -- PxgConstraintIdMapEntry
    constraintIdMapSize,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Get D6 joint forces = linearImpulse * oneOverDt."""
    numBlocks = cp.ceildiv(nbElements, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gti = bx * BLOCK_SIZE + tid
            if gti < nbElements:
                index = gpuIndices[gti] + cp.int32(0)

                rx = cp.float32(0.0)
                ry = cp.float32(0.0)
                rz = cp.float32(0.0)

                valid = cp.int32(0)
                if index < constraintIdMapSize:
                    jointDataId = constraintIdMap[index] + cp.int32(0)
                    if jointDataId != cp.int32(INVALID_ID):
                        valid = cp.int32(1)

                if valid != cp.int32(0):
                    rx = constraintWriteBackBuffer[index, CW_LIN_X] * oneOverDt
                    ry = constraintWriteBackBuffer[index, CW_LIN_Y] * oneOverDt
                    rz = constraintWriteBackBuffer[index, CW_LIN_Z] * oneOverDt

                data[gti, 0] = rx
                data[gti, 1] = ry
                data[gti, 2] = rz


# =====================================================================
# Kernel 20: getD6JointTorques
# =====================================================================
@cp.kernel
def getD6JointTorques(
    data,                          # float32[nbElements, 3] -- output PxVec3
    gpuIndices,                    # int32[nbElements]
    nbElements,
    constraintWriteBackBuffer,     # float32[bufSize, 8] -- PxgConstraintWriteback
    oneOverDt,                     # float32
    constraintIdMap,               # int32[constraintIdMapSize]
    constraintIdMapSize,
    BLOCK_SIZE: cp.constexpr = 256
):
    """Get D6 joint torques = angularImpulse * oneOverDt."""
    numBlocks = cp.ceildiv(nbElements, BLOCK_SIZE)
    with cp.Kernel(numBlocks, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            gti = bx * BLOCK_SIZE + tid
            if gti < nbElements:
                index = gpuIndices[gti] + cp.int32(0)

                rx = cp.float32(0.0)
                ry = cp.float32(0.0)
                rz = cp.float32(0.0)

                valid = cp.int32(0)
                if index < constraintIdMapSize:
                    jointDataId = constraintIdMap[index] + cp.int32(0)
                    if jointDataId != cp.int32(INVALID_ID):
                        valid = cp.int32(1)

                if valid != cp.int32(0):
                    rx = constraintWriteBackBuffer[index, CW_ANG_X] * oneOverDt
                    ry = constraintWriteBackBuffer[index, CW_ANG_Y] * oneOverDt
                    rz = constraintWriteBackBuffer[index, CW_ANG_Z] * oneOverDt

                data[gti, 0] = rx
                data[gti, 1] = ry
                data[gti, 2] = rz
