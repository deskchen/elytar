"""Capybara DSL port of preIntegration.cu — preIntegrationLaunch + initStaticKinematics.

Ports both kernels from the CUDA source.  The swizzled shared-memory loading
in the original CUDA is eliminated: each thread reads PxgBodySim fields
directly from the flat tensor body_sim_buffer[nodeIndex, offset].

Struct arrays are passed as flat int32 tensors.  Float fields are loaded via
thread.bitcast(int32_val, cp.float32).  Field offsets are defined as module
constants matching the C++ ABI (see PxgSolverBody.h, PxgBodySim.h, etc.).
"""

import capybara as cp

BLOCK_SIZE = 128

# ---------------------------------------------------------------------------
# PxgSolverBodyData layout — 96 bytes = 24 int32 slots
#   float4 initialAngVelXYZ_penBiasClamp  [0..3]
#   float4 initialLinVelXYZ_invMassW      [4..7]
#   PxAlignedTransform body2World         [8..15]  (q.xyzw at 8-11, p.xyzw at 12-15)
#   PxNodeIndex islandNodeIndex           [16] (mID)  [17] (mLinkID)
#   float  reportThreshold                [18]
#   float  maxImpulse                     [19]
#   uint32 flags                          [20]
#   float  offsetSlop                     [21]
#   padding                               [22..23]
# ---------------------------------------------------------------------------
SBD_INIT_ANG_X, SBD_INIT_ANG_Y, SBD_INIT_ANG_Z, SBD_PEN_BIAS = 0, 1, 2, 3
SBD_INIT_LIN_X, SBD_INIT_LIN_Y, SBD_INIT_LIN_Z, SBD_INV_MASS = 4, 5, 6, 7
SBD_B2W_QX, SBD_B2W_QY, SBD_B2W_QZ, SBD_B2W_QW = 8, 9, 10, 11
SBD_B2W_PX, SBD_B2W_PY, SBD_B2W_PZ, SBD_B2W_PW = 12, 13, 14, 15
SBD_NODE_ID = 16
SBD_NODE_LINK_ID = 17
SBD_REPORT_THRESHOLD = 18
SBD_MAX_IMPULSE = 19
SBD_FLAGS = 20
SBD_OFFSET_SLOP = 21

# ---------------------------------------------------------------------------
# PxgSolverTxIData layout — 64 bytes = 16 int32 slots
#   PxTransform deltaBody2World:  q.xyzw [0..3], p.xyz [4..6], pad [7]
#   PxMat33 sqrtInvInertia:       col0 [8..10], col1 [11..13], col2 [14..16]
#   ACTUALLY: PxTransform is (PxQuat=16 bytes + PxVec3=12 bytes = 28 bytes)
#   Then PxMat33 starts at byte 28 = int32 offset 7.
# ---------------------------------------------------------------------------
TXI_DELTA_QX, TXI_DELTA_QY, TXI_DELTA_QZ, TXI_DELTA_QW = 0, 1, 2, 3
TXI_DELTA_PX, TXI_DELTA_PY, TXI_DELTA_PZ = 4, 5, 6
TXI_SQRT_INV_00, TXI_SQRT_INV_10, TXI_SQRT_INV_20 = 7, 8, 9
TXI_SQRT_INV_01, TXI_SQRT_INV_11, TXI_SQRT_INV_21 = 10, 11, 12
TXI_SQRT_INV_02, TXI_SQRT_INV_12, TXI_SQRT_INV_22 = 13, 14, 15

# ---------------------------------------------------------------------------
# PxgBodySim layout — 240 bytes = 60 int32 slots
#   float4 linearVelocityXYZ_inverseMassW             [0..3]
#   float4 angularVelocityXYZ_maxPenBiasW             [4..7]
#   float4 maxLinVelSq_maxAngVelSq_linDamp_angDamp    [8..11]
#   float4 invInertiaXYZ_contactReportThreshW         [12..15]
#   float4 sleepLinVelAccXYZ_freezeCountW              [16..19]
#   float4 sleepAngVelAccXYZ_accelScaleW               [20..23]
#   float4 freezeThreshX_wakeCounterY_sleepThreshZ_bodySimIdxW  [24..27]
#   PxAlignedTransform body2World                      [28..35]
#   PxAlignedTransform body2Actor_maxImpulseW          [36..43]
#   uint32 articulationRemapId                         [44]
#   uint32 internalFlags                               [45]
#   uint16 lockFlags + uint16 disableGravity packed    [46]
#   float  offsetSlop                                  [47]
#   padding                                            [48..51]
#   float4 externalLinearAcceleration                  [52..55]
#   float4 externalAngularAcceleration                 [56..59]
# ---------------------------------------------------------------------------
BS_LIN_VEL_X, BS_LIN_VEL_Y, BS_LIN_VEL_Z, BS_INV_MASS_W = 0, 1, 2, 3
BS_ANG_VEL_X, BS_ANG_VEL_Y, BS_ANG_VEL_Z, BS_MAX_PEN_BIAS_W = 4, 5, 6, 7
BS_MAX_LIN_VEL_SQ, BS_MAX_ANG_VEL_SQ, BS_LIN_DAMP, BS_ANG_DAMP = 8, 9, 10, 11
BS_INV_INERTIA_X, BS_INV_INERTIA_Y, BS_INV_INERTIA_Z, BS_CONTACT_REPORT_THRESH = 12, 13, 14, 15
BS_SLEEP_LIN_X, BS_SLEEP_LIN_Y, BS_SLEEP_LIN_Z, BS_FREEZE_COUNT = 16, 17, 18, 19
BS_SLEEP_ANG_X, BS_SLEEP_ANG_Y, BS_SLEEP_ANG_Z, BS_ACCEL_SCALE = 20, 21, 22, 23
BS_FREEZE_THRESH, BS_WAKE_COUNTER, BS_SLEEP_THRESH = 24, 25, 26
BS_B2W_QX, BS_B2W_QY, BS_B2W_QZ, BS_B2W_QW = 28, 29, 30, 31
BS_B2W_PX, BS_B2W_PY, BS_B2W_PZ = 32, 33, 34
BS_MAX_IMPULSE = 43  # body2Actor p.w field
BS_INTERNAL_FLAGS = 45
BS_LOCK_DISABLE_GRAVITY = 46  # packed: lower 16 = lockFlags, upper 16 = disableGravity
BS_OFFSET_SLOP = 47
BS_EXT_LIN_ACC_X, BS_EXT_LIN_ACC_Y, BS_EXT_LIN_ACC_Z = 52, 53, 54
BS_EXT_ANG_ACC_X, BS_EXT_ANG_ACC_Y, BS_EXT_ANG_ACC_Z = 56, 57, 58

# ---------------------------------------------------------------------------
# PxsRigidBody flag constants (from PxsRigidBody.h) — used for internalFlags
# ---------------------------------------------------------------------------
eSPECULATIVE_CCD_INTERNAL = 1 << 6   # PxsRigidBody::eSPECULATIVE_CCD
eENABLE_GYROSCOPIC_INTERNAL = 1 << 7  # PxsRigidBody::eENABLE_GYROSCOPIC

# PxRigidBodyFlag (from PxRigidBody.h) — used for solverBodyData.flags output
eENABLE_SPECULATIVE_CCD_FLAG = 1 << 4   # PxRigidBodyFlag::eENABLE_SPECULATIVE_CCD
eENABLE_GYROSCOPIC_FORCES_FLAG = 1 << 10  # PxRigidBodyFlag::eENABLE_GYROSCOPIC_FORCES

# PxRigidDynamicLockFlag (from PxRigidDynamicLockFlag.h)
eLOCK_LINEAR_X = 1 << 0
eLOCK_LINEAR_Y = 1 << 1
eLOCK_LINEAR_Z = 1 << 2
eLOCK_ANGULAR_X = 1 << 3
eLOCK_ANGULAR_Y = 1 << 4
eLOCK_ANGULAR_Z = 1 << 5

# PX_INVALID_NODE for static body check
PX_INVALID_NODE = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# preIntegrationLaunch kernel
# ---------------------------------------------------------------------------
@cp.kernel
def preIntegrationLaunch(
    offset,                # int: thread index offset
    nb_solver_bodies,      # int: number of solver bodies
    dt,                    # float: timestep
    gravity_x,             # float: gravity X
    gravity_y,             # float: gravity Y
    gravity_z,             # float: gravity Z
    solver_body_data,      # int32[N, 24]: PxgSolverBodyData AoS
    solver_txidata,        # int32[N, 16]: PxgSolverTxIData AoS
    body_sim_buffer,       # int32[M, 60]: PxgBodySim AoS, indexed by nodeIndex
    island_node_indices,   # int32[N, 2]: PxNodeIndex array (mID, mLinkID per entry)
    out_transforms,        # int32[N, 8]: PxAlignedTransform output (q.xyzw + p.xyzw)
    out_velocity_pool,     # int32[2N, 4]: output velocities (lin at [0..N), ang at [N..2N))
    solver_body_indices,   # int32[M]: nodeIndex -> solver body index mapping
    grid_x,                # int: grid X dimension
):
    with cp.Kernel(grid_x, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            idx = bx * BLOCK_SIZE + tid
            a = idx + offset

            if a < nb_solver_bodies:
                # ---- Read island node index ----
                node_index = island_node_indices[a, 0] + cp.int32(0)

                # Write solver body index mapping
                solver_body_indices[node_index] = a

                ni = node_index

                # ---- Read PxgBodySim fields directly (no shared memory) ----
                lin_vel_x = thread.bitcast(body_sim_buffer[ni, BS_LIN_VEL_X] + cp.int32(0), cp.float32)
                lin_vel_y = thread.bitcast(body_sim_buffer[ni, BS_LIN_VEL_Y] + cp.int32(0), cp.float32)
                lin_vel_z = thread.bitcast(body_sim_buffer[ni, BS_LIN_VEL_Z] + cp.int32(0), cp.float32)
                inv_mass = thread.bitcast(body_sim_buffer[ni, BS_INV_MASS_W] + cp.int32(0), cp.float32)

                ang_vel_x = thread.bitcast(body_sim_buffer[ni, BS_ANG_VEL_X] + cp.int32(0), cp.float32)
                ang_vel_y = thread.bitcast(body_sim_buffer[ni, BS_ANG_VEL_Y] + cp.int32(0), cp.float32)
                ang_vel_z = thread.bitcast(body_sim_buffer[ni, BS_ANG_VEL_Z] + cp.int32(0), cp.float32)
                max_pen_bias = thread.bitcast(body_sim_buffer[ni, BS_MAX_PEN_BIAS_W] + cp.int32(0), cp.float32)

                max_lin_vel_sq = thread.bitcast(body_sim_buffer[ni, BS_MAX_LIN_VEL_SQ] + cp.int32(0), cp.float32)
                max_ang_vel_sq = thread.bitcast(body_sim_buffer[ni, BS_MAX_ANG_VEL_SQ] + cp.int32(0), cp.float32)
                lin_damp = thread.bitcast(body_sim_buffer[ni, BS_LIN_DAMP] + cp.int32(0), cp.float32)
                ang_damp = thread.bitcast(body_sim_buffer[ni, BS_ANG_DAMP] + cp.int32(0), cp.float32)

                inv_inertia_x = thread.bitcast(body_sim_buffer[ni, BS_INV_INERTIA_X] + cp.int32(0), cp.float32)
                inv_inertia_y = thread.bitcast(body_sim_buffer[ni, BS_INV_INERTIA_Y] + cp.int32(0), cp.float32)
                inv_inertia_z = thread.bitcast(body_sim_buffer[ni, BS_INV_INERTIA_Z] + cp.int32(0), cp.float32)
                contact_report_thresh = thread.bitcast(body_sim_buffer[ni, BS_CONTACT_REPORT_THRESH] + cp.int32(0), cp.float32)

                accel_scale = thread.bitcast(body_sim_buffer[ni, BS_ACCEL_SCALE] + cp.int32(0), cp.float32)

                b2w_qx = thread.bitcast(body_sim_buffer[ni, BS_B2W_QX] + cp.int32(0), cp.float32)
                b2w_qy = thread.bitcast(body_sim_buffer[ni, BS_B2W_QY] + cp.int32(0), cp.float32)
                b2w_qz = thread.bitcast(body_sim_buffer[ni, BS_B2W_QZ] + cp.int32(0), cp.float32)
                b2w_qw = thread.bitcast(body_sim_buffer[ni, BS_B2W_QW] + cp.int32(0), cp.float32)
                b2w_px = thread.bitcast(body_sim_buffer[ni, BS_B2W_PX] + cp.int32(0), cp.float32)
                b2w_py = thread.bitcast(body_sim_buffer[ni, BS_B2W_PY] + cp.int32(0), cp.float32)
                b2w_pz = thread.bitcast(body_sim_buffer[ni, BS_B2W_PZ] + cp.int32(0), cp.float32)

                max_impulse = thread.bitcast(body_sim_buffer[ni, BS_MAX_IMPULSE] + cp.int32(0), cp.float32)
                internal_flags = body_sim_buffer[ni, BS_INTERNAL_FLAGS] + cp.int32(0)

                lock_disable = body_sim_buffer[ni, BS_LOCK_DISABLE_GRAVITY] + cp.int32(0)
                lock_flags = lock_disable & cp.int32(0xFFFF)
                disable_gravity = (lock_disable >> cp.int32(16)) & cp.int32(0xFFFF)

                offset_slop = thread.bitcast(body_sim_buffer[ni, BS_OFFSET_SLOP] + cp.int32(0), cp.float32)

                lin_acc_x = thread.bitcast(body_sim_buffer[ni, BS_EXT_LIN_ACC_X] + cp.int32(0), cp.float32)
                lin_acc_y = thread.bitcast(body_sim_buffer[ni, BS_EXT_LIN_ACC_Y] + cp.int32(0), cp.float32)
                lin_acc_z = thread.bitcast(body_sim_buffer[ni, BS_EXT_LIN_ACC_Z] + cp.int32(0), cp.float32)
                ang_acc_x = thread.bitcast(body_sim_buffer[ni, BS_EXT_ANG_ACC_X] + cp.int32(0), cp.float32)
                ang_acc_y = thread.bitcast(body_sim_buffer[ni, BS_EXT_ANG_ACC_Y] + cp.int32(0), cp.float32)
                ang_acc_z = thread.bitcast(body_sim_buffer[ni, BS_EXT_ANG_ACC_Z] + cp.int32(0), cp.float32)

                # ============================================================
                # bodyCoreComputeUnconstrainedVelocity
                # ============================================================

                # Gravity acceleration (if not disabled)
                grav_acc_x = cp.float32(0.0)
                grav_acc_y = cp.float32(0.0)
                grav_acc_z = cp.float32(0.0)
                if disable_gravity == cp.int32(0):
                    grav_acc_x = gravity_x * accel_scale * dt
                    grav_acc_y = gravity_y * accel_scale * dt
                    grav_acc_z = gravity_z * accel_scale * dt

                # linearAccelTimesDT = gravityAccel + externalLinAccel * dt
                lin_acc_dt_x = grav_acc_x + lin_acc_x * dt
                lin_acc_dt_y = grav_acc_y + lin_acc_y * dt
                lin_acc_dt_z = grav_acc_z + lin_acc_z * dt

                # angularAccelTimesDT = externalAngAccel * dt
                ang_acc_dt_x = ang_acc_x * dt
                ang_acc_dt_y = ang_acc_y * dt
                ang_acc_dt_z = ang_acc_z * dt

                # Apply accelerations
                lin_vel_x = lin_vel_x + lin_acc_dt_x
                lin_vel_y = lin_vel_y + lin_acc_dt_y
                lin_vel_z = lin_vel_z + lin_acc_dt_z
                ang_vel_x = ang_vel_x + ang_acc_dt_x
                ang_vel_y = ang_vel_y + ang_acc_dt_y
                ang_vel_z = ang_vel_z + ang_acc_dt_z

                # Apply damping: fsel(x, x, 0) = max(x, 0) when x >= 0 return x else 0
                one_minus_lin_damp_dt = cp.float32(1.0) - lin_damp * dt
                one_minus_ang_damp_dt = cp.float32(1.0) - ang_damp * dt

                lin_multiplier = one_minus_lin_damp_dt
                if one_minus_lin_damp_dt < cp.float32(0.0):
                    lin_multiplier = cp.float32(0.0)
                ang_multiplier = one_minus_ang_damp_dt
                if one_minus_ang_damp_dt < cp.float32(0.0):
                    ang_multiplier = cp.float32(0.0)

                lin_vel_x = lin_vel_x * lin_multiplier
                lin_vel_y = lin_vel_y * lin_multiplier
                lin_vel_z = lin_vel_z * lin_multiplier
                ang_vel_x = ang_vel_x * ang_multiplier
                ang_vel_y = ang_vel_y * ang_multiplier
                ang_vel_z = ang_vel_z * ang_multiplier

                # Clamp angular velocity
                ang_vel_sq = ang_vel_x * ang_vel_x + ang_vel_y * ang_vel_y + ang_vel_z * ang_vel_z
                if ang_vel_sq > max_ang_vel_sq:
                    ang_scale = thread.sqrt(max_ang_vel_sq / ang_vel_sq)
                    ang_vel_x = ang_vel_x * ang_scale
                    ang_vel_y = ang_vel_y * ang_scale
                    ang_vel_z = ang_vel_z * ang_scale

                # Clamp linear velocity
                lin_vel_sq = lin_vel_x * lin_vel_x + lin_vel_y * lin_vel_y + lin_vel_z * lin_vel_z
                if lin_vel_sq > max_lin_vel_sq:
                    lin_scale = thread.sqrt(max_lin_vel_sq / lin_vel_sq)
                    lin_vel_x = lin_vel_x * lin_scale
                    lin_vel_y = lin_vel_y * lin_scale
                    lin_vel_z = lin_vel_z * lin_scale

                # Apply lock flags
                if lock_flags & cp.int32(eLOCK_LINEAR_X):
                    lin_vel_x = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_LINEAR_Y):
                    lin_vel_y = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_LINEAR_Z):
                    lin_vel_z = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_ANGULAR_X):
                    ang_vel_x = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_ANGULAR_Y):
                    ang_vel_y = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_ANGULAR_Z):
                    ang_vel_z = cp.float32(0.0)

                # ============================================================
                # computeSafeSqrtInertia
                # ============================================================
                safe_sqrt_inv_x = cp.float32(0.0)
                if inv_inertia_x != cp.float32(0.0):
                    safe_sqrt_inv_x = thread.sqrt(inv_inertia_x)
                safe_sqrt_inv_y = cp.float32(0.0)
                if inv_inertia_y != cp.float32(0.0):
                    safe_sqrt_inv_y = thread.sqrt(inv_inertia_y)
                safe_sqrt_inv_z = cp.float32(0.0)
                if inv_inertia_z != cp.float32(0.0):
                    safe_sqrt_inv_z = thread.sqrt(inv_inertia_z)

                # ============================================================
                # Build rotation matrix from body2World quaternion
                # PxMat33(PxQuat) — column-major rotation matrix
                # ============================================================
                qx = b2w_qx
                qy = b2w_qy
                qz = b2w_qz
                qw = b2w_qw

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

                # Rotation matrix M (column-major: M[row][col])
                # Column 0
                rot_00 = cp.float32(1.0) - (yy + zz)
                rot_10 = xy + wz
                rot_20 = xz - wy
                # Column 1
                rot_01 = xy - wz
                rot_11 = cp.float32(1.0) - (xx + zz)
                rot_21 = yz + wx
                # Column 2
                rot_02 = xz + wy
                rot_12 = yz - wx
                rot_22 = cp.float32(1.0) - (xx + yy)

                # ============================================================
                # transformInertiaTensor: result = M * diag(invD) * M^T
                # invD = safeSqrtInvInertia
                # ============================================================
                axx = safe_sqrt_inv_x * rot_00
                axy = safe_sqrt_inv_x * rot_10
                axz = safe_sqrt_inv_x * rot_20
                byx = safe_sqrt_inv_y * rot_01
                byy = safe_sqrt_inv_y * rot_11
                byz = safe_sqrt_inv_y * rot_21
                czx = safe_sqrt_inv_z * rot_02
                czy = safe_sqrt_inv_z * rot_12
                czz = safe_sqrt_inv_z * rot_22

                sqrt_inv_00 = axx * rot_00 + byx * rot_01 + czx * rot_02
                sqrt_inv_11 = axy * rot_10 + byy * rot_11 + czy * rot_12
                sqrt_inv_22 = axz * rot_20 + byz * rot_21 + czz * rot_22

                sqrt_inv_01 = axx * rot_10 + byx * rot_11 + czx * rot_12
                sqrt_inv_10 = sqrt_inv_01
                sqrt_inv_02 = axx * rot_20 + byx * rot_21 + czx * rot_22
                sqrt_inv_20 = sqrt_inv_02
                sqrt_inv_12 = axy * rot_20 + byy * rot_21 + czy * rot_22
                sqrt_inv_21 = sqrt_inv_12

                # ============================================================
                # Write linear velocity to gOutVelocityPool[a]
                # ============================================================
                out_velocity_pool[a, 0] = thread.bitcast(lin_vel_x, cp.int32)
                out_velocity_pool[a, 1] = thread.bitcast(lin_vel_y, cp.int32)
                out_velocity_pool[a, 2] = thread.bitcast(lin_vel_z, cp.int32)
                out_velocity_pool[a, 3] = thread.bitcast(inv_mass, cp.int32)

                # ============================================================
                # Compute sqrtInertia (inverse of safeSqrtInvInertia) for momocity
                # ============================================================
                sqrt_inertia_x = cp.float32(0.0)
                if safe_sqrt_inv_x != cp.float32(0.0):
                    sqrt_inertia_x = cp.float32(1.0) / safe_sqrt_inv_x
                sqrt_inertia_y = cp.float32(0.0)
                if safe_sqrt_inv_y != cp.float32(0.0):
                    sqrt_inertia_y = cp.float32(1.0) / safe_sqrt_inv_y
                sqrt_inertia_z = cp.float32(0.0)
                if safe_sqrt_inv_z != cp.float32(0.0):
                    sqrt_inertia_z = cp.float32(1.0) / safe_sqrt_inv_z

                # transformInertiaTensor for sqrtInertia: M * diag(sqrtInertiaV) * M^T
                si_axx = sqrt_inertia_x * rot_00
                si_axy = sqrt_inertia_x * rot_10
                si_axz = sqrt_inertia_x * rot_20
                si_byx = sqrt_inertia_y * rot_01
                si_byy = sqrt_inertia_y * rot_11
                si_byz = sqrt_inertia_y * rot_21
                si_czx = sqrt_inertia_z * rot_02
                si_czy = sqrt_inertia_z * rot_12
                si_czz = sqrt_inertia_z * rot_22

                si_00 = si_axx * rot_00 + si_byx * rot_01 + si_czx * rot_02
                si_11 = si_axy * rot_10 + si_byy * rot_11 + si_czy * rot_12
                si_22 = si_axz * rot_20 + si_byz * rot_21 + si_czz * rot_22
                si_01 = si_axx * rot_10 + si_byx * rot_11 + si_czx * rot_12
                si_10 = si_01
                si_02 = si_axx * rot_20 + si_byx * rot_21 + si_czx * rot_22
                si_20 = si_02
                si_12 = si_axy * rot_20 + si_byy * rot_21 + si_czy * rot_22
                si_21 = si_12

                # ============================================================
                # Gyroscopic forces (optional)
                # ============================================================
                out_ang_x = ang_vel_x
                out_ang_y = ang_vel_y
                out_ang_z = ang_vel_z

                if internal_flags & cp.int32(eENABLE_GYROSCOPIC_INTERNAL):
                    # Compute local inertia (1/invInertia, or 0 if invInertia==0)
                    local_inertia_x = cp.float32(0.0)
                    if inv_inertia_x != cp.float32(0.0):
                        local_inertia_x = cp.float32(1.0) / inv_inertia_x
                    local_inertia_y = cp.float32(0.0)
                    if inv_inertia_y != cp.float32(0.0):
                        local_inertia_y = cp.float32(1.0) / inv_inertia_y
                    local_inertia_z = cp.float32(0.0)
                    if inv_inertia_z != cp.float32(0.0):
                        local_inertia_z = cp.float32(1.0) / inv_inertia_z

                    # body2World.q.rotateInv(angVel) = conjugate(q).rotate(angVel)
                    # conjugate: (-qx, -qy, -qz, qw)
                    cqx = -qx
                    cqy = -qy
                    cqz = -qz
                    cqw = qw
                    # rotate angVel by conjugate quaternion
                    cx0 = cqy * out_ang_z - cqz * out_ang_y
                    cy0 = cqz * out_ang_x - cqx * out_ang_z
                    cz0 = cqx * out_ang_y - cqy * out_ang_x
                    local_ang_x = out_ang_x + cp.float32(2.0) * (cqw * cx0 + cqy * cz0 - cqz * cy0)
                    local_ang_y = out_ang_y + cp.float32(2.0) * (cqw * cy0 + cqz * cx0 - cqx * cz0)
                    local_ang_z = out_ang_z + cp.float32(2.0) * (cqw * cz0 + cqx * cy0 - cqy * cx0)

                    # origMom = localInertia * localAngVel (component-wise)
                    orig_mom_x = local_inertia_x * local_ang_x
                    orig_mom_y = local_inertia_y * local_ang_y
                    orig_mom_z = local_inertia_z * local_ang_z

                    # torque = -localAngVel x origMom
                    torque_x = -(local_ang_y * orig_mom_z - local_ang_z * orig_mom_y)
                    torque_y = -(local_ang_z * orig_mom_x - local_ang_x * orig_mom_z)
                    torque_z = -(local_ang_x * orig_mom_y - local_ang_y * orig_mom_x)

                    # newMom = origMom + torque * dt
                    new_mom_x = orig_mom_x + torque_x * dt
                    new_mom_y = orig_mom_y + torque_y * dt
                    new_mom_z = orig_mom_z + torque_z * dt

                    # ratio = |origMom| / |newMom|
                    orig_mom_mag = thread.sqrt(orig_mom_x * orig_mom_x + orig_mom_y * orig_mom_y + orig_mom_z * orig_mom_z)
                    new_mom_mag = thread.sqrt(new_mom_x * new_mom_x + new_mom_y * new_mom_y + new_mom_z * new_mom_z)

                    gyro_ratio = cp.float32(0.0)
                    if new_mom_mag > cp.float32(0.0):
                        gyro_ratio = orig_mom_mag / new_mom_mag
                    new_mom_x = new_mom_x * gyro_ratio
                    new_mom_y = new_mom_y * gyro_ratio
                    new_mom_z = new_mom_z * gyro_ratio

                    # newDeltaAngVel = body2World.q.rotate(invInertia * newMom - localAngVel)
                    delta_x = inv_inertia_x * new_mom_x - local_ang_x
                    delta_y = inv_inertia_y * new_mom_y - local_ang_y
                    delta_z = inv_inertia_z * new_mom_z - local_ang_z

                    # rotate delta by body2World.q
                    rx0 = qy * delta_z - qz * delta_y
                    ry0 = qz * delta_x - qx * delta_z
                    rz0 = qx * delta_y - qy * delta_x
                    rot_delta_x = delta_x + cp.float32(2.0) * (qw * rx0 + qy * rz0 - qz * ry0)
                    rot_delta_y = delta_y + cp.float32(2.0) * (qw * ry0 + qz * rx0 - qx * rz0)
                    rot_delta_z = delta_z + cp.float32(2.0) * (qw * rz0 + qx * ry0 - qy * rx0)

                    out_ang_x = out_ang_x + rot_delta_x
                    out_ang_y = out_ang_y + rot_delta_y
                    out_ang_z = out_ang_z + rot_delta_z

                # ============================================================
                # Write angular velocity in momocity format:
                # angVel_momocity = sqrtInertia * angVel
                # ============================================================
                mom_x = si_00 * out_ang_x + si_01 * out_ang_y + si_02 * out_ang_z
                mom_y = si_10 * out_ang_x + si_11 * out_ang_y + si_12 * out_ang_z
                mom_z = si_20 * out_ang_x + si_21 * out_ang_y + si_22 * out_ang_z

                out_velocity_pool[a + nb_solver_bodies, 0] = thread.bitcast(mom_x, cp.int32)
                out_velocity_pool[a + nb_solver_bodies, 1] = thread.bitcast(mom_y, cp.int32)
                out_velocity_pool[a + nb_solver_bodies, 2] = thread.bitcast(mom_z, cp.int32)
                out_velocity_pool[a + nb_solver_bodies, 3] = thread.bitcast(max_pen_bias, cp.int32)

                # ============================================================
                # Write output transform gTransforms[a]
                # ============================================================
                out_transforms[a, 0] = thread.bitcast(b2w_qx, cp.int32)
                out_transforms[a, 1] = thread.bitcast(b2w_qy, cp.int32)
                out_transforms[a, 2] = thread.bitcast(b2w_qz, cp.int32)
                out_transforms[a, 3] = thread.bitcast(b2w_qw, cp.int32)
                out_transforms[a, 4] = thread.bitcast(b2w_px, cp.int32)
                out_transforms[a, 5] = thread.bitcast(b2w_py, cp.int32)
                out_transforms[a, 6] = thread.bitcast(b2w_pz, cp.int32)
                out_transforms[a, 7] = cp.int32(0)

                # ============================================================
                # Write PxgSolverBodyData[a]
                # ============================================================
                solver_body_data[a, SBD_B2W_QX] = thread.bitcast(b2w_qx, cp.int32)
                solver_body_data[a, SBD_B2W_QY] = thread.bitcast(b2w_qy, cp.int32)
                solver_body_data[a, SBD_B2W_QZ] = thread.bitcast(b2w_qz, cp.int32)
                solver_body_data[a, SBD_B2W_QW] = thread.bitcast(b2w_qw, cp.int32)
                solver_body_data[a, SBD_B2W_PX] = thread.bitcast(b2w_px, cp.int32)
                solver_body_data[a, SBD_B2W_PY] = thread.bitcast(b2w_py, cp.int32)
                solver_body_data[a, SBD_B2W_PZ] = thread.bitcast(b2w_pz, cp.int32)
                solver_body_data[a, SBD_B2W_PW] = cp.int32(0)

                solver_body_data[a, SBD_INIT_LIN_X] = thread.bitcast(lin_vel_x, cp.int32)
                solver_body_data[a, SBD_INIT_LIN_Y] = thread.bitcast(lin_vel_y, cp.int32)
                solver_body_data[a, SBD_INIT_LIN_Z] = thread.bitcast(lin_vel_z, cp.int32)
                solver_body_data[a, SBD_INV_MASS] = thread.bitcast(inv_mass, cp.int32)

                solver_body_data[a, SBD_INIT_ANG_X] = thread.bitcast(out_ang_x, cp.int32)
                solver_body_data[a, SBD_INIT_ANG_Y] = thread.bitcast(out_ang_y, cp.int32)
                solver_body_data[a, SBD_INIT_ANG_Z] = thread.bitcast(out_ang_z, cp.int32)
                solver_body_data[a, SBD_PEN_BIAS] = thread.bitcast(max_pen_bias, cp.int32)

                solver_body_data[a, SBD_OFFSET_SLOP] = thread.bitcast(offset_slop, cp.int32)
                solver_body_data[a, SBD_REPORT_THRESHOLD] = thread.bitcast(contact_report_thresh, cp.int32)
                solver_body_data[a, SBD_NODE_ID] = node_index
                solver_body_data[a, SBD_NODE_LINK_ID] = cp.int32(0)
                solver_body_data[a, SBD_MAX_IMPULSE] = thread.bitcast(max_impulse, cp.int32)

                # Compute and write flags
                out_flags = cp.int32(0)
                if internal_flags & cp.int32(eSPECULATIVE_CCD_INTERNAL):
                    out_flags = out_flags | cp.int32(eENABLE_SPECULATIVE_CCD_FLAG)
                if internal_flags & cp.int32(eENABLE_GYROSCOPIC_INTERNAL):
                    out_flags = out_flags | cp.int32(eENABLE_GYROSCOPIC_FORCES_FLAG)
                solver_body_data[a, SBD_FLAGS] = out_flags

                # ============================================================
                # Write PxgSolverTxIData[a]
                # ============================================================
                solver_txidata[a, TXI_SQRT_INV_00] = thread.bitcast(sqrt_inv_00, cp.int32)
                solver_txidata[a, TXI_SQRT_INV_10] = thread.bitcast(sqrt_inv_10, cp.int32)
                solver_txidata[a, TXI_SQRT_INV_20] = thread.bitcast(sqrt_inv_20, cp.int32)
                solver_txidata[a, TXI_SQRT_INV_01] = thread.bitcast(sqrt_inv_01, cp.int32)
                solver_txidata[a, TXI_SQRT_INV_11] = thread.bitcast(sqrt_inv_11, cp.int32)
                solver_txidata[a, TXI_SQRT_INV_21] = thread.bitcast(sqrt_inv_21, cp.int32)
                solver_txidata[a, TXI_SQRT_INV_02] = thread.bitcast(sqrt_inv_02, cp.int32)
                solver_txidata[a, TXI_SQRT_INV_12] = thread.bitcast(sqrt_inv_12, cp.int32)
                solver_txidata[a, TXI_SQRT_INV_22] = thread.bitcast(sqrt_inv_22, cp.int32)

                # deltaBody2World = PxTransform(PxIdentity): q=(0,0,0,1), p=(0,0,0)
                solver_txidata[a, TXI_DELTA_QX] = cp.int32(0)
                solver_txidata[a, TXI_DELTA_QY] = cp.int32(0)
                solver_txidata[a, TXI_DELTA_QZ] = cp.int32(0)
                solver_txidata[a, TXI_DELTA_QW] = thread.bitcast(cp.float32(1.0), cp.int32)
                solver_txidata[a, TXI_DELTA_PX] = cp.int32(0)
                solver_txidata[a, TXI_DELTA_PY] = cp.int32(0)
                solver_txidata[a, TXI_DELTA_PZ] = cp.int32(0)


# ---------------------------------------------------------------------------
# initStaticKinematics kernel
# ---------------------------------------------------------------------------
@cp.kernel
def initStaticKinematics(
    nb_static_kinematics,  # int: number of static/kinematic bodies
    nb_solver_bodies,      # int: total solver bodies (for angular velocity offset)
    solver_body_data,      # int32[N, 24]: PxgSolverBodyData AoS
    solver_txidata,        # int32[N, 16]: PxgSolverTxIData AoS
    out_transforms,        # int32[N, 8]: PxAlignedTransform output
    out_velocity_pool,     # int32[2N, 4]: output velocities
    active_node_indices,   # int32[N, 2]: PxNodeIndex array (mID, mLinkID per entry)
    solver_body_indices,   # int32[M]: nodeIndex -> solver body index mapping
    grid_x,                # int: grid X dimension
):
    with cp.Kernel(grid_x, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            idx = bx * BLOCK_SIZE + tid

            if idx < nb_static_kinematics:
                # Check if not a static body (mID != PX_INVALID_NODE)
                node_id = active_node_indices[idx, 0] + cp.int32(0)
                if node_id != cp.int32(PX_INVALID_NODE):
                    solver_body_indices[node_id] = idx

                # Copy body2World from solverBodyData to gTransforms
                out_transforms[idx, 0] = solver_body_data[idx, SBD_B2W_QX] + cp.int32(0)
                out_transforms[idx, 1] = solver_body_data[idx, SBD_B2W_QY] + cp.int32(0)
                out_transforms[idx, 2] = solver_body_data[idx, SBD_B2W_QZ] + cp.int32(0)
                out_transforms[idx, 3] = solver_body_data[idx, SBD_B2W_QW] + cp.int32(0)
                out_transforms[idx, 4] = solver_body_data[idx, SBD_B2W_PX] + cp.int32(0)
                out_transforms[idx, 5] = solver_body_data[idx, SBD_B2W_PY] + cp.int32(0)
                out_transforms[idx, 6] = solver_body_data[idx, SBD_B2W_PZ] + cp.int32(0)
                out_transforms[idx, 7] = solver_body_data[idx, SBD_B2W_PW] + cp.int32(0)

                # Copy initialLinVelXYZ_invMassW to gOutVelocityPool[idx]
                out_velocity_pool[idx, 0] = solver_body_data[idx, SBD_INIT_LIN_X] + cp.int32(0)
                out_velocity_pool[idx, 1] = solver_body_data[idx, SBD_INIT_LIN_Y] + cp.int32(0)
                out_velocity_pool[idx, 2] = solver_body_data[idx, SBD_INIT_LIN_Z] + cp.int32(0)
                out_velocity_pool[idx, 3] = solver_body_data[idx, SBD_INV_MASS] + cp.int32(0)

                # Copy initialAngVelXYZ_penBiasClamp to gOutVelocityPool[idx + nbSolverBodies]
                out_velocity_pool[idx + nb_solver_bodies, 0] = solver_body_data[idx, SBD_INIT_ANG_X] + cp.int32(0)
                out_velocity_pool[idx + nb_solver_bodies, 1] = solver_body_data[idx, SBD_INIT_ANG_Y] + cp.int32(0)
                out_velocity_pool[idx + nb_solver_bodies, 2] = solver_body_data[idx, SBD_INIT_ANG_Z] + cp.int32(0)
                out_velocity_pool[idx + nb_solver_bodies, 3] = solver_body_data[idx, SBD_PEN_BIAS] + cp.int32(0)

                # Zero deltaBody2World: q = (0,0,0,1), p = (0,0,0)
                solver_txidata[idx, TXI_DELTA_QX] = cp.int32(0)
                solver_txidata[idx, TXI_DELTA_QY] = cp.int32(0)
                solver_txidata[idx, TXI_DELTA_QZ] = cp.int32(0)
                solver_txidata[idx, TXI_DELTA_QW] = thread.bitcast(cp.float32(1.0), cp.int32)
                solver_txidata[idx, TXI_DELTA_PX] = cp.int32(0)
                solver_txidata[idx, TXI_DELTA_PY] = cp.int32(0)
                solver_txidata[idx, TXI_DELTA_PZ] = cp.int32(0)

                # Zero sqrtInvInertia (PxMat33(PxZero))
                solver_txidata[idx, TXI_SQRT_INV_00] = cp.int32(0)
                solver_txidata[idx, TXI_SQRT_INV_10] = cp.int32(0)
                solver_txidata[idx, TXI_SQRT_INV_20] = cp.int32(0)
                solver_txidata[idx, TXI_SQRT_INV_01] = cp.int32(0)
                solver_txidata[idx, TXI_SQRT_INV_11] = cp.int32(0)
                solver_txidata[idx, TXI_SQRT_INV_21] = cp.int32(0)
                solver_txidata[idx, TXI_SQRT_INV_02] = cp.int32(0)
                solver_txidata[idx, TXI_SQRT_INV_12] = cp.int32(0)
                solver_txidata[idx, TXI_SQRT_INV_22] = cp.int32(0)
