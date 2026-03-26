"""Capybara DSL port of integration.cu — integrateCoreParallelLaunch.

Ports the rigid-body velocity integration and pose update kernel.
The host adapter in PxgCudaSolverCore.cpp unpacks PxgSolverCoreDesc and
PxgSolverSharedDesc into flat tensor/scalar arguments.

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

# ---------------------------------------------------------------------------
# PxgSolverTxIData layout — 64 bytes = 16 int32 slots
#   PxTransform deltaBody2World:  q.xyzw [0..3], p.xyz [4..6], pad [7]
#   PxMat33 sqrtInvInertia:       col0 [8..10], col1 [11..13], col2 [14..16]
#   ACTUALLY: PxTransform is (PxQuat=16 bytes + PxVec3=12 bytes = 28 bytes)
#   Then PxMat33 starts at byte 28 = int32 offset 7.
# ---------------------------------------------------------------------------
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
BS_ANG_VEL_X, BS_ANG_VEL_Y, BS_ANG_VEL_Z = 4, 5, 6
BS_INV_INERTIA_X, BS_INV_INERTIA_Y, BS_INV_INERTIA_Z = 12, 13, 14
BS_SLEEP_LIN_X, BS_SLEEP_LIN_Y, BS_SLEEP_LIN_Z, BS_FREEZE_COUNT = 16, 17, 18, 19
BS_SLEEP_ANG_X, BS_SLEEP_ANG_Y, BS_SLEEP_ANG_Z, BS_ACCEL_SCALE = 20, 21, 22, 23
BS_FREEZE_THRESH, BS_WAKE_COUNTER, BS_SLEEP_THRESH = 24, 25, 26
BS_B2W_QX, BS_B2W_QY, BS_B2W_QZ, BS_B2W_QW = 28, 29, 30, 31
BS_B2W_PX, BS_B2W_PY, BS_B2W_PZ = 32, 33, 34
BS_INTERNAL_FLAGS = 45
BS_LOCK_DISABLE_GRAVITY = 46  # packed: lower 16 = lockFlags, upper 16 = disableGravity
BS_EXT_LIN_ACC_X, BS_EXT_LIN_ACC_Y, BS_EXT_LIN_ACC_Z = 52, 53, 54
BS_EXT_ANG_ACC_X, BS_EXT_ANG_ACC_Y, BS_EXT_ANG_ACC_Z = 56, 57, 58

# ---------------------------------------------------------------------------
# PxgSolverBodySleepData layout — 8 bytes = 2 int32 slots
#   float  wakeCounter     [0]
#   uint32 internalFlags   [1]
# ---------------------------------------------------------------------------
SD_WAKE_COUNTER = 0
SD_INTERNAL_FLAGS = 1

# ---------------------------------------------------------------------------
# PxgBodySimVelocities layout — 32 bytes = 8 int32/float32 slots
#   float4 linearVelocity   [0..3]
#   float4 angularVelocity  [4..7]
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PxsRigidBody flag constants (from PxsRigidBody.h)
# ---------------------------------------------------------------------------
eDISABLE_GRAVITY_GPU    = 1 << 3   # 8
eFROZEN                 = 1 << 4   # 16
eENABLE_GYROSCOPIC      = 1 << 8   # 256
eRETAIN_ACCELERATION    = 1 << 9   # 512
eFREEZE_THIS_FRAME      = 1 << 5   # 32
eUNFREEZE_THIS_FRAME    = 1 << 6   # 64
eACTIVATE_THIS_FRAME    = 1 << 7   # 128
eDEACTIVATE_THIS_FRAME  = 1 << 10  # 1024
SLEEP_FLAG_MASK         = eDISABLE_GRAVITY_GPU | eFROZEN | eENABLE_GYROSCOPIC | eRETAIN_ACCELERATION
UNFROZEN_MASK           = eDISABLE_GRAVITY_GPU | eENABLE_GYROSCOPIC | eRETAIN_ACCELERATION

# PxRigidDynamicLockFlag (from PxRigidDynamicLockFlag.h)
eLOCK_LINEAR_X  = 1 << 0
eLOCK_LINEAR_Y  = 1 << 1
eLOCK_LINEAR_Z  = 1 << 2
eLOCK_ANGULAR_X = 1 << 3
eLOCK_ANGULAR_Y = 1 << 4
eLOCK_ANGULAR_Z = 1 << 5

# Sleep constants (from DySleepingConfigulation.h)
# All used inline as cp.float32(literal) inside the kernel body.
# PXD_FREEZE_INTERVAL  = 1.5
# PXD_SLEEP_DAMPING    = 0.5
# PXD_FREEZE_TOLERANCE = 0.25
# PXD_FREEZE_SCALE     = 0.1

# WAKE_COUNTER_RESET_TIME = 0.4  (used inline as cp.float32(0.4))


# NOTE: Struct arrays are passed as float32 tensors (most fields are float).
# For int fields (node_id, flags, lockFlags), use thread.bitcast(val, cp.int32).
# For writing int values to float tensor, use thread.bitcast(int_val, cp.float32).


# ---------------------------------------------------------------------------
# integrateCoreParallelLaunch kernel
# ---------------------------------------------------------------------------
@cp.kernel
def integrateCoreParallelLaunch(
    offset,                         # int: thread index offset
    motion_velocity_array,          # float32[2N, 4]: linear at [0..N), angular at [N..2N)
    num_solver_bodies,              # int: N
    solver_body_vel_pool,           # float32[2N, 4]: solver velocities
    solver_body_data,               # float32[N, 24]: PxgSolverBodyData AoS (bitcast for int fields)
    solver_txidata,                 # float32[N, 16]: PxgSolverTxIData AoS
    out_solver_velocity,            # float32[2N, 4]: output velocities
    out_body2world,                 # float32[N, 8]: output transforms (q.xyzw + p.xyzw)
    body_sim_buffer,                # float32[M, 60]: PxgBodySim AoS, indexed by nodeIndex
    sleep_data,                     # float32[N, 2]: PxgSolverBodySleepData (bitcast for flags)
    accumulated_delta_v_offset,     # int: offset into solver_body_vel_pool
    enable_stabilization,           # int: 0 or 1
    prev_velocities,                # float32[M, 8]: PxgBodySimVelocities
    prev_velocities_valid,          # int: 1 if prev_velocities is non-null, 0 otherwise
    dt,                             # float: timestep
    inv_dt,                         # float: 1/dt
    island_ids,                     # int32[M]: node -> island mapping
    island_static_touch_counts,     # int32[islands]: per-island static touch count
    num_counted_interactions,       # int32[M]: per-node interaction count
    grid_x,                         # int: grid X dimension
):
    with cp.Kernel(grid_x, threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            idx = bx * BLOCK_SIZE + tid
            a = idx + offset
            if a < num_solver_bodies:
                # ---- Read PxgSolverBodyData[a] ----
                # Force loads by adding 0.0 (avoids cp.ref / f32 yield mismatch in cp.if)
                node_id_f = solver_body_data[a, SBD_NODE_ID]
                node_id = thread.bitcast(node_id_f, cp.int32)

                init_lin_x = solver_body_data[a, SBD_INIT_LIN_X] + cp.float32(0.0)
                init_lin_y = solver_body_data[a, SBD_INIT_LIN_Y] + cp.float32(0.0)
                init_lin_z = solver_body_data[a, SBD_INIT_LIN_Z] + cp.float32(0.0)
                inv_mass   = solver_body_data[a, SBD_INV_MASS] + cp.float32(0.0)

                init_ang_x = solver_body_data[a, SBD_INIT_ANG_X] + cp.float32(0.0)
                init_ang_y = solver_body_data[a, SBD_INIT_ANG_Y] + cp.float32(0.0)
                init_ang_z = solver_body_data[a, SBD_INIT_ANG_Z] + cp.float32(0.0)

                # ---- Read PxgBodySim[nodeIndex] ----
                ni = node_id
                lock_disable_f = body_sim_buffer[ni, BS_LOCK_DISABLE_GRAVITY]
                lock_disable   = thread.bitcast(lock_disable_f, cp.int32)
                lock_flags     = lock_disable & cp.int32(0xFFFF)
                inv_inertia_x = body_sim_buffer[ni, BS_INV_INERTIA_X] + cp.float32(0.0)
                inv_inertia_y = body_sim_buffer[ni, BS_INV_INERTIA_Y] + cp.float32(0.0)
                inv_inertia_z = body_sim_buffer[ni, BS_INV_INERTIA_Z] + cp.float32(0.0)

                # ---- Read body2World from bodySim ----
                b2w_qx = body_sim_buffer[ni, BS_B2W_QX] + cp.float32(0.0)
                b2w_qy = body_sim_buffer[ni, BS_B2W_QY] + cp.float32(0.0)
                b2w_qz = body_sim_buffer[ni, BS_B2W_QZ] + cp.float32(0.0)
                b2w_qw = body_sim_buffer[ni, BS_B2W_QW] + cp.float32(0.0)
                b2w_px = body_sim_buffer[ni, BS_B2W_PX] + cp.float32(0.0)
                b2w_py = body_sim_buffer[ni, BS_B2W_PY] + cp.float32(0.0)
                b2w_pz = body_sim_buffer[ni, BS_B2W_PZ] + cp.float32(0.0)

                # ---- Read sqrtInvInertia from PxgSolverTxIData[a] ----
                m00 = solver_txidata[a, TXI_SQRT_INV_00] + cp.float32(0.0)
                m10 = solver_txidata[a, TXI_SQRT_INV_10] + cp.float32(0.0)
                m20 = solver_txidata[a, TXI_SQRT_INV_20] + cp.float32(0.0)
                m01 = solver_txidata[a, TXI_SQRT_INV_01] + cp.float32(0.0)
                m11 = solver_txidata[a, TXI_SQRT_INV_11] + cp.float32(0.0)
                m21 = solver_txidata[a, TXI_SQRT_INV_21] + cp.float32(0.0)
                m02 = solver_txidata[a, TXI_SQRT_INV_02] + cp.float32(0.0)
                m12 = solver_txidata[a, TXI_SQRT_INV_12] + cp.float32(0.0)
                m22 = solver_txidata[a, TXI_SQRT_INV_22] + cp.float32(0.0)

                # ---- Read solver body velocity ----
                vel_index = accumulated_delta_v_offset + a
                lin_vel_x = solver_body_vel_pool[vel_index, 0] + cp.float32(0.0)
                lin_vel_y = solver_body_vel_pool[vel_index, 1] + cp.float32(0.0)
                lin_vel_z = solver_body_vel_pool[vel_index, 2] + cp.float32(0.0)
                ang_vel_x = solver_body_vel_pool[vel_index + num_solver_bodies, 0] + cp.float32(0.0)
                ang_vel_y = solver_body_vel_pool[vel_index + num_solver_bodies, 1] + cp.float32(0.0)
                ang_vel_z = solver_body_vel_pool[vel_index + num_solver_bodies, 2] + cp.float32(0.0)

                # ---- Read motion velocity ----
                mot_lin_x = motion_velocity_array[a, 0] + cp.float32(0.0)
                mot_lin_y = motion_velocity_array[a, 1] + cp.float32(0.0)
                mot_lin_z = motion_velocity_array[a, 2] + cp.float32(0.0)
                mot_ang_x = motion_velocity_array[a + num_solver_bodies, 0] + cp.float32(0.0)
                mot_ang_y = motion_velocity_array[a + num_solver_bodies, 1] + cp.float32(0.0)
                mot_ang_z = motion_velocity_array[a + num_solver_bodies, 2] + cp.float32(0.0)

                # ============================================================
                # integrateCore() — begin
                # ============================================================

                # Construct body velocities from solver output (non-TGS path)
                # bodyLinearVelocity = initLinVel + solverBodyLinVel
                body_lin_x = init_lin_x + lin_vel_x
                body_lin_y = init_lin_y + lin_vel_y
                body_lin_z = init_lin_z + lin_vel_z

                # bodyAngularVelocity = sqrtInvInertia * solverBodyAngVel
                body_ang_x = m00 * ang_vel_x + m01 * ang_vel_y + m02 * ang_vel_z
                body_ang_y = m10 * ang_vel_x + m11 * ang_vel_y + m12 * ang_vel_z
                body_ang_z = m20 * ang_vel_x + m21 * ang_vel_y + m22 * ang_vel_z

                # Apply lock flags to output velocity
                if lock_flags & cp.int32(eLOCK_LINEAR_X):
                    body_lin_x = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_LINEAR_Y):
                    body_lin_y = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_LINEAR_Z):
                    body_lin_z = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_ANGULAR_X):
                    body_ang_x = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_ANGULAR_Y):
                    body_ang_y = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_ANGULAR_Z):
                    body_ang_z = cp.float32(0.0)

                # Update solver body velocities with lock flags
                lin_vel_x = body_lin_x - init_lin_x
                lin_vel_y = body_lin_y - init_lin_y
                lin_vel_z = body_lin_z - init_lin_z
                # Inverse transform for angular: angSolver = inv(sqrtInvInertia) * bodyAng
                # But CUDA just zeroes locked components of the solver vel directly
                if lock_flags & cp.int32(eLOCK_ANGULAR_X):
                    ang_vel_x = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_ANGULAR_Y):
                    ang_vel_y = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_ANGULAR_Z):
                    ang_vel_z = cp.float32(0.0)

                # Compute motion velocities for sleep checking
                # linearMotionVel = initLinVel + motionLinVel
                lm_x = init_lin_x + mot_lin_x
                lm_y = init_lin_y + mot_lin_y
                lm_z = init_lin_z + mot_lin_z
                # angularMotionVel = initAngVel + sqrtInvInertia * motionAngVel
                am_x = init_ang_x + (m00 * mot_ang_x + m01 * mot_ang_y + m02 * mot_ang_z)
                am_y = init_ang_y + (m10 * mot_ang_x + m11 * mot_ang_y + m12 * mot_ang_z)
                am_z = init_ang_z + (m20 * mot_ang_x + m21 * mot_ang_y + m22 * mot_ang_z)

                # Apply lock flags to motion velocities
                if lock_flags & cp.int32(eLOCK_LINEAR_X):
                    lm_x = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_LINEAR_Y):
                    lm_y = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_LINEAR_Z):
                    lm_z = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_ANGULAR_X):
                    am_x = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_ANGULAR_Y):
                    am_y = cp.float32(0.0)
                if lock_flags & cp.int32(eLOCK_ANGULAR_Z):
                    am_z = cp.float32(0.0)

                # ============================================================
                # sleepCheck / updateWakeCounter — begin
                # ============================================================
                island_id = island_ids[ni]
                static_touch_count = island_static_touch_counts[island_id]
                n_interactions = num_counted_interactions[ni]

                flags_f = body_sim_buffer[ni, BS_INTERNAL_FLAGS]
                flags = thread.bitcast(flags_f, cp.int32)
                flags = flags & cp.int32(SLEEP_FLAG_MASK)

                freeze_thresh_val = body_sim_buffer[ni, BS_FREEZE_THRESH] + cp.float32(0.0)
                wc                = body_sim_buffer[ni, BS_WAKE_COUNTER] + cp.float32(0.0)
                sleep_thresh_val  = body_sim_buffer[ni, BS_SLEEP_THRESH] + cp.float32(0.0)

                sleep_lin_x  = body_sim_buffer[ni, BS_SLEEP_LIN_X] + cp.float32(0.0)
                sleep_lin_y  = body_sim_buffer[ni, BS_SLEEP_LIN_Y] + cp.float32(0.0)
                sleep_lin_z  = body_sim_buffer[ni, BS_SLEEP_LIN_Z] + cp.float32(0.0)
                freeze_count = body_sim_buffer[ni, BS_FREEZE_COUNT] + cp.float32(0.0)
                sleep_ang_x  = body_sim_buffer[ni, BS_SLEEP_ANG_X] + cp.float32(0.0)
                sleep_ang_y  = body_sim_buffer[ni, BS_SLEEP_ANG_Y] + cp.float32(0.0)
                sleep_ang_z  = body_sim_buffer[ni, BS_SLEEP_ANG_Z] + cp.float32(0.0)
                accel_scale  = body_sim_buffer[ni, BS_ACCEL_SCALE] + cp.float32(0.0)

                freeze = cp.int32(0)  # bool
                already_updated_wc = cp.int32(0)
                has_static_touch = cp.int32(0)
                if static_touch_count != cp.int32(0):
                    has_static_touch = cp.int32(1)

                # Compute inertia for energy calculations
                inertia_x = cp.float32(1.0)
                if inv_inertia_x > cp.float32(0.0):
                    inertia_x = cp.float32(1.0) / inv_inertia_x
                inertia_y = cp.float32(1.0)
                if inv_inertia_y > cp.float32(0.0):
                    inertia_y = cp.float32(1.0) / inv_inertia_y
                inertia_z = cp.float32(1.0)
                if inv_inertia_z > cp.float32(0.0):
                    inertia_z = cp.float32(1.0) / inv_inertia_z

                inv_mass_check = inv_mass
                if inv_mass_check == cp.float32(0.0):
                    inv_mass_check = cp.float32(1.0)

                # Pre-define variables used in both stabilization branches
                sla_x = cp.float32(0.0)
                sla_y = cp.float32(0.0)
                sla_z = cp.float32(0.0)
                saa_x = cp.float32(0.0)
                saa_y = cp.float32(0.0)
                saa_z = cp.float32(0.0)
                s_angular = cp.float32(0.0)
                s_linear = cp.float32(0.0)
                norm_energy = cp.float32(0.0)
                threshold = cp.float32(0.0)
                factor = cp.float32(2.0)
                ratio = cp.float32(0.0)
                old_wc = cp.float32(0.0)
                half_reset = cp.float32(0.4) * cp.float32(0.5)

                if enable_stabilization != cp.int32(0):
                    # ---- Stabilization path ----
                    sla_x = lm_x
                    sla_y = lm_y
                    sla_z = lm_z
                    saa_x = am_x
                    saa_y = am_y
                    saa_z = am_z

                    # frameNormalizedEnergy
                    angular_e = (saa_x * saa_x * inertia_x + saa_y * saa_y * inertia_y + saa_z * saa_z * inertia_z) * inv_mass_check
                    linear_e  = sla_x * sla_x + sla_y * sla_y + sla_z * sla_z
                    frame_energy = cp.float32(0.5) * (angular_e + linear_e)

                    cluster_factor = n_interactions
                    if cluster_factor > cp.int32(10):
                        cluster_factor = cp.int32(10)

                    cf_f = cp.float32(0.0)
                    if has_static_touch != cp.int32(0):
                        cf_f = cp.float32(cluster_factor)

                    freeze_thresh_scaled = cf_f * freeze_thresh_val
                    new_freeze_count = freeze_count - dt
                    if new_freeze_count < cp.float32(0.0):
                        new_freeze_count = cp.float32(0.0)
                    freeze_count = new_freeze_count

                    settled = cp.int32(1)
                    new_accel_scale = accel_scale + dt
                    if new_accel_scale > cp.float32(1.0):
                        new_accel_scale = cp.float32(1.0)
                    accel_scale = new_accel_scale

                    if frame_energy >= freeze_thresh_scaled:
                        settled = cp.int32(0)
                        freeze_count = cp.float32(1.5)

                    if has_static_touch == cp.int32(0):
                        accel_scale = cp.float32(1.0)
                        settled = cp.int32(0)

                    if settled != cp.int32(0):
                        if cf_f > cp.float32(1.0):
                            damp = cp.float32(1.0) - (cp.float32(0.5) * dt)
                            lin_vel_x = lin_vel_x * damp
                            lin_vel_y = lin_vel_y * damp
                            lin_vel_z = lin_vel_z * damp
                            ang_vel_x = ang_vel_x * damp
                            ang_vel_y = ang_vel_y * damp
                            ang_vel_z = ang_vel_z * damp
                            accel_scale = accel_scale * cp.float32(0.75) + cp.float32(0.25) * cp.float32(0.1)

                        if freeze_count == cp.float32(0.0):
                            if frame_energy < (freeze_thresh_val * cp.float32(0.25)):
                                freeze = cp.int32(1)

                    if freeze != cp.int32(0):
                        was_not_frozen = cp.int32(0)
                        if (flags & cp.int32(eFROZEN)) == cp.int32(0):
                            was_not_frozen = cp.int32(1)
                        flags = flags | cp.int32(eFROZEN)
                        if was_not_frozen != cp.int32(0):
                            flags = flags | cp.int32(eFREEZE_THIS_FRAME)
                    else:
                        was_frozen = cp.int32(0)
                        if (flags & cp.int32(eFROZEN)) != cp.int32(0):
                            was_frozen = cp.int32(1)
                        flags = flags & cp.int32(UNFROZEN_MASK)
                        if was_frozen != cp.int32(0):
                            flags = flags | cp.int32(eUNFREEZE_THIS_FRAME)

                    # Sleep accumulation check
                    half_reset = cp.float32(0.4) * cp.float32(0.5)
                    if wc < half_reset or wc < dt:
                        sla_x = sla_x + sleep_lin_x
                        sla_y = sla_y + sleep_lin_y
                        sla_z = sla_z + sleep_lin_z
                        saa_x = saa_x + sleep_ang_x
                        saa_y = saa_y + sleep_ang_y
                        saa_z = saa_z + sleep_ang_z

                        if frame_energy >= sleep_thresh_val:
                            s_angular = (saa_x * saa_x * inertia_x + saa_y * saa_y * inertia_y + saa_z * saa_z * inertia_z) * inv_mass_check
                            s_linear  = sla_x * sla_x + sla_y * sla_y + sla_z * sla_z
                            norm_energy = cp.float32(0.5) * (s_angular + s_linear)
                            sleep_cf = cp.float32(cluster_factor) + cp.float32(1.0)
                            threshold = sleep_cf * sleep_thresh_val

                            if norm_energy >= threshold:
                                sla_x = cp.float32(0.0)
                                sla_y = cp.float32(0.0)
                                sla_z = cp.float32(0.0)
                                saa_x = cp.float32(0.0)
                                saa_y = cp.float32(0.0)
                                saa_z = cp.float32(0.0)
                                factor = cp.float32(2.0)
                                if sleep_thresh_val != cp.float32(0.0):
                                    ratio = norm_energy / threshold
                                    if ratio < cp.float32(2.0):
                                        factor = ratio
                                old_wc = wc
                                wc = factor * cp.float32(0.5) * cp.float32(0.4) + dt * (sleep_cf - cp.float32(1.0))
                                if old_wc == cp.float32(0.0):
                                    flags = flags | cp.int32(eACTIVATE_THIS_FRAME)
                                already_updated_wc = cp.int32(1)

                    sleep_lin_x = sla_x
                    sleep_lin_y = sla_y
                    sleep_lin_z = sla_z
                    sleep_ang_x = saa_x
                    sleep_ang_y = saa_y
                    sleep_ang_z = saa_z
                else:
                    # ---- Non-stabilization path ----
                    half_reset = cp.float32(0.4) * cp.float32(0.5)
                    if wc < half_reset or wc < dt:
                        sla_x = lm_x + sleep_lin_x
                        sla_y = lm_y + sleep_lin_y
                        sla_z = lm_z + sleep_lin_z

                        # For non-stabilization: angMotionVel is in world space,
                        # rotateInv by body2World.q before accumulation.
                        # q.rotateInv(v) = q.conjugate().rotate(v)
                        # conjugate: (-qx, -qy, -qz, qw)
                        cqx = -b2w_qx
                        cqy = -b2w_qy
                        cqz = -b2w_qz
                        cqw = b2w_qw
                        # rotate v by quaternion q: v' = q * v * q_conjugate
                        # Using the formula: v' = v + 2*q.w*(q.xyz cross v) + 2*(q.xyz cross (q.xyz cross v))
                        cx = cqy * am_z - cqz * am_y
                        cy = cqz * am_x - cqx * am_z
                        cz = cqx * am_y - cqy * am_x
                        ri_x = am_x + cp.float32(2.0) * (cqw * cx + cqy * cz - cqz * cy)
                        ri_y = am_y + cp.float32(2.0) * (cqw * cy + cqz * cx - cqx * cz)
                        ri_z = am_z + cp.float32(2.0) * (cqw * cz + cqx * cy - cqy * cx)

                        saa_x = ri_x + sleep_ang_x
                        saa_y = ri_y + sleep_ang_y
                        saa_z = ri_z + sleep_ang_z

                        s_angular = (saa_x * saa_x * inertia_x + saa_y * saa_y * inertia_y + saa_z * saa_z * inertia_z) * inv_mass_check
                        s_linear  = sla_x * sla_x + sla_y * sla_y + sla_z * sla_z
                        norm_energy = cp.float32(0.5) * (s_angular + s_linear)

                        cluster_factor_f = cp.float32(1.0) + cp.float32(n_interactions)
                        threshold = cluster_factor_f * sleep_thresh_val

                        if norm_energy >= threshold:
                            sla_x = cp.float32(0.0)
                            sla_y = cp.float32(0.0)
                            sla_z = cp.float32(0.0)
                            saa_x = cp.float32(0.0)
                            saa_y = cp.float32(0.0)
                            saa_z = cp.float32(0.0)
                            factor = cp.float32(2.0)
                            if threshold != cp.float32(0.0):
                                ratio = norm_energy / threshold
                                if ratio < cp.float32(2.0):
                                    factor = ratio
                            old_wc = wc
                            wc = factor * cp.float32(0.5) * cp.float32(0.4) + dt * (cluster_factor_f - cp.float32(1.0))
                            if old_wc == cp.float32(0.0):
                                flags = flags | cp.int32(eACTIVATE_THIS_FRAME)
                            already_updated_wc = cp.int32(1)

                        sleep_lin_x = sla_x
                        sleep_lin_y = sla_y
                        sleep_lin_z = sla_z
                        sleep_ang_x = saa_x
                        sleep_ang_y = saa_y
                        sleep_ang_z = saa_z

                # ---- Final wake counter update ----
                if already_updated_wc == cp.int32(0):
                    new_wc = wc - dt
                    if new_wc < cp.float32(0.0):
                        new_wc = cp.float32(0.0)
                    wc = new_wc

                wc_zero = cp.int32(0)
                if wc == cp.float32(0.0):
                    wc_zero = cp.int32(1)

                if wc_zero != cp.int32(0):
                    flags = flags | cp.int32(eDEACTIVATE_THIS_FRAME)
                    sleep_lin_x = cp.float32(0.0)
                    sleep_lin_y = cp.float32(0.0)
                    sleep_lin_z = cp.float32(0.0)
                    sleep_ang_x = cp.float32(0.0)
                    sleep_ang_y = cp.float32(0.0)
                    sleep_ang_z = cp.float32(0.0)

                # ---- Write back to bodySim ----
                body_sim_buffer[ni, BS_INTERNAL_FLAGS] = thread.bitcast(flags, cp.float32)
                body_sim_buffer[ni, BS_WAKE_COUNTER] = wc
                body_sim_buffer[ni, BS_SLEEP_LIN_X] = sleep_lin_x
                body_sim_buffer[ni, BS_SLEEP_LIN_Y] = sleep_lin_y
                body_sim_buffer[ni, BS_SLEEP_LIN_Z] = sleep_lin_z
                body_sim_buffer[ni, BS_FREEZE_COUNT] = freeze_count
                body_sim_buffer[ni, BS_SLEEP_ANG_X] = sleep_ang_x
                body_sim_buffer[ni, BS_SLEEP_ANG_Y] = sleep_ang_y
                body_sim_buffer[ni, BS_SLEEP_ANG_Z] = sleep_ang_z
                body_sim_buffer[ni, BS_ACCEL_SCALE] = accel_scale

                # Zero external accelerations if not retaining
                if (flags & cp.int32(eRETAIN_ACCELERATION)) == cp.int32(0):
                    body_sim_buffer[ni, BS_EXT_LIN_ACC_X] = cp.float32(0.0)
                    body_sim_buffer[ni, BS_EXT_LIN_ACC_Y] = cp.float32(0.0)
                    body_sim_buffer[ni, BS_EXT_LIN_ACC_Z] = cp.float32(0.0)
                    body_sim_buffer[ni, BS_EXT_ANG_ACC_X] = cp.float32(0.0)
                    body_sim_buffer[ni, BS_EXT_ANG_ACC_Y] = cp.float32(0.0)
                    body_sim_buffer[ni, BS_EXT_ANG_ACC_Z] = cp.float32(0.0)

                # Write back to sleepData
                sleep_data[a, SD_WAKE_COUNTER] = wc
                sleep_data[a, SD_INTERNAL_FLAGS] = thread.bitcast(flags, cp.float32)

                # ============================================================
                # Position & rotation integration (if not frozen)
                # ============================================================
                if freeze == cp.int32(0):
                    # Position: body2World.p += linearMotionVel * dt
                    b2w_px = b2w_px + lm_x * dt
                    b2w_py = b2w_py + lm_y * dt
                    b2w_pz = b2w_pz + lm_z * dt

                    # Rotation: closed-form quaternion integration
                    w_sq = am_x * am_x + am_y * am_y + am_z * am_z
                    w = thread.sqrt(w_sq)

                    if w > cp.float32(0.0):
                        half_angle = dt * w * cp.float32(0.5)
                        s = thread.sin(half_angle)
                        c = thread.cos(half_angle)
                        inv_w = s / w

                        # quatVel = (angularMotionVel * inv_w, 0)
                        qv_x = am_x * inv_w
                        qv_y = am_y * inv_w
                        qv_z = am_z * inv_w

                        # result = quatVel * body2World.q + body2World.q * c
                        # Quaternion multiply: q1 * q2
                        # q1 = (qv_x, qv_y, qv_z, 0), q2 = (b2w_qx, b2w_qy, b2w_qz, b2w_qw)
                        # result.x = 0*b2w_qw + qv_x*1 ... actually use the full formula
                        # (a,b,c,d)*(e,f,g,h) where d,h are scalar parts:
                        # scalar: d*h - (a*e+b*f+c*g)
                        # vector: d*(e,f,g) + h*(a,b,c) + (b*g-c*f, c*e-a*g, a*f-b*e)
                        # quatVel has w=0, so:
                        # result.w = 0*b2w_qw - (qv_x*b2w_qx + qv_y*b2w_qy + qv_z*b2w_qz)
                        # result.xyz = 0*(b2w_qx,y,z) + b2w_qw*(qv_x,y,z) + cross(qv, b2w_q.xyz)
                        cross_x = qv_y * b2w_qz - qv_z * b2w_qy
                        cross_y = qv_z * b2w_qx - qv_x * b2w_qz
                        cross_z = qv_x * b2w_qy - qv_y * b2w_qx

                        res_x = b2w_qw * qv_x + cross_x
                        res_y = b2w_qw * qv_y + cross_y
                        res_z = b2w_qw * qv_z + cross_z
                        res_w = -(qv_x * b2w_qx + qv_y * b2w_qy + qv_z * b2w_qz)

                        # Add scaled original quaternion: result += body2World.q * c
                        res_x = res_x + b2w_qx * c
                        res_y = res_y + b2w_qy * c
                        res_z = res_z + b2w_qz * c
                        res_w = res_w + b2w_qw * c

                        # Normalize
                        mag_sq = res_x * res_x + res_y * res_y + res_z * res_z + res_w * res_w
                        inv_mag = thread.rsqrt(mag_sq)
                        b2w_qx = res_x * inv_mag
                        b2w_qy = res_y * inv_mag
                        b2w_qz = res_z * inv_mag
                        b2w_qw = res_w * inv_mag

                # ---- Write output solver velocity ----
                out_solver_velocity[a, 0] = lin_vel_x
                out_solver_velocity[a, 1] = lin_vel_y
                out_solver_velocity[a, 2] = lin_vel_z
                out_solver_velocity[a, 3] = cp.float32(0.0)
                out_solver_velocity[a + num_solver_bodies, 0] = ang_vel_x
                out_solver_velocity[a + num_solver_bodies, 1] = ang_vel_y
                out_solver_velocity[a + num_solver_bodies, 2] = ang_vel_z
                out_solver_velocity[a + num_solver_bodies, 3] = cp.float32(0.0)

                # ---- Write output body2World ----
                out_body2world[a, 0] = b2w_qx
                out_body2world[a, 1] = b2w_qy
                out_body2world[a, 2] = b2w_qz
                out_body2world[a, 3] = b2w_qw
                out_body2world[a, 4] = b2w_px
                out_body2world[a, 5] = b2w_py
                out_body2world[a, 6] = b2w_pz
                out_body2world[a, 7] = cp.float32(0.0)

                # ---- Write prev velocities if valid ----
                if prev_velocities_valid != cp.int32(0):
                    # Store pre-integration velocities from bodySim
                    prev_velocities[ni, 0] = body_sim_buffer[ni, BS_LIN_VEL_X]
                    prev_velocities[ni, 1] = body_sim_buffer[ni, BS_LIN_VEL_Y]
                    prev_velocities[ni, 2] = body_sim_buffer[ni, BS_LIN_VEL_Z]
                    prev_velocities[ni, 3] = body_sim_buffer[ni, BS_INV_MASS_W]
                    prev_velocities[ni, 4] = body_sim_buffer[ni, BS_ANG_VEL_X]
                    prev_velocities[ni, 5] = body_sim_buffer[ni, BS_ANG_VEL_Y]
                    prev_velocities[ni, 6] = body_sim_buffer[ni, BS_ANG_VEL_Z]
                    prev_velocities[ni, 7] = cp.float32(0.0)

                # ---- Write back to bodySim: updated velocities and transform ----
                body_sim_buffer[ni, BS_LIN_VEL_X] = body_lin_x
                body_sim_buffer[ni, BS_LIN_VEL_Y] = body_lin_y
                body_sim_buffer[ni, BS_LIN_VEL_Z] = body_lin_z
                body_sim_buffer[ni, BS_ANG_VEL_X] = body_ang_x
                body_sim_buffer[ni, BS_ANG_VEL_Y] = body_ang_y
                body_sim_buffer[ni, BS_ANG_VEL_Z] = body_ang_z
                body_sim_buffer[ni, BS_B2W_QX] = b2w_qx
                body_sim_buffer[ni, BS_B2W_QY] = b2w_qy
                body_sim_buffer[ni, BS_B2W_QZ] = b2w_qz
                body_sim_buffer[ni, BS_B2W_QW] = b2w_qw
                body_sim_buffer[ni, BS_B2W_PX] = b2w_px
                body_sim_buffer[ni, BS_B2W_PY] = b2w_py
                body_sim_buffer[ni, BS_B2W_PZ] = b2w_pz
