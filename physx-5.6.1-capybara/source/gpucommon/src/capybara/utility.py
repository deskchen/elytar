"""Capybara DSL port of gpucommon/CUDA/utility.cu.

This file ports the whole utility.cu translation unit as a first PTX
replacement candidate. It keeps kernel names aligned with CUDA:

- interleaveBuffers          — ABI-compatible (drop-in PTX replacement OK)
- normalVectorsAreaWeighted  — flat-arg adapter (requires host-side unpack)
- zeroNormals                — flat-arg adapter (requires host-side unpack)
- normalizeNormals           — flat-arg adapter (requires host-side unpack)
- interpolateSkinnedClothVertices   — flat-arg adapter (requires host-side unpack)
- interpolateSkinnedSoftBodyVertices — flat-arg adapter (requires host-side unpack)

ABI status:
  interleaveBuffers: argument order and data layout match the CUDA signature
  (float4* vertices, float4* normals, PxU32 length, PxVec3* result).

  All skinning kernels: the original CUDA kernels take a single
  PxTrimeshSkinningGpuData* / PxTetmeshSkinningGpuData* per batch.  This
  port uses flat per-batch arguments instead.  PxgDeformableSkinning.cpp
  (guarded by ELYTAR_CAPYBARA_SKINNING) unpacks the struct on the host and
  launches one Capybara kernel call per batch.  See docs/PHYSX_PORT_PRACTICE.md.

Parameter notes for skinning kernels:
  guide_vertices  — float32[N,4]  PxVec4 stride (x,y,z,invMass); only xyz read
  guide_normals   — float32[N,3]  compact PxVec3
  guide_triangles — int32[M*3]    flat PxU32 triangle index buffer
  skin_info (cloth)  — int32[K,4] reinterpret of PxTriangleMeshEmbeddingInfo:
                         col0=uv.x bits, col1=uv.y bits,
                         col2=offsetAlongNormal bits, col3=guideTriangleId
  skin_info (volume) — int32[K,4] reinterpret of PxTetrahedronMeshEmbeddingInfo:
                         col0=uvw.x bits, col1=uvw.y bits, col2=uvw.z bits,
                         col3=guideTetrahedronId
  skinned_vertices — float32[K,3] compact PxVec3 output (pinned host memory,
                      GPU-writable because it is allocated with cuMemHostAlloc)
"""

import capybara as cp


BLOCK_SIZE = 256


@cp.inline
def _normalize_safe(thread, x, y, z):
    """Unit vector if |v|^2 is significant, else ~0.  Also returns magnitude.

    Avoid ``thread.where`` here: inside ``@cp.inline`` there may be no unit scope, so
    ``cp.select`` operands mix bare scalars with ``!cp.val`` and MLIR verification fails.

    Branchless mask approximates ``v/|v|`` when ``|v|^2 >> eps`` and fades out when tiny.
    """
    mag2 = x * x + y * y + z * z
    eps = cp.float32(1.0e-20)
    tiny = cp.float32(1.0e-30)
    mask = mag2 / (mag2 + eps)
    inv = thread.rsqrt(mag2 + tiny)
    mag = thread.sqrt(mag2 + tiny) * mask
    return (
        x * mask * inv,
        y * mask * inv,
        z * mask * inv,
        mag,
    )


@cp.inline
def _tanh_approx(thread, x):
    two_x = cp.float32(2.0) * x
    e = thread.exp(two_x)
    return (e - cp.float32(1.0)) / (e + cp.float32(1.0))


@cp.inline
def _evaluate_point_phong(
    thread,
    ax, ay, az,
    bx, by, bz,
    cx, cy, cz,
    uvx, uvy, uvz,
    nAx, nAy, nAz,
    nBx, nBy, nBz,
    nCx, nCy, nCz,
    half_surface_thickness,
):
    qx = uvx * ax + uvy * bx + uvz * cx
    qy = uvx * ay + uvy * by + uvz * cy
    qz = uvx * az + uvy * bz + uvz * cz

    scale1 = (qx - ax) * nAx + (qy - ay) * nAy + (qz - az) * nAz
    projAx = qx - scale1 * nAx
    projAy = qy - scale1 * nAy
    projAz = qz - scale1 * nAz

    scale2 = (qx - bx) * nBx + (qy - by) * nBy + (qz - bz) * nBz
    projBx = qx - scale2 * nBx
    projBy = qy - scale2 * nBy
    projBz = qz - scale2 * nBz

    scale3 = (qx - cx) * nCx + (qy - cy) * nCy + (qz - cz) * nCz
    projCx = qx - scale3 * nCx
    projCy = qy - scale3 * nCy
    projCz = qz - scale3 * nCz

    qStarx = uvx * projAx + uvy * projBx + uvz * projCx
    qStary = uvx * projAy + uvy * projBy + uvz * projCy
    qStarz = uvx * projAz + uvy * projBz + uvz * projCz

    dirx = qStarx - qx
    diry = qStary - qy
    dirz = qStarz - qz
    ndx, ndy, ndz, dir_mag = _normalize_safe(thread, dirx, diry, dirz)

    alpha = cp.float32(0.625)
    # CUDA: offset = |dir| * alpha; ratio = tanh(offset / h); offset = ratio * h
    # Branchless: use (h+eps) to avoid div-by-zero, then multiply back by h.
    eps_h = cp.float32(1.0e-30)
    raw_offset = dir_mag * alpha
    ratio_raw = _tanh_approx(thread, raw_offset / (half_surface_thickness + eps_h))
    offset = ratio_raw * half_surface_thickness

    return qx + offset * ndx, qy + offset * ndy, qz + offset * ndz


@cp.kernel
def interleaveBuffers(vertices, normals, length, interleaved_result):
    """Match CUDA ABI: (const float4* vertices, const float4* normals,
    PxU32 length, PxVec3* interleavedResultBuffer).

    CUDA reads float4 (16-byte stride, .w ignored).  Capybara uses
    float32[N,4] to preserve the stride, reading only columns 0-2.
    Output is PxVec3 = float32[2N,3] (12-byte stride).
    """
    with cp.Kernel(cp.ceildiv(length, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, thread in block.threads():
            idx = bx * BLOCK_SIZE + tid
            if idx < length:
                vx = vertices[idx, 0]
                vy = vertices[idx, 1]
                vz = vertices[idx, 2]
                nx = normals[idx, 0]
                ny = normals[idx, 1]
                nz = normals[idx, 2]
                interleaved_result[2 * idx, 0] = vx
                interleaved_result[2 * idx, 1] = vy
                interleaved_result[2 * idx, 2] = vz
                interleaved_result[2 * idx + 1, 0] = nx
                interleaved_result[2 * idx + 1, 1] = ny
                interleaved_result[2 * idx + 1, 2] = nz


@cp.kernel
def zeroNormals(guide_normals, guide_vertices_count, grid_x):
    """Flat-arg replacement for zeroNormals(PxTrimeshSkinningGpuData*).

    Host unpacks one batch struct and launches per batch.

    Args:
        guide_normals       float32[N,3]  compact PxVec3 normal buffer
        guide_vertices_count u32          number of guide vertices (= normals)
        grid_x              u32          gridDim.x (total x-blocks for this launch)
    """
    with cp.Kernel(grid_x, threads=BLOCK_SIZE) as (bx, block):
        x_dim = grid_x * BLOCK_SIZE
        start = bx * BLOCK_SIZE
        for tid, thread in block.threads():
            idx = start + tid
            while idx < guide_vertices_count:
                guide_normals[idx, 0] = cp.float32(0.0)
                guide_normals[idx, 1] = cp.float32(0.0)
                guide_normals[idx, 2] = cp.float32(0.0)
                idx += x_dim


@cp.kernel
def normalVectorsAreaWeighted(guide_vertices, guide_normals, guide_triangles,
                               nb_guide_triangles, grid_x):
    """Flat-arg replacement for normalVectorsAreaWeighted(PxTrimeshSkinningGpuData*).

    Accumulates area-weighted face normals into guide_normals via atomic add.
    Host unpacks one batch struct and launches per batch.

    Args:
        guide_vertices      float32[N,4]  PxVec4 layout (x,y,z,invMass); only xyz used
        guide_normals       float32[N,3]  compact PxVec3 normal accumulation buffer
        guide_triangles     int32[M*3]    flat triangle index buffer (3 ints per tri)
        nb_guide_triangles  u32           number of guide triangles
        grid_x              u32           gridDim.x
    """
    with cp.Kernel(grid_x, threads=BLOCK_SIZE) as (bx, block):
        # Atomic target grammar in current Capybara expects fully-indexed view
        # forms for tuple subscripts (view[i, j]) in thread.atomic_add.
        guide_normals_2d = guide_normals.view(guide_normals.shape[0], 3)
        x_dim = grid_x * BLOCK_SIZE
        start = bx * BLOCK_SIZE
        for tid, thread in block.threads():
            idx = start + tid
            while idx < nb_guide_triangles:
                tri_base = 3 * idx
                i0 = guide_triangles[tri_base]
                i1 = guide_triangles[tri_base + 1]
                i2 = guide_triangles[tri_base + 2]

                p0x = guide_vertices[i0, 0]
                p0y = guide_vertices[i0, 1]
                p0z = guide_vertices[i0, 2]
                p1x = guide_vertices[i1, 0]
                p1y = guide_vertices[i1, 1]
                p1z = guide_vertices[i1, 2]
                p2x = guide_vertices[i2, 0]
                p2y = guide_vertices[i2, 1]
                p2z = guide_vertices[i2, 2]

                e10x = p1x - p0x
                e10y = p1y - p0y
                e10z = p1z - p0z
                e20x = p2x - p0x
                e20y = p2y - p0y
                e20z = p2z - p0z

                nx = e10y * e20z - e10z * e20y
                ny = e10z * e20x - e10x * e20z
                nz = e10x * e20y - e10y * e20x

                thread.atomic_add(guide_normals_2d[i0, 0], nx)
                thread.atomic_add(guide_normals_2d[i0, 1], ny)
                thread.atomic_add(guide_normals_2d[i0, 2], nz)
                thread.atomic_add(guide_normals_2d[i1, 0], nx)
                thread.atomic_add(guide_normals_2d[i1, 1], ny)
                thread.atomic_add(guide_normals_2d[i1, 2], nz)
                thread.atomic_add(guide_normals_2d[i2, 0], nx)
                thread.atomic_add(guide_normals_2d[i2, 1], ny)
                thread.atomic_add(guide_normals_2d[i2, 2], nz)

                idx += x_dim


@cp.kernel
def normalizeNormals(guide_normals, guide_vertices_count, grid_x):
    """Flat-arg replacement for normalizeNormals(PxTrimeshSkinningGpuData*).

    Host unpacks one batch struct and launches per batch.

    Args:
        guide_normals       float32[N,3]  compact PxVec3 normal buffer
        guide_vertices_count u32          number of guide vertices (= normals)
        grid_x              u32           gridDim.x
    """
    with cp.Kernel(grid_x, threads=BLOCK_SIZE) as (bx, block):
        x_dim = grid_x * BLOCK_SIZE
        start = bx * BLOCK_SIZE
        for tid, thread in block.threads():
            idx = start + tid
            while idx < guide_vertices_count:
                x = guide_normals[idx, 0]
                y = guide_normals[idx, 1]
                z = guide_normals[idx, 2]
                x, y, z, _ = _normalize_safe(thread, x, y, z)
                guide_normals[idx, 0] = x
                guide_normals[idx, 1] = y
                guide_normals[idx, 2] = z
                idx += x_dim


@cp.kernel
def interpolateSkinnedClothVertices(
    guide_vertices,
    guide_normals,
    guide_triangles,
    skin_info,
    skinned_vertices,
    skinned_count,
    half_surface_thickness,
    grid_x,
):
    """Flat-arg replacement for interpolateSkinnedClothVertices(PxTrimeshSkinningGpuData*).

    Host unpacks one batch struct and launches per batch.

    Args:
        guide_vertices          float32[N,4]  PxVec4 layout; only xyz used
        guide_normals           float32[N,3]  compact PxVec3 normals (pre-normalized)
        guide_triangles         int32[M*3]    flat triangle index buffer
        skin_info               int32[K,4]    reinterpret of PxTriangleMeshEmbeddingInfo:
                                              [uv.x bits, uv.y bits,
                                               offsetAlongNormal bits, guideTriangleId]
        skinned_vertices        float32[K,3]  output (compact PxVec3, pinned memory)
        skinned_count           u32           number of skinned vertices
        half_surface_thickness  f32           halfSurfaceThickness field
        grid_x                  u32           gridDim.x
    """
    with cp.Kernel(grid_x, threads=BLOCK_SIZE) as (bx, block):
        x_dim = grid_x * BLOCK_SIZE
        start = bx * BLOCK_SIZE
        for tid, thread in block.threads():
            idx = start + tid
            while idx < skinned_count:
                # Unpack PxTriangleMeshEmbeddingInfo via bitcast.
                # Load into i32 temps first: passing tensor[i,j] directly to
                # thread.bitcast gives a ScalarRefTy (memory ref) rather than
                # a loaded value, causing a width-mismatch error.
                s0 = skin_info[idx, 0]
                s1 = skin_info[idx, 1]
                s2 = skin_info[idx, 2]
                tri_id = skin_info[idx, 3]
                uvx = thread.bitcast(s0, cp.float32)
                uvy = thread.bitcast(s1, cp.float32)
                offset_along_n = thread.bitcast(s2, cp.float32)

                uvz = cp.float32(1.0) - uvx - uvy

                # Clamp barycentrics to [0,1] and renormalize (matches CUDA uvwProj)
                uvpx = uvx if uvx > cp.float32(0.0) else cp.float32(0.0)
                uvpy = uvy if uvy > cp.float32(0.0) else cp.float32(0.0)
                uvpz = uvz if uvz > cp.float32(0.0) else cp.float32(0.0)
                s = uvpx + uvpy + uvpz
                if s > cp.float32(0.0):
                    invs = cp.float32(1.0) / s
                    uvpx = uvpx * invs
                    uvpy = uvpy * invs
                    uvpz = uvpz * invs

                tri_base = 3 * tri_id
                i0 = guide_triangles[tri_base]
                i1 = guide_triangles[tri_base + 1]
                i2 = guide_triangles[tri_base + 2]

                ax = guide_vertices[i0, 0]
                ay = guide_vertices[i0, 1]
                az = guide_vertices[i0, 2]
                bxv = guide_vertices[i1, 0]
                byv = guide_vertices[i1, 1]
                bzv = guide_vertices[i1, 2]
                cx = guide_vertices[i2, 0]
                cy = guide_vertices[i2, 1]
                cz = guide_vertices[i2, 2]

                nAx = guide_normals[i0, 0]
                nAy = guide_normals[i0, 1]
                nAz = guide_normals[i0, 2]
                nBx = guide_normals[i1, 0]
                nBy = guide_normals[i1, 1]
                nBz = guide_normals[i1, 2]
                nCx = guide_normals[i2, 0]
                nCy = guide_normals[i2, 1]
                nCz = guide_normals[i2, 2]

                nx = uvpx * nAx + uvpy * nBx + uvpz * nCx
                ny = uvpx * nAy + uvpy * nBy + uvpz * nCy
                nz = uvpx * nAz + uvpy * nBz + uvpz * nCz
                nx, ny, nz, _ = _normalize_safe(thread, nx, ny, nz)

                ppx, ppy, ppz = _evaluate_point_phong(
                    thread,
                    ax, ay, az,
                    bxv, byv, bzv,
                    cx, cy, cz,
                    uvpx, uvpy, uvpz,
                    nAx, nAy, nAz,
                    nBx, nBy, nBz,
                    nCx, nCy, nCz,
                    half_surface_thickness,
                )

                # pointUVW = uvw (unclamped) interpolated position
                puvx = uvx * ax + uvy * bxv + uvz * cx
                puvy = uvx * ay + uvy * byv + uvz * cy
                puvz = uvx * az + uvy * bzv + uvz * cz
                # pointUVWProj = uvwProj (clamped) interpolated position
                pprojx = uvpx * ax + uvpy * bxv + uvpz * cx
                pprojy = uvpx * ay + uvpy * byv + uvpz * cy
                pprojz = uvpx * az + uvpy * bzv + uvpz * cz

                skinned_vertices[idx, 0] = ppx + offset_along_n * nx + puvx - pprojx
                skinned_vertices[idx, 1] = ppy + offset_along_n * ny + puvy - pprojy
                skinned_vertices[idx, 2] = ppz + offset_along_n * nz + puvz - pprojz

                idx += x_dim


@cp.kernel
def interpolateSkinnedSoftBodyVertices(
    guide_vertices,
    guide_tetrahedra,
    skin_info,
    skinned_vertices,
    skinned_count,
    grid_x,
):
    """Flat-arg replacement for interpolateSkinnedSoftBodyVertices(PxTetmeshSkinningGpuData*).

    Host unpacks one batch struct and launches per batch.

    Args:
        guide_vertices   float32[N,4]  PxVec4 layout; only xyz used
        guide_tetrahedra int32[T*4]    flat tet index buffer (4 ints per tet)
        skin_info        int32[K,4]    reinterpret of PxTetrahedronMeshEmbeddingInfo:
                                       [uvw.x bits, uvw.y bits, uvw.z bits, guideTetrahedronId]
        skinned_vertices float32[K,3]  output (compact PxVec3, pinned memory)
        skinned_count    u32           number of skinned vertices
        grid_x           u32           gridDim.x
    """
    with cp.Kernel(grid_x, threads=BLOCK_SIZE) as (bx, block):
        x_dim = grid_x * BLOCK_SIZE
        start = bx * BLOCK_SIZE
        for tid, thread in block.threads():
            idx = start + tid
            while idx < skinned_count:
                # Unpack PxTetrahedronMeshEmbeddingInfo via bitcast.
                # Load into i32 temps first (same reason as cloth kernel above).
                s0 = skin_info[idx, 0]
                s1 = skin_info[idx, 1]
                s2 = skin_info[idx, 2]
                tet_id = skin_info[idx, 3]
                ux = thread.bitcast(s0, cp.float32)
                uy = thread.bitcast(s1, cp.float32)
                uz = thread.bitcast(s2, cp.float32)
                s = cp.float32(1.0) - ux - uy - uz

                tet_base = 4 * tet_id
                i0 = guide_tetrahedra[tet_base]
                i1 = guide_tetrahedra[tet_base + 1]
                i2 = guide_tetrahedra[tet_base + 2]
                i3 = guide_tetrahedra[tet_base + 3]

                x = (
                    ux * guide_vertices[i0, 0]
                    + uy * guide_vertices[i1, 0]
                    + uz * guide_vertices[i2, 0]
                    + s  * guide_vertices[i3, 0]
                )
                y = (
                    ux * guide_vertices[i0, 1]
                    + uy * guide_vertices[i1, 1]
                    + uz * guide_vertices[i2, 1]
                    + s  * guide_vertices[i3, 1]
                )
                z = (
                    ux * guide_vertices[i0, 2]
                    + uy * guide_vertices[i1, 2]
                    + uz * guide_vertices[i2, 2]
                    + s  * guide_vertices[i3, 2]
                )

                skinned_vertices[idx, 0] = x
                skinned_vertices[idx, 1] = y
                skinned_vertices[idx, 2] = z

                idx += x_dim
