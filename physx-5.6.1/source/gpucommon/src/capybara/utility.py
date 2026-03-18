"""Capybara port of PhysX gpucommon/CUDA/utility.cu.

Kernel names are preserved:
  - interleaveBuffers
  - normalVectorsAreaWeighted
  - zeroNormals
  - normalizeNormals
  - interpolateSkinnedClothVertices
  - interpolateSkinnedSoftBodyVertices

This port uses explicit SoA tensors plus per-batch offsets/counts.
"""

import capybara as cp


@cp.struct
class PxTrimeshSkinningGpuData:
    guide_vertices_offset: cp.int32
    guide_vertices_count: cp.int32
    guide_normals_offset: cp.int32
    guide_normals_count: cp.int32
    guide_triangles_offset: cp.int32
    skinning_info_offset: cp.int32
    skinned_vertices_offset: cp.int32
    skinned_vertices_count: cp.int32
    half_surface_thickness: cp.float32
    nb_guide_triangles: cp.int32


@cp.struct
class PxTetmeshSkinningGpuData:
    guide_vertices_offset: cp.int32
    guide_tetrahedra_offset: cp.int32
    skinning_info_offset: cp.int32
    skinned_vertices_offset: cp.int32
    skinned_vertices_count: cp.int32


@cp.inline
def _normalize3(thread: cp.Threads, x, y, z):
    mag2 = x * x + y * y + z * z
    if mag2 > 0.0:
        inv_mag = thread.rsqrt(mag2)
        return x * inv_mag, y * inv_mag, z * inv_mag
    return 0.0, 0.0, 0.0


@cp.inline
def _tanh_approx(thread: cp.Threads, x):
    e2x = thread.exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)


@cp.inline
def _evaluate_point_phong(
    thread: cp.Threads,
    ax,
    ay,
    az,
    bx,
    by,
    bz,
    cx,
    cy,
    cz,
    uvx,
    uvy,
    uvz,
    nAx,
    nAy,
    nAz,
    nBx,
    nBy,
    nBz,
    nCx,
    nCy,
    nCz,
    half_surface_thickness,
    alpha: cp.constexpr = 0.625,
):
    qx = uvx * ax + uvy * bx + uvz * cx
    qy = uvx * ay + uvy * by + uvz * cy
    qz = uvx * az + uvy * bz + uvz * cz

    s1 = (qx - ax) * nAx + (qy - ay) * nAy + (qz - az) * nAz
    s2 = (qx - bx) * nBx + (qy - by) * nBy + (qz - bz) * nBz
    s3 = (qx - cx) * nCx + (qy - cy) * nCy + (qz - cz) * nCz

    projAx = qx - s1 * nAx
    projAy = qy - s1 * nAy
    projAz = qz - s1 * nAz
    projBx = qx - s2 * nBx
    projBy = qy - s2 * nBy
    projBz = qz - s2 * nBz
    projCx = qx - s3 * nCx
    projCy = qy - s3 * nCy
    projCz = qz - s3 * nCz

    qsx = uvx * projAx + uvy * projBx + uvz * projCx
    qsy = uvx * projAy + uvy * projBy + uvz * projCy
    qsz = uvx * projAz + uvy * projBz + uvz * projCz

    dirx = qsx - qx
    diry = qsy - qy
    dirz = qsz - qz
    dirx, diry, dirz = _normalize3(thread, dirx, diry, dirz)

    raw_offset = alpha
    if half_surface_thickness > 0.0:
        ratio = raw_offset / half_surface_thickness
        ratio = _tanh_approx(thread, ratio)
        raw_offset = ratio * half_surface_thickness
    else:
        raw_offset = 0.0

    return qx + raw_offset * dirx, qy + raw_offset * diry, qz + raw_offset * dirz


@cp.kernel
def interleaveBuffers(vertices, normals, length, interleavedResultBuffer, BLOCK_SIZE: cp.constexpr = 256):
    """Port of interleaveBuffers.

    vertices: [N, 4], normals: [N, 4], interleavedResultBuffer: [2N, 3]
    """
    with cp.Kernel(cp.ceildiv(length, BLOCK_SIZE), threads=BLOCK_SIZE) as (bx, block):
        for tid, _thread in block.threads():
            threadIndex = bx * BLOCK_SIZE + tid
            if threadIndex < length:
                interleavedResultBuffer[2 * threadIndex, 0] = vertices[threadIndex, 0]
                interleavedResultBuffer[2 * threadIndex, 1] = vertices[threadIndex, 1]
                interleavedResultBuffer[2 * threadIndex, 2] = vertices[threadIndex, 2]
                interleavedResultBuffer[2 * threadIndex + 1, 0] = normals[threadIndex, 0]
                interleavedResultBuffer[2 * threadIndex + 1, 1] = normals[threadIndex, 1]
                interleavedResultBuffer[2 * threadIndex + 1, 2] = normals[threadIndex, 2]


@cp.kernel
def normalVectorsAreaWeighted(
    data,
    guide_vertices,
    guide_normals,
    guide_triangles,
    max_nb_guide_triangles,
    BLOCK_SIZE: cp.constexpr = 256,
):
    """Port of normalVectorsAreaWeighted.

    guide_vertices: [V, 3]
    guide_normals: [V, 3]
    guide_triangles: [T, 3] (local per-batch indices)
    """
    grid_x = cp.ceildiv(max_nb_guide_triangles, BLOCK_SIZE)
    with cp.Kernel(grid_x, data.shape[0], threads=BLOCK_SIZE) as (bx, by, block):
        d = data[by]
        stride = grid_x * BLOCK_SIZE
        for tid, thread in block.threads():
            threadIndex = bx * BLOCK_SIZE + tid
            while threadIndex < d.nb_guide_triangles:
                tri_idx = d.guide_triangles_offset + threadIndex
                i0 = guide_triangles[tri_idx, 0]
                i1 = guide_triangles[tri_idx, 1]
                i2 = guide_triangles[tri_idx, 2]

                g0 = d.guide_vertices_offset + i0
                g1 = d.guide_vertices_offset + i1
                g2 = d.guide_vertices_offset + i2

                p0x = guide_vertices[g0, 0]
                p0y = guide_vertices[g0, 1]
                p0z = guide_vertices[g0, 2]
                p1x = guide_vertices[g1, 0]
                p1y = guide_vertices[g1, 1]
                p1z = guide_vertices[g1, 2]
                p2x = guide_vertices[g2, 0]
                p2y = guide_vertices[g2, 1]
                p2z = guide_vertices[g2, 2]

                e10x = p1x - p0x
                e10y = p1y - p0y
                e10z = p1z - p0z
                e20x = p2x - p0x
                e20y = p2y - p0y
                e20z = p2z - p0z

                nx = e10y * e20z - e10z * e20y
                ny = e10z * e20x - e10x * e20z
                nz = e10x * e20y - e10y * e20x

                n0 = d.guide_normals_offset + i0
                n1 = d.guide_normals_offset + i1
                n2 = d.guide_normals_offset + i2

                thread.atomic_add(guide_normals[n0, 0], nx)
                thread.atomic_add(guide_normals[n0, 1], ny)
                thread.atomic_add(guide_normals[n0, 2], nz)
                thread.atomic_add(guide_normals[n1, 0], nx)
                thread.atomic_add(guide_normals[n1, 1], ny)
                thread.atomic_add(guide_normals[n1, 2], nz)
                thread.atomic_add(guide_normals[n2, 0], nx)
                thread.atomic_add(guide_normals[n2, 1], ny)
                thread.atomic_add(guide_normals[n2, 2], nz)

                threadIndex += stride


@cp.kernel
def zeroNormals(
    data,
    guide_normals,
    max_nb_guide_vertices,
    BLOCK_SIZE: cp.constexpr = 256,
):
    """Port of zeroNormals."""
    grid_x = cp.ceildiv(max_nb_guide_vertices, BLOCK_SIZE)
    with cp.Kernel(grid_x, data.shape[0], threads=BLOCK_SIZE) as (bx, by, block):
        d = data[by]
        stride = grid_x * BLOCK_SIZE
        for tid, _thread in block.threads():
            threadIndex = bx * BLOCK_SIZE + tid
            while threadIndex < d.guide_vertices_count:
                gi = d.guide_normals_offset + threadIndex
                guide_normals[gi, 0] = 0.0
                guide_normals[gi, 1] = 0.0
                guide_normals[gi, 2] = 0.0
                threadIndex += stride


@cp.kernel
def normalizeNormals(
    data,
    guide_normals,
    max_nb_guide_vertices,
    BLOCK_SIZE: cp.constexpr = 256,
):
    """Port of normalizeNormals."""
    grid_x = cp.ceildiv(max_nb_guide_vertices, BLOCK_SIZE)
    with cp.Kernel(grid_x, data.shape[0], threads=BLOCK_SIZE) as (bx, by, block):
        d = data[by]
        stride = grid_x * BLOCK_SIZE
        for tid, thread in block.threads():
            threadIndex = bx * BLOCK_SIZE + tid
            while threadIndex < d.guide_vertices_count:
                gi = d.guide_normals_offset + threadIndex
                nx = guide_normals[gi, 0]
                ny = guide_normals[gi, 1]
                nz = guide_normals[gi, 2]
                nx, ny, nz = _normalize3(thread, nx, ny, nz)
                guide_normals[gi, 0] = nx
                guide_normals[gi, 1] = ny
                guide_normals[gi, 2] = nz
                threadIndex += stride


@cp.kernel
def interpolateSkinnedClothVertices(
    data,
    guide_vertices,
    guide_normals,
    guide_triangles,
    skin_uv,
    skin_offset_along_normal,
    skin_tri_id,
    skinned_vertices,
    max_nb_skinned_vertices,
    BLOCK_SIZE: cp.constexpr = 256,
):
    """Port of interpolateSkinnedClothVertices.

    skin_uv: [S, 2]
    skin_offset_along_normal: [S]
    skin_tri_id: [S]
    skinned_vertices: [S, 3]
    """
    grid_x = cp.ceildiv(max_nb_skinned_vertices, BLOCK_SIZE)
    with cp.Kernel(grid_x, data.shape[0], threads=BLOCK_SIZE) as (bx, by, block):
        d = data[by]
        stride = grid_x * BLOCK_SIZE
        for tid, thread in block.threads():
            threadIndex = bx * BLOCK_SIZE + tid
            while threadIndex < d.skinned_vertices_count:
                info_idx = d.skinning_info_offset + threadIndex
                tri_id = skin_tri_id[info_idx]
                tri_idx = d.guide_triangles_offset + tri_id

                l0 = guide_triangles[tri_idx, 0]
                l1 = guide_triangles[tri_idx, 1]
                l2 = guide_triangles[tri_idx, 2]

                g0 = d.guide_vertices_offset + l0
                g1 = d.guide_vertices_offset + l1
                g2 = d.guide_vertices_offset + l2
                n0 = d.guide_normals_offset + l0
                n1 = d.guide_normals_offset + l1
                n2 = d.guide_normals_offset + l2

                uvx = skin_uv[info_idx, 0]
                uvy = skin_uv[info_idx, 1]
                uvz = 1.0 - uvx - uvy

                uvpx = uvx if uvx > 0.0 else 0.0
                uvpy = uvy if uvy > 0.0 else 0.0
                uvpz = uvz if uvz > 0.0 else 0.0
                uvpsum = uvpx + uvpy + uvpz
                if uvpsum > 0.0:
                    inv = 1.0 / uvpsum
                    uvpx *= inv
                    uvpy *= inv
                    uvpz *= inv

                nAx = guide_normals[n0, 0]
                nAy = guide_normals[n0, 1]
                nAz = guide_normals[n0, 2]
                nBx = guide_normals[n1, 0]
                nBy = guide_normals[n1, 1]
                nBz = guide_normals[n1, 2]
                nCx = guide_normals[n2, 0]
                nCy = guide_normals[n2, 1]
                nCz = guide_normals[n2, 2]

                normalx = uvpx * nAx + uvpy * nBx + uvpz * nCx
                normaly = uvpx * nAy + uvpy * nBy + uvpz * nCy
                normalz = uvpx * nAz + uvpy * nBz + uvpz * nCz
                normalx, normaly, normalz = _normalize3(thread, normalx, normaly, normalz)

                a0x = guide_vertices[g0, 0]
                a0y = guide_vertices[g0, 1]
                a0z = guide_vertices[g0, 2]
                a1x = guide_vertices[g1, 0]
                a1y = guide_vertices[g1, 1]
                a1z = guide_vertices[g1, 2]
                a2x = guide_vertices[g2, 0]
                a2y = guide_vertices[g2, 1]
                a2z = guide_vertices[g2, 2]

                ppx, ppy, ppz = _evaluate_point_phong(
                    thread,
                    a0x,
                    a0y,
                    a0z,
                    a1x,
                    a1y,
                    a1z,
                    a2x,
                    a2y,
                    a2z,
                    uvpx,
                    uvpy,
                    uvpz,
                    nAx,
                    nAy,
                    nAz,
                    nBx,
                    nBy,
                    nBz,
                    nCx,
                    nCy,
                    nCz,
                    d.half_surface_thickness,
                )

                puvx = uvx * a0x + uvy * a1x + uvz * a2x
                puvy = uvx * a0y + uvy * a1y + uvz * a2y
                puvz = uvx * a0z + uvy * a1z + uvz * a2z

                pprojx = uvpx * a0x + uvpy * a1x + uvpz * a2x
                pprojy = uvpx * a0y + uvpy * a1y + uvpz * a2y
                pprojz = uvpx * a0z + uvpy * a1z + uvpz * a2z

                offset_along = skin_offset_along_normal[info_idx]
                outx = ppx + offset_along * normalx + puvx - pprojx
                outy = ppy + offset_along * normaly + puvy - pprojy
                outz = ppz + offset_along * normalz + puvz - pprojz

                out_idx = d.skinned_vertices_offset + threadIndex
                skinned_vertices[out_idx, 0] = outx
                skinned_vertices[out_idx, 1] = outy
                skinned_vertices[out_idx, 2] = outz

                threadIndex += stride


@cp.kernel
def interpolateSkinnedSoftBodyVertices(
    data,
    guide_vertices,
    guide_tetrahedra,
    skin_uvw,
    skin_tet_id,
    skinned_vertices,
    max_nb_skinned_vertices,
    BLOCK_SIZE: cp.constexpr = 256,
):
    """Port of interpolateSkinnedSoftBodyVertices.

    guide_tetrahedra: [T, 4] local per-batch indices
    skin_uvw: [S, 3], skin_tet_id: [S], skinned_vertices: [S, 3]
    """
    grid_x = cp.ceildiv(max_nb_skinned_vertices, BLOCK_SIZE)
    with cp.Kernel(grid_x, data.shape[0], threads=BLOCK_SIZE) as (bx, by, block):
        d = data[by]
        stride = grid_x * BLOCK_SIZE
        for tid, _thread in block.threads():
            threadIndex = bx * BLOCK_SIZE + tid
            while threadIndex < d.skinned_vertices_count:
                info_idx = d.skinning_info_offset + threadIndex
                tet_id = skin_tet_id[info_idx]
                tet_idx = d.guide_tetrahedra_offset + tet_id

                l0 = guide_tetrahedra[tet_idx, 0]
                l1 = guide_tetrahedra[tet_idx, 1]
                l2 = guide_tetrahedra[tet_idx, 2]
                l3 = guide_tetrahedra[tet_idx, 3]

                g0 = d.guide_vertices_offset + l0
                g1 = d.guide_vertices_offset + l1
                g2 = d.guide_vertices_offset + l2
                g3 = d.guide_vertices_offset + l3

                u = skin_uvw[info_idx, 0]
                v = skin_uvw[info_idx, 1]
                w = skin_uvw[info_idx, 2]
                s = 1.0 - u - v - w

                p0x = guide_vertices[g0, 0]
                p0y = guide_vertices[g0, 1]
                p0z = guide_vertices[g0, 2]
                p1x = guide_vertices[g1, 0]
                p1y = guide_vertices[g1, 1]
                p1z = guide_vertices[g1, 2]
                p2x = guide_vertices[g2, 0]
                p2y = guide_vertices[g2, 1]
                p2z = guide_vertices[g2, 2]
                p3x = guide_vertices[g3, 0]
                p3y = guide_vertices[g3, 1]
                p3z = guide_vertices[g3, 2]

                out_idx = d.skinned_vertices_offset + threadIndex
                skinned_vertices[out_idx, 0] = u * p0x + v * p1x + w * p2x + s * p3x
                skinned_vertices[out_idx, 1] = u * p0y + v * p1y + w * p2y + s * p3y
                skinned_vertices[out_idx, 2] = u * p0z + v * p1z + w * p2z + s * p3z

                threadIndex += stride
