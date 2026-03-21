"""Capybara DSL port of gpucommon/CUDA/utility.cu.

This file ports the whole utility.cu translation unit as a first PTX
replacement candidate. It keeps kernel names aligned with CUDA:

- interleaveBuffers
- normalVectorsAreaWeighted
- zeroNormals
- normalizeNormals
- interpolateSkinnedClothVertices
- interpolateSkinnedSoftBodyVertices

Important: this draft uses an explicit-array/offset model for skinning data
instead of raw C++ pointer-rich structs (PxTypedBoundedData/Px*EmbeddingInfo).
That keeps the kernel logic apple-to-apple, but ABI parity requires a host-side
packing adapter before drop-in replacement.
"""

import capybara as cp


BLOCK_SIZE = 256


@cp.struct
class TrimeshSkinningBatch:
    guide_vertices_offset: cp.int32
    guide_vertices_count: cp.int32
    guide_normals_offset: cp.int32
    guide_triangles_offset: cp.int32
    skinned_vertices_offset: cp.int32
    skinned_vertices_count: cp.int32
    skinning_info_offset: cp.int32
    half_surface_thickness: cp.float32
    nb_guide_triangles: cp.int32


@cp.struct
class TetmeshSkinningBatch:
    guide_vertices_offset: cp.int32
    guide_tetrahedra_offset: cp.int32
    skinned_vertices_offset: cp.int32
    skinned_vertices_count: cp.int32
    skinning_info_offset: cp.int32


@cp.inline
def _normalize_safe(thread, x, y, z):
    """Unit vector if |v|^2 is significant, else ~0.

    Avoid ``thread.where`` here: inside ``@cp.inline`` there may be no unit scope, so
    ``cp.select`` operands mix bare scalars with ``!cp.val`` and MLIR verification fails.

    Branchless mask approximates ``v/|v|`` when ``|v|^2 >> eps`` and fades out when tiny.
    """
    mag2 = x * x + y * y + z * z
    eps = cp.float32(1.0e-20)
    tiny = cp.float32(1.0e-30)
    mask = mag2 / (mag2 + eps)
    inv = thread.rsqrt(mag2 + tiny)
    return (
        x * mask * inv,
        y * mask * inv,
        z * mask * inv,
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
    ndx, ndy, ndz = _normalize_safe(thread, dirx, diry, dirz)

    alpha = cp.float32(0.625)
    # Branchless: offset = tanh(alpha/h)*h for h>0, else 0 — use (h+eps) in tanh and *h.
    # Avoid ``thread.where`` in @cp.inline (same cp.select / scope issue as _normalize_safe).
    eps_h = cp.float32(1.0e-30)
    ratio_raw = _tanh_approx(thread, alpha / (half_surface_thickness + eps_h))
    offset = ratio_raw * half_surface_thickness

    return qx + offset * ndx, qy + offset * ndy, qz + offset * ndz


@cp.kernel
def interleaveBuffers(vertices, normals, interleaved_result, length):
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
def normalVectorsAreaWeighted(
    batches, guide_vertices, guide_normals, guide_triangles, grid_x, num_batches
):
    with cp.Kernel(grid_x, num_batches, threads=BLOCK_SIZE) as (bx, by, block):
        # Atomic target grammar in current Capybara expects fully-indexed view
        # forms for tuple subscripts (view[i, j]) in thread.atomic_add.
        guide_normals_2d = guide_normals.view(guide_normals.shape[0], 3)
        guide_vertices_offset = batches.guide_vertices_offset[by]
        guide_normals_offset = batches.guide_normals_offset[by]
        guide_triangles_offset = batches.guide_triangles_offset[by]
        nb_guide_triangles = batches.nb_guide_triangles[by]
        x_dim = grid_x * BLOCK_SIZE
        start = bx * BLOCK_SIZE
        loop_end = nb_guide_triangles
        for tid, thread in block.threads():
            idx = start + tid
            while idx < loop_end:
                tri_base = guide_triangles_offset + 3 * idx
                i0 = guide_triangles[tri_base]
                i1 = guide_triangles[tri_base + 1]
                i2 = guide_triangles[tri_base + 2]

                p0x = guide_vertices[guide_vertices_offset + i0, 0]
                p0y = guide_vertices[guide_vertices_offset + i0, 1]
                p0z = guide_vertices[guide_vertices_offset + i0, 2]
                p1x = guide_vertices[guide_vertices_offset + i1, 0]
                p1y = guide_vertices[guide_vertices_offset + i1, 1]
                p1z = guide_vertices[guide_vertices_offset + i1, 2]
                p2x = guide_vertices[guide_vertices_offset + i2, 0]
                p2y = guide_vertices[guide_vertices_offset + i2, 1]
                p2z = guide_vertices[guide_vertices_offset + i2, 2]

                e10x = p1x - p0x
                e10y = p1y - p0y
                e10z = p1z - p0z
                e20x = p2x - p0x
                e20y = p2y - p0y
                e20z = p2z - p0z

                nx = e10y * e20z - e10z * e20y
                ny = e10z * e20x - e10x * e20z
                nz = e10x * e20y - e10y * e20x

                n0 = guide_normals_offset + i0
                n1 = guide_normals_offset + i1
                n2 = guide_normals_offset + i2
                thread.atomic_add(guide_normals_2d[n0, 0], nx)
                thread.atomic_add(guide_normals_2d[n0, 1], ny)
                thread.atomic_add(guide_normals_2d[n0, 2], nz)
                thread.atomic_add(guide_normals_2d[n1, 0], nx)
                thread.atomic_add(guide_normals_2d[n1, 1], ny)
                thread.atomic_add(guide_normals_2d[n1, 2], nz)
                thread.atomic_add(guide_normals_2d[n2, 0], nx)
                thread.atomic_add(guide_normals_2d[n2, 1], ny)
                thread.atomic_add(guide_normals_2d[n2, 2], nz)

                idx += x_dim


@cp.kernel
def zeroNormals(batches, guide_normals, grid_x, num_batches):
    with cp.Kernel(grid_x, num_batches, threads=BLOCK_SIZE) as (bx, by, block):
        guide_vertices_count = batches.guide_vertices_count[by]
        guide_normals_offset = batches.guide_normals_offset[by]
        x_dim = grid_x * BLOCK_SIZE
        start = bx * BLOCK_SIZE
        loop_end = guide_vertices_count
        for tid, thread in block.threads():
            idx = start + tid
            while idx < loop_end:
                n = guide_normals_offset + idx
                guide_normals[n, 0] = cp.float32(0.0)
                guide_normals[n, 1] = cp.float32(0.0)
                guide_normals[n, 2] = cp.float32(0.0)
                idx += x_dim


@cp.kernel
def normalizeNormals(batches, guide_normals, grid_x, num_batches):
    with cp.Kernel(grid_x, num_batches, threads=BLOCK_SIZE) as (bx, by, block):
        guide_vertices_count = batches.guide_vertices_count[by]
        guide_normals_offset = batches.guide_normals_offset[by]
        x_dim = grid_x * BLOCK_SIZE
        start = bx * BLOCK_SIZE
        loop_end = guide_vertices_count
        for tid, thread in block.threads():
            idx = start + tid
            while idx < loop_end:
                n = guide_normals_offset + idx
                x = guide_normals[n, 0]
                y = guide_normals[n, 1]
                z = guide_normals[n, 2]
                x, y, z = _normalize_safe(thread, x, y, z)
                guide_normals[n, 0] = x
                guide_normals[n, 1] = y
                guide_normals[n, 2] = z
                idx += x_dim


@cp.kernel
def interpolateSkinnedClothVertices(
    batches,
    guide_vertices,
    guide_normals,
    guide_triangles,
    skin_uv,
    skin_offset_along_normal,
    skin_tri_id,
    skinned_vertices,
    grid_x,
    num_batches,
):
    with cp.Kernel(grid_x, num_batches, threads=BLOCK_SIZE) as (bx, by, block):
        guide_vertices_offset = batches.guide_vertices_offset[by]
        guide_normals_offset = batches.guide_normals_offset[by]
        guide_triangles_offset = batches.guide_triangles_offset[by]
        skinned_vertices_offset = batches.skinned_vertices_offset[by]
        skinned_vertices_count = batches.skinned_vertices_count[by]
        skinning_info_offset = batches.skinning_info_offset[by]
        half_surface_thickness = batches.half_surface_thickness[by]
        x_dim = grid_x * BLOCK_SIZE
        start = bx * BLOCK_SIZE
        loop_end = skinned_vertices_count
        for tid, thread in block.threads():
            idx = start + tid
            while idx < loop_end:
                info_idx = skinning_info_offset + idx
                uvx = skin_uv[info_idx, 0]
                uvy = skin_uv[info_idx, 1]
                tri_id = skin_tri_id[info_idx]
                offset_along_n = skin_offset_along_normal[info_idx]
                uvz = cp.float32(1.0) - uvx - uvy

                uvpx = uvx if uvx > cp.float32(0.0) else cp.float32(0.0)
                uvpy = uvy if uvy > cp.float32(0.0) else cp.float32(0.0)
                uvpz = uvz if uvz > cp.float32(0.0) else cp.float32(0.0)
                s = uvpx + uvpy + uvpz
                if s > cp.float32(0.0):
                    invs = cp.float32(1.0) / s
                    uvpx *= invs
                    uvpy *= invs
                    uvpz *= invs

                tri_base = guide_triangles_offset + 3 * tri_id
                i0 = guide_triangles[tri_base]
                i1 = guide_triangles[tri_base + 1]
                i2 = guide_triangles[tri_base + 2]

                a0 = guide_vertices_offset + i0
                a1 = guide_vertices_offset + i1
                a2 = guide_vertices_offset + i2
                ax = guide_vertices[a0, 0]
                ay = guide_vertices[a0, 1]
                az = guide_vertices[a0, 2]
                bxv = guide_vertices[a1, 0]
                byv = guide_vertices[a1, 1]
                bzv = guide_vertices[a1, 2]
                cx = guide_vertices[a2, 0]
                cy = guide_vertices[a2, 1]
                cz = guide_vertices[a2, 2]

                n0 = guide_normals_offset + i0
                n1 = guide_normals_offset + i1
                n2 = guide_normals_offset + i2
                nAx = guide_normals[n0, 0]
                nAy = guide_normals[n0, 1]
                nAz = guide_normals[n0, 2]
                nBx = guide_normals[n1, 0]
                nBy = guide_normals[n1, 1]
                nBz = guide_normals[n1, 2]
                nCx = guide_normals[n2, 0]
                nCy = guide_normals[n2, 1]
                nCz = guide_normals[n2, 2]

                nx = uvpx * nAx + uvpy * nBx + uvpz * nCx
                ny = uvpx * nAy + uvpy * nBy + uvpz * nCy
                nz = uvpx * nAz + uvpy * nBz + uvpz * nCz
                nx, ny, nz = _normalize_safe(thread, nx, ny, nz)

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

                puvx = uvx * ax + uvy * bxv + uvz * cx
                puvy = uvx * ay + uvy * byv + uvz * cy
                puvz = uvx * az + uvy * bzv + uvz * cz
                pprojx = uvpx * ax + uvpy * bxv + uvpz * cx
                pprojy = uvpx * ay + uvpy * byv + uvpz * cy
                pprojz = uvpx * az + uvpy * bzv + uvpz * cz

                out = skinned_vertices_offset + idx
                skinned_vertices[out, 0] = ppx + offset_along_n * nx + puvx - pprojx
                skinned_vertices[out, 1] = ppy + offset_along_n * ny + puvy - pprojy
                skinned_vertices[out, 2] = ppz + offset_along_n * nz + puvz - pprojz

                idx += x_dim


@cp.kernel
def interpolateSkinnedSoftBodyVertices(
    batches,
    guide_vertices,
    guide_tetrahedra,
    skin_uvw,
    skin_tet_id,
    skinned_vertices,
    grid_x,
    num_batches,
):
    with cp.Kernel(grid_x, num_batches, threads=BLOCK_SIZE) as (bx, by, block):
        guide_vertices_offset = batches.guide_vertices_offset[by]
        guide_tetrahedra_offset = batches.guide_tetrahedra_offset[by]
        skinned_vertices_offset = batches.skinned_vertices_offset[by]
        skinned_vertices_count = batches.skinned_vertices_count[by]
        skinning_info_offset = batches.skinning_info_offset[by]
        x_dim = grid_x * BLOCK_SIZE
        start = bx * BLOCK_SIZE
        loop_end = skinned_vertices_count
        for tid, thread in block.threads():
            idx = start + tid
            while idx < loop_end:
                info_idx = skinning_info_offset + idx
                ux = skin_uvw[info_idx, 0]
                uy = skin_uvw[info_idx, 1]
                uz = skin_uvw[info_idx, 2]
                tet_id = skin_tet_id[info_idx]
                s = cp.float32(1.0) - ux - uy - uz

                tet_base = guide_tetrahedra_offset + 4 * tet_id
                i0 = guide_tetrahedra[tet_base]
                i1 = guide_tetrahedra[tet_base + 1]
                i2 = guide_tetrahedra[tet_base + 2]
                i3 = guide_tetrahedra[tet_base + 3]

                p0 = guide_vertices_offset + i0
                p1 = guide_vertices_offset + i1
                p2 = guide_vertices_offset + i2
                p3 = guide_vertices_offset + i3

                x = (
                    ux * guide_vertices[p0, 0]
                    + uy * guide_vertices[p1, 0]
                    + uz * guide_vertices[p2, 0]
                    + s * guide_vertices[p3, 0]
                )
                y = (
                    ux * guide_vertices[p0, 1]
                    + uy * guide_vertices[p1, 1]
                    + uz * guide_vertices[p2, 1]
                    + s * guide_vertices[p3, 1]
                )
                z = (
                    ux * guide_vertices[p0, 2]
                    + uy * guide_vertices[p1, 2]
                    + uz * guide_vertices[p2, 2]
                    + s * guide_vertices[p3, 2]
                )

                out = skinned_vertices_offset + idx
                skinned_vertices[out, 0] = x
                skinned_vertices[out, 1] = y
                skinned_vertices[out, 2] = z

                idx += x_dim

