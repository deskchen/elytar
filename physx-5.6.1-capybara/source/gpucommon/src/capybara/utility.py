"""Capybara DSL port of gpucommon/CUDA/utility.cu.

Kernel names match CUDA for PTX replacement. interleaveBuffers is ABI-compatible
(drop-in PTX replacement). Skinning kernels use flat-arg adapters requiring
host-side struct unpacking (one launch per batch via PxgDeformableSkinning.cpp
guarded by ELYTAR_CAPYBARA_SKINNING).

Parameter notes for skinning kernels:
  guide_vertices  — float32[N,4]  PxVec4 stride (x,y,z,invMass); only xyz read
  guide_normals   — float32[N,3]  compact PxVec3
  guide_triangles — int32[M*3]    flat PxU32 triangle index buffer
  skin_info (cloth)  — int32[K,4] reinterpret of PxTriangleMeshEmbeddingInfo
  skin_info (volume) — int32[K,4] reinterpret of PxTetrahedronMeshEmbeddingInfo
  skinned_vertices — float32[K,3] compact PxVec3 output (pinned host memory)
"""

import capybara as cp
from physx_math import PxVec3


BLOCK_SIZE = 256


@cp.inline
def _tanh_approx(thread, x):
    two_x = cp.float32(2.0) * x
    e = thread.exp(two_x)
    return (e - cp.float32(1.0)) / (e + cp.float32(1.0))


@cp.inline
def _evaluate_point_phong(thread, a, b, c, uvw, nA, nB, nC, half_surface_thickness):
    """Phong interpolation with asymptotic normal offset (alpha=0.625)."""
    q = a.scale(uvw.x)
    q = q.add(b.scale(uvw.y))
    q = q.add(c.scale(uvw.z))

    da = q.sub(a)
    projA = q.sub(nA.scale(da.dot(nA)))
    db = q.sub(b)
    projB = q.sub(nB.scale(db.dot(nB)))
    dc = q.sub(c)
    projC = q.sub(nC.scale(dc.dot(nC)))

    qStar = projA.scale(uvw.x)
    qStar = qStar.add(projB.scale(uvw.y))
    qStar = qStar.add(projC.scale(uvw.z))
    diff = qStar.sub(q)
    nd, dir_mag = diff.normalize_safe(thread)

    alpha = cp.float32(0.625)
    eps_h = cp.float32(1.0e-30)
    raw_offset = dir_mag * alpha
    ratio_raw = _tanh_approx(thread, raw_offset / (half_surface_thickness + eps_h))
    offset = ratio_raw * half_surface_thickness
    return q.add(nd.scale(offset))


@cp.kernel
def interleaveBuffers(vertices, normals, length, interleaved_result):
    """CUDA ABI: (float4* vertices, float4* normals, PxU32 length, PxVec3* result).
    Reads float4 [N,4] (w ignored). Writes PxVec3 [2N,3]."""
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
    with cp.Kernel(grid_x, threads=BLOCK_SIZE) as (bx, block):
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

                p0 = PxVec3(x=guide_vertices[i0, 0], y=guide_vertices[i0, 1], z=guide_vertices[i0, 2])
                p1 = PxVec3(x=guide_vertices[i1, 0], y=guide_vertices[i1, 1], z=guide_vertices[i1, 2])
                p2 = PxVec3(x=guide_vertices[i2, 0], y=guide_vertices[i2, 1], z=guide_vertices[i2, 2])
                e10 = p1.sub(p0)
                e20 = p2.sub(p0)
                n = e10.cross(e20)

                thread.atomic_add(guide_normals_2d[i0, 0], n.x)
                thread.atomic_add(guide_normals_2d[i0, 1], n.y)
                thread.atomic_add(guide_normals_2d[i0, 2], n.z)
                thread.atomic_add(guide_normals_2d[i1, 0], n.x)
                thread.atomic_add(guide_normals_2d[i1, 1], n.y)
                thread.atomic_add(guide_normals_2d[i1, 2], n.z)
                thread.atomic_add(guide_normals_2d[i2, 0], n.x)
                thread.atomic_add(guide_normals_2d[i2, 1], n.y)
                thread.atomic_add(guide_normals_2d[i2, 2], n.z)

                idx += x_dim


@cp.kernel
def normalizeNormals(guide_normals, guide_vertices_count, grid_x):
    with cp.Kernel(grid_x, threads=BLOCK_SIZE) as (bx, block):
        x_dim = grid_x * BLOCK_SIZE
        start = bx * BLOCK_SIZE
        for tid, thread in block.threads():
            idx = start + tid
            while idx < guide_vertices_count:
                v = PxVec3(x=guide_normals[idx, 0], y=guide_normals[idx, 1], z=guide_normals[idx, 2])
                v, _ = v.normalize_safe(thread)
                guide_normals[idx, 0] = v.x
                guide_normals[idx, 1] = v.y
                guide_normals[idx, 2] = v.z
                idx += x_dim


@cp.kernel
def interpolateSkinnedClothVertices(
    guide_vertices, guide_normals, guide_triangles, skin_info,
    skinned_vertices, skinned_count, half_surface_thickness, grid_x,
):
    """Flat-arg cloth skinning. Host unpacks PxTrimeshSkinningGpuData per batch."""
    with cp.Kernel(grid_x, threads=BLOCK_SIZE) as (bx, block):
        x_dim = grid_x * BLOCK_SIZE
        start = bx * BLOCK_SIZE
        for tid, thread in block.threads():
            idx = start + tid
            while idx < skinned_count:
                # Unpack PxTriangleMeshEmbeddingInfo via bitcast
                s0 = skin_info[idx, 0]
                s1 = skin_info[idx, 1]
                s2 = skin_info[idx, 2]
                tri_id = skin_info[idx, 3]
                uvx = thread.bitcast(s0, cp.float32)
                uvy = thread.bitcast(s1, cp.float32)
                offset_along_n = thread.bitcast(s2, cp.float32)
                uvz = cp.float32(1.0) - uvx - uvy

                # Clamp barycentrics to [0,1] and renormalize (scalar — avoids
                # struct reassignment inside conditional)
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

                a = PxVec3(x=guide_vertices[i0, 0], y=guide_vertices[i0, 1], z=guide_vertices[i0, 2])
                b = PxVec3(x=guide_vertices[i1, 0], y=guide_vertices[i1, 1], z=guide_vertices[i1, 2])
                c = PxVec3(x=guide_vertices[i2, 0], y=guide_vertices[i2, 1], z=guide_vertices[i2, 2])
                nA = PxVec3(x=guide_normals[i0, 0], y=guide_normals[i0, 1], z=guide_normals[i0, 2])
                nB = PxVec3(x=guide_normals[i1, 0], y=guide_normals[i1, 1], z=guide_normals[i1, 2])
                nC = PxVec3(x=guide_normals[i2, 0], y=guide_normals[i2, 1], z=guide_normals[i2, 2])

                n = nA.scale(uvpx)
                n = n.add(nB.scale(uvpy))
                n = n.add(nC.scale(uvpz))
                n, _ = n.normalize_safe(thread)

                uvwP = PxVec3(x=uvpx, y=uvpy, z=uvpz)
                pp = _evaluate_point_phong(thread, a, b, c, uvwP, nA, nB, nC,
                                           half_surface_thickness)
                puvw = a.scale(uvx)
                puvw = puvw.add(b.scale(uvy))
                puvw = puvw.add(c.scale(uvz))
                pproj = a.scale(uvpx)
                pproj = pproj.add(b.scale(uvpy))
                pproj = pproj.add(c.scale(uvpz))
                result = pp.add(n.scale(offset_along_n))
                result = result.add(puvw.sub(pproj))

                skinned_vertices[idx, 0] = result.x
                skinned_vertices[idx, 1] = result.y
                skinned_vertices[idx, 2] = result.z

                idx += x_dim


@cp.kernel
def interpolateSkinnedSoftBodyVertices(
    guide_vertices, guide_tetrahedra, skin_info,
    skinned_vertices, skinned_count, grid_x,
):
    """Flat-arg volume skinning. Host unpacks PxTetmeshSkinningGpuData per batch."""
    with cp.Kernel(grid_x, threads=BLOCK_SIZE) as (bx, block):
        x_dim = grid_x * BLOCK_SIZE
        start = bx * BLOCK_SIZE
        for tid, thread in block.threads():
            idx = start + tid
            while idx < skinned_count:
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

                p0 = PxVec3(x=guide_vertices[i0, 0], y=guide_vertices[i0, 1], z=guide_vertices[i0, 2])
                p1 = PxVec3(x=guide_vertices[i1, 0], y=guide_vertices[i1, 1], z=guide_vertices[i1, 2])
                p2 = PxVec3(x=guide_vertices[i2, 0], y=guide_vertices[i2, 1], z=guide_vertices[i2, 2])
                p3 = PxVec3(x=guide_vertices[i3, 0], y=guide_vertices[i3, 1], z=guide_vertices[i3, 2])
                pt = p0.scale(ux)
                pt = pt.add(p1.scale(uy))
                pt = pt.add(p2.scale(uz))
                pt = pt.add(p3.scale(s))

                skinned_vertices[idx, 0] = pt.x
                skinned_vertices[idx, 1] = pt.y
                skinned_vertices[idx, 2] = pt.z

                idx += x_dim
