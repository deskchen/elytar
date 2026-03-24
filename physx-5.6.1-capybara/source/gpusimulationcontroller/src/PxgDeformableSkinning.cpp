// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Copyright (c) 2008-2025 NVIDIA Corporation. All rights reserved.
// Copyright (c) 2004-2008 AGEIA Technologies, Inc. All rights reserved.
// Copyright (c) 2001-2004 NovodeX AG. All rights reserved.

#include "PxgDeformableSkinning.h"

#include "foundation/PxUserAllocated.h"

#include "PxPhysXGpu.h"
#include "PxgKernelWrangler.h"
#include "PxgKernelIndices.h"
#include "GuAABBTree.h"
#include "foundation/PxMathUtils.h"


namespace physx
{
	PxgDeformableSkinning::PxgDeformableSkinning(PxgKernelLauncher& kernelLauncher)
	{
		mKernelLauncher = kernelLauncher;
	}

	// ---------------------------------------------------------------------------
	// CUDA path: single struct-pointer launch (original ABI)
	// ---------------------------------------------------------------------------

	void PxgDeformableSkinning::computeNormalVectors(
		PxTrimeshSkinningGpuData* skinningDataArrayD, PxU32 arrayLength,
		CUstream stream, PxU32 numGpuThreads)
	{
#if defined(ELYTAR_CAPYBARA_SKINNING)
		// Without a host mirror we must copy from device.  This path is the
		// fallback for callers that did not supply the host pointer overload.
		PxArray<PxTrimeshSkinningGpuData> hostBuf(arrayLength);
		mKernelLauncher.getCudaContextManager()->getCudaContext()->memcpyDtoH(
			hostBuf.begin(), reinterpret_cast<CUdeviceptr>(skinningDataArrayD),
			arrayLength * sizeof(PxTrimeshSkinningGpuData));
		computeNormalVectors(skinningDataArrayD, hostBuf.begin(), arrayLength, stream, numGpuThreads);
#else
		physx::PxScopedCudaLock _lock(*mKernelLauncher.getCudaContextManager());

		const PxU32 numThreadsPerBlock = 256;
		const PxU32 numBlocks = (numGpuThreads + numThreadsPerBlock - 1) / numThreadsPerBlock;

		const PxU32 numThreadsPerBlockSmallKernels = 1024;
		const PxU32 numBlocksSmallKernels = (numGpuThreads + numThreadsPerBlockSmallKernels - 1) / numThreadsPerBlockSmallKernels;

		mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_ZeroNormals, numBlocksSmallKernels, arrayLength, 1, numThreadsPerBlockSmallKernels, 1, 1, 0, stream,
			skinningDataArrayD);

		mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_ComputeNormals, numBlocks, arrayLength, 1, numThreadsPerBlock, 1, 1, 0, stream,
			skinningDataArrayD);

		mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_NormalizeNormals, numBlocksSmallKernels, arrayLength, 1, numThreadsPerBlockSmallKernels, 1, 1, 0, stream,
			skinningDataArrayD);
#endif
	}

	void PxgDeformableSkinning::evaluateVerticesEmbeddedIntoSurface(
		PxTrimeshSkinningGpuData* skinningDataArrayD, PxU32 arrayLength,
		CUstream stream, PxU32 numGpuThreads)
	{
#if defined(ELYTAR_CAPYBARA_SKINNING)
		PxArray<PxTrimeshSkinningGpuData> hostBuf(arrayLength);
		mKernelLauncher.getCudaContextManager()->getCudaContext()->memcpyDtoH(
			hostBuf.begin(), reinterpret_cast<CUdeviceptr>(skinningDataArrayD),
			arrayLength * sizeof(PxTrimeshSkinningGpuData));
		evaluateVerticesEmbeddedIntoSurface(skinningDataArrayD, hostBuf.begin(), arrayLength, stream, numGpuThreads);
#else
		physx::PxScopedCudaLock _lock(*mKernelLauncher.getCudaContextManager());
		const PxU32 numThreadsPerBlock = 256;
		const PxU32 numBlocks = (numGpuThreads + numThreadsPerBlock - 1) / numThreadsPerBlock;
		mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_InterpolateSkinnedClothVertices, numBlocks, arrayLength, 1, numThreadsPerBlock, 1, 1, 0, stream,
			skinningDataArrayD);
#endif
	}

	void PxgDeformableSkinning::evaluateVerticesEmbeddedIntoVolume(
		PxTetmeshSkinningGpuData* skinningDataArrayD, PxU32 arrayLength,
		CUstream stream, PxU32 numGpuThreads)
	{
#if defined(ELYTAR_CAPYBARA_SKINNING)
		PxArray<PxTetmeshSkinningGpuData> hostBuf(arrayLength);
		mKernelLauncher.getCudaContextManager()->getCudaContext()->memcpyDtoH(
			hostBuf.begin(), reinterpret_cast<CUdeviceptr>(skinningDataArrayD),
			arrayLength * sizeof(PxTetmeshSkinningGpuData));
		evaluateVerticesEmbeddedIntoVolume(skinningDataArrayD, hostBuf.begin(), arrayLength, stream, numGpuThreads);
#else
		physx::PxScopedCudaLock _lock(*mKernelLauncher.getCudaContextManager());
		const PxU32 numThreadsPerBlock = 256;
		const PxU32 numBlocks = (numGpuThreads + numThreadsPerBlock - 1) / numThreadsPerBlock;
		mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_InterpolateSkinnedSoftBodyVertices, numBlocks, arrayLength, 1, numThreadsPerBlock, 1, 1, 0, stream,
			skinningDataArrayD);
#endif
	}

#if defined(ELYTAR_CAPYBARA_SKINNING)
	// ---------------------------------------------------------------------------
	// Capybara path: host-pointer overloads — flat per-batch launches
	//
	// The caller (e.g. SnippetDeformableSurfaceSkinning) already has a host-side
	// mirror of the struct array (HostAndDeviceBuffer::mHostData, which is pinned
	// memory).  We unpack the GPU sub-buffer pointers from that mirror and launch
	// one Capybara kernel call per batch, matching the flat PTX parameter layout.
	//
	// Capybara kernel parameter mapping:
	//   guide_vertices  — d.guideVerticesD.data   (PxVec4 stride, float32[N,4])
	//   guide_normals   — d.guideNormalsD.data    (compact PxVec3, float32[N,3])
	//   guide_triangles — d.guideTrianglesD        (flat PxU32 array)
	//   skin_info       — d.skinningInfoPerVertexD (PxTriangle/TetEmbeddingInfo AoS
	//                     reinterpreted as int32[K,4])
	//   skinned_vertices— d.skinnedVerticesD.data  (compact PxVec3, pinned memory)
	// ---------------------------------------------------------------------------

	void PxgDeformableSkinning::computeNormalVectors(
		PxTrimeshSkinningGpuData* /*skinningDataArrayD*/,
		const PxTrimeshSkinningGpuData* skinningDataArrayH,
		PxU32 arrayLength, CUstream stream, PxU32 numGpuThreads)
	{
		physx::PxScopedCudaLock _lock(*mKernelLauncher.getCudaContextManager());

		const PxU32 numThreadsPerBlock      = 256;
		const PxU32 numBlocks               = (numGpuThreads + numThreadsPerBlock - 1) / numThreadsPerBlock;
		const PxU32 numThreadsSmall         = 1024;
		const PxU32 numBlocksSmall          = (numGpuThreads + numThreadsSmall - 1) / numThreadsSmall;

		for (PxU32 i = 0; i < arrayLength; i++)
		{
			const PxTrimeshSkinningGpuData& d = skinningDataArrayH[i];

			// guideVerticesD.count is authoritative for vertex/normal count;
			// guideNormalsD.count is 0 (not set by the snippet — see PHYSX_PORT_PRACTICE.md).
			PxVec3* normData       = d.guideNormalsD.data;
			PxU32   vertCount      = d.guideVerticesD.count;
			PxU32   gridXSmall     = numBlocksSmall;

			mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_ZeroNormals,
				numBlocksSmall, 1, 1, numThreadsSmall, 1, 1, 0, stream,
				normData, vertCount, gridXSmall);

			PxVec3* vertData       = d.guideVerticesD.data;
			PxU32*  triData        = d.guideTrianglesD;
			PxU32   nbTris         = d.nbGuideTriangles;
			PxU32   gridX          = numBlocks;

			mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_ComputeNormals,
				numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0, stream,
				vertData, normData, triData, nbTris, gridX);

			mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_NormalizeNormals,
				numBlocksSmall, 1, 1, numThreadsSmall, 1, 1, 0, stream,
				normData, vertCount, gridXSmall);
		}
	}

	void PxgDeformableSkinning::evaluateVerticesEmbeddedIntoSurface(
		PxTrimeshSkinningGpuData* /*skinningDataArrayD*/,
		const PxTrimeshSkinningGpuData* skinningDataArrayH,
		PxU32 arrayLength, CUstream stream, PxU32 numGpuThreads)
	{
		physx::PxScopedCudaLock _lock(*mKernelLauncher.getCudaContextManager());

		const PxU32 numThreadsPerBlock = 256;
		const PxU32 numBlocks          = (numGpuThreads + numThreadsPerBlock - 1) / numThreadsPerBlock;

		for (PxU32 i = 0; i < arrayLength; i++)
		{
			const PxTrimeshSkinningGpuData& d = skinningDataArrayH[i];

			PxVec3*                       vertData    = d.guideVerticesD.data;
			PxVec3*                       normData    = d.guideNormalsD.data;
			PxU32*                        triData     = d.guideTrianglesD;
			PxTriangleMeshEmbeddingInfo*  skinInfo    = d.skinningInfoPerVertexD;
			PxVec3*                       skinnedData = d.skinnedVerticesD.data;
			PxU32                         skinnedCnt  = d.skinnedVerticesD.count;
			PxReal                        halfThick   = d.halfSurfaceThickness;
			PxU32                         gridX       = numBlocks;

			mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_InterpolateSkinnedClothVertices,
				numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0, stream,
				vertData, normData, triData, skinInfo,
				skinnedData, skinnedCnt, halfThick, gridX);
		}
	}

	void PxgDeformableSkinning::evaluateVerticesEmbeddedIntoVolume(
		PxTetmeshSkinningGpuData* /*skinningDataArrayD*/,
		const PxTetmeshSkinningGpuData* skinningDataArrayH,
		PxU32 arrayLength, CUstream stream, PxU32 numGpuThreads)
	{
		physx::PxScopedCudaLock _lock(*mKernelLauncher.getCudaContextManager());

		const PxU32 numThreadsPerBlock = 256;
		const PxU32 numBlocks          = (numGpuThreads + numThreadsPerBlock - 1) / numThreadsPerBlock;

		for (PxU32 i = 0; i < arrayLength; i++)
		{
			const PxTetmeshSkinningGpuData& d = skinningDataArrayH[i];

			PxVec3*                          vertData    = d.guideVerticesD.data;
			PxU32*                           tetData     = d.guideTetrahedraD;
			PxTetrahedronMeshEmbeddingInfo*  skinInfo    = d.skinningInfoPerVertexD;
			PxVec3*                          skinnedData = d.skinnedVerticesD.data;
			PxU32                            skinnedCnt  = d.skinnedVerticesD.count;
			PxU32                            gridX       = numBlocks;

			mKernelLauncher.launchKernelXYZ(PxgKernelIds::util_InterpolateSkinnedSoftBodyVertices,
				numBlocks, 1, 1, numThreadsPerBlock, 1, 1, 0, stream,
				vertData, tetData, skinInfo, skinnedData, skinnedCnt, gridX);
		}
	}

#endif // ELYTAR_CAPYBARA_SKINNING
}
