"""Capybara port of PhysX gpucommon/CUDA/MemCopyBalanced.cu.

This file preserves kernel names and control-flow intent.
Memory addresses in CopyDesc are represented as element offsets into
explicit word buffers for DSL portability.
"""

import capybara as cp


@cp.struct
class CopyDesc:
    # Original CUDA stores raw size_t pointers; this port stores u64 offsets
    # into src_words/dst_words buffers.
    dest: cp.uint64
    source: cp.uint64
    bytes: cp.uint64
    pad: cp.uint64


@cp.kernel
def MemCopyBalanced(
    desc,
    src_words,
    dst_words,
    count,
    COPY_KERNEL_WARPS_PER_BLOCK: cp.constexpr = 4,
    BLOCK_SIZE: cp.constexpr = 128,
):
    """Port of MemCopyBalanced kernel + copyBalanced template body.

    Expected launch config for parity:
      BLOCK_SIZE == 32 * COPY_KERNEL_WARPS_PER_BLOCK
    """
    with cp.Kernel(count, threads=BLOCK_SIZE) as (bx, block):
        # Shared descriptor tile: one descriptor per warp in the block.
        shared_desc = block.alloc_struct(CopyDesc, COPY_KERNEL_WARPS_PER_BLOCK)

        # Map to CUDA's (threadIdx.y, threadIdx.x) with y=warp_idx, x=lane.
        for warp_idx, lane, thread in block.threads(
            COPY_KERNEL_WARPS_PER_BLOCK, 32, barrier=True
        ):
            if lane == 0:
                shared_desc.dest[warp_idx] = desc.dest[bx]
                shared_desc.source[warp_idx] = desc.source[bx]
                shared_desc.bytes[warp_idx] = desc.bytes[bx]
                shared_desc.pad[warp_idx] = desc.pad[bx]

        for warp_idx, lane, _thread in block.threads(
            COPY_KERNEL_WARPS_PER_BLOCK, 32
        ):
            src_base = cp.int32(shared_desc.source[warp_idx])
            dst_base = cp.int32(shared_desc.dest[warp_idx])
            size_words = cp.int32(shared_desc.bytes[warp_idx]) // 4

            group_thread_idx = lane + warp_idx * 32
            stride = 32 * COPY_KERNEL_WARPS_PER_BLOCK
            a = group_thread_idx
            while a < size_words:
                dst_words[dst_base + a] = src_words[src_base + a]
                a += stride


@cp.kernel
def clampMaxValue(value, maxValue, BLOCK_SIZE: cp.constexpr = 32):
    """Port of clampMaxValue kernel."""
    with cp.Kernel(1, threads=BLOCK_SIZE) as (_bx, block):
        for tid, _thread in block.threads()[:1]:
            if value[0] > maxValue:
                value[0] = maxValue


@cp.kernel
def clampMaxValues(value0, value1, value2, maxValue, BLOCK_SIZE: cp.constexpr = 32):
    """Port of clampMaxValues kernel."""
    with cp.Kernel(1, threads=BLOCK_SIZE) as (_bx, block):
        for tid, _thread in block.threads()[:1]:
            if value0[0] > maxValue:
                value0[0] = maxValue
            if value1[0] > maxValue:
                value1[0] = maxValue
            if value2[0] > maxValue:
                value2[0] = maxValue
