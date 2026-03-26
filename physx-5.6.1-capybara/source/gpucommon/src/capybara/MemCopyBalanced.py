"""Capybara DSL port of MemCopyBalanced.cu — clamp kernels only.

The main MemCopyBalanced kernel is kept as CUDA (shared memory + 2D
warp-level copy).  Only clampMaxValue and clampMaxValues are ported.
"""

import capybara as cp

# --------------------------------------------------------------------------
# clampMaxValue: clamp a single PxU32 device value to a maximum.
# CUDA: extern "C" __global__ void clampMaxValue(PxU32* value, const PxU32 maxValue)
# Launch: grid=(1,1,1) block=(1,1,1)
# --------------------------------------------------------------------------
@cp.kernel
def clampMaxValue(value, max_value):
    with cp.Kernel(1, threads=1) as (bx, block):
        for tid, thread in block.threads():
            v = value[0]
            if v > max_value:
                value[0] = max_value


# --------------------------------------------------------------------------
# clampMaxValues: clamp three PxU32 device values to a maximum.
# CUDA: extern "C" __global__ void clampMaxValues(
#           PxU32* value0, PxU32* value1, PxU32* value2, const PxU32 maxValue)
# Launch: grid=(1,1,1) block=(1,1,1)
# --------------------------------------------------------------------------
@cp.kernel
def clampMaxValues(value0, value1, value2, max_value):
    with cp.Kernel(1, threads=1) as (bx, block):
        for tid, thread in block.threads():
            v0 = value0[0]
            if v0 > max_value:
                value0[0] = max_value
            v1 = value1[0]
            if v1 > max_value:
                value1[0] = max_value
            v2 = value2[0]
            if v2 > max_value:
                value2[0] = max_value
