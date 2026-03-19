#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# generate_ptx.sh  --  Generate PTX files from PhysX .cu kernel files.
#
# Usage:
#   ./scripts/generate_ptx.sh --all                        # all 61 .cu files
#   ./scripts/generate_ptx.sh --cu <path>                  # single .cu file
#   ./scripts/generate_ptx.sh --list "integration;solver"  # named stems only
#   ./scripts/generate_ptx.sh --all --arch compute_90
#
# Options:
#   --all               Process all 61 .cu files across all 6 GPU sub-libraries
#   --cu <path>         Process a single .cu file (absolute or relative to repo root)
#   --list "s1;s2;..."  Process only the named kernel stems (semicolon-separated)
#   --arch <val>        GPU compute capability (default: $PX_PTX_ARCH or compute_86)
#                       Lab GPUs: RTX 3090 = compute_86, H200 = compute_90
#
# Output:  Each <stem>.ptx is written next to its CUDA/ dir inside a PTX/
#          sibling directory:
#            source/<module>/src/CUDA/<stem>.cu
#            source/<module>/src/PTX/<stem>.ptx   (output)
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PHYSX_DIR="${PHYSX_DIR:-${ROOT_DIR}/physx-5.6.1-capybara}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
NVCC="${CUDA_PATH}/bin/nvcc"
PX_PTX_ARCH="${PX_PTX_ARCH:-compute_86}"

MODE=""
SINGLE_CU=""
LIST_STEMS=""   # semicolon-separated stems for --list mode

while [[ $# -gt 0 ]]; do
    case "$1" in
        --all)         MODE="all";    shift ;;
        --cu)          MODE="single"; SINGLE_CU="$2"; shift 2 ;;
        --list)        MODE="list";   LIST_STEMS="$2"; shift 2 ;;
        --arch)        PX_PTX_ARCH="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,22p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown argument: $1.  Use --help for usage."; exit 1 ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "Usage: $0 [--all | --cu <file> | --list \"stem1;stem2\"] [--arch compute_XX]"
    exit 1
fi

# ---- Pre-flight checks ----------------------------------------------------
if [[ ! -x "${NVCC}" ]]; then
    echo "ERROR: nvcc not found at ${NVCC}."
    exit 1
fi

# ---- Superset of all include paths across all 6 sub-libraries -------------
# Using the union ensures every .cu compiles regardless of sub-library.
INCLUDE_FLAGS=(
    -I"${CUDA_PATH}/include"
    -I"${PHYSX_DIR}/include"
    -I"${PHYSX_DIR}/source/common/include"
    -I"${PHYSX_DIR}/source/common/src"
    -I"${PHYSX_DIR}/source/physxgpu/include"
    -I"${PHYSX_DIR}/source/geomutils/include"
    -I"${PHYSX_DIR}/source/geomutils/src"
    -I"${PHYSX_DIR}/source/geomutils/src/contact"
    -I"${PHYSX_DIR}/source/geomutils/src/pcm"
    -I"${PHYSX_DIR}/source/geomutils/src/mesh"
    -I"${PHYSX_DIR}/source/lowlevel/api/include"
    -I"${PHYSX_DIR}/source/lowlevel/api/include/windows"
    -I"${PHYSX_DIR}/source/lowlevel/common/include/pipeline"
    -I"${PHYSX_DIR}/source/lowlevel/common/include/utils"
    -I"${PHYSX_DIR}/source/lowlevelaabb/include"
    -I"${PHYSX_DIR}/source/lowleveldynamics/include"
    -I"${PHYSX_DIR}/source/lowleveldynamics/shared"
    -I"${PHYSX_DIR}/source/lowleveldynamics/src"
    -I"${PHYSX_DIR}/source/lowlevel/software/include"
    -I"${PHYSX_DIR}/source/gpusolver/include"
    -I"${PHYSX_DIR}/source/gpusolver/src/CUDA"
    -I"${PHYSX_DIR}/source/gpubroadphase/include"
    -I"${PHYSX_DIR}/source/gpunarrowphase/include"
    -I"${PHYSX_DIR}/source/gpunarrowphase/src/CUDA"
    -I"${PHYSX_DIR}/source/gpucommon/include"
    -I"${PHYSX_DIR}/source/gpucommon/src/CUDA"
    -I"${PHYSX_DIR}/source/gpusimulationcontroller/include"
    -I"${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA"
    -I"${PHYSX_DIR}/source/gpuarticulation/include"
    -I"${PHYSX_DIR}/source/gpuarticulation/src/CUDA"
    -I"${PHYSX_DIR}/source/cudamanager/include"
)

# CUDA compile flags matching PhysX's own build (from cmakegpu/CMakeLists.txt)
# Note: host-compiler flags like -Wno-* must NOT be passed bare to nvcc.
# Use -Xcompiler -Wno-... syntax if needed. For PTX extraction we keep it minimal.
CUDA_FLAGS=(
    -use_fast_math
    -ftz=true
    -prec-div=false
    -prec-sqrt=false
    -D_CONSOLE
    -DNDEBUG
)

# ---- Complete list of all 61 kernel .cu files -----------------------------
# Grouped by sub-library for clarity.
ALL_CU_FILES=(
    # gpusolver (12)
    "${PHYSX_DIR}/source/gpusolver/src/CUDA/accumulateThresholdStream.cu"
    "${PHYSX_DIR}/source/gpusolver/src/CUDA/artiConstraintPrep2.cu"
    "${PHYSX_DIR}/source/gpusolver/src/CUDA/constraintBlockPrep.cu"
    "${PHYSX_DIR}/source/gpusolver/src/CUDA/constraintBlockPrePrep.cu"
    "${PHYSX_DIR}/source/gpusolver/src/CUDA/constraintBlockPrepTGS.cu"
    "${PHYSX_DIR}/source/gpusolver/src/CUDA/integration.cu"
    "${PHYSX_DIR}/source/gpusolver/src/CUDA/integrationTGS.cu"
    "${PHYSX_DIR}/source/gpusolver/src/CUDA/preIntegration.cu"
    "${PHYSX_DIR}/source/gpusolver/src/CUDA/preIntegrationTGS.cu"
    "${PHYSX_DIR}/source/gpusolver/src/CUDA/solver.cu"
    "${PHYSX_DIR}/source/gpusolver/src/CUDA/solverMultiBlock.cu"
    "${PHYSX_DIR}/source/gpusolver/src/CUDA/solverMultiBlockTGS.cu"
    # gpubroadphase (2)
    "${PHYSX_DIR}/source/gpubroadphase/src/CUDA/aggregate.cu"
    "${PHYSX_DIR}/source/gpubroadphase/src/CUDA/broadphase.cu"
    # gpucommon (3)
    "${PHYSX_DIR}/source/gpucommon/src/CUDA/MemCopyBalanced.cu"
    "${PHYSX_DIR}/source/gpucommon/src/CUDA/radixSortImpl.cu"
    "${PHYSX_DIR}/source/gpucommon/src/CUDA/utility.cu"
    # gpunarrowphase (25)
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/compressOutputContacts.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/convexCoreCollision.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/convexHeightfield.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/convexHFMidphase.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/convexMesh.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/convexMeshCorrelate.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/convexMeshMidphase.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/convexMeshOutput.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/convexMeshPostProcess.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/cudaBox.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/cudaGJKEPA.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/cudaParticleSystem.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/cudaSphere.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/femClothClothMidPhase.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/femClothHFMidPhase.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/femClothMidPhase.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/femClothPrimitives.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/pairManagement.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/particleSystemHFMidPhaseCG.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/particleSystemMeshMidphase.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/softbodyHFMidPhase.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/softbodyMidPhase.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/softbodyPrimitives.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/softbodySoftbodyMidPhase.cu"
    "${PHYSX_DIR}/source/gpunarrowphase/src/CUDA/trimeshCollision.cu"
    # gpusimulationcontroller (15)
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/algorithms.cu"
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/anisotropy.cu"
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/diffuseParticles.cu"
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/FEMCloth.cu"
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/FEMClothConstraintPrep.cu"
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/FEMClothExternalSolve.cu"
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/isosurfaceExtraction.cu"
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/particlesystem.cu"
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/rigidDeltaAccum.cu"
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/SDFConstruction.cu"
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/softBody.cu"
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/softBodyGM.cu"
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/sparseGridStandalone.cu"
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/updateBodiesAndShapes.cu"
    "${PHYSX_DIR}/source/gpusimulationcontroller/src/CUDA/updateTransformAndBoundArray.cu"
    # gpuarticulation (4)
    "${PHYSX_DIR}/source/gpuarticulation/src/CUDA/articulationDirectGpuApi.cu"
    "${PHYSX_DIR}/source/gpuarticulation/src/CUDA/forwardDynamic2.cu"
    "${PHYSX_DIR}/source/gpuarticulation/src/CUDA/internalConstraints2.cu"
    "${PHYSX_DIR}/source/gpuarticulation/src/CUDA/inverseDynamic.cu"
)

# ---- Process a single .cu file --------------------------------------------
# NOTE: bash disables set -e inside functions invoked as the condition of an
# 'if' statement.  We therefore explicitly check all exit codes ourselves.
process_cu_file() {
    local cu_file="$1"

    # Resolve to absolute path
    if [[ ! "${cu_file}" = /* ]]; then
        cu_file="${ROOT_DIR}/${cu_file}"
    fi

    if [[ ! -f "${cu_file}" ]]; then
        echo "  ERROR: .cu file not found: ${cu_file}"
        return 1
    fi

    # Output PTX goes in PTX/ sibling to CUDA/ within the same src/ dir
    local cu_dir    cu_dir="$(dirname "${cu_file}")"
    local src_dir   src_dir="$(dirname "${cu_dir}")"
    local ptx_dir="${src_dir}/PTX"
    mkdir -p "${ptx_dir}"

    local stem stem="$(basename "${cu_file}" .cu)"
    local ptx_file="${ptx_dir}/${stem}.ptx"

    echo "  Compiling: ${stem}.cu  ->  ${ptx_file##"${ROOT_DIR}/"}"

    local nvcc_exit=0
    "${NVCC}" -ptx \
        -arch="${PX_PTX_ARCH}" \
        "${CUDA_FLAGS[@]}" \
        "${INCLUDE_FLAGS[@]}" \
        "${cu_file}" \
        -o "${ptx_file}" || nvcc_exit=$?

    if [[ "${nvcc_exit}" -ne 0 ]]; then
        echo "  ERROR: nvcc exited ${nvcc_exit} for ${stem}.cu — PTX not generated."
        echo "         Check include paths and CUDA flags above."
        return 1
    fi

    # Verify at least one kernel entry point exists in the output
    local n_entries
    n_entries=$(grep -c '\.entry ' "${ptx_file}" 2>/dev/null || true)
    if [[ "${n_entries}" -eq 0 ]]; then
        echo "  ERROR: nvcc succeeded but no .entry found in ${ptx_file}"
        echo "         The .cu may have no __global__ kernels, or compilation was partial."
        return 1
    fi

    echo "    -> OK (${n_entries} kernel entries, $(wc -c < "${ptx_file}") bytes)"
    return 0
}

# ---- Main -----------------------------------------------------------------

echo "=== Elytar PTX Generator ==="
echo "  arch    : ${PX_PTX_ARCH}"
echo "  nvcc    : ${NVCC} ($(${NVCC} --version 2>&1 | grep 'release' | head -1 | sed 's/.*release //' | sed 's/,.*//'))"
echo ""

if [[ "${MODE}" == "all" ]]; then
    echo "Mode: ALL (${#ALL_CU_FILES[@]} files)"
    echo ""
    ok=0; fail=0
    for cu_file in "${ALL_CU_FILES[@]}"; do
        if process_cu_file "${cu_file}"; then
            ((ok++)) || true
        else
            ((fail++)) || true
        fi
    done
    echo ""
    echo "=== Done: ${ok} succeeded, ${fail} failed ==="
    [[ "${fail}" -eq 0 ]] || exit 1

elif [[ "${MODE}" == "single" ]]; then
    echo "Mode: SINGLE"
    echo ""
    process_cu_file "${SINGLE_CU}"
    echo ""
    echo "=== Done ==="

elif [[ "${MODE}" == "list" ]]; then
    # Build a lookup set from the semicolon-separated stem list
    declare -A _stem_set
    IFS=';' read -ra _stems <<< "${LIST_STEMS}"
    for _s in "${_stems[@]}"; do
        _stem_set["${_s}"]=1
    done
    echo "Mode: LIST (${#_stems[@]} stems: ${LIST_STEMS})"
    echo ""
    ok=0; fail=0; skip=0
    for cu_file in "${ALL_CU_FILES[@]}"; do
        _stem="$(basename "${cu_file}" .cu)"
        if [[ -n "${_stem_set[${_stem}]+x}" ]]; then
            if process_cu_file "${cu_file}"; then
                ((ok++)) || true
            else
                ((fail++)) || true
            fi
        else
            ((skip++)) || true
        fi
    done
    echo ""
    echo "=== Done: ${ok} succeeded, ${fail} failed, ${skip} skipped ==="
    [[ "${fail}" -eq 0 ]] || exit 1
fi
