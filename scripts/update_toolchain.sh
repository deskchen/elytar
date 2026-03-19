#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
PHYSX_DIR="${PHYSX_DIR:-${ROOT_DIR}/physx-5.6.1-capybara}"
SAPIEN_DIR="${SAPIEN_DIR:-${ROOT_DIR}/sapien-3.0.2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# PhysX 5.6 uses linux-clang preset.
PHYSX_PRESET="${PHYSX_PRESET:-linux-clang}"
PHYSX_CONFIG="${PHYSX_CONFIG:-profile}"
# PX_PTX_REPLACE_LIST: semicolon-separated kernel stems to build from PTX.
# Use "all" to replace all 61, empty string for none (pure .cu build).
# e.g. PX_PTX_REPLACE_LIST="integration;solver" ./scripts/update_toolchain.sh
PX_PTX_REPLACE_LIST="${PX_PTX_REPLACE_LIST:-}"
PX_PTX_ARCH="${PX_PTX_ARCH:-compute_86}"
SAPIEN_BUILD_MODE="${SAPIEN_BUILD_MODE:---profile}"
SAPIEN_BUILD_DIR="${SAPIEN_BUILD_DIR:-docker_sapien_build}"
SAPIEN_LOCAL_BUILD_MARKER="${SAPIEN_LOCAL_BUILD_MARKER:-elytar}"
PHYSX_BUILD_DIR="${PHYSX_DIR}/compiler/${PHYSX_PRESET}-${PHYSX_CONFIG}"

export CC=clang
export CXX=clang++
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
export SAPIEN_DIR
export SAPIEN_LOCAL_BUILD_MARKER
export SAPIEN_PHYSX5_DIR="${SAPIEN_PHYSX5_DIR:-${PHYSX_DIR}}"

# Extract the local PhysX version from its header so the wheel can embed it.
_px_ver_header="${PHYSX_DIR}/include/foundation/PxPhysicsVersion.h"
if [[ -f "${_px_ver_header}" ]]; then
  _px_major=$(sed -n 's/^#define PX_PHYSICS_VERSION_MAJOR \([0-9]*\)/\1/p' "${_px_ver_header}")
  _px_minor=$(sed -n 's/^#define PX_PHYSICS_VERSION_MINOR \([0-9]*\)/\1/p' "${_px_ver_header}")
  _px_bugfix=$(sed -n 's/^#define PX_PHYSICS_VERSION_BUGFIX \([0-9]*\)/\1/p' "${_px_ver_header}")
  export SAPIEN_LOCAL_PHYSX_VERSION="${_px_major}.${_px_minor}.${_px_bugfix}"
else
  export SAPIEN_LOCAL_PHYSX_VERSION=""
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script supports Linux only."
  exit 1
fi

if [[ ! -d "${PHYSX_DIR}" || ! -d "${SAPIEN_DIR}" ]]; then
  echo "Expected both ${PHYSX_DIR} and ${SAPIEN_DIR} to exist."
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${PYTHON_BIN}"
  exit 1
fi

if ! command -v clang >/dev/null 2>&1 || ! command -v clang++ >/dev/null 2>&1; then
  echo "clang/clang++ are required in the dev image."
  exit 1
fi

if [[ ! -x "${CUDA_PATH}/bin/nvcc" ]]; then
  echo "CUDA toolkit not found at ${CUDA_PATH}. GPU-enabled PhysX build requires nvcc."
  exit 1
fi

case "${PHYSX_CONFIG}" in
  debug|checked|profile|release) ;;
  *)
    echo "Invalid PHYSX_CONFIG=${PHYSX_CONFIG}. Use one of: debug, checked, profile, release."
    exit 1
    ;;
esac

if [[ ! -f "${PHYSX_BUILD_DIR}/CMakeCache.txt" ]]; then
  echo "[1/4] Generate PhysX build files (${PHYSX_PRESET})"
  (
    cd "${PHYSX_DIR}"
    ./generate_projects.sh "${PHYSX_PRESET}"
  )
else
  echo "[1/4] Reusing existing PhysX build files (${PHYSX_PRESET}-${PHYSX_CONFIG})"
fi

if [[ ! -d "${PHYSX_BUILD_DIR}" ]]; then
  echo "Expected PhysX build directory not found: ${PHYSX_BUILD_DIR}"
  exit 1
fi

# When PTX kernels are requested, verify the .ptx files already exist on disk.
# PTX must be produced externally (e.g. by scripts/generate_ptx.sh or a DSL compiler)
# before invoking this script.
if [[ -n "${PX_PTX_REPLACE_LIST}" ]]; then
  _ptx_missing=0
  declare -A _ptx_src_dirs=(
    [gpusolver]="${PHYSX_DIR}/source/gpusolver/src/PTX"
    [gpubroadphase]="${PHYSX_DIR}/source/gpubroadphase/src/PTX"
    [gpunarrowphase]="${PHYSX_DIR}/source/gpunarrowphase/src/PTX"
    [gpusimulationcontroller]="${PHYSX_DIR}/source/gpusimulationcontroller/src/PTX"
    [gpuarticulation]="${PHYSX_DIR}/source/gpuarticulation/src/PTX"
    [gpucommon]="${PHYSX_DIR}/source/gpucommon/src/PTX"
  )
  declare -A _ptx_stem_to_mod
  for _s in accumulateThresholdStream artiConstraintPrep2 constraintBlockPrep constraintBlockPrePrep constraintBlockPrepTGS integration integrationTGS preIntegration preIntegrationTGS solver solverMultiBlock solverMultiBlockTGS; do _ptx_stem_to_mod[$_s]=gpusolver; done
  for _s in aggregate broadphase; do _ptx_stem_to_mod[$_s]=gpubroadphase; done
  for _s in compressOutputContacts convexCoreCollision convexHeightfield convexHFMidphase convexMesh convexMeshCorrelate convexMeshMidphase convexMeshOutput convexMeshPostProcess cudaBox cudaGJKEPA cudaParticleSystem cudaSphere femClothClothMidPhase femClothHFMidPhase femClothMidPhase femClothPrimitives pairManagement particleSystemHFMidPhaseCG particleSystemMeshMidphase softbodyHFMidPhase softbodyMidPhase softbodyPrimitives softbodySoftbodyMidPhase trimeshCollision; do _ptx_stem_to_mod[$_s]=gpunarrowphase; done
  for _s in algorithms anisotropy diffuseParticles FEMCloth FEMClothConstraintPrep FEMClothExternalSolve isosurfaceExtraction particlesystem rigidDeltaAccum SDFConstruction softBody softBodyGM sparseGridStandalone updateBodiesAndShapes updateTransformAndBoundArray; do _ptx_stem_to_mod[$_s]=gpusimulationcontroller; done
  for _s in articulationDirectGpuApi forwardDynamic2 internalConstraints2 inverseDynamic; do _ptx_stem_to_mod[$_s]=gpuarticulation; done
  for _s in MemCopyBalanced radixSortImpl utility; do _ptx_stem_to_mod[$_s]=gpucommon; done

  if [[ "${PX_PTX_REPLACE_LIST}" == "all" ]]; then
    _check_stems=("${!_ptx_stem_to_mod[@]}")
  else
    IFS=';' read -ra _check_stems <<< "${PX_PTX_REPLACE_LIST}"
  fi

  for _stem in "${_check_stems[@]}"; do
    _mod="${_ptx_stem_to_mod[$_stem]:-}"
    if [[ -z "${_mod}" ]]; then
      echo "ERROR: Unknown kernel stem '${_stem}' in PX_PTX_REPLACE_LIST."
      exit 1
    fi
    _ptx_file="${_ptx_src_dirs[$_mod]}/${_stem}.ptx"
    if [[ ! -f "${_ptx_file}" ]]; then
      echo "ERROR: PTX file not found: ${_ptx_file}"
      ((_ptx_missing++)) || true
    fi
  done

  if [[ "${_ptx_missing}" -gt 0 ]]; then
    echo "  ${_ptx_missing} PTX file(s) missing. Generate them first, e.g.:"
    echo "    scripts/generate_ptx.sh --list \"${PX_PTX_REPLACE_LIST}\""
    exit 1
  fi
fi

echo "[2/4] Configure PhysX (snippets/PVD off, PTX_LIST=${PX_PTX_REPLACE_LIST:-none})"
cmake -S "${PHYSX_DIR}/compiler/public" -B "${PHYSX_BUILD_DIR}" \
  -DPX_BUILDSNIPPETS=FALSE \
  -DPX_BUILDPVDRUNTIME=FALSE \
  -DPX_PTX_REPLACE_LIST="${PX_PTX_REPLACE_LIST}" \
  -DPX_PTX_ARCH="${PX_PTX_ARCH}"

echo "[3/4] Build PhysX (${PHYSX_CONFIG}, PTX_LIST=${PX_PTX_REPLACE_LIST:-none})"
cmake --build "${PHYSX_BUILD_DIR}" -- -j"$(nproc)"

PHYSX_LIB_DIR=""
for _arch in linux.clang linux.x86_64; do
  _candidate="${PHYSX_DIR}/bin/${_arch}/${PHYSX_CONFIG}"
  if [[ -f "${_candidate}/libPhysX_static_64.a" ]]; then
    PHYSX_LIB_DIR="${_candidate}"
    break
  fi
done
if [[ -z "${PHYSX_LIB_DIR}" ]]; then
  echo "PhysX static library not found under ${PHYSX_DIR}/bin/*/${PHYSX_CONFIG}/"
  exit 1
fi
echo "  PhysX libs at: ${PHYSX_LIB_DIR}"

# ---------------------------------------------------------------------------
# PTX hybrid verification — confirms that exactly the configured stems were
# built from PTX and that any others used the original .cu path.
# ---------------------------------------------------------------------------
verify_ptx_mode() {
  local lib_dir="$1"   # e.g. physx-5.6.1-capybara/bin/linux.clang/profile
  local build_dir="$2" # e.g. physx-5.6.1-capybara/compiler/linux-clang-profile
  local replace_list="$3"  # value of PX_PTX_REPLACE_LIST

  local ok=true
  local n_pass=0
  local n_fail=0

  _ptx_check_pass() { echo "    [PASS] $*"; ((n_pass++)) || true; }
  _ptx_check_fail() { echo "    [FAIL] $*"; ((n_fail++)) || true; ok=false; }

  # Build a stem -> module mapping for all 61 kernels
  declare -A _stem_to_mod
  local _solver_stems=( accumulateThresholdStream artiConstraintPrep2 constraintBlockPrep constraintBlockPrePrep constraintBlockPrepTGS integration integrationTGS preIntegration preIntegrationTGS solver solverMultiBlock solverMultiBlockTGS )
  local _broadphase_stems=( aggregate broadphase )
  local _narrowphase_stems=( compressOutputContacts convexCoreCollision convexHeightfield convexHFMidphase convexMesh convexMeshCorrelate convexMeshMidphase convexMeshOutput convexMeshPostProcess cudaBox cudaGJKEPA cudaParticleSystem cudaSphere femClothClothMidPhase femClothHFMidPhase femClothMidPhase femClothPrimitives pairManagement particleSystemHFMidPhaseCG particleSystemMeshMidphase softbodyHFMidPhase softbodyMidPhase softbodyPrimitives softbodySoftbodyMidPhase trimeshCollision )
  local _simctrl_stems=( algorithms anisotropy diffuseParticles FEMCloth FEMClothConstraintPrep FEMClothExternalSolve isosurfaceExtraction particlesystem rigidDeltaAccum SDFConstruction softBody softBodyGM sparseGridStandalone updateBodiesAndShapes updateTransformAndBoundArray )
  local _articulation_stems=( articulationDirectGpuApi forwardDynamic2 internalConstraints2 inverseDynamic )
  local _common_stems=( MemCopyBalanced radixSortImpl utility )

  for s in "${_solver_stems[@]}";       do _stem_to_mod["$s"]="gpusolver";               done
  for s in "${_broadphase_stems[@]}";   do _stem_to_mod["$s"]="gpubroadphase";           done
  for s in "${_narrowphase_stems[@]}";  do _stem_to_mod["$s"]="gpunarrowphase";          done
  for s in "${_simctrl_stems[@]}";      do _stem_to_mod["$s"]="gpusimulationcontroller"; done
  for s in "${_articulation_stems[@]}"; do _stem_to_mod["$s"]="gpuarticulation";         done
  for s in "${_common_stems[@]}";       do _stem_to_mod["$s"]="gpucommon";               done

  # Resolve PX_PTX_REPLACE_LIST to a concrete list of stems
  declare -A _ptx_stems
  if [[ "${replace_list}" == "all" ]]; then
    for s in "${!_stem_to_mod[@]}"; do _ptx_stems["$s"]=1; done
  else
    IFS=';' read -ra _listed <<< "${replace_list}"
    for s in "${_listed[@]}"; do _ptx_stems["$s"]=1; done
  fi

  local n_expected_ptx="${#_ptx_stems[@]}"

  echo ""
  echo "=== PTX hybrid verification ==="
  echo "  arch        : ${PX_PTX_ARCH}"
  echo "  PTX stems   : ${n_expected_ptx} (PX_PTX_REPLACE_LIST=${replace_list})"
  echo "  libs        : ${lib_dir}"
  echo "  build       : ${build_dir}"
  echo ""

  # ------------------------------------------------------------------
  # Check 1: Each configured PTX stem has a .ptx file on disk
  # ------------------------------------------------------------------
  echo "  [1] PTX source files (generated by generate_ptx.sh):"
  local -A _mod_ptx_dirs=(
    [gpusolver]="${PHYSX_DIR}/source/gpusolver/src/PTX"
    [gpubroadphase]="${PHYSX_DIR}/source/gpubroadphase/src/PTX"
    [gpunarrowphase]="${PHYSX_DIR}/source/gpunarrowphase/src/PTX"
    [gpusimulationcontroller]="${PHYSX_DIR}/source/gpusimulationcontroller/src/PTX"
    [gpuarticulation]="${PHYSX_DIR}/source/gpuarticulation/src/PTX"
    [gpucommon]="${PHYSX_DIR}/source/gpucommon/src/PTX"
  )
  for stem in "${!_ptx_stems[@]}"; do
    local mod="${_stem_to_mod[$stem]:-}"
    if [[ -z "$mod" ]]; then
      _ptx_check_fail "Unknown stem '${stem}' in PX_PTX_REPLACE_LIST"
      continue
    fi
    local ptx_file="${_mod_ptx_dirs[$mod]}/${stem}.ptx"
    if [[ -f "${ptx_file}" ]]; then
      _ptx_check_pass "${stem}.ptx found (${mod})"
    else
      _ptx_check_fail "${stem}.ptx missing at ${ptx_file} — run generate_ptx.sh"
    fi
  done

  # ------------------------------------------------------------------
  # Check 2: Per-stem build artifacts exist in elytar_ptx/
  # (Directory may contain stale files from previous runs — only check
  #  the stems that are currently configured.)
  # ------------------------------------------------------------------
  echo ""
  echo "  [2] Auto-generated CMake stubs (configure-time, elytar_ptx/):"
  local elytar_dir
  elytar_dir=$(find "${build_dir}" -maxdepth 3 -name "elytar_ptx" -type d 2>/dev/null | head -1)
  if [[ -z "${elytar_dir}" ]]; then
    elytar_dir="${build_dir}/elytar_ptx"
  fi
  if [[ -d "${elytar_dir}" ]]; then
    for stem in "${!_ptx_stems[@]}"; do
      if [[ -f "${elytar_dir}/${stem}_ptx_register.cpp" ]]; then
        _ptx_check_pass "${stem}_ptx_register.cpp found"
      else
        _ptx_check_fail "${stem}_ptx_register.cpp missing in ${elytar_dir}"
      fi
      if [[ -f "${elytar_dir}/${stem}.fatbin" ]]; then
        _ptx_check_pass "${stem}.fatbin found"
      else
        _ptx_check_fail "${stem}.fatbin missing in ${elytar_dir}"
      fi
    done
  else
    _ptx_check_fail "elytar_ptx/ not found under ${build_dir} — cmake may not have run with PTX stems"
  fi

  # ------------------------------------------------------------------
  # Check 3: Per-library symbol counts match configured PTX stems
  # ------------------------------------------------------------------
  echo ""
  echo "  [3] Symbol verification (nm on built static libraries):"

  # Map library name -> list of stems belonging to it
  declare -A _lib_stems
  _lib_stems[libPhysXSolverGpu_static_64.a]="${_solver_stems[*]}"
  _lib_stems[libPhysXBroadphaseGpu_static_64.a]="${_broadphase_stems[*]}"
  _lib_stems[libPhysXNarrowphaseGpu_static_64.a]="${_narrowphase_stems[*]}"
  _lib_stems[libPhysXSimulationControllerGpu_static_64.a]="${_simctrl_stems[*]}"
  _lib_stems[libPhysXArticulationGpu_static_64.a]="${_articulation_stems[*]}"
  _lib_stems[libPhysXCommonGpu_static_64.a]="${_common_stems[*]}"

  for lib_name in "${!_lib_stems[@]}"; do
    local lib="${lib_dir}/${lib_name}"
    # Try fallback without _64
    if [[ ! -f "${lib}" ]]; then
      lib="${lib_dir}/${lib_name/_static_64.a/_static.a}"
    fi
    if [[ ! -f "${lib}" ]]; then
      _ptx_check_fail "${lib_name}: not found in ${lib_dir}"
      continue
    fi

    # Count how many stems for this lib are in the PTX list
    local expected_for_lib=0
    for s in ${_lib_stems[$lib_name]}; do
      if [[ -n "${_ptx_stems[$s]+x}" ]]; then
        ((expected_for_lib++)) || true
      fi
    done

    local lib_base n_fatbin_syms
    lib_base="$(basename "${lib}")"
    n_fatbin_syms=$(nm --defined-only "${lib}" 2>/dev/null | grep -c "_fatbin$" || true)

    if [[ "${n_fatbin_syms}" -eq "${expected_for_lib}" ]]; then
      if [[ "${expected_for_lib}" -eq 0 ]]; then
        _ptx_check_pass "${lib_base}: 0 PTX stems configured (all .cu) — confirmed no *_fatbin symbols"
      else
        _ptx_check_pass "${lib_base}: ${n_fatbin_syms}/${expected_for_lib} *_fatbin symbols (PTX kernels confirmed)"
      fi
    else
      _ptx_check_fail "${lib_base}: found ${n_fatbin_syms} *_fatbin symbols, expected ${expected_for_lib}"
    fi

    # Secondary (informational): static-init entries may be >1 per TU
    # (constructor + destructor), so only check presence, not exact count.
    if [[ "${expected_for_lib}" -gt 0 ]]; then
      local n_constructors
      n_constructors=$(nm --defined-only "${lib}" 2>/dev/null | grep -c "_ptx_register\.cpp" || true)
      if [[ "${n_constructors}" -ge "${expected_for_lib}" ]]; then
        _ptx_check_pass "${lib_base}: ${n_constructors} static-init entries for ptx_register TUs"
      else
        _ptx_check_fail "${lib_base}: ${n_constructors} static-init entries, expected >= ${expected_for_lib}"
      fi
    fi
  done

  # ------------------------------------------------------------------
  # Summary
  # ------------------------------------------------------------------
  echo ""
  if [[ "${ok}" == "true" ]]; then
    echo "  PTX verification PASSED (${n_pass} checks, 0 failures)"
    echo "  -> ${n_expected_ptx}/61 kernels from PTX, remainder from .cu."
  else
    echo "  PTX verification FAILED (${n_pass} passed, ${n_fail} failed)"
    echo "  -> See [FAIL] entries above."
    return 1
  fi
  echo ""
}

if [[ -n "${PX_PTX_REPLACE_LIST}" ]]; then
  verify_ptx_mode "${PHYSX_LIB_DIR}" "${PHYSX_BUILD_DIR}" "${PX_PTX_REPLACE_LIST}"
fi

echo "[4/4] Build SAPIEN wheel using local PhysX"
(
  cd "${SAPIEN_DIR}"

  # Force CMake reconfigure so it picks up the (potentially changed) PhysX tree.
  SAPIEN_CMAKE_BUILD="${SAPIEN_BUILD_DIR}/_sapien_build"
  if [[ -f "${SAPIEN_CMAKE_BUILD}/CMakeCache.txt" ]]; then
    cached_dir=$(sed -n 's/^SAPIEN_PHYSX5_DIR:STRING=//p' "${SAPIEN_CMAKE_BUILD}/CMakeCache.txt")
    if [[ "${cached_dir}" != "${SAPIEN_PHYSX5_DIR}" ]]; then
      echo "  PhysX dir changed (${cached_dir} → ${SAPIEN_PHYSX5_DIR}), cleaning SAPIEN build cache"
      rm -rf "${SAPIEN_CMAKE_BUILD}"
    fi
  fi

  "${PYTHON_BIN}" -m pip install -U pip setuptools wheel
  rm -f dist/sapien-*.whl
  "${PYTHON_BIN}" setup.py bdist_wheel "${SAPIEN_BUILD_MODE}" --build-dir="${SAPIEN_BUILD_DIR}"
  # Symlink so C++ IDE/clangd finds compile_commands.json when editing sapien C++ sources.
  if [[ -f "${SAPIEN_BUILD_DIR}/_sapien_build/compile_commands.json" ]]; then
    ln -sf "${SAPIEN_BUILD_DIR}/_sapien_build/compile_commands.json" "${SAPIEN_DIR}/compile_commands.json"
  fi
)

LATEST_WHEEL="$(
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import sys

dist_dir = Path(os.environ.get("SAPIEN_DIR", "/workspace/sapien-3.0.2"), "dist")
cp_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
pattern = f"sapien-*-{cp_tag}-{cp_tag}-*.whl"
wheels = sorted(dist_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
print(wheels[-1] if wheels else "")
PY
)"

if [[ -z "${LATEST_WHEEL}" || ! -f "${LATEST_WHEEL}" ]]; then
  echo "No wheel matching ${PYTHON_BIN} found under ${SAPIEN_DIR}/dist."
  exit 1
fi

echo "Reinstall SAPIEN wheel: ${LATEST_WHEEL}"
"${PYTHON_BIN}" -m pip uninstall -y sapien >/dev/null 2>&1 || true
"${PYTHON_BIN}" -m pip install --force-reinstall "${LATEST_WHEEL}"

echo ""
echo "=============================="
echo " Verifying toolchain update (PhysX + SAPIEN + wheel)"
echo "=============================="
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/verify_toolchain.py"
