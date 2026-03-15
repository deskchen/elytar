#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
PHYSX_DIR="${PHYSX_DIR:-${ROOT_DIR}/physx-5.6.1}"
SAPIEN_DIR="${SAPIEN_DIR:-${ROOT_DIR}/sapien-3.0.2}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# PhysX 5.6 uses linux-clang preset.
PHYSX_PRESET="${PHYSX_PRESET:-linux-clang}"
PHYSX_CONFIG="${PHYSX_CONFIG:-profile}"
PX_USE_PTX_KERNELS="${PX_USE_PTX_KERNELS:-OFF}"
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

# When PTX mode is enabled, generate PTX before building.
if [[ "${PX_USE_PTX_KERNELS}" == "ON" ]]; then
  echo "[2/5] Generate PTX from .cu (arch=${PX_PTX_ARCH})"
  PX_PTX_ARCH="${PX_PTX_ARCH}" \
  PHYSX_DIR="${PHYSX_DIR}" \
  CUDA_PATH="${CUDA_PATH}" \
    "${ROOT_DIR}/scripts/generate_ptx.sh" --all
  _ptx_step_offset=1
else
  _ptx_step_offset=0
fi

_step=$((2 + _ptx_step_offset))
echo "[${_step}/$((4 + _ptx_step_offset))] Configure PhysX (snippets/PVD off, PTX=${PX_USE_PTX_KERNELS})"
cmake -S "${PHYSX_DIR}/compiler/public" -B "${PHYSX_BUILD_DIR}" \
  -DPX_BUILDSNIPPETS=FALSE \
  -DPX_BUILDPVDRUNTIME=FALSE \
  -DPX_USE_PTX_KERNELS="${PX_USE_PTX_KERNELS}" \
  -DPX_PTX_ARCH="${PX_PTX_ARCH}"

_step=$((3 + _ptx_step_offset))
echo "[${_step}/$((4 + _ptx_step_offset))] Build PhysX (${PHYSX_CONFIG}, PTX=${PX_USE_PTX_KERNELS})"
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
# PTX mode verification — run after the PhysX build to confirm the pipeline
# actually used PTX-generated code rather than the original .cu files.
# ---------------------------------------------------------------------------
verify_ptx_mode() {
  local lib_dir="$1"   # e.g. physx-5.6.1/bin/linux.clang/profile
  local build_dir="$2" # e.g. physx-5.6.1/compiler/linux-clang-profile

  local ok=true
  local n_pass=0
  local n_fail=0

  _ptx_check_pass() { echo "    [PASS] $*"; ((n_pass++)) || true; }
  _ptx_check_fail() { echo "    [FAIL] $*"; ((n_fail++)) || true; ok=false; }

  echo ""
  echo "=== PTX mode verification ==="
  echo "  arch   : ${PX_PTX_ARCH}"
  echo "  libs   : ${lib_dir}"
  echo "  build  : ${build_dir}"
  echo ""

  # ------------------------------------------------------------------
  # Check 1: PTX source files exist for all 6 sub-libraries
  # ------------------------------------------------------------------
  echo "  [1] PTX source files (generated by generate_ptx.sh):"
  local -A ptx_dirs=(
    [gpusolver]="${PHYSX_DIR}/source/gpusolver/src/PTX"
    [gpubroadphase]="${PHYSX_DIR}/source/gpubroadphase/src/PTX"
    [gpunarrowphase]="${PHYSX_DIR}/source/gpunarrowphase/src/PTX"
    [gpusimulationcontroller]="${PHYSX_DIR}/source/gpusimulationcontroller/src/PTX"
    [gpuarticulation]="${PHYSX_DIR}/source/gpuarticulation/src/PTX"
    [gpucommon]="${PHYSX_DIR}/source/gpucommon/src/PTX"
  )
  local -A ptx_expected_counts=(
    [gpusolver]=12 [gpubroadphase]=2 [gpunarrowphase]=25
    [gpusimulationcontroller]=15 [gpuarticulation]=4 [gpucommon]=3
  )
  for mod in "${!ptx_dirs[@]}"; do
    local ptx_dir="${ptx_dirs[$mod]}"
    if [[ -d "${ptx_dir}" ]]; then
      local count
      count=$(find "${ptx_dir}" -name "*.ptx" 2>/dev/null | wc -l)
      local expected="${ptx_expected_counts[$mod]}"
      if [[ "${count}" -ge "${expected}" ]]; then
        _ptx_check_pass "${mod}: ${count}/${expected} .ptx files found"
      else
        _ptx_check_fail "${mod}: only ${count}/${expected} .ptx files (run generate_ptx.sh --all)"
      fi
    else
      _ptx_check_fail "${mod}: PTX directory missing: ${ptx_dir}"
    fi
  done

  # ------------------------------------------------------------------
  # Check 2: Auto-generated stub files exist in the build tree
  # ------------------------------------------------------------------
  echo ""
  echo "  [2] Auto-generated CMake stubs (configure-time, elytar_ptx/):"
  # The generated files live in CMAKE_CURRENT_BINARY_DIR/elytar_ptx which is
  # a sub-directory under build_dir (e.g. sdk_gpu_source_bin/elytar_ptx).
  local elytar_dir
  elytar_dir=$(find "${build_dir}" -maxdepth 3 -name "elytar_ptx" -type d 2>/dev/null | head -1)
  if [[ -z "${elytar_dir}" ]]; then
    elytar_dir="${build_dir}/elytar_ptx"  # fallback for error message
  fi
  if [[ -d "${elytar_dir}" ]]; then
    local n_stubs
    n_stubs=$(find "${elytar_dir}" -name "*_ptx_register.cpp" 2>/dev/null | wc -l)
    local n_fatbins
    n_fatbins=$(find "${elytar_dir}" -name "*.fatbin" 2>/dev/null | wc -l)
    if [[ "${n_stubs}" -ge 61 ]]; then
      _ptx_check_pass "Registration stubs: ${n_stubs} *_ptx_register.cpp found"
    else
      _ptx_check_fail "Only ${n_stubs}/61 registration stubs found in ${elytar_dir}"
    fi
    if [[ "${n_fatbins}" -ge 61 ]]; then
      _ptx_check_pass "Fatbins: ${n_fatbins} *.fatbin found (PTX compiled to binary)"
    else
      _ptx_check_fail "Only ${n_fatbins}/61 .fatbin files found — build may have failed"
    fi
  else
    _ptx_check_fail "elytar_ptx/ not found at ${elytar_dir} — cmake may not have run in PTX mode"
  fi

  # ------------------------------------------------------------------
  # Check 3: Symbol check — *_fatbin data symbols must be in the libs
  # ------------------------------------------------------------------
  echo ""
  echo "  [3] Symbol verification (nm on built static libraries):"

  local -a gpu_libs=(
    "libPhysXSolverGpu_static_64.a"
    "libPhysXBroadphaseGpu_static_64.a"
    "libPhysXNarrowphaseGpu_static_64.a"
    "libPhysXSimulationControllerGpu_static_64.a"
    "libPhysXArticulationGpu_static_64.a"
    "libPhysXCommonGpu_static_64.a"
  )
  # Fallback: non-_64 names
  local -a gpu_libs_alt=(
    "libPhysXSolverGpu_static.a"
    "libPhysXBroadphaseGpu_static.a"
    "libPhysXNarrowphaseGpu_static.a"
    "libPhysXSimulationControllerGpu_static.a"
    "libPhysXArticulationGpu_static.a"
    "libPhysXCommonGpu_static.a"
  )

  for i in "${!gpu_libs[@]}"; do
    local lib="${lib_dir}/${gpu_libs[$i]}"
    if [[ ! -f "${lib}" ]]; then
      lib="${lib_dir}/${gpu_libs_alt[$i]}"
    fi
    if [[ ! -f "${lib}" ]]; then
      _ptx_check_fail "${gpu_libs[$i]}: not found in ${lib_dir}"
      continue
    fi
    local lib_base
    lib_base="$(basename "${lib}")"
    # In PTX mode the registrar struct lives inside an anonymous namespace, so its
    # mangled name never contains "Elytar_".  The reliable indicator is the embedded
    # fatbin byte-array symbol (type D, named <stem>_fatbin) that bin2c emits and that
    # is ONLY present when a _ptx_register.cpp object was compiled into the archive.
    # In original-mode builds nvcc embeds fatbinaries differently (no <stem>_fatbin).
    local n_fatbin_syms
    n_fatbin_syms=$(nm --defined-only "${lib}" 2>/dev/null | grep -c "_fatbin$" || true)
    if [[ "${n_fatbin_syms}" -gt 0 ]]; then
      _ptx_check_pass "${lib_base}: ${n_fatbin_syms} *_fatbin data symbols found (PTX mode confirmed)"
    else
      _ptx_check_fail "${lib_base}: no *_fatbin symbols — library may have been built in original mode"
    fi
    # Secondary: every PTX-mode TU also emits a static-init function whose demangled
    # name contains "_ptx_register.cpp".  Count them as a sanity cross-check.
    local n_constructors
    n_constructors=$(nm --defined-only "${lib}" 2>/dev/null | grep -c "_ptx_register\.cpp" || true)
    if [[ "${n_constructors}" -gt 0 ]]; then
      _ptx_check_pass "${lib_base}: ${n_constructors} static-init entries for ptx_register TUs"
    fi
  done

  # ------------------------------------------------------------------
  # Summary
  # ------------------------------------------------------------------
  echo ""
  if [[ "${ok}" == "true" ]]; then
    echo "  PTX verification PASSED (${n_pass} checks, 0 failures)"
    echo "  -> PhysX was built using pre-generated PTX for all 61 kernel files."
  else
    echo "  PTX verification FAILED (${n_pass} passed, ${n_fail} failed)"
    echo "  -> See [FAIL] entries above. Run generate_ptx.sh --all then rebuild."
    return 1
  fi
  echo ""
}

if [[ "${PX_USE_PTX_KERNELS}" == "ON" ]]; then
  verify_ptx_mode "${PHYSX_LIB_DIR}" "${PHYSX_BUILD_DIR}"
fi

_step=$((4 + _ptx_step_offset))
echo "[${_step}/$((4 + _ptx_step_offset))] Build SAPIEN wheel using local PhysX"
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
