#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace}"
PHYSX_DIR="${PHYSX_DIR:-${ROOT_DIR}/physx}"
SAPIEN_DIR="${SAPIEN_DIR:-${ROOT_DIR}/sapien}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PHYSX_PRESET="${PHYSX_PRESET:-linux-clang}"
PHYSX_CONFIG="${PHYSX_CONFIG:-checked}"
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

prune_physx_other_configs() {
  local cfg
  for cfg in debug checked profile release; do
    if [[ "${cfg}" != "${PHYSX_CONFIG}" ]]; then
      rm -rf "${PHYSX_DIR}/compiler/${PHYSX_PRESET}-${cfg}"
      rm -rf "${PHYSX_DIR}/bin/linux.x86_64/${cfg}"
    fi
  done
}

if [[ ! -f "${PHYSX_BUILD_DIR}/CMakeCache.txt" ]]; then
  echo "[1/4] Generate PhysX build files (${PHYSX_PRESET})"
  (
    cd "${PHYSX_DIR}"
    ./generate_projects.sh "${PHYSX_PRESET}"
  )
  # PhysX generator creates all configs by design; keep only the selected one.
  prune_physx_other_configs
else
  echo "[1/4] Reusing existing PhysX build files (${PHYSX_PRESET}-${PHYSX_CONFIG})"
  prune_physx_other_configs
fi

if [[ ! -d "${PHYSX_BUILD_DIR}" ]]; then
  echo "Expected PhysX build directory not found: ${PHYSX_BUILD_DIR}"
  exit 1
fi

echo "[2/4] Disable PhysX snippets/PVD for toolchain build"
cmake -S "${PHYSX_DIR}/compiler/public" -B "${PHYSX_BUILD_DIR}" \
  -DPX_BUILDSNIPPETS=FALSE \
  -DPX_BUILDPVDRUNTIME=FALSE

echo "[3/4] Build PhysX (${PHYSX_CONFIG})"
cmake --build "${PHYSX_BUILD_DIR}" -- -j"$(nproc)"

PHYSX_LIB_DIR="${PHYSX_DIR}/bin/linux.x86_64/${PHYSX_CONFIG}"
if [[ ! -f "${PHYSX_LIB_DIR}/libPhysX_static_64.a" ]]; then
  echo "PhysX static library not found at ${PHYSX_LIB_DIR}/libPhysX_static_64.a"
  exit 1
fi

echo "[4/4] Build SAPIEN wheel using local PhysX"
(
  cd "${SAPIEN_DIR}"
  "${PYTHON_BIN}" -m pip install -U pip setuptools wheel
  rm -f dist/sapien-*.whl
  "${PYTHON_BIN}" setup.py bdist_wheel "${SAPIEN_BUILD_MODE}" --build-dir="${SAPIEN_BUILD_DIR}"
)

LATEST_WHEEL="$(
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import sys

dist_dir = Path(os.environ.get("SAPIEN_DIR", "/workspace/sapien"), "dist")
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
