#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-elytar-dev:cuda12.8-clang}"
CONTAINER_NAME="${CONTAINER_NAME:-elytar-dev}"

# Ensure host NVIDIA display is ready for --render (skip if inside Docker)
if [[ ! -f "/.dockerenv" ]] && [[ -f "${ROOT_DIR}/scripts/host_display_setup.sh" ]]; then
  "${ROOT_DIR}/scripts/host_display_setup.sh" >/dev/null 2>&1 || true
fi

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  docker start "${CONTAINER_NAME}" >/dev/null
    docker exec -it \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}" \
    -e __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}" \
    "${CONTAINER_NAME}" \
    zsh
else
  docker run -it \
    --name "${CONTAINER_NAME}" \
    --network host \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute,display \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}" \
    -e __EGL_VENDOR_LIBRARY_FILENAMES="${__EGL_VENDOR_LIBRARY_FILENAMES:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}" \
    -v "${ROOT_DIR}:/workspace" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -w /workspace \
    --entrypoint "" \
    "${IMAGE_NAME}" \
    zsh
fi
