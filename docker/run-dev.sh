#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-elytar-dev:cuda12.8-clang}"
CONTAINER_NAME="${CONTAINER_NAME:-elytar-dev}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Reattaching to existing container '${CONTAINER_NAME}'..."
  docker start -ai "${CONTAINER_NAME}"
else
  echo "Creating new container '${CONTAINER_NAME}'..."
  docker run -it \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute \
    -v "${ROOT_DIR}:/workspace" \
    -w /workspace \
    "${IMAGE_NAME}" \
    bash
fi
