#!/usr/bin/env bash
set -euo pipefail

echo "=== Part 1: Compile Capybara PTX ==="
conda run -n triton-dev python scripts/compile_capybara_ptx.py -v
echo "Part 1 passed."

echo "=== Part 2: Build & test PhysX in Docker ==="
CONTAINER_NAME="${CONTAINER_NAME:-elytar-dev}"
docker start "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker exec -u dev -w /workspace "${CONTAINER_NAME}" bash -c '
  ELYTAR_PHYSX_ONLY=1 \
  PHYSX_DIR="/workspace/physx-5.6.1-capybara" \
  PX_PTX_SOURCE=capybara \
  ELYTAR_BUILD_PHYSX_SNIPPETS=1 \
  ./scripts/update_toolchain.sh
'
echo "Part 2 passed."

echo "=== All verification passed ==="
