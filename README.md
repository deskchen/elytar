# Elytar Linux Dev Environment

This repository is configured for a Linux-only NVIDIA GPU workflow.

## What this provides

- A CUDA + clang dev image: `docker/Dockerfile.dev`
- A container launcher: `scripts/run-dev.sh`
- A one-command toolchain updater: `scripts/update_toolchain.sh`

The update script rebuilds:

1. PhysX from local source (`physx/`) with the `linux-clang` preset (GPU enabled).
2. SAPIEN from local source (`sapien/`) against that local PhysX build.
3. A fresh Python wheel, then reinstalls it in the container Python environment.

## Prerequisites

- Linux host
- Docker
- NVIDIA Container Toolkit (`--gpus all` support)
- NVIDIA driver compatible with CUDA 12.8 container toolchain

## Build the dev image

From repo root:

```bash
docker build -f docker/Dockerfile.dev -t elytar-dev:cuda12.8-clang .
```

If you previously built the image, rebuild it after dependency changes in `docker/Dockerfile.dev`.

## Start the dev container

```bash
./scripts/run-dev.sh
```

The launcher mounts this repo to `/workspace` and starts in `/workspace/sapien`.

## Update the toolchain (PhysX + SAPIEN + wheel)

Inside the container:

```bash
/workspace/scripts/update_toolchain.sh
```

## Useful environment overrides

- `PHYSX_CONFIG` (default: `checked`, supports `debug|checked|profile|release`)
- `PHYSX_PRESET` (default: `linux-clang`)
- `SAPIEN_BUILD_MODE` (default: `--profile`)
- `SAPIEN_BUILD_DIR` (default: `docker_sapien_build`)
- `PYTHON_BIN` (default: `python3`, builds/installs wheel for this one interpreter only)

Example:

```bash
PHYSX_CONFIG=checked SAPIEN_BUILD_MODE="" PYTHON_BIN=python3.11 /workspace/scripts/update_toolchain.sh
```
