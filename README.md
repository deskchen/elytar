# Elytar

Linux dev environment for GPU PhysX benchmarking with SAPIEN. PhysX and SAPIEN are built from local source (no prebuilt downloads).

## How to run

**1. Build the dev image** (from repo root):

```bash
docker build -f docker/Dockerfile.dev -t elytar-dev:cuda12.8-clang .
```

**2. Start the container:**

```bash
./scripts/run-dev.sh
```

**3. Build the toolchain** (inside the container):

```bash
/workspace/scripts/update_toolchain.sh
```

**4. Run the benchmark:**

```bash
python3 -m benchmark.sapien.run --tasks cube_stack --difficulty easy --steps 5 --output-dir benchmark/sapien/results
```

List tasks: `python3 -m benchmark.sapien.run --list-tasks`

## PhysX snippets A/B (two trees)

For snippet benchmarking, compare binaries built from:
- `physx-5.6.1` (baseline `.cu`)
- `physx-5.6.1-capybara` (PTX replacement with `*.capybara.ptx`)

Example build commands:

```bash
# Baseline (.cu)
PHYSX_DIR="/workspace/physx-5.6.1" \
PX_PTX_REPLACE_LIST="" \
ELYTAR_BUILD_PHYSX_SNIPPETS=1 \
./scripts/update_toolchain.sh

# Capybara PTX
python3 scripts/compile_capybara_ptx.py -v
PHYSX_DIR="/workspace/physx-5.6.1-capybara" \
PX_PTX_REPLACE_LIST="utility" \
PX_PTX_SOURCE=capybara \
ELYTAR_BUILD_PHYSX_SNIPPETS=1 \
./scripts/update_toolchain.sh
```

Then benchmark chosen snippet binaries with `./benchmark/physx_snippets/run_ptx_ab.sh`.