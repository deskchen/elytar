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
python3 benchmark/run.py --tasks cube_stack --difficulty easy --steps 5 --output-dir benchmark/results
```

List tasks: `python3 benchmark/run.py --list-tasks`
