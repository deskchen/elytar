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
python3 -m benchmark.run --tasks cube_stack --difficulty easy --steps 5 --output-dir benchmark/results
```

List tasks: `python3 -m benchmark.run --list-tasks`

## C++ checker (clangd / IntelliSense)

The SAPIEN build exports **compile_commands.json** and the toolchain script symlinks it to `sapien/compile_commands.json`. After you run `scripts/update_toolchain.sh` at least once, the C++ extension (clangd) should use it automatically when you open files under `sapien/src/`. If the checker still reports false errors (e.g. “undeclared identifier” on valid code), point the extension at that file: in VS Code/Cursor set `C_Cpp.default.compileCommands` to `${workspaceFolder}/sapien/compile_commands.json`, or for clangd create `sapien/.clangd` with `CompileFlags: { CompilationDatabase: "compile_commands.json" }`.

**Why the checker still fails:** `compile_commands.json` is generated **inside the container** (paths like `/workspace/sapien/...`). If you open the repo on the **host**, clangd runs on the host and those paths don't match. **Fix: run the IDE inside the container.** Install the **Dev Containers** extension, then **Reopen in Container** (Command Palette). The repo includes `.devcontainer/devcontainer.json` so the same image and GPU settings are used; clangd then runs in the container and the C++ checker works.
