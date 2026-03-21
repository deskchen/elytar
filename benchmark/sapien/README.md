# SAPIEN GPU Benchmarks

This folder is the nested home for the existing Python/SAPIEN benchmark suite.

## Entrypoint

- `python3 -m benchmark.sapien.run`

## Quick start

```bash
python3 -m benchmark.sapien.run --tasks cube_stack,pouring_balls
```

Sweep helper:

```bash
TASK=cube_stack STEPS=2000 ./benchmark/sapien/run_sweep.sh
```

Solver-ratio plot helper:

```bash
python3 -m benchmark.sapien.plot_solver_ratio \
  --input benchmark/sapien/results/solver_ratio_history.csv
```

Use `--output-dir benchmark/sapien/results` to keep SAPIEN benchmark artifacts in
this subfolder.

