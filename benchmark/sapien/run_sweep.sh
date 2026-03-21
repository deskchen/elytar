#!/usr/bin/env bash
# Sweep num_envs from 2 to 1024 and run benchmark.
# Run from repo root: ./benchmark/sapien/run_sweep.sh

set -euo pipefail

# Configurable variables
TASK="${TASK:-cube_stack}"
STEPS="${STEPS:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark/sapien/results}"
PREFIX="${PREFIX:-solver_ratio}"

for n in 2 4 8 16 32 64 128 256 512 1024; do
  echo "=== num_envs=$n ==="
  python3 -m benchmark.sapien.run \
    --tasks "$TASK" \
    --num-envs "$n" \
    --steps "$STEPS" \
    --output-dir "$OUTPUT_DIR" \
    --prefix "$PREFIX"
done

echo "Done. History: ${OUTPUT_DIR}/${PREFIX}_history.csv"

