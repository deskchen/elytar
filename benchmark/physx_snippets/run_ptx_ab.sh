#!/usr/bin/env bash
set -euo pipefail

# Repeated A/B wall-clock runner for PhysX snippet binaries.
#
# Required:
#   BENCH_CMD_A: command for variant A (usually vanilla .cu tree binary)
#   BENCH_CMD_B: command for variant B (usually capybara-tree PTX binary)
#
# Optional:
#   LABEL_A         default: vanilla_cu
#   LABEL_B         default: capybara_ptx
#   REPS            default: 10
#   STEPS_PER_RUN   default: 100
#   OUTPUT_CSV      default: benchmark/physx_snippets/results/ptx_ab_results.csv
#   RUN_ID          default: timestamp

BENCH_CMD_A="${BENCH_CMD_A:-}"
BENCH_CMD_B="${BENCH_CMD_B:-}"
LABEL_A="${LABEL_A:-vanilla_cu}"
LABEL_B="${LABEL_B:-capybara_ptx}"
REPS="${REPS:-1}"
STEPS_PER_RUN="${STEPS_PER_RUN:-10}"
OUTPUT_CSV="${OUTPUT_CSV:-benchmark/physx_snippets/results/ptx_ab_results.csv}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

if [[ -z "${BENCH_CMD_A}" || -z "${BENCH_CMD_B}" ]]; then
  echo "ERROR: BENCH_CMD_A and BENCH_CMD_B are required."
  echo "Example:"
  echo "  BENCH_CMD_A=\"/workspace/physx-5.6.1/bin/linux.x86_64/profile/SnippetIsosurface_64\" BENCH_CMD_B=\"/workspace/physx-5.6.1-capybara/bin/linux.x86_64/profile/SnippetIsosurface_64\" ./benchmark/physx_snippets/run_ptx_ab.sh"
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_CSV}")"
if [[ ! -f "${OUTPUT_CSV}" || ! -s "${OUTPUT_CSV}" ]]; then
  echo "run_id,variant,rep,elapsed_s,throughput_steps_per_s,steps_per_run,command" > "${OUTPUT_CSV}"
fi

run_one() {
  local label="$1"
  local cmd="$2"
  local rep="$3"

  local start_ns end_ns elapsed_s throughput
  start_ns=$(date +%s%N)
  bash -lc "${cmd}" >/dev/null
  end_ns=$(date +%s%N)

  elapsed_s=$(python3 - <<'PY' "${start_ns}" "${end_ns}"
import sys
s=int(sys.argv[1]); e=int(sys.argv[2])
print((e-s)/1e9)
PY
)
  throughput=$(python3 - <<'PY' "${STEPS_PER_RUN}" "${elapsed_s}"
import sys
steps=float(sys.argv[1]); elapsed=float(sys.argv[2])
print(0.0 if elapsed <= 0 else steps/elapsed)
PY
)

  echo "${RUN_ID},${label},${rep},${elapsed_s},${throughput},${STEPS_PER_RUN},\"${cmd}\"" >> "${OUTPUT_CSV}"
  echo "[${label}] rep=${rep} elapsed=${elapsed_s}s throughput=${throughput} steps/s"
}

for ((i=1; i<=REPS; i++)); do
  run_one "${LABEL_A}" "${BENCH_CMD_A}" "${i}"
  run_one "${LABEL_B}" "${BENCH_CMD_B}" "${i}"
done

echo "Wrote ${OUTPUT_CSV}"

