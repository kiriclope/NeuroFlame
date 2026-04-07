#!/usr/bin/env bash
set -u

# Usage:
#   bash scripts/run_dual_analysis_seeds.sh N [RUNS_ROOT] [PATTERN] [OUT_ROOT] [PLOT_ROOT] [DEVICE] [AGG_CSV]
#
# Example:
#   bash scripts/run_dual_analysis_seeds.sh \
#     5 \
#     runs/dual \
#     "*.pth" \
#     results/dual_analysis \
#     results/dual_plots \
#     cuda:0 \
#     results/dual_analysis/all_seeds.csv

N="${1:?Please provide N}"
RUNS_ROOT="${2:-runs/dual}"
PATTERN="${3:-*.pth}"
OUT_ROOT="${4:-results/dual_analysis}"
PLOT_ROOT="${5:-results/dual_plots}"
DEVICE="${6:-cuda:0}"
AGG_CSV="${7:-}"

LOG_DIR="${OUT_ROOT}"
LOG_FILE="${LOG_DIR}/run_dual_analysis_seeds.log"

mkdir -p "${OUT_ROOT}" "${PLOT_ROOT}" "${LOG_DIR}"

echo "num_seeds: ${N}" | tee "${LOG_FILE}"
echo "runs_root: ${RUNS_ROOT}" | tee -a "${LOG_FILE}"
echo "pattern:   ${PATTERN}" | tee -a "${LOG_FILE}"
echo "out_root:  ${OUT_ROOT}" | tee -a "${LOG_FILE}"
echo "plot_root: ${PLOT_ROOT}" | tee -a "${LOG_FILE}"
echo "device:    ${DEVICE}" | tee -a "${LOG_FILE}"
echo "agg_csv:   ${AGG_CSV:-<none>}" | tee -a "${LOG_FILE}"

successful=()
failed=()
skipped=()

for ((seed=1; seed<=N; seed++)); do
  CHECKPOINT_DIR="${RUNS_ROOT}/seed${seed}"
  OUT_CSV="${OUT_ROOT}/seed${seed}.csv"
  PLOT_DIR="${PLOT_ROOT}/seed${seed}"

  echo "" | tee -a "${LOG_FILE}"
  echo "=== Seed ${seed} ===" | tee -a "${LOG_FILE}"

  if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Skipping seed ${seed}: missing directory ${CHECKPOINT_DIR}" | tee -a "${LOG_FILE}"
    skipped+=("${seed}")
    continue
  fi

  mkdir -p "${PLOT_DIR}"

  if bash scripts/run_dual_analysis.sh \
    "${CHECKPOINT_DIR}" \
    "${PATTERN}" \
    "${OUT_CSV}" \
    "${PLOT_DIR}" \
    "${DEVICE}" 2>&1 | tee -a "${LOG_FILE}"
  then
    echo "Seed ${seed} succeeded" | tee -a "${LOG_FILE}"
    successful+=("${seed}")
  else
    echo "Seed ${seed} failed" | tee -a "${LOG_FILE}"
    failed+=("${seed}")
  fi
done

echo "" | tee -a "${LOG_FILE}"
echo "=== Summary ===" | tee -a "${LOG_FILE}"
echo "successful: ${successful[*]:-<none>}" | tee -a "${LOG_FILE}"
echo "failed:     ${failed[*]:-<none>}" | tee -a "${LOG_FILE}"
echo "skipped:    ${skipped[*]:-<none>}" | tee -a "${LOG_FILE}"

# Optional aggregation
if [[ -n "${AGG_CSV}" ]]; then
  echo "" | tee -a "${LOG_FILE}"
  echo "Aggregating CSVs into ${AGG_CSV}" | tee -a "${LOG_FILE}"

  mkdir -p "$(dirname "${AGG_CSV}")"

  first_file=""
  for ((seed=1; seed<=N; seed++)); do
    csv="${OUT_ROOT}/seed${seed}.csv"
    if [[ -f "${csv}" ]]; then
      first_file="${csv}"
      break
    fi
  done

  if [[ -z "${first_file}" ]]; then
    echo "No CSV files found to aggregate." | tee -a "${LOG_FILE}"
  else
    head -n 1 "${first_file}" > "${AGG_CSV}"

    for ((seed=1; seed<=N; seed++)); do
      csv="${OUT_ROOT}/seed${seed}.csv"
      if [[ -f "${csv}" ]]; then
        tail -n +2 "${csv}" >> "${AGG_CSV}"
      fi
    done

    echo "Wrote aggregated CSV: ${AGG_CSV}" | tee -a "${LOG_FILE}"
  fi
fi

# Exit nonzero only if something failed
if [[ "${#failed[@]}" -gt 0 ]]; then
  exit 1
fi
