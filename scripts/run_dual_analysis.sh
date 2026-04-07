#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_dual_analysis.sh CHECKPOINT_DIR [PATTERN] [OUT_CSV] [PLOT_DIR] [DEVICE]
#
# Example:
#   bash scripts/run_dual_analysis.sh \
#     runs/dual/seed1 \
#     "*.pth" \
#     results/dual_analysis.csv \
#     results/dual_plots \
#     cuda:0

CHECKPOINT_DIR="${1:-runs/dual}"
PATTERN="${2:-*.pth}"
OUT_CSV="${3:-results/dual_analysis.csv}"
PLOT_DIR="${4:-results/dual_plots}"
DEVICE="${5:-cuda:0}"

echo "checkpoint_dir: ${CHECKPOINT_DIR}"
echo "pattern:        ${PATTERN}"
echo "out_csv:        ${OUT_CSV}"
echo "plot_dir:       ${PLOT_DIR}"
echo "device:         ${DEVICE}"

python -m src.test.dual.run_analysis \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --pattern "${PATTERN}" \
  --device "${DEVICE}" \
  --out "${OUT_CSV}" \
  --plot-dir "${PLOT_DIR}"

echo "done"
