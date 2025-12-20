#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-models/gpt2_rigorous.nfpt}"
REPORT_PATH="${2:-reports/gpt2_induction_sound_scan.txt}"
EXTRA_ARGS=("${@:3}")

python scripts/scan_gpt2_induction_sound.py \
  --model "$MODEL_PATH" \
  --output "$REPORT_PATH" \
  "${EXTRA_ARGS[@]}"

echo "Report written to $REPORT_PATH"
