#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-models/gpt2_rigorous.nfpt}"
REPORT_PATH="${2:-reports/gpt2_induction_sound_scan.txt}"
EXTRA_ARGS=()
if [ "$#" -gt 2 ]; then
  EXTRA_ARGS=("${@:3}")
fi

PYTHON_BIN="python"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if ! "$PYTHON_BIN" scripts/ensure_gelu_kind.py --check "$MODEL_PATH"; then
  PATCHED_PATH="${MODEL_PATH%.nfpt}_with_gelu_kind.nfpt"
  "$PYTHON_BIN" scripts/ensure_gelu_kind.py "$MODEL_PATH" \
    --output "$PATCHED_PATH" \
    --default tanh
  MODEL_PATH="$PATCHED_PATH"
fi

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  "$PYTHON_BIN" scripts/scan_gpt2_induction_sound.py \
    --model "$MODEL_PATH" \
    --output "$REPORT_PATH" \
    "${EXTRA_ARGS[@]}"
else
  "$PYTHON_BIN" scripts/scan_gpt2_induction_sound.py \
    --model "$MODEL_PATH" \
    --output "$REPORT_PATH"
fi

echo "Report written to $REPORT_PATH"
