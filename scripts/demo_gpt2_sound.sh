#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-models/gpt2.nfpt}"
REPORT_PATH="${2:-reports/gpt2_sound_demo.txt}"

mkdir -p "$(dirname "$MODEL_PATH")" "$(dirname "$REPORT_PATH")"

PYTHON_BIN="python"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [ ! -f "$MODEL_PATH" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv run python scripts/export_gpt2.py "$MODEL_PATH"
  else
    "$PYTHON_BIN" scripts/export_gpt2.py "$MODEL_PATH"
  fi
fi

if ! "$PYTHON_BIN" scripts/ensure_gelu_kind.py --check "$MODEL_PATH"; then
  PATCHED_PATH="${MODEL_PATH%.nfpt}_with_gelu_kind.nfpt"
  "$PYTHON_BIN" scripts/ensure_gelu_kind.py "$MODEL_PATH" \
    --output "$PATCHED_PATH" \
    --default tanh
  MODEL_PATH="$PATCHED_PATH"
fi

lake exe nfp certify "$MODEL_PATH" --output "$REPORT_PATH"
echo "Report written to $REPORT_PATH"
