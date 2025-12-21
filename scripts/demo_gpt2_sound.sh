#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-models/gpt2.nfpt}"
REPORT_PATH="${2:-reports/gpt2_sound_demo.txt}"

mkdir -p "$(dirname "$MODEL_PATH")" "$(dirname "$REPORT_PATH")"

if [ ! -f "$MODEL_PATH" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv run python scripts/export_gpt2.py "$MODEL_PATH"
  else
    python scripts/export_gpt2.py "$MODEL_PATH"
  fi
fi

lake exe nfp certify "$MODEL_PATH" --output "$REPORT_PATH"
echo "Report written to $REPORT_PATH"
