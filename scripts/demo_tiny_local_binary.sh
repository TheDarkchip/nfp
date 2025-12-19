#!/usr/bin/env bash
set -euo pipefail

MODEL_TEXT="${1:-tests/fixtures/tiny_sound_model.nfpt}"
INPUT_TEXT="${2:-tests/fixtures/tiny_sound_input.nfpt}"
BINARY_PATH="${3:-tests/fixtures/tiny_sound_binary.nfpt}"
REPORT_PATH="${4:-reports/tiny_sound_local_binary.txt}"

mkdir -p "$(dirname "$BINARY_PATH")" "$(dirname "$REPORT_PATH")"

if [ "${USE_UV:-0}" = "1" ] && command -v uv >/dev/null 2>&1; then
  uv run python scripts/convert_text_fixture_to_binary.py \
    --model "$MODEL_TEXT" \
    --input "$INPUT_TEXT" \
    --output "$BINARY_PATH"
else
  python scripts/convert_text_fixture_to_binary.py \
    --model "$MODEL_TEXT" \
    --input "$INPUT_TEXT" \
    --output "$BINARY_PATH"
fi

lake exe nfp certify "$BINARY_PATH" --delta 0.05 --output "$REPORT_PATH"
echo "Report written to $REPORT_PATH"
