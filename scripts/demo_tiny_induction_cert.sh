#!/usr/bin/env bash
set -euo pipefail

MODEL_TEXT="${1:-tests/fixtures/tiny_sound_model.nfpt}"
INPUT_TEXT="${2:-tests/fixtures/tiny_sound_input.nfpt}"
BINARY_PATH="${3:-tests/fixtures/tiny_sound_binary.nfpt}"
REPORT_PATH="${4:-reports/tiny_induction_cert.txt}"

mkdir -p "$(dirname "$BINARY_PATH")" "$(dirname "$REPORT_PATH")"

PYTHON_BIN="python"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" scripts/convert_text_fixture_to_binary.py \
  --model "$MODEL_TEXT" \
  --input "$INPUT_TEXT" \
  --output "$BINARY_PATH"

lake exe nfp induction_cert "$BINARY_PATH" \
  --layer1 0 --head1 0 --layer2 0 --head2 0 --coord 0 \
  --offset1 -1 --offset2 -1 --target 2 --negative 1 \
  --delta 0.05 --output "$REPORT_PATH"

echo "Report written to $REPORT_PATH"
