#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

# Reproducible circuit-level verification for the top GPT-2 small induction heads.
# Uses a fixed token pattern (period 16) and direction search over vocab 1000-1100.

period=16
seq=$((2 * period))
reports_dir="reports"
prev_tokens="${reports_dir}/prev.tokens"
ind_tokens="${reports_dir}/induction32.tokens"
prev_cert="${reports_dir}/tl_scan/circuit_certs/prev/L4H11.cert"

mkdir -p "${reports_dir}" "${reports_dir}/tl_scan/circuit_certs/prev" "${reports_dir}/tl_scan/circuit_certs/ind"

python - <<'PY'
from pathlib import Path

period = 16
seq = 2 * period
prev_tokens = Path("reports/prev.tokens")
ind_tokens = Path("reports/induction32.tokens")

prev_tokens.parent.mkdir(parents=True, exist_ok=True)
ind_tokens.parent.mkdir(parents=True, exist_ok=True)

prev = [1] * seq
induction = list(range(1000, 1000 + period)) * 2

with prev_tokens.open("w", encoding="ascii") as f:
    f.write(f"seq {seq}\n")
    for idx, tok in enumerate(prev):
        f.write(f"token {idx} {tok}\n")

with ind_tokens.open("w", encoding="ascii") as f:
    f.write(f"seq {seq}\n")
    for idx, tok in enumerate(induction):
        f.write(f"token {idx} {tok}\n")
PY

python scripts/build_gpt2_induction_cert.py \
  --output "${prev_cert}" \
  --layer 4 --head 11 \
  --seq "${seq}" --pattern-length "${period}" \
  --tokens-in "${prev_tokens}" \
  --active-eps-max 1 --min-margin -10000 \
  --search-direction --direction-vocab-min 1000 --direction-vocab-max 1100 --direction-max-candidates 0 \
  --omit-unembed-rows

pairs=("5 5" "6 9" "5 1" "7 10" "7 2")
for pair in "${pairs[@]}"; do
  set -- ${pair}
  layer="$1"
  head="$2"
  ind_cert="${reports_dir}/tl_scan/circuit_certs/ind/L${layer}H${head}.cert"
  python scripts/build_gpt2_induction_cert.py \
    --output "${ind_cert}" \
    --layer "${layer}" --head "${head}" \
    --seq "${seq}" --pattern-length "${period}" \
    --tokens-in "${ind_tokens}" --prev-shift \
    --active-eps-max 1 --min-margin -10000 \
    --search-direction --direction-vocab-min 1000 --direction-vocab-max 1100 --direction-max-candidates 0 \
    --omit-unembed-rows

  lake exe nfp induction verify \
    --cert "${ind_cert}" \
    --min-margin -10000 --max-eps 1 --min-logit-diff 0

  lake exe nfp induction verify-circuit \
    --prev-cert "${prev_cert}" \
    --ind-cert "${ind_cert}" \
    --period "${period}" \
    --tokens "${ind_tokens}"
done
