# NFP User-Facing Demo (Induction Head Certification)

This demo shows a full, reproducible path from **untrusted certificate generation**
to **trusted Lean verification** for an induction head.

## 0. Prerequisites

- Lean 4 / Lake (pinned in `lean-toolchain`)
- Python with: `numpy`, `torch`, `transformers`

## 1. Build

```bash
lake build -q --wfail
lake build nfp -q --wfail
```

## 2. Generate an explicit certificate (untrusted)

This uses a repeated pattern (period repeated twice) and searches for a
non-vacuous logit-diff direction.

```bash
python scripts/build_gpt2_induction_cert.py \
  --output reports/gpt2_induction.cert \
  --layer 0 --head 5 --seq 32 --pattern-length 16 \
  --random-pattern --seed 0 \
  --active-eps-max 1/2 \
  --search-direction --direction-vocab-min 1000 --direction-vocab-max 2000 \
  --direction-min-lb 1/10 \
  --direction-report-out reports/direction_report.txt --direction-topk 10 \
  --tokens-out reports/gpt2_induction.tokens
```

Expected output includes:
- a certificate (`reports/gpt2_induction.cert`)
- a ranked direction report (`reports/direction_report.txt`)
- a token list (`reports/gpt2_induction.tokens`)

## 3. Verify with the Lean checker (trusted)

This enforces a **non-vacuous** logit-diff lower bound and checks `prev/active`
against the token list.

```bash
lake exe nfp induction verify \
  --cert reports/gpt2_induction.cert \
  --min-logit-diff 1/10 \
  --tokens reports/gpt2_induction.tokens
```

Expected output (example):

```
ok: onehot-approx (proxy) certificate checked (seq=32, active=15, margin=..., eps=..., logitDiffLB=...)
```

## Notes

- Everything in `scripts/` is **untrusted witness generation**.
- The Lean CLI **only verifies** explicit certificates and token semantics.
- To emit an induction-aligned certificate, add `--kind induction-aligned` when generating.
  Induction-aligned verification uses the prefix-matching stripe-mean and copying
  metrics (not softmax-margin/onehot gates).
