# What Induction-Head Certificates Do (and Do Not) Claim

This document summarizes the **useful, limited guarantees** provided by an induction‑head
certificate, without overselling.

Two certificate kinds are supported:
- `onehot-approx` (proxy bounds only)
- `induction-aligned` (adds a periodic prompt check via `period`)

## What the certificate **does** guarantee

If the Lean checker accepts a certificate, then:
- **Kind `onehot-approx`:** softmax‑margin bounds and one‑hot‑style weight bounds hold on the
  specified active queries; value‑interval bounds hold; and a logit‑diff lower bound is verified
  if `--min-logit-diff` is used with direction metadata.
- **Kind `induction-aligned`:** the periodic prompt semantics hold (`active`/`prev` match the
  declared period) and the **prefix matching score** (stripe‑mean) is evaluated on the full
  second repeat (gated by `--min-stripe-mean`, default `0`).

These are **formal, exact** statements about the explicit certificate data.

## Why this is useful

- **Quantitative guarantees:** the bounds are numeric and can be gated (e.g., require a strictly
  positive logit‑diff lower bound).
- **Reproducibility:** certificates are explicit artifacts that can be re‑checked later.
- **Comparability:** bounds provide a principled way to compare heads or settings.
- **Soundness boundary clarity:** generation is untrusted, verification is trusted.

## What the certificate **does not** guarantee

- **No full‑model claim:** this is a head‑level certificate; it does not imply end‑to‑end model
  behavior.
- **Input‑specific:** guarantees apply only to the specified inputs / token patterns.
- **No onehot bounds for induction‑aligned:** prefix-matching metrics do not certify one‑hot
  attention or positive score margins on every query.
- **Untrusted semantics:** unless you pass `--tokens`, the token sequence is not verified. For
  `kind onehot-approx`, this means `prev`/`active` are unchecked against tokens; for
  `kind induction-aligned`, only the periodic prompt is checked (not actual token periodicity).
- **Direction is untrusted:** `direction-target` / `direction-negative` are supplied metadata.

## Optional token verification

If you provide a token list to the CLI (`--tokens`), the checker verifies:
- `kind onehot-approx`: `prev`/`active` match **previous‑occurrence semantics**.
- `kind induction-aligned`: the token sequence is periodic with the declared `period`.

This strengthens the link to the induction‑head diagnostic while keeping the trusted checker
lightweight.
