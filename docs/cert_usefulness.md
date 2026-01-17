# What Induction-Head Certificates Do (and Do Not) Claim

This document summarizes the **useful, limited guarantees** provided by an induction‑head
certificate, without overselling.

## What the certificate **does** guarantee

If the Lean checker accepts a certificate, then:
- **Softmax‑margin bounds** hold on the specified active queries (the `prev` score dominates other
  keys by the declared margin).
- **One‑hot‑style weight bounds** hold on those queries (non‑`prev` weights are bounded by `ε`).
- **Value‑interval bounds** hold for the supplied value ranges.
- **Logit‑diff lower bound** holds for the supplied direction (if direction metadata is present
  and the checker is run with `--min-logit-diff`).

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
- **Untrusted semantics:** unless you pass `--tokens`, the `prev` and `active` sets are not
  verified against a token sequence.
- **Direction is untrusted:** `direction-target` / `direction-negative` are supplied metadata.

## Optional token verification

If you provide a token list to the CLI (`--tokens`), the checker verifies that `prev` and `active`
match **previous‑occurrence semantics** for that token sequence. This strengthens the link to the
induction‑head diagnostic while keeping the trusted checker lightweight.
