# CLAIMS

This file lists what is formally proven in Lean, what is soundly checked by the trusted checker,
what is untrusted/heuristic, and what is not yet proven in the tabula rasa rewrite.

## Proven in Lean

- Circuit core definitions and semantics (typed circuits, evaluation, interfaces).
- Softmax-margin certificate soundness: `checkSoftmaxMarginCert` implies
  `SoftmaxMarginBoundsOn`.
- Value-range certificate soundness: `checkValueRangeCert` implies `ValueRangeBounds`.
- Induction-head certificate soundness: `checkInductionHeadCert` implies
  `InductionHeadCertBounds`.
- Logit-diff lower bound lemmas: `logitDiffLowerBound_le`, `logitDiffLowerBoundAt_le`, and
  `logitDiffLowerBoundWeightedAt_le`.
- The head logit-diff equals the direction dot product of the head output
  (`headLogitDiff_eq_direction_dot_headOutput`).
- Row-stochastic attention/one-hot bounds for induction heads and related interval lemmas.

## Soundly checked by the trusted CLI

- `nfp induction certify`, `nfp induction certify_nonvacuous`, and
  `nfp induction head_cert_check` verify explicit induction-head certificates from a single
  cert file, optionally enforcing minimum `active`, `margin`, `eps`, and logit-diff gates.

## Untrusted / heuristic

- Python helpers that generate explicit induction-head certificates from GPT-2 weights or
  `.nfpt` files: `scripts/build_gpt2_induction_cert.py`,
  `scripts/build_gpt2_induction_cert_from_binary.py`.
- Exporters and dataset generators for `.nfpt` model files.
- Any choice of prompts, directions, or candidate heads used by certificate generators.

## Not yet proven

- A verified extraction pipeline from model weights to explicit certificates.
- Model-level claims about logits or Jacobians derived from certificates.
- A full bridge from explicit head certificates to complete model semantics.
