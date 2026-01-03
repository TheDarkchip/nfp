# CLAIMS

This file lists what is formally proven in Lean, what is soundly checked by the trusted checker,
what is untrusted/heuristic, and what is not yet proven in the tabula rasa rewrite.

## Proven in Lean

- Circuit core definitions and semantics (typed circuits, evaluation, interfaces).
- Softmax-margin certificate soundness: `checkSoftmaxMarginCert` implies
  `SoftmaxMarginBoundsOn`.
- Value-range certificate soundness: `checkValueRangeCert` implies `ValueRangeBounds`.
- Logit-diff lower bound lemma: `logitDiffLowerBound_le`.
- Downstream linear certificate soundness: `checkDownstreamLinearCert` implies
  `DownstreamLinearBounds`.
- Residual-interval certificate soundness: `checkResidualIntervalCert` implies
  `ResidualIntervalBounds`.
- Row-sum matrix norm bounds for `mulVec` under uniform input magnitude.

## Soundly checked by the trusted CLI

- `nfp induction certify` verifies softmax-margin certificates, value-range certificates,
  and computes a logit-diff lower bound.
- `nfp induction certify_sound` recomputes `eps`/`margin` and `lo`/`hi` from raw entries
  and verifies the resulting certificates.
- `nfp induction certify_head` recomputes scores/values from exact head inputs and verifies
  the resulting induction certificate (experimental, potentially slow).
- `nfp induction certify_head_model` reads a model binary, derives head inputs in Lean,
  and verifies the resulting induction certificate (includes attention projection biases
  and derives `prev`/active from the stored token sequence by default).
- `nfp induction certify_end_to_end` composes a head-level logit-diff lower bound with a
  downstream error certificate (arithmetic consistency only).
- `nfp induction certify_end_to_end_matrix` computes a downstream bound from a matrix payload
  using verified row-sum norms, then composes it with the head-level logit-diff lower bound.
- `nfp induction certify_end_to_end_model` derives a downstream matrix from an `NFP_BINARY_V1`
  model file (unembedding direction only), computes a downstream error bound from a
  residual-interval certificate, and composes it with the head-level logit-diff lower bound.

## Untrusted / heuristic

- Python helpers that generate certificates from GPT-2 weights or head inputs:
  `scripts/build_gpt2_induction_cert.py`, `scripts/build_gpt2_head_inputs.py`,
  `scripts/build_downstream_linear_cert.py`.
- The head-input extractor now emits attention projection biases and LayerNorm metadata, but
  the Lean-side computation still ignores LayerNorm and the shared attention output bias.
- Residual-interval certificates are generated externally (unchecked beyond consistency).
- Any downstream error bound provided externally (outside the matrix-payload path).

## Not yet proven

- End-to-end claims about GPT-2 logits or Jacobians derived from certificates.
- Sound, verified downstream bounds computed from GPT-2 weights inside Lean.
- A bridge theorem connecting certificate validity to full circuit/model semantics.
