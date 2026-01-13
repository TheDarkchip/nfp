# CLAIMS

This file lists what is formally proven in Lean, what is soundly checked by the trusted checker,
what is untrusted/heuristic, and what is not yet proven in the tabula rasa rewrite.

## Proven in Lean

- Circuit core definitions and semantics (typed circuits, evaluation, interfaces).
- Softmax-margin certificate soundness: `checkSoftmaxMarginCert` implies
  `SoftmaxMarginBoundsOn`.
- Value-range certificate soundness: `checkValueRangeCert` implies `ValueRangeBounds`.
- Induction-head certificate soundness: `InductionHeadCertSound` holds whenever
  `buildInductionCertFromHeadCoreWith?` returns a certificate for the given inputs.
- Logit-diff lower bound lemmas: `logitDiffLowerBound_le` and
  `logitDiffLowerBoundFromCert_le`.
- Bridge lemmas composing head logit-diff bounds with head outputs and residual
  interval bounds: `headLogitDiff_eq_direction_dot_headOutput` and
  `logitDiffLowerBound_with_residual`, plus interval-composition
  `logitDiffLowerBound_with_output_intervals`.
- Downstream linear certificate soundness: `checkDownstreamLinearCert` implies
  `DownstreamLinearBounds`.
- Residual-interval certificate soundness: `checkResidualIntervalCert` implies
  `ResidualIntervalBounds`.
- GPT-2 residual interval bounds from model slices are sound for
  `transformerStackFinalReal` on active positions (`gpt2ResidualIntervalBoundsActive_sound`).
- End-to-end direction-dot lower bounds on `transformerStackFinalReal` can be derived by
  composing head logit-diff bounds with head/output intervals
  (`logitDiffLowerBound_end_to_end_gpt2`).
- Row-sum matrix norm bounds for `mulVec` under uniform input magnitude.
- Tanh-GELU bounds and interval propagation through MLP layers.
- Interval bounds for multi-head attention and full transformer-layer residual blocks.
- Interval bounds for transformer stacks and final LayerNorm outputs.

## Soundly checked by the trusted CLI

- `nfp induction certify` verifies head-level induction certificates from either a head-input
  file or a model binary, and can compute a logit-diff lower bound.
- `nfp induction certify_nonvacuous` requires a strictly positive logit-diff lower bound.
- `nfp induction advanced certify_sound` recomputes `eps`/`margin` and `lo`/`hi` from raw
  entries and verifies the resulting certificates.
- `nfp induction advanced certify_head` recomputes scores/values from exact head inputs and
  verifies the resulting induction certificate (experimental, potentially slow).
- `nfp induction advanced certify_head_model` reads a model binary, derives head inputs in Lean,
  and verifies the resulting induction certificate (includes attention projection biases and
  derives `prev`/active from the stored token sequence by default, and builds the logit-diff
  direction vector from the target/negative unembedding columns).
- `nfp induction advanced certify_head_model_auto` derives the logit-diff direction from the
  prompt tokens stored in the model file before running the same head-input checker (the
  direction vector still uses the unembedding columns).
- `nfp induction advanced certify_end_to_end` composes a head-level logit-diff lower bound with
  a downstream error certificate (arithmetic consistency only).
- `nfp induction advanced certify_end_to_end_matrix` computes a downstream bound from a matrix
  payload using verified row-sum norms, then composes it with the head-level logit-diff lower
  bound.
- `nfp induction advanced certify_end_to_end_model` derives the unembedding direction from an
  `NFP_BINARY_V1` model file, computes a downstream error bound from either a supplied
  residual-interval certificate or a verified model-derived interval, and composes it with the
  head-level logit-diff lower bound.

## Untrusted / heuristic

- Python helpers that generate certificates from GPT-2 weights or head inputs:
  `scripts/build_gpt2_induction_cert.py`, `scripts/build_gpt2_head_inputs.py`,
  `scripts/build_downstream_linear_cert.py`.
- The head-input extractor now emits attention projection biases and LayerNorm metadata, but
  the Lean-side computation still ignores LayerNorm and the shared attention output bias.
- External residual-interval scripts remain untrusted; model-derived bounds are now available.
- Any downstream error bound provided externally (outside the matrix-payload path).

## Not yet proven

- End-to-end claims about GPT-2 logits or Jacobians derived from certificates.
- Sound, verified downstream bounds computed from GPT-2 weights inside Lean.
- A full end-to-end bridge from head certificates to full-model logit bounds
  (beyond the head-output + residual-interval composition).
