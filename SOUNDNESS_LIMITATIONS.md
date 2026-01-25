# SOUNDNESS_LIMITATIONS

This file tracks **current limitations** and **remaining work** for the tabula rasa rewrite.
It is intentionally brief and focused on the soundness boundary.

## Current limitations

- The trusted CLI only **checks explicit certificates**; it does not search for witnesses or
  run model evaluation.
- Induction certificates are **head-level** (softmax-margin + value-interval + logit-diff lower
  bound) and conditional on the supplied `prev`, `active`, and `direction` inputs. For
  `kind induction-aligned`, the checker instead evaluates prefix-matching (stripe-mean)
  and copying on the full period using the supplied `copy-logit` data.
  These do **not** yet imply full model behavior.
- Value/LN bounds are tied to the pre-LN inputs in the certificate (`embed`). If `model-resid`
  and `model-ln` data are provided, the checker verifies post-LN residuals against those inputs;
  if only embeddings are supplied, the certificate is a proxy for true residual-stream inputs.
- Direction metadata (`direction-target`, `direction-negative`) is untrusted and assumes that the
  unembedding columns represent token logits.
- Any direction search performed by Python helpers is untrusted witness generation; only the
  resulting explicit certificate is checked by the Lean CLI.
- The active set is user-supplied (or defaulted by the parser); bounds only hold for
  `q ∈ active`. You can optionally verify token semantics via
  `nfp induction verify --tokens ...` (previous-occurrence for `onehot-approx`,
  periodicity for `induction-aligned`).
- Performance: checking large certificates can be expensive for long sequences.

## Remaining work

- Prove or verify that `prev`, `active`, and `direction` are derived from token-level semantics.
- Add a verified extraction pipeline from model weights to explicit certificates.
- Extend the bridge from head-level certificates to full circuit/model semantics.
