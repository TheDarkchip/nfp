# SOUNDNESS_LIMITATIONS

This file tracks **current limitations** and **remaining work** for the tabula rasa rewrite.
It is intentionally brief and focused on the soundness boundary.

## Current limitations

- The trusted CLI only **checks explicit certificates**; it does not search for witnesses or
  run model evaluation.
- Induction certificates are **head-level** (softmax-margin + value-interval + logit-diff lower
  bound) and conditional on the supplied `prev`, `active`, and `direction` inputs. They do **not**
  yet imply full model behavior.
- Direction metadata (`direction-target`, `direction-negative`) is untrusted and assumes that the
  unembedding columns represent token logits.
- The active set is user-supplied (or defaulted by the parser); bounds only hold for
  `q ∈ active`.
- Performance: checking large certificates can be expensive for long sequences.

## Remaining work

- Prove or verify that `prev`, `active`, and `direction` are derived from token-level semantics.
- Add a verified extraction pipeline from model weights to explicit certificates.
- Extend the bridge from head-level certificates to full circuit/model semantics.
