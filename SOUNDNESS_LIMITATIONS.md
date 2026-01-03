# SOUNDNESS_LIMITATIONS

This file tracks **current limitations** and **remaining work** for the tabula rasa rewrite.
It is intentionally brief and focused on the soundness boundary.

## Current limitations

- The trusted CLI only **checks certificates**; it does not search for witnesses or run a model.
- Induction certificates are **head-level** (softmax-margin + value-range + logit-diff lower bound).
  They do **not** yet imply end-to-end model behavior.
- Downstream error bounds can be computed from a **matrix payload** inside Lean. A model-based
  path exists, but it currently uses only the unembedding direction and relies on an external
  **residual-bound certificate** (per-coordinate absolute bounds).
- The `certify_head` path uses a **head-input file** extracted by an untrusted script; the extractor
  currently ignores LayerNorm and bias terms, so it is not end-to-end faithful.
- Performance: exact head-input recomputation in Lean can be slow for nontrivial sequence lengths.
- There is no bridge theorem connecting certificate validity to a full circuit/model semantics
  statement (for example, a formal statement about logits under a transformer block stack).

## Remaining work

- Compute the downstream bound **inside Lean** from model weights and certified residual
  bounds (not just matrix payloads), and wire this into `certify_end_to_end`.
- Replace untrusted residual-bound generation with a verified derivation from upstream bounds.
- Replace untrusted extraction with a verified parser for model weight slices.
- Add a formal bridge from certificates to circuit semantics and (eventually) to end-to-end
  transformer claims.
- Improve performance for the exact head-input path without weakening soundness.
