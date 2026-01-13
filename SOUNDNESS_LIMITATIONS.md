# SOUNDNESS_LIMITATIONS

This file tracks **current limitations** and **remaining work** for the tabula rasa rewrite.
It is intentionally brief and focused on the soundness boundary.

## Current limitations

- The trusted CLI only **checks certificates**; it does not search for witnesses or run a model.
- Induction certificates are **head-level** (softmax-margin + value-range + logit-diff lower bound),
  and they are conditional on the supplied `prev`, `active`, and `direction` inputs. They do **not**
  yet imply end-to-end model behavior.
- Downstream error bounds can be computed from a **matrix payload** inside Lean. A model-based
  path exists, but it currently uses only the unembedding direction and derives residual
  intervals via conservative interval propagation (ignoring attention-score structure),
  which can be loose.
- The `certify_head` path uses a **head-input file** extracted by an untrusted script; the extractor
  now includes attention projection biases and LayerNorm metadata, but the Lean-side computation
  still ignores the shared attention output bias.
- The `certify_head_model` path derives head inputs from the model binary in Lean, includes
  attention projection biases and LayerNorm metadata, and derives `prev`/active from the stored
  token sequence by default, but still ignores the shared attention output bias. It currently
  requires `head_dim` to be a perfect square to represent the scale as an exact rational.
- The `certify_head_model_auto` path derives the logit-diff direction from the stored prompt
  tokens using a heuristic; use explicit direction tokens for fixed claims.
- The certification does not yet prove end-to-end behavioral induction claims. For
  `certify_head_model` with `period? = none`, `prev` is derived from tokens and is the maximal
  prior match, but other inputs (head-input files or explicit periods) still rely on supplied
  `prev` maps. The chosen direction still assumes the unembedding columns encode token logits.
- There is now a sound interval-composition lemma that combines head logit-diff bounds with
  head/output intervals via subtraction, but it does not model how head outputs propagate
  through subsequent LN/MLP blocks (so tight end-to-end claims remain open).
- Performance: exact head-input recomputation in Lean can be slow for nontrivial sequence lengths.
- There is no bridge theorem connecting certificate validity to a full circuit/model semantics
  statement (for example, a formal statement about logits under a transformer block stack).

## Remaining work

- Tighten model-derived residual intervals (e.g., use attention-weight certificates or
  score-aware bounds) to avoid vacuity.
- Replace untrusted extraction with a verified parser for model weight slices.
- Prove or verify that `prev` and `direction` are derived from token-level semantics.
- Add a formal bridge from certificates to circuit semantics and (eventually) to end-to-end
  transformer claims.
- Improve performance for the exact head-input path without weakening soundness.
