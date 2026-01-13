# Induction Head Certification Audit

Goal: assess whether the current Lean proofs justify the claim that we can certify
induction heads, and spell out the scope and limitations of that claim.

## Formal proof chain (Lean)

- `buildInductionCertFromHeadCoreWith?` returns a certificate under explicit guards
  (`lnEps > 0`, `sqrtLower lnEps > 0`, `dModel ≠ 0`, `active.Nonempty`), so the
  computation is only claimed when these preconditions hold
  (`Nfp/Sound/Induction/Core.lean`).
- `buildInductionHeadInputs_def` shows the model-derived head inputs are
  definitional: `prev`/`active` are computed from tokens (or a fixed period),
  and the `direction` vector is the unembedding-column difference for the
  provided target/negative token ids (`Nfp/IO/NfptPure.lean`).
- `buildInductionHeadInputs_prev_spec_of_active` and
  `prevOfTokens_spec_of_active` prove that when `period? = none`,
  every active query has a maximal prior matching token in `prev`
  (`Nfp/IO/NfptPure.lean`, `Nfp/Model/InductionPrompt.lean`).
- `buildInductionCertFromHeadWith?` wraps the core computation and returns
  a proof-carrying certificate `⟨c, InductionHeadCertSound inputs c⟩`
  (`Nfp/Sound/Induction/HeadOutput.lean`).
- `buildInductionCertFromHeadCoreWith?_sound` proves that any returned certificate
  satisfies `InductionHeadCertSound`, i.e. the softmax-margin bounds, one-hot
  bounds, and value-interval bounds that define the head-level certificate
  (`Nfp/Sound/Induction/CoreSound.lean`).
- `buildInductionLogitLowerBoundFromHead?` and
  `buildInductionLogitLowerBoundNonvacuous?` lift the head certificate to a
  logit-diff lower bound; the key lemma `logitDiffLowerBoundFromCert_le` shows
  the bound is sound on active queries (`Nfp/Sound/Induction/LogitDiff.lean`).
- `logitDiffLowerBound_end_to_end_gpt2` combines head logit-diff bounds, head
  output intervals, and GPT-2 stack output intervals to give a direction lower
  bound on `transformerStackFinalReal`
  (`Nfp/Sound/Induction/EndToEnd.lean`, `Nfp/Sound/Bounds/Transformer.lean`).

## Mechanistic mapping (Transformer Circuits)

The mechanistic induction-head story is a QK/OV decomposition:
- QK: identify a matching prior token (prefix-matching attention).
- OV: write the continuation token (or logit-diff direction) into the residual stream.

The certificate aligns to that decomposition:
- The softmax-margin bounds constrain the QK pattern so that attention to the
  chosen `prev` index dominates other keys (mechanistic “prefix match”).
- The value-interval bounds and logit-diff lower bound constrain the OV path in
  the chosen direction, so the head’s contribution increases the target logit
  relative to the negative logit.

This is direct mechanistic evidence in the Transformer Circuits sense: it ties
parameters (Q/K/V/O + LayerNorm) to certified bounds on attention and value
contributions, but only for the specific inputs and direction supplied.

Sources referenced for the mechanistic framing:
- `transformer-circuits-framework.md` (QK/OV decomposition).
- `induction-heads.md` (induction head behavior definition).
- `foundations.md` (reverse-engineering framing and feature decomposition).

## Preconditions and scope limits

These proofs are sufficient for a **conditional** certification claim:
if the inputs are correct and the builder returns a certificate, then the
head-level bounds hold. They are **not** sufficient for a global claim that a
head “is an induction head” without additional assumptions.

Key assumptions and limitations:
- For `certify_head_model` with `period? = none`, `prev`/`active` are derived
  from tokens and `prev` is the maximal prior match. For head-input files or
  when `period?` is set explicitly, `prev` remains a user-supplied input.
- The certificate proves a logit-diff bound along the supplied `direction`
  vector. For model-derived inputs, this vector is the target-minus-negative
  unembedding column difference, but we still assume that the unembedding
  matrix represents the model’s logit map.
- The active set is user-supplied and can be strict; bounds only hold for
  `q ∈ active`, not all positions.
- There is now a formal bridge from head logit-diff bounds plus residual interval
  bounds to a direction lower bound on `headOutput + residual`, but full
  end-to-end model logits still require verified residual bounds through the
  rest of the stack.
  We now have a theorem packaging GPT-2 residual interval bounds derived from
  model slices into a sound `ResidualIntervalCert`, but it is not yet connected
  to the head-level logit-diff contribution inside the full stack.
  A new lemma composes head logit-diff bounds with *both* head-output intervals
  and downstream output intervals, yielding a sound lower bound on the direction
  dot of the downstream output (via interval subtraction), and we now instantiate
  this for GPT-2 stack outputs via `logitDiffLowerBound_end_to_end_gpt2`.

## Conclusion

Yes—**within the formal scope** of the current definitions, the proofs are
enough to claim that we can certify induction-head behavior at the head level:
they certify attention to a specified `prev` index and a logit-diff lower bound
along a specified direction. We now have a bridge that composes those bounds
with residual interval bounds to certify `headOutput + residual`, but we still
need a proof that the inputs correspond to the behavioral induction-head
definition on actual sequences and that residual bounds are derived from full
model semantics.

## Next steps

- Formalize the relationship between `directionSpec` and the logit-diff vector
  derived from unembedding (so the certified direction matches token-level claims).
- Add a proof or verified derivation that the `prev` mapping corresponds to the
  induction pattern for a given prompt sequence.
- Extend the bridge to full transformer stacks by deriving residual interval
  bounds from verified layer/block semantics.
