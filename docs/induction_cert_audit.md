# Induction Head Certification Audit

Goal: assess whether the current Lean proofs justify the claim that we can certify
induction heads, and spell out the scope and limitations of that claim.

## Formal proof chain (Lean)

- Explicit induction-head certificates are parsed from text in
  `Nfp/IO/InductionHead/Cert.lean`.
- `checkInductionHeadCert` and `checkInductionHeadCert_sound` show that a
  passing certificate satisfies `InductionHeadCertBounds`
  (`Nfp/Circuit/Cert/InductionHead.lean`).
- `logitDiffLowerBoundAt` plus `logitDiffLowerBoundAt_le` give a certified lower
  bound on the logit-diff contribution derived from the certificate’s values
  (`Nfp/Circuit/Cert/LogitDiff.lean`).
- `headLogitDiff_eq_direction_dot_headOutput`, `logitDiffLowerBound_with_residual`,
  and `logitDiffLowerBound_with_output_intervals` compose head-level logit-diff
  bounds with output intervals (`Nfp/Sound/Induction/LogitDiff.lean`).
- `logitDiffLowerBound_end_to_end_gpt2` instantiates the composition for GPT-2
  stack outputs (`Nfp/Sound/Induction/EndToEnd.lean`).

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

## Preconditions and scope limits

These proofs are sufficient for a **conditional** certification claim:
if the explicit certificate passes the checker, then the head-level bounds hold.
They are **not** sufficient for a global claim that a head “is an induction head”
without additional assumptions.

Key assumptions and limitations:
- `prev`, `active`, and `direction` are user-supplied or produced by untrusted
  scripts; Lean does not (yet) verify their derivation from token-level semantics.
- The active set can be strict; bounds only hold for `q ∈ active`, not all positions.
- The direction metadata assumes the unembedding columns encode the model’s logit map.
- End-to-end claims rely on external residual/downstream interval certificates; the
  current checker only verifies those certificates once provided.

## Conclusion

Yes—**within the formal scope** of the current definitions, the proofs are
enough to claim that we can certify induction-head behavior at the head level:
they certify attention to a specified `prev` index and a logit-diff lower bound
along a specified direction, conditional on an explicit certificate.

## Next steps

- Add a verified extraction pipeline from model weights to explicit certificates.
- Prove that `prev`, `active`, and `direction` correspond to token-level semantics.
- Tighten residual/downstream interval bounds to strengthen end-to-end claims.
