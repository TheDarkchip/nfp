# Induction Head Certification Audit

Goal: assess whether the current Lean proofs justify the claim that we can certify
induction heads, and spell out the scope and limitations of that claim.

## Formal proof chain (Lean)

- Explicit induction-head certificates are parsed from text in
  `Nfp/IO/InductionHead/Cert.lean` (sequence indices are 0-based in the payload). The parser
  supports `kind onehot-approx` (proxy bounds) and `kind induction-aligned` (periodic prompt).
- `checkInductionHeadCert` and `checkInductionHeadCert_sound` show that a
  passing certificate satisfies `InductionHeadCertBounds`
  (`Nfp/Circuit/Cert/InductionHead.lean`).
- `logitDiffLowerBoundAt` plus `logitDiffLowerBoundAt_le` give a certified lower
  bound on the logit-diff contribution derived from the certificate’s values
  (`Nfp/Circuit/Cert/LogitDiff.lean`).
- `headLogitDiff_eq_direction_dot_headOutput` connects the logit-diff definition
  to head-output semantics (`Nfp/Sound/Induction/LogitDiff.lean`).

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

## Literature alignment

We follow the standard induction-head diagnostic setup from the literature:
repeated-token sequences (a pattern repeated twice). The diagnostic script
`scripts/diagnose_induction_heads.py` now aligns with TransformerLens detection,
using the induction detection pattern (duplicate-token mask shifted right) and
the `"mul"` scoring metric (fraction of attention on the detection pattern). On
repeated patterns, this corresponds to the shifted stripe target (`q -> q -
period + 1`) rather than the unshifted stripe (`q -> q - period`). The
certificate generator still uses repeated patterns for its inputs. For `kind
induction-aligned`, Lean additionally checks that `active`/`prev` match the
declared `period` and evaluates stripe-mean (prefix matching score) and the
copying score on the full second repeat.

## Preconditions and scope limits

These proofs are sufficient for a **conditional** certification claim:
if the explicit certificate passes the checker, then the head-level bounds hold.
They are **not** sufficient for a global claim that a head “is an induction head”
without additional assumptions.

Key assumptions and limitations:
- `prev`, `active`, and `direction` are user-supplied or produced by untrusted
  scripts. For `kind onehot-approx`, Lean does not (yet) verify their derivation
  from token-level semantics; for `kind induction-aligned`, it checks that
  `prev`/`active` match the declared periodic prompt and applies prefix-matching
  and copying metrics derived from the certificate payload.
- The active set can be strict; bounds only hold for `q ∈ active`, not all positions.
- The direction metadata assumes the unembedding columns encode the model’s logit map.

Optional safeguard:
- If a token list is supplied to the CLI (`--tokens`), the checker verifies
  previous-occurrence semantics for `kind onehot-approx`, and periodicity for
  `kind induction-aligned`.

## Conclusion

Yes—**within the formal scope** of the current definitions, the proofs are
enough to claim that we can certify induction-head behavior at the head level:
they certify attention to a specified `prev` index and a logit-diff lower bound
along a specified direction, conditional on an explicit certificate.

## Next steps

- Add a verified extraction pipeline from model weights to explicit certificates.
- Prove that `prev`, `active`, and `direction` correspond to token-level semantics.

## References

- Elhage et al., “A Mathematical Framework for Transformer Circuits.”
  Link: `https://transformer-circuits.pub/2021/framework/index.html`
- Olsson et al., “In-context Learning and Induction Heads.”
  Link: `https://arxiv.org/abs/2209.11895`
