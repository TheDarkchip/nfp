-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Model.InductionPrompt

/-!
Circuit-level induction prompt specifications.

These wrappers name the *shifted* prev/active maps that correspond to the
canonical induction-head circuit (previous-token head feeding induction head).
They are definitional aliases over `InductionPrevSpec*Shift`, but make the
intended mechanistic interpretation explicit for later lemmas.
-/

public section

namespace Nfp

namespace Model

/--
Circuit-level shifted-prev spec for periodic prompts.

This matches the canonical induction circuit: a previous-token head shifts
the match by one position, so the induction head attends to `q - period + 1`.
-/
structure InductionCircuitSpecPeriodShift {seq dModel dHead : Nat}
    (period : Nat) (inputs : InductionHeadInputs seq dModel dHead) : Prop where
  /-- The underlying shifted-period prev/active spec. -/
  prev_spec : InductionPrevSpecPeriodShift (seq := seq) period inputs

/--
Circuit-level shifted-prev spec for token-based prompts.

This corresponds to the canonical induction circuit with a previous-token head,
so the induction head uses the shifted-token map.
-/
structure InductionCircuitSpecTokensShift {seq dModel dHead : Nat}
    (tokens : Fin seq → Nat) (inputs : InductionHeadInputs seq dModel dHead) : Prop where
  /-- The underlying shifted-token prev/active spec. -/
  prev_spec : InductionPrevSpecTokensShift (seq := seq) tokens inputs

/--
Lift a shifted-period circuit spec to the token-based spec for diagnostic prompts.
-/
theorem InductionCircuitSpecTokensShift_of_diag {dModel dHead : Nat} {period : Nat}
    (tokens : Fin (2 * period) → Nat)
    (inputs : InductionHeadInputs (2 * period) dModel dHead)
    (hdiag : InductionDiagnosticTokens period tokens)
    (hper : 0 < period)
    (hspec : InductionCircuitSpecPeriodShift (seq := 2 * period) period inputs) :
    InductionCircuitSpecTokensShift (seq := 2 * period) tokens inputs := by
  refine ⟨?_,⟩
  exact
    InductionPrevSpecTokensShift_of_diag (tokens := tokens) hdiag hper hspec.prev_spec

end Model

end Nfp
