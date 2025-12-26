-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Unbundled.Rat

namespace Nfp.Sound

/-!
# Portfolio bounds

Combinators for selecting the best bound among sound candidates.
-/

/-- Best upper bound among candidates (never worse than `base`). -/
def ubBest (base : Rat) (cands : Array Rat) : Rat :=
  cands.foldl min base

theorem ubBest_def (base : Rat) (cands : Array Rat) :
    ubBest base cands = cands.foldl min base := rfl

/-- `ubBest` never exceeds its baseline upper bound. -/
theorem ubBest_le_base (base : Rat) (cands : Array Rat) : ubBest base cands ≤ base := by
  classical
  have hList : cands.toList.foldl min base ≤ base := by
    induction cands.toList generalizing base with
    | nil => simp
    | cons x xs ih =>
        simp only [List.foldl]
        have h := ih (base := min base x)
        exact le_trans h (min_le_left _ _)
  have hArray : cands.foldl min base ≤ base := by
    simpa [Array.foldl_toList] using hList
  simpa [ubBest] using hArray

/-- Best lower bound among candidates (never worse than `base`). -/
def lbBest (base : Rat) (cands : Array Rat) : Rat :=
  cands.foldl max base

theorem lbBest_def (base : Rat) (cands : Array Rat) :
    lbBest base cands = cands.foldl max base := rfl

/-- `lbBest` never undercuts its baseline lower bound. -/
theorem lbBest_ge_base (base : Rat) (cands : Array Rat) : base ≤ lbBest base cands := by
  classical
  have hList : base ≤ cands.toList.foldl max base := by
    induction cands.toList generalizing base with
    | nil => simp
    | cons x xs ih =>
        simp only [List.foldl]
        have h := ih (base := max base x)
        exact le_trans (le_max_left _ _) h
  have hArray : base ≤ cands.foldl max base := by
    simpa [Array.foldl_toList] using hList
  simpa [lbBest] using hArray

end Nfp.Sound
