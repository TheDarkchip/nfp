-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Unbundled.Rat
import Init.Data.Array.Lemmas

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
  have hArray : cands.foldl min base ≤ base := by
    refine Array.foldl_induction (as := cands)
        (motive := fun _ acc => acc ≤ base) (init := base) (f := fun acc x => min acc x) ?h0 ?hf
    · exact le_rfl
    · intro i acc hacc
      exact le_trans (min_le_left _ _) hacc
  simpa [ubBest] using hArray

/-- Best lower bound among candidates (never worse than `base`). -/
def lbBest (base : Rat) (cands : Array Rat) : Rat :=
  cands.foldl max base

theorem lbBest_def (base : Rat) (cands : Array Rat) :
    lbBest base cands = cands.foldl max base := rfl

/-- `lbBest` never undercuts its baseline lower bound. -/
theorem lbBest_ge_base (base : Rat) (cands : Array Rat) : base ≤ lbBest base cands := by
  classical
  have hArray : base ≤ cands.foldl max base := by
    refine Array.foldl_induction (as := cands)
        (motive := fun _ acc => base ≤ acc) (init := base) (f := fun acc x => max acc x) ?h0 ?hf
    · exact le_rfl
    · intro i acc hacc
      exact le_trans hacc (le_max_left _ _)
  simpa [lbBest] using hArray

end Nfp.Sound
