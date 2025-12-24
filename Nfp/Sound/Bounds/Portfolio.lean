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

/-- Best lower bound among candidates (never worse than `base`). -/
def lbBest (base : Rat) (cands : Array Rat) : Rat :=
  cands.foldl max base

theorem lbBest_def (base : Rat) (cands : Array Rat) :
    lbBest base cands = cands.foldl max base := rfl

end Nfp.Sound
