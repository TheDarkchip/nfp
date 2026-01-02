-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Fintype.BigOperators
import Nfp.Mixer.Basic
import Nfp.System.Dag

/-!
Local mixing systems on finite DAGs.
-/

open scoped BigOperators

namespace Nfp

universe u

/-- A local mixing system on a DAG.
`weight i j` is the contribution from `j` into `i`. -/
structure LocalSystem (ι : Type u) [Fintype ι] where
  /-- The underlying DAG describing allowed dependencies. -/
  dag : Dag ι
  /-- Mixing weights for each target/source pair. -/
  weight : ι → ι → Mass
  /-- Weights vanish off the edge relation. -/
  support : ∀ i j, ¬ dag.rel j i → weight i j = 0
  /-- Each row is a probability vector. -/
  row_sum : ∀ i, (∑ j, weight i j) = 1

attribute [simp] LocalSystem.row_sum

namespace LocalSystem

variable {ι : Type u} [Fintype ι]

/-- View a local system as a global mixer. -/
def toMixer (L : LocalSystem ι) : Mixer ι ι :=
  { weight := L.weight
    row_sum := L.row_sum }

/-- Off-edge weights are zero. -/
theorem weight_eq_zero_of_not_parent (L : LocalSystem ι) {i j : ι} (h : ¬ L.dag.rel j i) :
    L.weight i j = 0 :=
  L.support i j h

end LocalSystem

end Nfp
