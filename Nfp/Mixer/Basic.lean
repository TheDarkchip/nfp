-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Prob.Basic

/-!
Row-stochastic mixers.
-/

public section

open scoped BigOperators

namespace Nfp

universe u

/-- A row-stochastic mixer from `ι` to `κ`. -/
structure Mixer (ι κ : Type u) [Fintype ι] [Fintype κ] where
  /-- Nonnegative weights for each source/target pair. -/
  weight : ι → κ → Mass
  /-- Each row is a probability vector. -/
  row_sum : ∀ i, (∑ k, weight i k) = 1

attribute [simp] Mixer.row_sum

namespace Mixer

variable {ι κ : Type u} [Fintype ι] [Fintype κ]

instance : CoeFun (Mixer ι κ) (fun _ => ι → κ → Mass) := ⟨Mixer.weight⟩

/-- The row of a mixer as a probability vector. -/
def row (M : Mixer ι κ) (i : ι) : ProbVec κ :=
  { mass := fun k => M.weight i k
    sum_mass := M.row_sum i }

end Mixer

end Nfp

end
