-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Core
import Mathlib.Data.Fintype.BigOperators

/-!
Probability vectors on finite types.
-/

open scoped BigOperators

namespace Nfp

universe u

/-- A probability vector on a finite type. -/
structure ProbVec (ι : Type u) [Fintype ι] where
  /-- Mass assigned to each point. -/
  mass : ι → Mass
  /-- Total mass is exactly one. -/
  sum_mass : (∑ i, mass i) = 1

attribute [simp] ProbVec.sum_mass

namespace ProbVec

variable {ι : Type u} [Fintype ι]

instance : CoeFun (ProbVec ι) (fun _ => ι → Mass) := ⟨ProbVec.mass⟩

end ProbVec

end Nfp
