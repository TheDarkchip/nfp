-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Ring.Basic
import Mathlib.Data.Finset.Attach
import Mathlib.Data.Fintype.BigOperators

/-!
Basic gate combinators for aggregating parent values.
-/

namespace Nfp

namespace Circuit

namespace Gates

universe u v

variable {ι : Type u} {α : Type v}

/-- Sum of parent values. -/
def sumParents (parents : Finset ι) (rec : ∀ j, j ∈ parents → α)
    [AddCommMonoid α] : α :=
  parents.attach.sum fun j => rec j.1 j.2

/-- Weighted sum of parent values using weights `w`. -/
def weightedSumParents (parents : Finset ι) (w : ι → α)
    (rec : ∀ j, j ∈ parents → α) [Semiring α] : α :=
  parents.attach.sum fun j => w j.1 * rec j.1 j.2

/-- Affine combination of parent values with weights `w` and bias `b`. -/
def affineParents (parents : Finset ι) (w : ι → α) (b : α)
    (rec : ∀ j, j ∈ parents → α) [Semiring α] : α :=
  weightedSumParents parents w rec + b

end Gates

end Circuit

end Nfp
