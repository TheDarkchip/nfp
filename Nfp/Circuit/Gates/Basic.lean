-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Algebra.Ring.Basic
public import Mathlib.Data.Finset.Attach
public import Mathlib.Data.Fintype.BigOperators

/-!
Basic gate combinators for aggregating parent values.
-/

public section

namespace Nfp

namespace Circuit

namespace Gates

universe u v

variable {Node : Type u} {Val : Type v}

/-- Sum of parent values. -/
def sumParents (parents : Finset Node) (rec : ∀ j, j ∈ parents → Val)
    [AddCommMonoid Val] : Val :=
  parents.attach.sum fun j => rec j.1 j.2

/-- Weighted sum of parent values using weights `w`. -/
def weightedSumParents (parents : Finset Node) (w : Node → Val)
    (rec : ∀ j, j ∈ parents → Val) [Semiring Val] : Val :=
  parents.attach.sum fun j => w j.1 * rec j.1 j.2

/-- Affine combination of parent values with weights `w` and bias `b`. -/
def affineParents (parents : Finset Node) (w : Node → Val) (b : Val)
    (rec : ∀ j, j ∈ parents → Val) [Semiring Val] : Val :=
  weightedSumParents parents w rec + b

end Gates

end Circuit

end Nfp
