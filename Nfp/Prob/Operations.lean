-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Prob.Basic
public import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
Basic constructions on probability vectors.
-/

@[expose] public section

open scoped BigOperators

namespace Nfp
namespace ProbVec

universe u

variable {ι : Type u} [Fintype ι]

/-- Factor a constant out of a sum. -/
private lemma sum_mul_const (a : Mass) (p : ι → Mass) :
    (∑ i, a * p i) = a * ∑ i, p i := by
  simpa using
    (Finset.mul_sum (a := a) (s := (Finset.univ : Finset ι))
      (f := fun i => p i)).symm

/-- The pure distribution at a single point. -/
def pure (i0 : ι) [DecidableEq ι] : ProbVec ι := by
  refine
    { mass := Pi.single i0 (1 : Mass)
      sum_mass := ?_ }
  simp

@[simp] theorem mass_pure (i0 i : ι) [DecidableEq ι] :
    (pure i0).mass i = if i = i0 then 1 else 0 := by
  by_cases h : i = i0 <;> simp [pure, Pi.single, h]

/-- Convex combination of two probability vectors with weights that sum to one. -/
def mix (a b : Mass) (h : a + b = 1) (p q : ProbVec ι) : ProbVec ι :=
  { mass := fun i => a * p.mass i + b * q.mass i
    sum_mass := by
      classical
      simp [Finset.sum_add_distrib, sum_mul_const, h] }

end ProbVec
end Nfp

end
