-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Prob.Basic
import Mathlib.Algebra.BigOperators.Ring.Finset

/-!
Basic constructions on probability vectors.
-/

open scoped BigOperators

namespace Nfp
namespace ProbVec

universe u

variable {ι : Type u} [Fintype ι]

/-- The pure distribution at a single point. -/
def pure (i0 : ι) [DecidableEq ι] : ProbVec ι := by
  refine
    { mass := fun i => if i = i0 then 1 else 0
      sum_mass := ?_ }
  exact (Fintype.sum_ite_eq' (ι := ι) (i := i0) (f := fun _ => (1 : Mass)))

@[simp] theorem mass_pure (i0 i : ι) [DecidableEq ι] :
    (pure i0).mass i = if i = i0 then 1 else 0 := rfl

/-- Convex combination of two probability vectors with weights that sum to one. -/
def mix (a b : Mass) (h : a + b = 1) (p q : ProbVec ι) : ProbVec ι :=
  { mass := fun i => a * p.mass i + b * q.mass i
    sum_mass := by
      classical
      calc
        ∑ i, (a * p.mass i + b * q.mass i)
            = (∑ i, a * p.mass i) + (∑ i, b * q.mass i) := by
                simp [Finset.sum_add_distrib]
        _ = a * ∑ i, p.mass i + b * ∑ i, q.mass i := by
              have ha : (∑ i, a * p.mass i) = a * ∑ i, p.mass i := by
                simpa using
                  (Finset.mul_sum (a := a) (s := (Finset.univ : Finset ι))
                    (f := fun i => p.mass i)).symm
              have hb : (∑ i, b * q.mass i) = b * ∑ i, q.mass i := by
                simpa using
                  (Finset.mul_sum (a := b) (s := (Finset.univ : Finset ι))
                    (f := fun i => q.mass i)).symm
              simp [ha, hb]
        _ = a * 1 + b * 1 := by simp
        _ = 1 := by simp [h] }

end ProbVec
end Nfp
