import Mathlib.Data.NNReal.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Finset.Basic

/-
Basic probability-friendly definitions used across the NFP development.
We work with finite types and nonnegative reals `NNReal` from mathlib.
-/

namespace Nfp

open scoped BigOperators
open Finset

/-- A probability vector on a finite type `ι` is a nonnegative function summing to 1. -/
structure ProbVec (ι : Type*) [Fintype ι] where
  mass : ι → NNReal
  norm_one : (∑ i, mass i) = (1 : NNReal)

namespace ProbVec

variable {ι : Type*} [Fintype ι]

@[simp] theorem sum_mass (p : ProbVec ι) : (∑ i, p.mass i) = 1 := p.norm_one

@[ext]
theorem ext {p q : ProbVec ι} (h : ∀ i, p.mass i = q.mass i) : p = q := by
  cases p; cases q; simp only [mk.injEq]; funext i; exact h i

theorem mass_le_one (p : ProbVec ι) (i : ι) : p.mass i ≤ 1 := by
  have h := p.sum_mass
  calc p.mass i ≤ ∑ j, p.mass j := Finset.single_le_sum (by simp) (Finset.mem_univ i)
    _ = 1 := h

end ProbVec

end Nfp
