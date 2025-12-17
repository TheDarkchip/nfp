-- SPDX-License-Identifier: AGPL-3.0-or-later

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

/-- The Dirac/point-mass probability vector at `i0`. -/
noncomputable def pure (i0 : ι) : ProbVec ι := by
  classical
  refine
    {
      mass := fun i => if i = i0 then 1 else 0
      norm_one := ?_
    }
  simp

@[simp] lemma pure_mass_self (i0 : ι) : (pure (ι := ι) i0).mass i0 = 1 := by
  classical
  simp [pure]

@[simp] lemma pure_mass_ne_self {i0 i : ι} (h : i ≠ i0) : (pure (ι := ι) i0).mass i = 0 := by
  classical
  simp [pure, h]

/-- Convex mixture of probability vectors using coefficient `c ∈ [0,1]`. -/
noncomputable def mix (c : NNReal) (hc : c ≤ 1) (p q : ProbVec ι) : ProbVec ι :=
  {
    mass := fun i => c * p.mass i + (1 - c) * q.mass i
    norm_one := by
      classical
      calc
        (∑ i, (c * p.mass i + (1 - c) * q.mass i))
            = (∑ i, c * p.mass i) + (∑ i, (1 - c) * q.mass i) := by
                simp [Finset.sum_add_distrib]
        _   = c * (∑ i, p.mass i) + (1 - c) * (∑ i, q.mass i) := by
                have hp : (∑ i, c * p.mass i) = c * (∑ i, p.mass i) := by
                  simpa using
                    (Finset.mul_sum (s := (Finset.univ : Finset ι)) (f := fun i : ι => p.mass i)
                      (a := c)).symm
                have hq :
                    (∑ i, (1 - c) * q.mass i) = (1 - c) * (∑ i, q.mass i) := by
                  simpa using
                    (Finset.mul_sum (s := (Finset.univ : Finset ι)) (f := fun i : ι => q.mass i)
                      (a := (1 - c))).symm
                simp [hp, hq]
        _   = c * 1 + (1 - c) * 1 := by
                simp [ProbVec.sum_mass]
        _   = c + (1 - c) := by
                simp
        _   = 1 := by
                simpa using (add_tsub_cancel_of_le hc)
  }

@[simp] lemma mix_mass (c : NNReal) (hc : c ≤ 1) (p q : ProbVec ι) (i : ι) :
    (mix (ι := ι) c hc p q).mass i = c * p.mass i + (1 - c) * q.mass i := rfl

end ProbVec

end Nfp
