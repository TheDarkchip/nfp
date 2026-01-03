-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Data.Finset.Card

/-!
Real-valued softmax utilities and margin-based bounds.

These lemmas provide the analytical bridge from score gaps to softmax weight
upper bounds.
-/

namespace Nfp

namespace Circuit

open scoped BigOperators

noncomputable section

variable {seq : Nat}

/-- Real softmax over a finite score vector. -/
def softmax (scores : Fin seq → Real) (k : Fin seq) : Real :=
  Real.exp (scores k) / ∑ j, Real.exp (scores j)

private lemma softmax_denom_pos [NeZero seq] (scores : Fin seq → Real) :
    0 < ∑ j, Real.exp (scores j) := by
  classical
  have hnonempty : (Finset.univ : Finset (Fin seq)).Nonempty := by
    refine ⟨⟨0, ?_⟩, by simp⟩
    exact Nat.pos_of_ne_zero (NeZero.ne seq)
  exact Finset.sum_pos (fun _ _ => Real.exp_pos _) hnonempty

lemma softmax_nonneg [NeZero seq] (scores : Fin seq → Real) (k : Fin seq) :
    0 ≤ softmax scores k := by
  have hdenom : 0 < ∑ j, Real.exp (scores j) := softmax_denom_pos scores
  exact (div_nonneg (Real.exp_pos _).le (le_of_lt hdenom))

lemma softmax_sum_one [NeZero seq] (scores : Fin seq → Real) :
    (∑ k, softmax scores k) = 1 := by
  classical
  have hdenom : (∑ j, Real.exp (scores j)) ≠ 0 :=
    ne_of_gt (softmax_denom_pos scores)
  have hsum :
      (∑ k, Real.exp (scores k) / ∑ j, Real.exp (scores j)) =
        (∑ k, Real.exp (scores k)) / ∑ j, Real.exp (scores j) := by
    simpa using
      (Finset.sum_div (Finset.univ) (fun k => Real.exp (scores k))
        (∑ j, Real.exp (scores j))).symm
  calc
    ∑ k, softmax scores k
        = ∑ k, Real.exp (scores k) / ∑ j, Real.exp (scores j) := by
            simp [softmax]
    _ = (∑ k, Real.exp (scores k)) / ∑ j, Real.exp (scores j) := hsum
    _ = 1 := by
        simp [hdenom]

lemma softmax_le_one [NeZero seq] (scores : Fin seq → Real) (k : Fin seq) :
    softmax scores k ≤ 1 := by
  classical
  have hdenom_pos : 0 < ∑ j, Real.exp (scores j) := softmax_denom_pos scores
  have hnum_le : Real.exp (scores k) ≤ ∑ j, Real.exp (scores j) := by
    have hnonneg : ∀ j ∈ (Finset.univ : Finset (Fin seq)), 0 ≤ Real.exp (scores j) :=
      fun _ _ => (Real.exp_pos _).le
    simpa using (Finset.single_le_sum hnonneg (by simp))
  have hdiv := (div_le_one hdenom_pos).2 hnum_le
  simpa [softmax] using hdiv

lemma exp_neg_le_inv_one_add {m : Real} (hm : 0 ≤ m) :
    Real.exp (-m) ≤ 1 / (1 + m) := by
  have hpos : 0 < 1 + m := add_pos_of_pos_of_nonneg zero_lt_one hm
  have hle : 1 + m ≤ Real.exp m := by
    simpa [add_comm] using (Real.add_one_le_exp m)
  have hdiv : 1 / Real.exp m ≤ 1 / (1 + m) :=
    one_div_le_one_div_of_le hpos hle
  simpa [Real.exp_neg] using hdiv

lemma softmax_other_le_exp_neg [NeZero seq] (scores : Fin seq → Real)
    {prev k : Fin seq} {m : Real} (hmargin : scores k + m ≤ scores prev) :
    softmax scores k ≤ Real.exp (-m) := by
  classical
  let denom : Real := ∑ j, Real.exp (scores j)
  have hdenom_pos : 0 < denom := softmax_denom_pos scores
  have hdenom_ge : Real.exp (scores prev) ≤ denom := by
    have hnonneg : ∀ j ∈ (Finset.univ : Finset (Fin seq)), 0 ≤ Real.exp (scores j) :=
      fun _ _ => (Real.exp_pos _).le
    simpa [denom] using
      (Finset.single_le_sum hnonneg (by simp : prev ∈ (Finset.univ : Finset (Fin seq))))
  have hinv : 1 / denom ≤ 1 / Real.exp (scores prev) :=
    one_div_le_one_div_of_le (Real.exp_pos _) hdenom_ge
  have hmul :=
    mul_le_mul_of_nonneg_left hinv (Real.exp_pos (scores k)).le
  have hratio :
      Real.exp (scores k) / Real.exp (scores prev) =
        Real.exp (scores k - scores prev) := by
    symm
    exact Real.exp_sub (scores k) (scores prev)
  have hk : scores k ≤ scores prev - m := (le_sub_iff_add_le).2 hmargin
  have hdiff : scores k - scores prev ≤ -m := by
    have hsub := sub_le_sub_right hk (scores prev)
    simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using hsub
  have hle : Real.exp (scores k - scores prev) ≤ Real.exp (-m) :=
    Real.exp_le_exp.mpr hdiff
  have hsoft :
      Real.exp (scores k) / denom ≤ Real.exp (scores k) / Real.exp (scores prev) := by
    simpa [denom, div_eq_mul_inv] using hmul
  calc
    softmax scores k
        = Real.exp (scores k) / denom := by
            simp [softmax, denom]
    _ ≤ Real.exp (scores k) / Real.exp (scores prev) := hsoft
    _ = Real.exp (scores k - scores prev) := hratio
    _ ≤ Real.exp (-m) := hle

lemma softmax_other_le_inv_one_add [NeZero seq] (scores : Fin seq → Real)
    {prev k : Fin seq} {m : Real} (hm : 0 ≤ m) (hmargin : scores k + m ≤ scores prev) :
    softmax scores k ≤ 1 / (1 + m) :=
  (softmax_other_le_exp_neg (scores := scores) hmargin).trans (exp_neg_le_inv_one_add hm)

end

end Circuit

end Nfp
