-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Algebra.BigOperators.Field
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Analysis.Complex.Exponential
public import Mathlib.Data.Finset.Card

/-!
Real-valued softmax utilities and margin-based bounds.

These lemmas provide the analytical bridge from score gaps to softmax weight
upper bounds.
-/

public section

namespace Nfp

namespace Circuit

open scoped BigOperators
noncomputable section

variable {seq : Nat}

/-- Real softmax over a finite score vector. -/
def softmax (scores : Fin seq → Real) (k : Fin seq) : Real :=
  Real.exp (scores k) / ∑ j, Real.exp (scores j)

/--
Real softmax over a finite score vector, restricted to an allowed key set.

Keys outside `allow` receive weight `0`.
-/
def softmaxMasked (scores : Fin seq → Real) (allow : Fin seq → Prop) (k : Fin seq) : Real := by
  classical
  exact if h : allow k then
    Real.exp (scores k) /
      (Finset.univ.filter allow).sum (fun j => Real.exp (scores j))
  else
    0

private lemma softmax_denom_pos [NeZero seq] (scores : Fin seq → Real) :
    0 < ∑ j, Real.exp (scores j) := by
  classical
  have hnonempty : (Finset.univ : Finset (Fin seq)).Nonempty := by
    refine ⟨⟨0, ?_⟩, by simp⟩
    exact Nat.pos_of_ne_zero (NeZero.ne seq)
  exact Finset.sum_pos (fun _ _ => Real.exp_pos _) hnonempty

private lemma softmaxMasked_denom_pos [NeZero seq] (scores : Fin seq → Real)
    (allow : Fin seq → Prop) [DecidablePred allow] (hallow : ∃ k, allow k) :
    0 < (Finset.univ.filter allow).sum (fun j => Real.exp (scores j)) := by
  classical
  rcases hallow with ⟨k, hk⟩
  have hnonempty : (Finset.univ.filter allow : Finset (Fin seq)).Nonempty := by
    refine ⟨k, ?_⟩
    simp [hk]
  exact Finset.sum_pos (fun _ _ => Real.exp_pos _) hnonempty

lemma softmax_nonneg [NeZero seq] (scores : Fin seq → Real) (k : Fin seq) :
    0 ≤ softmax scores k := by
  have hdenom : 0 < ∑ j, Real.exp (scores j) := softmax_denom_pos scores
  exact (div_nonneg (Real.exp_pos _).le (le_of_lt hdenom))

lemma softmaxMasked_nonneg [NeZero seq] (scores : Fin seq → Real)
    (allow : Fin seq → Prop) (k : Fin seq) :
    0 ≤ softmaxMasked scores allow k := by
  classical
  by_cases h : allow k
  · have hdenom :
        0 < (Finset.univ.filter allow).sum (fun j => Real.exp (scores j)) :=
      softmaxMasked_denom_pos (scores := scores) (allow := allow) ⟨k, h⟩
    have hdiv :
        0 ≤
          Real.exp (scores k) /
            (Finset.univ.filter allow).sum (fun j => Real.exp (scores j)) :=
      div_nonneg (Real.exp_pos _).le (le_of_lt hdenom)
    simpa [softmaxMasked, h] using hdiv
  · simp [softmaxMasked, h]

lemma softmaxMasked_eq_zero_of_not_allow (scores : Fin seq → Real)
    (allow : Fin seq → Prop) (k : Fin seq) (h : ¬ allow k) :
    softmaxMasked scores allow k = 0 := by
  classical
  simp [softmaxMasked, h]

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

lemma softmaxMasked_sum_one [NeZero seq] (scores : Fin seq → Real)
    (allow : Fin seq → Prop) (hallow : ∃ k, allow k) :
    (∑ k, softmaxMasked scores allow k) = 1 := by
  classical
  let denom : Real :=
    (Finset.univ.filter allow).sum (fun j => Real.exp (scores j))
  have hdenom : denom ≠ 0 :=
    ne_of_gt (softmaxMasked_denom_pos (scores := scores) (allow := allow) hallow)
  have hsum :
      ∑ k, softmaxMasked scores allow k =
        (Finset.univ : Finset (Fin seq)).sum
          (fun k => if allow k then Real.exp (scores k) / denom else 0) := by
    simp [softmaxMasked, denom]
  calc
    ∑ k, softmaxMasked scores allow k
        =
        (Finset.univ : Finset (Fin seq)).sum
          (fun k => if allow k then Real.exp (scores k) / denom else 0) := hsum
    _ =
        (Finset.univ.filter allow).sum (fun k => Real.exp (scores k) / denom) := by
          simpa using
            (Finset.sum_filter (s := (Finset.univ : Finset (Fin seq))) (p := allow)
              (f := fun k => Real.exp (scores k) / denom)).symm
    _ =
        (Finset.univ.filter allow).sum (fun k => Real.exp (scores k)) / denom := by
          simpa using
            (Finset.sum_div (Finset.univ.filter allow) (fun k => Real.exp (scores k)) denom).symm
    _ = 1 := by
          simp [denom, hdenom]

/-- Real-valued row-stochastic weights with explicit nonnegativity and row-sum proofs.
    Kept separate from `ProbVec` because softmax outputs `Real` rather than `NNReal`. -/
structure SoftmaxWeights (seq : Nat) [NeZero seq] where
  /-- Weight assigned to each query/key pair. -/
  weights : Fin seq → Fin seq → Real
  /-- All weights are nonnegative. -/
  nonneg : ∀ q k, 0 ≤ weights q k
  /-- Each row sums to one. -/
  sum_one : ∀ q, (∑ k, weights q k) = 1

/-- Package softmax weights with row-stochastic proofs. -/
def softmaxWeights [NeZero seq] (scores : Fin seq → Fin seq → Real) :
    SoftmaxWeights seq :=
  { weights := fun q k => softmax (scores q) k
    nonneg := by
      intro q k
      simpa using softmax_nonneg (scores := scores q) k
    sum_one := by
      intro q
      simpa using softmax_sum_one (scores := scores q) }

/-- Definitional unfolding for `softmaxWeights.weights`. -/
theorem softmaxWeights_weights [NeZero seq] (scores : Fin seq → Fin seq → Real) :
    (softmaxWeights scores).weights = fun q k => softmax (scores q) k := by
  rfl

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

lemma softmaxMasked_le_one [NeZero seq] (scores : Fin seq → Real)
    (allow : Fin seq → Prop) (k : Fin seq) :
    softmaxMasked scores allow k ≤ 1 := by
  classical
  by_cases h : allow k
  · have hdenom_pos :
        0 < (Finset.univ.filter allow).sum (fun j => Real.exp (scores j)) :=
      softmaxMasked_denom_pos (scores := scores) (allow := allow) ⟨k, h⟩
    have hnum_le :
        Real.exp (scores k) ≤
          (Finset.univ.filter allow).sum (fun j => Real.exp (scores j)) := by
      have hnonneg :
          ∀ j ∈ (Finset.univ.filter allow : Finset (Fin seq)), 0 ≤ Real.exp (scores j) :=
        fun _ _ => (Real.exp_pos _).le
      have hk : k ∈ (Finset.univ.filter allow : Finset (Fin seq)) := by
        simp [h]
      simpa using (Finset.single_le_sum hnonneg hk)
    have hdiv := (div_le_one hdenom_pos).2 hnum_le
    simpa [softmaxMasked, h] using hdiv
  · simp [softmaxMasked, h]

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

lemma softmaxMasked_other_le_exp_neg [NeZero seq] (scores : Fin seq → Real)
    (allow : Fin seq → Prop) {prev k : Fin seq} {m : Real}
    (hprev : allow prev) (hmargin : scores k + m ≤ scores prev) :
    softmaxMasked scores allow k ≤ Real.exp (-m) := by
  classical
  by_cases hk : allow k
  · let denom : Real :=
      (Finset.univ.filter allow).sum (fun j => Real.exp (scores j))
    have hdenom_pos : 0 < denom :=
      softmaxMasked_denom_pos (scores := scores) (allow := allow) ⟨prev, hprev⟩
    have hdenom_ge : Real.exp (scores prev) ≤ denom := by
      have hnonneg :
          ∀ j ∈ (Finset.univ.filter allow : Finset (Fin seq)), 0 ≤ Real.exp (scores j) :=
        fun _ _ => (Real.exp_pos _).le
      have hprev' : prev ∈ (Finset.univ.filter allow : Finset (Fin seq)) := by
        simp [hprev]
      simpa [denom] using (Finset.single_le_sum hnonneg hprev')
    have hinv : 1 / denom ≤ 1 / Real.exp (scores prev) :=
      one_div_le_one_div_of_le (Real.exp_pos _) hdenom_ge
    have hmul :=
      mul_le_mul_of_nonneg_left hinv (Real.exp_pos (scores k)).le
    have hratio :
        Real.exp (scores k) / Real.exp (scores prev) =
          Real.exp (scores k - scores prev) := by
      symm
      exact Real.exp_sub (scores k) (scores prev)
    have hk' : scores k ≤ scores prev - m := (le_sub_iff_add_le).2 hmargin
    have hdiff : scores k - scores prev ≤ -m := by
      have hsub := sub_le_sub_right hk' (scores prev)
      simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using hsub
    have hle : Real.exp (scores k - scores prev) ≤ Real.exp (-m) :=
      Real.exp_le_exp.mpr hdiff
    have hsoft :
        Real.exp (scores k) / denom ≤ Real.exp (scores k) / Real.exp (scores prev) := by
      simpa [denom, div_eq_mul_inv] using hmul
    calc
      softmaxMasked scores allow k
          = Real.exp (scores k) / denom := by
              simp [softmaxMasked, hk, denom]
      _ ≤ Real.exp (scores k) / Real.exp (scores prev) := hsoft
      _ = Real.exp (scores k - scores prev) := hratio
      _ ≤ Real.exp (-m) := hle
  · have hpos : 0 ≤ Real.exp (-m) := (Real.exp_pos _).le
    have hzero : softmaxMasked scores allow k = 0 := by
      simp [softmaxMasked, hk]
    simpa [hzero] using hpos

lemma softmax_other_le_inv_one_add [NeZero seq] (scores : Fin seq → Real)
    {prev k : Fin seq} {m : Real} (hm : 0 ≤ m) (hmargin : scores k + m ≤ scores prev) :
    softmax scores k ≤ 1 / (1 + m) :=
  (softmax_other_le_exp_neg (scores := scores) hmargin).trans (exp_neg_le_inv_one_add hm)

lemma softmaxMasked_other_le_inv_one_add [NeZero seq] (scores : Fin seq → Real)
    (allow : Fin seq → Prop) {prev k : Fin seq} {m : Real}
    (hm : 0 ≤ m) (hprev : allow prev) (hmargin : scores k + m ≤ scores prev) :
    softmaxMasked scores allow k ≤ 1 / (1 + m) :=
  (softmaxMasked_other_le_exp_neg (scores := scores) (allow := allow) hprev hmargin).trans
    (exp_neg_le_inv_one_add hm)

end

end Circuit

end Nfp
