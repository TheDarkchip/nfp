-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.Order.Ring.Basic
import Mathlib.Data.Rat.BigOperators
import Mathlib.Data.Rat.Cast.Order
import Nfp.Core.Basic

/-!
Mean/variance helpers for LayerNorm bounds.

This module isolates the dyadic and real mean/variance definitions and their
basic lemmas to keep `LayerNorm` bounds modular.
-/

namespace Nfp

namespace Sound

namespace Bounds

open scoped BigOperators

/-- Sum as a rational, used for exact mean/variance computations. -/
def sumRat {n : Nat} (x : Fin n → Dyadic) : Rat :=
  ∑ i, (x i : Rat)

/-- Exact mean as a rational (defaults to `0` when `n = 0`). -/
def meanRat {n : Nat} (x : Fin n → Dyadic) : Rat :=
  if n = 0 then
    0
  else
    (sumRat x) / n

/-- Mean rounded down to dyadic precision (defaults to `0` when `n = 0`). -/
def mean {n : Nat} (x : Fin n → Dyadic) : Dyadic :=
  if n = 0 then
    0
  else
    dyadicOfRatDown (meanRat x)

/-- Mean rounded up to dyadic precision (defaults to `0` when `n = 0`). -/
def meanUpper {n : Nat} (x : Fin n → Dyadic) : Dyadic :=
  if n = 0 then
    0
  else
    dyadicOfRatUp (meanRat x)

/-- Unfold `mean` when `n ≠ 0`. -/
theorem mean_def {n : Nat} (x : Fin n → Dyadic) (h : n ≠ 0) :
    mean x = dyadicOfRatDown (meanRat x) := by
  simp [mean, h]

/-- Unfold `meanUpper` when `n ≠ 0`. -/
theorem meanUpper_def {n : Nat} (x : Fin n → Dyadic) (h : n ≠ 0) :
    meanUpper x = dyadicOfRatUp (meanRat x) := by
  simp [meanUpper, h]

/-- Exact variance as a rational (defaults to `0` when `n = 0`). -/
def varianceRat {n : Nat} (x : Fin n → Dyadic) : Rat :=
  if n = 0 then
    0
  else
    let μ := meanRat x
    (∑ i, ((x i : Rat) - μ) ^ 2) / n

/-- Variance rounded down to dyadic precision (defaults to `0` when `n = 0`). -/
def variance {n : Nat} (x : Fin n → Dyadic) : Dyadic :=
  if n = 0 then
    0
  else
    dyadicOfRatDown (varianceRat x)

/-- Variance rounded up to dyadic precision (defaults to `0` when `n = 0`). -/
def varianceUpper {n : Nat} (x : Fin n → Dyadic) : Dyadic :=
  if n = 0 then
    0
  else
    dyadicOfRatUp (varianceRat x)

/-- Unfold `variance` when `n ≠ 0`. -/
theorem variance_def {n : Nat} (x : Fin n → Dyadic) (h : n ≠ 0) :
    variance x = dyadicOfRatDown (varianceRat x) := by
  simp [variance, h]

/-! Interval helpers. -/

/-- Absolute value bound from endpoint bounds. -/
theorem abs_le_max_of_bounds {α : Type _} [Ring α] [LinearOrder α] [IsOrderedRing α]
    {a b z : α}
    (hlo : a ≤ z) (hhi : z ≤ b) :
    |z| ≤ max |a| |b| := by
  have hleft : -max |a| |b| ≤ z := by
    have hneg : -max |a| |b| ≤ a := by
      have hneg' : -max |a| |b| ≤ -|a| := by
        exact neg_le_neg (le_max_left _ _)
      have hneg'' : -|a| ≤ a := by
        have h : -a ≤ |a| := neg_le_abs a
        simpa using (neg_le_neg h)
      exact le_trans hneg' hneg''
    exact le_trans hneg hlo
  have hright : z ≤ max |a| |b| := by
    have hb : b ≤ |b| := by
      exact le_abs_self b
    have hb' : b ≤ max |a| |b| := le_trans hb (le_max_right _ _)
    exact le_trans hhi hb'
  exact (abs_le.mpr ⟨hleft, hright⟩)

/-! Real-valued mean and variance. -/

/-- Mean of a real vector (defaults to `0` when `n = 0`). -/
noncomputable def meanReal {n : Nat} (x : Fin n → Real) : Real :=
  if n = 0 then
    0
  else
    (∑ i, x i) / n

/-- Unfold `meanReal` when `n ≠ 0`. -/
theorem meanReal_def {n : Nat} (x : Fin n → Real) (h : n ≠ 0) :
    meanReal x = (∑ i, x i) / n := by
  simp [meanReal, h]

/-- `meanReal` agrees with `mean` after casting. -/
theorem meanReal_eq_meanRat {n : Nat} (x : Fin n → Dyadic) :
    meanReal (fun i => (x i : Real)) = (meanRat x : Real) := by
  by_cases h : n = 0
  · simp [meanReal, meanRat, h]
  · have hsum :
        (sumRat x : Real) = ∑ i, (x i : Real) := by
      classical
      unfold sumRat
      simp [dyadicToReal, Rat.cast_sum]
    have hmean : (meanRat x : Real) = (sumRat x : Real) / n := by
      simp [meanRat, h]
    have hreal : meanReal (fun i => (x i : Real)) = (∑ i, (x i : Real)) / n := by
      simp [meanReal, h]
    simpa [hmean, hsum] using hreal

/-- Mean is monotone under pointwise order (real inputs). -/
theorem meanReal_le_meanReal {n : Nat} (x y : Fin n → Real) (hne : n ≠ 0)
    (hxy : ∀ i, x i ≤ y i) : meanReal x ≤ meanReal y := by
  classical
  have hsum : (∑ i, x i) ≤ ∑ i, y i := by
    refine Finset.sum_le_sum ?_
    intro i _
    exact hxy i
  have hden : 0 ≤ (n : Real) := by
    simp
  have hdiv : (∑ i, x i) / n ≤ (∑ i, y i) / n :=
    div_le_div_of_nonneg_right hsum hden
  simpa [meanReal, hne] using hdiv

/-- Mean monotonicity for dyadic inputs, interpreted in reals. -/
theorem meanRat_le_meanRat_real {n : Nat} (x y : Fin n → Dyadic) (hne : n ≠ 0)
    (hxy : ∀ i, x i ≤ y i) :
    (meanRat x : Real) ≤ (meanRat y : Real) := by
  have hreal :
      meanReal (fun i => (x i : Real)) ≤ meanReal (fun i => (y i : Real)) := by
    refine meanReal_le_meanReal (x := fun i => (x i : Real)) (y := fun i => (y i : Real)) hne ?_
    intro i
    exact dyadicToReal_le_of_le (hxy i)
  simpa [meanReal_eq_meanRat] using hreal

/-- Variance of a real vector (defaults to `0` when `n = 0`). -/
noncomputable def varianceReal {n : Nat} (x : Fin n → Real) : Real :=
  if n = 0 then
    0
  else
    let μ := meanReal x
    (∑ i, (x i - μ) ^ 2) / n

/-- Unfold `varianceReal` when `n ≠ 0`. -/
theorem varianceReal_def {n : Nat} (x : Fin n → Real) (h : n ≠ 0) :
    varianceReal x =
      let μ := meanReal x
      (∑ i, (x i - μ) ^ 2) / n := by
  simp [varianceReal, h]

/-- Variance is nonnegative when `n ≠ 0`. -/
theorem varianceReal_nonneg {n : Nat} (x : Fin n → Real) (h : n ≠ 0) :
    0 ≤ varianceReal x := by
  classical
  have hsum : 0 ≤ ∑ i, (x i - meanReal x) ^ 2 := by
    refine Finset.sum_nonneg ?_
    intro i _
    exact sq_nonneg (x i - meanReal x)
  have hden : 0 ≤ (n : Real) := by
    simp
  have hdiv : 0 ≤ (∑ i, (x i - meanReal x) ^ 2) / n :=
    div_nonneg hsum hden
  simpa [varianceReal_def x h] using hdiv

theorem varianceReal_eq_varianceRat {n : Nat} (x : Fin n → Dyadic) :
    varianceReal (fun i => (x i : Real)) = (varianceRat x : Real) := by
  by_cases h : n = 0
  · simp [varianceReal, varianceRat, h]
  · have hmean := meanReal_eq_meanRat (n := n) x
    have hsum :
        (∑ i, ((x i : Real) - (meanRat x : Real)) ^ 2) =
          (∑ i, ((x i : Rat) - meanRat x) ^ 2 : Rat) := by
      classical
      simp [dyadicToReal, Rat.cast_sum]
    have hreal : varianceReal (fun i => (x i : Real)) =
        (∑ i, ((x i : Real) - meanReal (fun j => (x j : Real))) ^ 2) / n := by
      simp [varianceReal, h]
    have hrat : (varianceRat x : Real) =
        (∑ i, ((x i : Rat) - meanRat x) ^ 2 : Rat) / n := by
      simp [varianceRat, h]
    calc
      varianceReal (fun i => (x i : Real))
          = (∑ i, ((x i : Real) - meanReal (fun j => (x j : Real))) ^ 2) / n := hreal
      _ = (∑ i, ((x i : Real) - (meanRat x : Real)) ^ 2) / n := by
            simp [hmean]
      _ = (∑ i, ((x i : Rat) - meanRat x) ^ 2 : Rat) / n := by
            simp [hsum]
      _ = (varianceRat x : Real) := hrat.symm

/-- Variance is nonnegative when `n ≠ 0`, interpreted in reals. -/
theorem varianceRat_nonneg_real {n : Nat} (x : Fin n → Dyadic) (hne : n ≠ 0) :
    0 ≤ (varianceRat x : Real) := by
  have hreal := varianceReal_nonneg (x := fun i => (x i : Real)) hne
  simpa [varianceReal_eq_varianceRat] using hreal

/-- Absolute mean bound from per-coordinate bounds (real inputs). -/
theorem meanReal_abs_le_bound {n : Nat} (x : Fin n → Real) (bound : Dyadic)
    (hne : n ≠ 0) (hbound : ∀ i, |x i| ≤ (bound : Real)) :
    |meanReal x| ≤ (bound : Real) := by
  classical
  have hsum_abs :
      |∑ i : Fin n, x i| ≤ ∑ i : Fin n, |x i| := by
    simpa using
      (Finset.abs_sum_le_sum_abs
        (f := fun i : Fin n => x i)
        (s := (Finset.univ : Finset (Fin n))))
  have hsum_bound : ∑ i : Fin n, |x i| ≤ ∑ i : Fin n, (bound : Real) := by
    refine Finset.sum_le_sum ?_
    intro i _
    exact hbound i
  have hsum_le : |∑ i : Fin n, x i| ≤ (n : Real) * (bound : Real) := by
    have hsum := le_trans hsum_abs hsum_bound
    simpa [Finset.sum_const, Finset.card_univ, mul_comm] using hsum
  have hpos : 0 < (n : Real) := by
    exact (Nat.cast_pos (α := Real)).2 (Nat.pos_of_ne_zero hne)
  have hsum_le' : |∑ i : Fin n, x i| ≤ (bound : Real) * (n : Real) := by
    simpa [mul_comm] using hsum_le
  have hdiv : |∑ i : Fin n, x i| / (n : Real) ≤ (bound : Real) := by
    exact (div_le_iff₀ hpos).2 hsum_le'
  have habs_mean :
      |(∑ i : Fin n, x i) / (n : Real)| ≤ (bound : Real) := by
    simpa [abs_div, abs_of_nonneg (le_of_lt hpos)] using hdiv
  simpa [meanReal_def x hne] using habs_mean

end Bounds

end Sound

end Nfp
