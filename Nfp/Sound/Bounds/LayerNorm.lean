-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Data.Nat.Sqrt
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Rat.Cast.Order

/-!
LayerNorm interval bounds for exact rational inputs.

This module computes rational interval bounds for LayerNorm outputs and proves
those bounds sound for real-valued LayerNorm semantics.
-/

namespace Nfp

namespace Sound

namespace Bounds

open scoped BigOperators

/-- Mean of a finite vector (defaults to `0` when `n = 0`). -/
def mean {n : Nat} (x : Fin n → Rat) : Rat :=
  if n = 0 then
    0
  else
    (∑ i, x i) / n

/-- Unfold `mean` when `n ≠ 0`. -/
theorem mean_def {n : Nat} (x : Fin n → Rat) (h : n ≠ 0) :
    mean x = (∑ i, x i) / n := by
  simp [mean, h]

/-- Variance of a finite vector (defaults to `0` when `n = 0`). -/
def variance {n : Nat} (x : Fin n → Rat) : Rat :=
  if n = 0 then
    0
  else
    let μ := mean x
    (∑ i, (x i - μ) ^ 2) / n

/-- Unfold `variance` when `n ≠ 0`. -/
theorem variance_def {n : Nat} (x : Fin n → Rat) (h : n ≠ 0) :
    variance x =
      let μ := mean x
      (∑ i, (x i - μ) ^ 2) / n := by
  simp [variance, h]

/-- Variance is nonnegative when `n ≠ 0`. -/
theorem variance_nonneg {n : Nat} (x : Fin n → Rat) (h : n ≠ 0) :
    0 ≤ variance x := by
  classical
  have hsum : 0 ≤ ∑ i, (x i - mean x) ^ 2 := by
    refine Finset.sum_nonneg ?_
    intro i _
    exact sq_nonneg (x i - mean x)
  have hden : 0 ≤ (n : Rat) := by
    exact_mod_cast (Nat.zero_le n)
  have hdiv : 0 ≤ (∑ i, (x i - mean x) ^ 2) / n :=
    div_nonneg hsum hden
  simpa [variance_def x h] using hdiv

/-! Square-root bounds. -/

/-- Base rational lower bound for a square root. -/
def sqrtLowerBase (q : Rat) : Rat :=
  let num := q.num.natAbs
  let den := q.den
  let a := Nat.sqrt num
  let b := Nat.sqrt den
  (a : Rat) / (b + 1 : Rat)

/-- Base rational upper bound for a square root. -/
def sqrtUpperBase (q : Rat) : Rat :=
  let num := q.num.natAbs
  let den := q.den
  let a := Nat.sqrt num
  let b := Nat.sqrt den
  (a + 1 : Rat) / (b : Rat)

/-- Alternate rational lower bound for a square root. -/
def sqrtLowerAlt (q : Rat) : Rat :=
  let num := q.num.natAbs
  let den := q.den
  let a := Nat.sqrt (num * den)
  (a : Rat) / den

/-- Alternate rational upper bound for a square root. -/
def sqrtUpperAlt (q : Rat) : Rat :=
  let num := q.num.natAbs
  let den := q.den
  let a := Nat.sqrt (num * den)
  (a + 1 : Rat) / den

/-- Rational lower bound for a square root (tighter of two bounds). -/
def sqrtLower (q : Rat) : Rat :=
  max (sqrtLowerBase q) (sqrtLowerAlt q)

/-- Rational upper bound for a square root (tighter of two bounds). -/
def sqrtUpper (q : Rat) : Rat :=
  min (sqrtUpperBase q) (sqrtUpperAlt q)

/-- `sqrtLowerBase` is nonnegative. -/
theorem sqrtLowerBase_nonneg (q : Rat) : 0 ≤ sqrtLowerBase q := by
  classical
  unfold sqrtLowerBase
  have hden : 0 ≤ (Nat.sqrt q.den + 1 : Rat) := by
    exact_mod_cast (Nat.zero_le _)
  have hnum : 0 ≤ (Nat.sqrt q.num.natAbs : Rat) := by
    exact_mod_cast (Nat.zero_le _)
  exact div_nonneg hnum hden

/-! Strict positivity helpers. -/

/-! Base bounds. -/

/-- `sqrtLowerBase` is positive when its input is positive. -/
theorem sqrtLowerBase_pos {q : Rat} (hq : 0 < q) : 0 < sqrtLowerBase q := by
  classical
  unfold sqrtLowerBase
  have hnum_pos : 0 < (Nat.sqrt q.num.natAbs : Rat) := by
    have hnum_pos' : 0 < q.num.natAbs := by
      have hnum : 0 < q.num := (Rat.num_pos (a := q)).2 hq
      exact Int.natAbs_pos.mpr hnum.ne'
    exact_mod_cast (Nat.sqrt_pos.2 hnum_pos')
  have hden_pos : 0 < (Nat.sqrt q.den + 1 : Rat) := by
    exact_mod_cast (Nat.succ_pos _)
  exact div_pos hnum_pos hden_pos

/-- `sqrtUpperBase` is nonnegative. -/
theorem sqrtUpperBase_nonneg (q : Rat) : 0 ≤ sqrtUpperBase q := by
  classical
  unfold sqrtUpperBase
  have hden : 0 ≤ (Nat.sqrt q.den : Rat) := by
    exact_mod_cast (Nat.zero_le _)
  have hnum : 0 ≤ (Nat.sqrt q.num.natAbs + 1 : Rat) := by
    exact_mod_cast (Nat.zero_le _)
  exact div_nonneg hnum hden

/-- `sqrtUpperBase` is always positive. -/
theorem sqrtUpperBase_pos (q : Rat) : 0 < sqrtUpperBase q := by
  classical
  unfold sqrtUpperBase
  have hnum_pos : 0 < (Nat.sqrt q.num.natAbs + 1 : Rat) := by
    exact_mod_cast (Nat.succ_pos _)
  have hden_pos : 0 < (Nat.sqrt q.den : Rat) := by
    have hden : 0 < q.den := q.den_pos
    exact_mod_cast (Nat.sqrt_pos.2 hden)
  exact div_pos hnum_pos hden_pos

/-! Alternate bounds. -/

/-- `sqrtLowerAlt` is nonnegative. -/
theorem sqrtLowerAlt_nonneg (q : Rat) : 0 ≤ sqrtLowerAlt q := by
  classical
  unfold sqrtLowerAlt
  have hnum : 0 ≤ (Nat.sqrt (q.num.natAbs * q.den) : Rat) := by
    exact_mod_cast (Nat.zero_le _)
  have hden : 0 ≤ (q.den : Rat) := by
    exact_mod_cast (Nat.zero_le _)
  exact div_nonneg hnum hden

/-- `sqrtLowerAlt` is positive when its input is positive. -/
theorem sqrtLowerAlt_pos {q : Rat} (hq : 0 < q) : 0 < sqrtLowerAlt q := by
  classical
  unfold sqrtLowerAlt
  have hnum_pos : 0 < (Nat.sqrt (q.num.natAbs * q.den) : Rat) := by
    have hnum_pos' : 0 < q.num.natAbs := by
      have hnum : 0 < q.num := (Rat.num_pos (a := q)).2 hq
      exact Int.natAbs_pos.mpr hnum.ne'
    have hden_pos : 0 < q.den := q.den_pos
    have hmul_pos : 0 < q.num.natAbs * q.den := by
      exact Nat.mul_pos hnum_pos' hden_pos
    exact_mod_cast (Nat.sqrt_pos.2 hmul_pos)
  have hden_pos : 0 < (q.den : Rat) := by
    exact_mod_cast q.den_pos
  exact div_pos hnum_pos hden_pos

/-- `sqrtUpperAlt` is nonnegative. -/
theorem sqrtUpperAlt_nonneg (q : Rat) : 0 ≤ sqrtUpperAlt q := by
  classical
  unfold sqrtUpperAlt
  have hnum : 0 ≤ (Nat.sqrt (q.num.natAbs * q.den) + 1 : Rat) := by
    exact_mod_cast (Nat.zero_le _)
  have hden : 0 ≤ (q.den : Rat) := by
    exact_mod_cast (Nat.zero_le _)
  exact div_nonneg hnum hden

/-- `sqrtUpperAlt` is always positive. -/
theorem sqrtUpperAlt_pos (q : Rat) : 0 < sqrtUpperAlt q := by
  classical
  unfold sqrtUpperAlt
  have hnum_pos : 0 < (Nat.sqrt (q.num.natAbs * q.den) + 1 : Rat) := by
    exact_mod_cast (Nat.succ_pos _)
  have hden_pos : 0 < (q.den : Rat) := by
    exact_mod_cast q.den_pos
  exact div_pos hnum_pos hden_pos

/-! Combined bounds. -/

/-- `sqrtLower` is nonnegative. -/
theorem sqrtLower_nonneg (q : Rat) : 0 ≤ sqrtLower q := by
  have hbase : 0 ≤ sqrtLowerBase q := sqrtLowerBase_nonneg q
  exact le_trans hbase (le_max_left _ _)

/-- `sqrtLower` is positive when its input is positive. -/
theorem sqrtLower_pos {q : Rat} (hq : 0 < q) : 0 < sqrtLower q := by
  have hbase : 0 < sqrtLowerBase q := sqrtLowerBase_pos hq
  exact lt_of_lt_of_le hbase (le_max_left _ _)

/-- `sqrtUpper` is nonnegative. -/
theorem sqrtUpper_nonneg (q : Rat) : 0 ≤ sqrtUpper q := by
  have hbase : 0 ≤ sqrtUpperBase q := sqrtUpperBase_nonneg q
  have halt : 0 ≤ sqrtUpperAlt q := sqrtUpperAlt_nonneg q
  exact le_min hbase halt

/-- `sqrtUpper` is always positive. -/
theorem sqrtUpper_pos (q : Rat) : 0 < sqrtUpper q := by
  have hbase : 0 < sqrtUpperBase q := sqrtUpperBase_pos q
  have halt : 0 < sqrtUpperAlt q := sqrtUpperAlt_pos q
  exact lt_min hbase halt

/-- Square-root lower bound in reals. -/
theorem sqrtLowerBase_le_real_sqrt {q : Rat} (hq : 0 ≤ q) :
    (sqrtLowerBase q : Real) ≤ Real.sqrt (q : Real) := by
  classical
  -- Set up numerator/denominator witnesses.
  set num : Nat := q.num.natAbs
  set den : Nat := q.den
  set a : Nat := Nat.sqrt num
  set b : Nat := Nat.sqrt den
  have hden_pos : 0 < (den : Real) := by
    exact_mod_cast q.den_pos
  have hbpos : 0 < (b + 1 : Real) := by
    exact_mod_cast (Nat.succ_pos b)
  have hnum_le : (a ^ 2 : Real) ≤ num := by
    exact_mod_cast (Nat.sqrt_le' num)
  have hden_le : (den : Real) ≤ (b + 1) ^ 2 := by
    exact_mod_cast (le_of_lt (Nat.lt_succ_sqrt' den))
  have hmul : (a ^ 2 : Real) * den ≤ (num : Real) * (b + 1) ^ 2 := by
    have hden_nonneg : 0 ≤ (den : Real) := by exact_mod_cast (Nat.zero_le den)
    have hnum_nonneg : 0 ≤ (num : Real) := by exact_mod_cast (Nat.zero_le num)
    exact mul_le_mul hnum_le hden_le hden_nonneg hnum_nonneg
  have hbpos2 : 0 < (b + 1 : Real) ^ 2 := by
    nlinarith [hbpos]
  have hdiv : (a ^ 2 : Real) / (b + 1) ^ 2 ≤ (num : Real) / den := by
    exact (div_le_div_iff₀ hbpos2 hden_pos).2 hmul
  have hpow : ((a : Real) / (b + 1 : Real)) ^ 2 = (a ^ 2 : Real) / (b + 1) ^ 2 := by
    simp [pow_two, div_mul_div_comm]
  have hq_cast : (q : Real) = (num : Real) / den := by
    have hnum_nonneg : 0 ≤ q.num := by
      exact (Rat.num_nonneg (q := q)).2 hq
    have hnum_eq : (num : Int) = q.num := by
      simpa [num] using (Int.natAbs_of_nonneg hnum_nonneg)
    have hnum_cast : (q.num : Real) = (num : Real) := by
      exact_mod_cast hnum_eq.symm
    have hq_rat : (q : Real) = (q.num : Real) / q.den := by
      simp [Rat.cast_def]
    simpa [hnum_cast, den] using hq_rat
  have hsq : ((a : Real) / (b + 1 : Real)) ^ 2 ≤ (q : Real) := by
    simpa [hpow, hq_cast, den, num] using hdiv
  have hnonneg : 0 ≤ (a : Real) / (b + 1 : Real) := by
    have hnum_nonneg : 0 ≤ (a : Real) := by exact_mod_cast (Nat.zero_le a)
    have hden_nonneg : 0 ≤ (b + 1 : Real) := by exact_mod_cast (Nat.zero_le (b + 1))
    exact div_nonneg hnum_nonneg hden_nonneg
  have hq_nonneg : 0 ≤ (q : Real) := by exact_mod_cast hq
  have hle : (a : Real) / (b + 1 : Real) ≤ Real.sqrt (q : Real) :=
    (Real.le_sqrt hnonneg hq_nonneg).2 hsq
  simpa [sqrtLowerBase, num, den, a, b] using hle

/-- Square-root upper bound in reals. -/
theorem real_sqrt_le_sqrtUpperBase {q : Rat} (hq : 0 ≤ q) :
    Real.sqrt (q : Real) ≤ (sqrtUpperBase q : Real) := by
  classical
  set num : Nat := q.num.natAbs
  set den : Nat := q.den
  set a : Nat := Nat.sqrt num
  set b : Nat := Nat.sqrt den
  have hden_pos : 0 < (den : Real) := by
    exact_mod_cast q.den_pos
  have hbpos : 0 < (b : Real) := by
    have hb : 0 < b := by
      have hden : 0 < den := q.den_pos
      exact (Nat.sqrt_pos).2 hden
    exact_mod_cast hb
  have hnum_lt : (num : Real) < (a + 1) ^ 2 := by
    exact_mod_cast (Nat.lt_succ_sqrt' num)
  have hden_le : (b ^ 2 : Real) ≤ den := by
    exact_mod_cast (Nat.sqrt_le' den)
  have hmul : (num : Real) * (b ^ 2) ≤ (a + 1) ^ 2 * den := by
    have hb2_nonneg : 0 ≤ (b ^ 2 : Real) := by
      exact sq_nonneg (b : Real)
    have hsq_nonneg : 0 ≤ (a + 1 : Real) ^ 2 := by
      exact sq_nonneg (a + 1 : Real)
    exact mul_le_mul (le_of_lt hnum_lt) hden_le hb2_nonneg hsq_nonneg
  have hbpos2 : 0 < (b : Real) ^ 2 := by
    nlinarith [hbpos]
  have hdiv : (num : Real) / den ≤ (a + 1) ^ 2 / (b : Real) ^ 2 := by
    exact (div_le_div_iff₀ hden_pos hbpos2).2 hmul
  have hpow : ((a + 1 : Real) / (b : Real)) ^ 2 = (a + 1) ^ 2 / (b : Real) ^ 2 := by
    simp [pow_two, div_mul_div_comm]
  have hq_cast : (q : Real) = (num : Real) / den := by
    have hnum_nonneg : 0 ≤ q.num := by
      exact (Rat.num_nonneg (q := q)).2 hq
    have hnum_eq : (num : Int) = q.num := by
      simpa [num] using (Int.natAbs_of_nonneg hnum_nonneg)
    have hnum_cast : (q.num : Real) = (num : Real) := by
      exact_mod_cast hnum_eq.symm
    have hq_rat : (q : Real) = (q.num : Real) / q.den := by
      simp [Rat.cast_def]
    simpa [hnum_cast, den] using hq_rat
  have hsq : (q : Real) ≤ ((a + 1 : Real) / (b : Real)) ^ 2 := by
    simpa [hpow, hq_cast, den, num] using hdiv
  have hnonneg : 0 ≤ ((a + 1 : Real) / (b : Real)) := by
    have hnum_nonneg : 0 ≤ (a + 1 : Real) := by exact_mod_cast (Nat.zero_le (a + 1))
    have hden_nonneg : 0 ≤ (b : Real) := by exact_mod_cast (Nat.zero_le b)
    exact div_nonneg hnum_nonneg hden_nonneg
  have hle : Real.sqrt (q : Real) ≤ (a + 1 : Real) / (b : Real) :=
    (Real.sqrt_le_iff).2 ⟨hnonneg, hsq⟩
  simpa [sqrtUpperBase, num, den, a, b] using hle

/-- Alternate square-root lower bound in reals. -/
theorem sqrtLowerAlt_le_real_sqrt {q : Rat} (hq : 0 ≤ q) :
    (sqrtLowerAlt q : Real) ≤ Real.sqrt (q : Real) := by
  classical
  set num : Nat := q.num.natAbs
  set den : Nat := q.den
  set a : Nat := Nat.sqrt (num * den)
  have hden_pos : 0 < (den : Real) := by
    exact_mod_cast q.den_pos
  have hnumden_le : (a ^ 2 : Real) ≤ (num * den : Nat) := by
    exact_mod_cast (Nat.sqrt_le' (num * den))
  have hmul : (a ^ 2 : Real) ≤ (num : Real) * den := by
    simpa [num, den, Nat.cast_mul] using hnumden_le
  have hden_pos2 : 0 < (den : Real) ^ 2 := by
    nlinarith [hden_pos]
  have hdiv :
      (a ^ 2 : Real) / (den : Real) ^ 2 ≤ (num : Real) * den / (den : Real) ^ 2 := by
    have hmul' :
        (a ^ 2 : Real) * (den : Real) ^ 2 ≤ (num : Real) * den * (den : Real) ^ 2 := by
      have hden_sq_nonneg : 0 ≤ (den : Real) ^ 2 := by
        exact sq_nonneg (den : Real)
      exact mul_le_mul_of_nonneg_right hmul hden_sq_nonneg
    exact (div_le_div_iff₀ hden_pos2 hden_pos2).2 hmul'
  have hden_ne : (den : Real) ≠ 0 := by
    exact_mod_cast q.den_pos.ne'
  have hq_cast : (q : Real) = (num : Real) * den / (den : Real) ^ 2 := by
    have hnum_nonneg : 0 ≤ q.num := by
      exact (Rat.num_nonneg (q := q)).2 hq
    have hnum_eq : (num : Int) = q.num := by
      simpa [num] using (Int.natAbs_of_nonneg hnum_nonneg)
    have hnum_cast : (q.num : Real) = (num : Real) := by
      exact_mod_cast hnum_eq.symm
    have hq_rat : (q : Real) = (q.num : Real) / q.den := by
      simp [Rat.cast_def]
    have hq_eq :
        (num : Real) / den = (num : Real) * den / (den : Real) ^ 2 := by
      field_simp [hden_ne]
    simpa [hnum_cast, den, hq_eq] using hq_rat
  have hsq : ((a : Real) / (den : Real)) ^ 2 ≤ (q : Real) := by
    simpa [hq_cast, pow_two, div_mul_div_comm] using hdiv
  have hnonneg : 0 ≤ (a : Real) / (den : Real) := by
    have hnum_nonneg : 0 ≤ (a : Real) := by exact_mod_cast (Nat.zero_le a)
    have hden_nonneg : 0 ≤ (den : Real) := by exact_mod_cast (Nat.zero_le den)
    exact div_nonneg hnum_nonneg hden_nonneg
  have hq_nonneg : 0 ≤ (q : Real) := by exact_mod_cast hq
  have hle : (a : Real) / (den : Real) ≤ Real.sqrt (q : Real) :=
    (Real.le_sqrt hnonneg hq_nonneg).2 hsq
  simpa [sqrtLowerAlt, num, den, a] using hle

/-- Alternate square-root upper bound in reals. -/
theorem real_sqrt_le_sqrtUpperAlt {q : Rat} (hq : 0 ≤ q) :
    Real.sqrt (q : Real) ≤ (sqrtUpperAlt q : Real) := by
  classical
  set num : Nat := q.num.natAbs
  set den : Nat := q.den
  set a : Nat := Nat.sqrt (num * den)
  have hden_pos : 0 < (den : Real) := by
    exact_mod_cast q.den_pos
  have hnumden_lt : (num * den : Real) < (a + 1) ^ 2 := by
    exact_mod_cast (Nat.lt_succ_sqrt' (num * den))
  have hmul : (num : Real) * den ≤ (a + 1 : Real) ^ 2 := by
    exact le_of_lt hnumden_lt
  have hden_pos2 : 0 < (den : Real) ^ 2 := by
    nlinarith [hden_pos]
  have hdiv :
      (num : Real) * den / (den : Real) ^ 2 ≤ (a + 1 : Real) ^ 2 / (den : Real) ^ 2 := by
    have hmul' :
        (num : Real) * den * (den : Real) ^ 2 ≤ (a + 1 : Real) ^ 2 * (den : Real) ^ 2 := by
      have hden_sq_nonneg : 0 ≤ (den : Real) ^ 2 := by
        exact sq_nonneg (den : Real)
      exact mul_le_mul_of_nonneg_right hmul hden_sq_nonneg
    exact (div_le_div_iff₀ hden_pos2 hden_pos2).2 hmul'
  have hden_ne : (den : Real) ≠ 0 := by
    exact_mod_cast q.den_pos.ne'
  have hq_cast : (q : Real) = (num : Real) * den / (den : Real) ^ 2 := by
    have hnum_nonneg : 0 ≤ q.num := by
      exact (Rat.num_nonneg (q := q)).2 hq
    have hnum_eq : (num : Int) = q.num := by
      simpa [num] using (Int.natAbs_of_nonneg hnum_nonneg)
    have hnum_cast : (q.num : Real) = (num : Real) := by
      exact_mod_cast hnum_eq.symm
    have hq_rat : (q : Real) = (q.num : Real) / q.den := by
      simp [Rat.cast_def]
    have hq_eq :
        (num : Real) / den = (num : Real) * den / (den : Real) ^ 2 := by
      field_simp [hden_ne]
    simpa [hnum_cast, den, hq_eq] using hq_rat
  have hpow :
      ((a + 1 : Real) / (den : Real)) ^ 2 =
        (a + 1 : Real) ^ 2 / (den : Real) ^ 2 := by
    simp [pow_two, div_mul_div_comm]
  have hsq : (q : Real) ≤ ((a + 1 : Real) / (den : Real)) ^ 2 := by
    simpa [hq_cast, hpow] using hdiv
  have hnonneg : 0 ≤ ((a + 1 : Real) / (den : Real)) := by
    have hnum_nonneg : 0 ≤ (a + 1 : Real) := by exact_mod_cast (Nat.zero_le (a + 1))
    have hden_nonneg : 0 ≤ (den : Real) := by exact_mod_cast (Nat.zero_le den)
    exact div_nonneg hnum_nonneg hden_nonneg
  have hle : Real.sqrt (q : Real) ≤ (a + 1 : Real) / (den : Real) :=
    (Real.sqrt_le_iff).2 ⟨hnonneg, hsq⟩
  simpa [sqrtUpperAlt, num, den, a] using hle

/-- Square-root lower bound in reals (tighter of two bounds). -/
theorem sqrtLower_le_real_sqrt {q : Rat} (hq : 0 ≤ q) :
    (sqrtLower q : Real) ≤ Real.sqrt (q : Real) := by
  have hbase := sqrtLowerBase_le_real_sqrt (q := q) hq
  have halt := sqrtLowerAlt_le_real_sqrt (q := q) hq
  simpa [sqrtLower] using (max_le_iff).2 ⟨hbase, halt⟩

/-- Square-root upper bound in reals (tighter of two bounds). -/
theorem real_sqrt_le_sqrtUpper {q : Rat} (hq : 0 ≤ q) :
    Real.sqrt (q : Real) ≤ (sqrtUpper q : Real) := by
  have hbase := real_sqrt_le_sqrtUpperBase (q := q) hq
  have halt := real_sqrt_le_sqrtUpperAlt (q := q) hq
  simpa [sqrtUpper] using (le_min_iff).2 ⟨hbase, halt⟩

/-- Bounds for multiplying a scalar by a bounded value. -/
def scaleInterval (x lo hi : Rat) : Rat × Rat :=
  if 0 ≤ x then
    (x * lo, x * hi)
  else
    (x * hi, x * lo)

/-- `scaleInterval` bounds a product. -/
theorem scaleInterval_bounds {x lo hi y : Rat}
    (hlo : lo ≤ y) (hhi : y ≤ hi) :
    let bounds := scaleInterval x lo hi
    bounds.1 ≤ x * y ∧ x * y ≤ bounds.2 := by
  by_cases hx : 0 ≤ x
  · have h1 : x * lo ≤ x * y := by
      exact mul_le_mul_of_nonneg_left hlo hx
    have h2 : x * y ≤ x * hi := by
      exact mul_le_mul_of_nonneg_left hhi hx
    simp [scaleInterval, hx, h1, h2]
  · have hx' : x ≤ 0 := le_of_lt (lt_of_not_ge hx)
    have h1 : x * hi ≤ x * y := by
      exact mul_le_mul_of_nonpos_left hhi hx'
    have h2 : x * y ≤ x * lo := by
      exact mul_le_mul_of_nonpos_left hlo hx'
    simp [scaleInterval, hx, h1, h2]

/-- `scaleInterval` bounds interpreted in the reals. -/
theorem scaleInterval_bounds_real {x lo hi : Rat} {y : Real}
    (hlo : (lo : Real) ≤ y) (hhi : y ≤ (hi : Real)) :
    let bounds := scaleInterval x lo hi
    (bounds.1 : Real) ≤ (x : Real) * y ∧ (x : Real) * y ≤ (bounds.2 : Real) := by
  by_cases hx : 0 ≤ x
  · have h1 : (x : Real) * (lo : Real) ≤ (x : Real) * y := by
      exact mul_le_mul_of_nonneg_left hlo (by exact_mod_cast hx)
    have h2 : (x : Real) * y ≤ (x : Real) * (hi : Real) := by
      exact mul_le_mul_of_nonneg_left hhi (by exact_mod_cast hx)
    simp [scaleInterval, hx, h1, h2]
  · have hx' : x ≤ 0 := le_of_lt (lt_of_not_ge hx)
    have h1 : (x : Real) * (hi : Real) ≤ (x : Real) * y := by
      exact mul_le_mul_of_nonpos_left hhi (by exact_mod_cast hx')
    have h2 : (x : Real) * y ≤ (x : Real) * (lo : Real) := by
      exact mul_le_mul_of_nonpos_left hlo (by exact_mod_cast hx')
    simp [scaleInterval, hx, h1, h2]

/-- Real-valued LayerNorm output for a vector. -/
noncomputable def layerNormReal {n : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat) (x : Fin n → Rat) : Fin n → Real :=
  if n = 0 then
    fun _ => 0
  else
    let μ : Real := mean x
    let varEps : Real := (variance x + eps : Rat)
    let invStd : Real := (Real.sqrt varEps)⁻¹
    fun i => (gamma i : Real) * ((x i : Real) - μ) * invStd + (beta i : Real)

/-- Interval bounds for LayerNorm outputs. -/
def layerNormBounds {n : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat) (x : Fin n → Rat) :
    (Fin n → Rat) × (Fin n → Rat) :=
  if n = 0 then
    (fun _ => 0, fun _ => 0)
  else
    let μ := mean x
    let var := variance x
    let varEps := var + eps
    let sLo := sqrtLower varEps
    let sHi := sqrtUpper varEps
    let invLo := sHi⁻¹
    let invHi := sLo⁻¹
    let normBounds : Fin n → Rat × Rat := fun i =>
      let centered := x i - μ
      scaleInterval centered invLo invHi
    let outBounds : Fin n → Rat × Rat := fun i =>
      let nb := normBounds i
      let sb := scaleInterval (gamma i) nb.1 nb.2
      (sb.1 + beta i, sb.2 + beta i)
    (fun i => (outBounds i).1, fun i => (outBounds i).2)

/-- `layerNormBounds` soundness for real LayerNorm outputs. -/
theorem layerNormBounds_spec {n : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat) (x : Fin n → Rat)
    (hne : n ≠ 0) (heps : 0 < eps) :
    let bounds := layerNormBounds eps gamma beta x
    ∀ i,
      (bounds.1 i : Real) ≤ layerNormReal eps gamma beta x i ∧
        layerNormReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds i
  have hvar_nonneg : 0 ≤ variance x := variance_nonneg x hne
  have hvarEps_pos : 0 < variance x + eps := by
    exact add_pos_of_nonneg_of_pos hvar_nonneg heps
  have hvarEps_nonneg : 0 ≤ variance x + eps := by
    exact le_of_lt hvarEps_pos
  let varEps : Rat := variance x + eps
  let sLo : Rat := sqrtLower varEps
  let sHi : Rat := sqrtUpper varEps
  let invLo : Rat := sHi⁻¹
  let invHi : Rat := sLo⁻¹
  let invStd : Real := (Real.sqrt (varEps : Real))⁻¹
  have hsLo : (sLo : Real) ≤ Real.sqrt (varEps : Real) := by
    have hsLo' := sqrtLower_le_real_sqrt (q := varEps) hvarEps_nonneg
    simpa [sLo, varEps, Rat.cast_add] using hsLo'
  have hsHi : Real.sqrt (varEps : Real) ≤ (sHi : Real) := by
    have hsHi' := real_sqrt_le_sqrtUpper (q := varEps) hvarEps_nonneg
    simpa [sHi, varEps, Rat.cast_add] using hsHi'
  have hsqrt_pos : 0 < Real.sqrt (varEps : Real) := by
    exact Real.sqrt_pos.2 (by exact_mod_cast hvarEps_pos)
  have hsLo_pos : 0 < (sLo : Real) := by
    exact_mod_cast (sqrtLower_pos (q := varEps) hvarEps_pos)
  have hsHi_ne : (sHi : Rat) ≠ 0 := ne_of_gt (sqrtUpper_pos varEps)
  have hsLo_ne : (sLo : Rat) ≠ 0 := ne_of_gt (sqrtLower_pos (q := varEps) hvarEps_pos)
  have hcast_inv_hi : (invLo : Real) = (sHi : Real)⁻¹ := by
    have hnum_ne : (sHi.num : Real) ≠ 0 := by
      exact_mod_cast (Rat.num_ne_zero (q := sHi)).2 hsHi_ne
    have hcast := Rat.cast_inv_of_ne_zero (q := sHi) hnum_ne
    dsimp [invLo]
    exact hcast
  have hcast_inv_lo : (invHi : Real) = (sLo : Real)⁻¹ := by
    have hnum_ne : (sLo.num : Real) ≠ 0 := by
      exact_mod_cast (Rat.num_ne_zero (q := sLo)).2 hsLo_ne
    have hcast := Rat.cast_inv_of_ne_zero (q := sLo) hnum_ne
    dsimp [invHi]
    exact hcast
  have hinv_lo : (invLo : Real) ≤ invStd := by
    have hcalc : (sHi : Real)⁻¹ ≤ invStd := by
      have h := one_div_le_one_div_of_le hsqrt_pos hsHi
      simpa [one_div, invStd] using h
    simpa [hcast_inv_hi] using hcalc
  have hinv_hi : invStd ≤ (invHi : Real) := by
    have hcalc : invStd ≤ (sLo : Real)⁻¹ := by
      have h := one_div_le_one_div_of_le hsLo_pos hsLo
      simpa [one_div, invStd] using h
    simpa [hcast_inv_lo] using hcalc
  let μ : Rat := mean x
  let centered : Rat := x i - μ
  let nb : Rat × Rat := scaleInterval centered invLo invHi
  have hnb : (nb.1 : Real) ≤ (centered : Real) * invStd ∧
      (centered : Real) * invStd ≤ (nb.2 : Real) := by
    have hscale := scaleInterval_bounds_real (x := centered)
      (lo := invLo) (hi := invHi) (y := invStd) hinv_lo hinv_hi
    simpa [nb] using hscale
  let sb : Rat × Rat := scaleInterval (gamma i) nb.1 nb.2
  have hsb :
      (sb.1 : Real) ≤ (gamma i : Real) * ((centered : Real) * invStd) ∧
        (gamma i : Real) * ((centered : Real) * invStd) ≤ (sb.2 : Real) := by
    have hscale := scaleInterval_bounds_real (x := gamma i)
      (lo := nb.1) (hi := nb.2) (y := (centered : Real) * invStd) hnb.1 hnb.2
    simpa [sb] using hscale
  let lo : Rat := sb.1 + beta i
  let hi : Rat := sb.2 + beta i
  have hreal :
      layerNormReal eps gamma beta x i =
        (gamma i : Real) * ((centered : Real) * invStd) + (beta i : Real) := by
    calc
      layerNormReal eps gamma beta x i =
          (gamma i : Real) * ((x i : Real) - μ) * invStd + (beta i : Real) := by
            simp [layerNormReal, hne, μ, invStd, varEps]
      _ = (gamma i : Real) * (((x i : Real) - μ) * invStd) + (beta i : Real) := by
            simp [mul_assoc]
      _ = (gamma i : Real) * ((centered : Real) * invStd) + (beta i : Real) := by
            simp [centered]
  have hlo : (lo : Real) ≤ layerNormReal eps gamma beta x i := by
    have hlo' : (sb.1 : Real) ≤ (gamma i : Real) * ((centered : Real) * invStd) := hsb.1
    have hlo'' : (lo : Real) ≤
        (gamma i : Real) * ((centered : Real) * invStd) + (beta i : Real) := by
      simpa [lo] using add_le_add_right hlo' (beta i : Real)
    simpa [hreal] using hlo''
  have hhi : layerNormReal eps gamma beta x i ≤ (hi : Real) := by
    have hhi' : (gamma i : Real) * ((centered : Real) * invStd) ≤ (sb.2 : Real) := hsb.2
    have hhi'' :
        (gamma i : Real) * ((centered : Real) * invStd) + (beta i : Real) ≤ (hi : Real) := by
      simpa [hi] using add_le_add_right hhi' (beta i : Real)
    simpa [hreal] using hhi''
  simpa [bounds, layerNormBounds, hne, μ, varEps, invLo, invHi, centered, nb, sb, lo, hi] using
    And.intro hlo hhi

end Bounds

end Sound

end Nfp
