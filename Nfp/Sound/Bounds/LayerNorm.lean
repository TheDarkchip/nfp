-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.Order.Ring.Basic
import Mathlib.Data.Nat.Sqrt
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Rat.BigOperators
import Mathlib.Data.Rat.Cast.Order
import Nfp.Core.Basic
import Nfp.Sound.Bounds.LayerNorm.MeanVariance
import Nfp.Sound.Linear.FinFold

/-!
LayerNorm interval bounds for dyadic inputs.

This module computes dyadic interval bounds for LayerNorm outputs and proves
those bounds sound for real-valued LayerNorm semantics.
-/

namespace Nfp

namespace Sound

namespace Bounds

open scoped BigOperators

/-! Square-root bounds. -/

lemma dyadic_nat_cast_nonneg (n : Nat) : (0 : Dyadic) ≤ (n : Dyadic) := by
  simp

lemma dyadic_nat_cast_pos {n : Nat} (h : 0 < n) : (0 : Dyadic) < (n : Dyadic) := by
  exact (Nat.cast_pos (α := Dyadic)).2 h

/-- Base rational lower bound for a square root. -/
def sqrtLowerBase (q : Dyadic) : Dyadic :=
  let num := q.toRat.num.natAbs
  let den := q.toRat.den
  let a := Nat.sqrt num
  let b := Nat.sqrt den
  dyadicOfRatDown ((a : Rat) / (b + 1))

/-- Base rational upper bound for a square root. -/
def sqrtUpperBase (q : Dyadic) : Dyadic :=
  let num := q.toRat.num.natAbs
  let den := q.toRat.den
  let a := Nat.sqrt num
  let b := Nat.sqrt den
  dyadicOfRatUp ((a + 1 : Rat) / b)

/-- Alternate rational lower bound for a square root. -/
def sqrtLowerAlt (q : Dyadic) : Dyadic :=
  let num := q.toRat.num.natAbs
  let den := q.toRat.den
  let a := Nat.sqrt (num * den)
  dyadicOfRatDown ((a : Rat) / den)

/-- Alternate rational upper bound for a square root. -/
def sqrtUpperAlt (q : Dyadic) : Dyadic :=
  let num := q.toRat.num.natAbs
  let den := q.toRat.den
  let a := Nat.sqrt (num * den)
  dyadicOfRatUp ((a + 1 : Rat) / den)

/-- Dyadicional lower bound for a square root (tighter of two bounds). -/
def sqrtLower (q : Dyadic) : Dyadic :=
  max (sqrtLowerBase q) (sqrtLowerAlt q)

/-- Dyadicional upper bound for a square root (tighter of two bounds). -/
def sqrtUpper (q : Dyadic) : Dyadic :=
  min (sqrtUpperBase q) (sqrtUpperAlt q)

/-- `sqrtLowerBase` is nonnegative. -/
theorem sqrtLowerBase_nonneg (q : Dyadic) : 0 ≤ sqrtLowerBase q := by
  classical
  unfold sqrtLowerBase
  have hnum : 0 ≤ (Nat.sqrt q.toRat.num.natAbs : Rat) := by
    exact_mod_cast (Nat.zero_le (Nat.sqrt q.toRat.num.natAbs))
  have hden : 0 ≤ (Nat.sqrt q.toRat.den + 1 : Rat) := by
    exact_mod_cast (Nat.zero_le (Nat.sqrt q.toRat.den + 1))
  have hrat : 0 ≤ (Nat.sqrt q.toRat.num.natAbs : Rat) / (Nat.sqrt q.toRat.den + 1) := by
    exact div_nonneg hnum hden
  exact dyadicOfRatDown_nonneg hrat

/-! Strict positivity helpers. -/

/-! Base bounds. -/


/-- `sqrtUpperBase` is nonnegative. -/
theorem sqrtUpperBase_nonneg (q : Dyadic) : 0 ≤ sqrtUpperBase q := by
  classical
  unfold sqrtUpperBase
  have hnum : 0 ≤ (Nat.sqrt q.toRat.num.natAbs + 1 : Rat) := by
    exact_mod_cast (Nat.zero_le (Nat.sqrt q.toRat.num.natAbs + 1))
  have hden : 0 ≤ (Nat.sqrt q.toRat.den : Rat) := by
    exact_mod_cast (Nat.zero_le (Nat.sqrt q.toRat.den))
  have hrat :
      0 ≤ (Nat.sqrt q.toRat.num.natAbs + 1 : Rat) / (Nat.sqrt q.toRat.den) := by
    exact div_nonneg hnum hden
  exact dyadicOfRatUp_nonneg hrat

/-- `sqrtUpperBase` is always positive. -/
theorem sqrtUpperBase_pos (q : Dyadic) : 0 < sqrtUpperBase q := by
  classical
  unfold sqrtUpperBase
  have hnum_pos : (0 : Rat) < (Nat.sqrt q.toRat.num.natAbs + 1 : Rat) := by
    exact_mod_cast (Nat.succ_pos (Nat.sqrt q.toRat.num.natAbs))
  have hden_pos : (0 : Rat) < (Nat.sqrt q.toRat.den : Rat) := by
    have hden : 0 < q.toRat.den := q.toRat.den_pos
    exact_mod_cast (Nat.sqrt_pos.2 hden)
  have hrat_pos :
      (0 : Rat) < (Nat.sqrt q.toRat.num.natAbs + 1 : Rat) / (Nat.sqrt q.toRat.den) := by
    exact div_pos hnum_pos hden_pos
  exact dyadicOfRatUp_pos hrat_pos

/-! Alternate bounds. -/

/-- `sqrtLowerAlt` is nonnegative. -/
theorem sqrtLowerAlt_nonneg (q : Dyadic) : 0 ≤ sqrtLowerAlt q := by
  classical
  unfold sqrtLowerAlt
  have hnum : 0 ≤ (Nat.sqrt (q.toRat.num.natAbs * q.toRat.den) : Rat) := by
    exact_mod_cast (Nat.zero_le (Nat.sqrt (q.toRat.num.natAbs * q.toRat.den)))
  have hden : 0 ≤ (q.toRat.den : Rat) := by
    exact_mod_cast (Nat.zero_le q.toRat.den)
  have hrat :
      0 ≤ (Nat.sqrt (q.toRat.num.natAbs * q.toRat.den) : Rat) / q.toRat.den := by
    exact div_nonneg hnum hden
  exact dyadicOfRatDown_nonneg hrat


/-- `sqrtUpperAlt` is nonnegative. -/
theorem sqrtUpperAlt_nonneg (q : Dyadic) : 0 ≤ sqrtUpperAlt q := by
  classical
  unfold sqrtUpperAlt
  have hnum : 0 ≤ (Nat.sqrt (q.toRat.num.natAbs * q.toRat.den) + 1 : Rat) := by
    exact_mod_cast (Nat.zero_le (Nat.sqrt (q.toRat.num.natAbs * q.toRat.den) + 1))
  have hden : 0 ≤ (q.toRat.den : Rat) := by
    exact_mod_cast (Nat.zero_le q.toRat.den)
  have hrat :
      0 ≤ (Nat.sqrt (q.toRat.num.natAbs * q.toRat.den) + 1 : Rat) / q.toRat.den := by
    exact div_nonneg hnum hden
  exact dyadicOfRatUp_nonneg hrat

/-- `sqrtUpperAlt` is always positive. -/
theorem sqrtUpperAlt_pos (q : Dyadic) : 0 < sqrtUpperAlt q := by
  classical
  unfold sqrtUpperAlt
  have hnum_pos :
      (0 : Rat) < (Nat.sqrt (q.toRat.num.natAbs * q.toRat.den) + 1 : Rat) := by
    exact_mod_cast (Nat.succ_pos (Nat.sqrt (q.toRat.num.natAbs * q.toRat.den)))
  have hden_pos : (0 : Rat) < (q.toRat.den : Rat) := by
    exact_mod_cast q.toRat.den_pos
  have hrat_pos :
      (0 : Rat) <
        (Nat.sqrt (q.toRat.num.natAbs * q.toRat.den) + 1 : Rat) / q.toRat.den := by
    exact div_pos hnum_pos hden_pos
  exact dyadicOfRatUp_pos hrat_pos

/-! Combined bounds. -/

/-- `sqrtLower` is nonnegative. -/
theorem sqrtLower_nonneg (q : Dyadic) : 0 ≤ sqrtLower q := by
  have hbase : 0 ≤ sqrtLowerBase q := sqrtLowerBase_nonneg q
  exact le_trans hbase (le_max_left _ _)


/-- `sqrtUpper` is nonnegative. -/
theorem sqrtUpper_nonneg (q : Dyadic) : 0 ≤ sqrtUpper q := by
  have hbase : 0 ≤ sqrtUpperBase q := sqrtUpperBase_nonneg q
  have halt : 0 ≤ sqrtUpperAlt q := sqrtUpperAlt_nonneg q
  exact le_min hbase halt

/-- `sqrtUpper` is always positive. -/
theorem sqrtUpper_pos (q : Dyadic) : 0 < sqrtUpper q := by
  have hbase : 0 < sqrtUpperBase q := sqrtUpperBase_pos q
  have halt : 0 < sqrtUpperAlt q := sqrtUpperAlt_pos q
  exact lt_min hbase halt

/-- Square-root lower bound in reals. -/
theorem sqrtLowerBase_le_real_sqrt {q : Dyadic} (hq : 0 ≤ q) :
    (sqrtLowerBase q : Real) ≤ Real.sqrt (q : Real) := by
  classical
  -- Set up numerator/denominator witnesses.
  set num : Nat := q.toRat.num.natAbs
  set den : Nat := q.toRat.den
  set a : Nat := Nat.sqrt num
  set b : Nat := Nat.sqrt den
  have hden_pos : 0 < (den : Real) := by
    exact_mod_cast q.toRat.den_pos
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
    have hnum_nonneg : 0 ≤ q.toRat.num := by
      have hq' : (0 : Rat) ≤ q.toRat :=
        (Dyadic.toRat_le_toRat_iff (x := 0) (y := q)).2 hq
      exact (Rat.num_nonneg (q := q.toRat)).2 hq'
    have hnum_eq : (num : Int) = q.toRat.num := by
      simpa [num] using (Int.natAbs_of_nonneg hnum_nonneg)
    have hnum_cast : (q.toRat.num : Real) = (num : Real) := by
      exact_mod_cast hnum_eq.symm
    have hq_rat : (q : Real) = (q.toRat.num : Real) / q.toRat.den := by
      simp [dyadicToReal, Rat.cast_def]
    simpa [hnum_cast, den] using hq_rat
  have hsq : ((a : Real) / (b + 1 : Real)) ^ 2 ≤ (q : Real) := by
    simpa [hpow, hq_cast, den, num] using hdiv
  have hnonneg : 0 ≤ (a : Real) / (b + 1 : Real) := by
    have hnum_nonneg : 0 ≤ (a : Real) := by exact_mod_cast (Nat.zero_le a)
    have hden_nonneg : 0 ≤ (b + 1 : Real) := by exact_mod_cast (Nat.zero_le (b + 1))
    exact div_nonneg hnum_nonneg hden_nonneg
  have hq_nonneg : 0 ≤ (q : Real) := by
    exact dyadicToReal_nonneg_of_nonneg hq
  have hle : (a : Real) / (b + 1 : Real) ≤ Real.sqrt (q : Real) :=
    (Real.le_sqrt hnonneg hq_nonneg).2 hsq
  have hdown :
      (sqrtLowerBase q : Real) ≤ (a : Real) / (b + 1 : Real) := by
    have hdown' :
        dyadicToReal (dyadicOfRatDown ((a : Rat) / (b + 1))) ≤
          (a : Real) / (b + 1 : Real) := by
      simpa using dyadicOfRatDown_le_real ((a : Rat) / (b + 1))
    simpa [sqrtLowerBase, num, den, a, b] using hdown'
  exact le_trans hdown hle

/-- Square-root upper bound in reals. -/
theorem real_sqrt_le_sqrtUpperBase {q : Dyadic} (hq : 0 ≤ q) :
    Real.sqrt (q : Real) ≤ (sqrtUpperBase q : Real) := by
  classical
  set num : Nat := q.toRat.num.natAbs
  set den : Nat := q.toRat.den
  set a : Nat := Nat.sqrt num
  set b : Nat := Nat.sqrt den
  have hden_pos : 0 < (den : Real) := by
    exact_mod_cast q.toRat.den_pos
  have hbpos : 0 < (b : Real) := by
    have hb : 0 < b := by
      have hden : 0 < den := q.toRat.den_pos
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
    have hnum_nonneg : 0 ≤ q.toRat.num := by
      have hq' : (0 : Rat) ≤ q.toRat :=
        (Dyadic.toRat_le_toRat_iff (x := 0) (y := q)).2 hq
      exact (Rat.num_nonneg (q := q.toRat)).2 hq'
    have hnum_eq : (num : Int) = q.toRat.num := by
      simpa [num] using (Int.natAbs_of_nonneg hnum_nonneg)
    have hnum_cast : (q.toRat.num : Real) = (num : Real) := by
      exact_mod_cast hnum_eq.symm
    have hq_rat : (q : Real) = (q.toRat.num : Real) / q.toRat.den := by
      simp [dyadicToReal, Rat.cast_def]
    simpa [hnum_cast, den] using hq_rat
  have hsq : (q : Real) ≤ ((a + 1 : Real) / (b : Real)) ^ 2 := by
    simpa [hpow, hq_cast, den, num] using hdiv
  have hnonneg : 0 ≤ ((a + 1 : Real) / (b : Real)) := by
    have hnum_nonneg : 0 ≤ (a + 1 : Real) := by exact_mod_cast (Nat.zero_le (a + 1))
    have hden_nonneg : 0 ≤ (b : Real) := by exact_mod_cast (Nat.zero_le b)
    exact div_nonneg hnum_nonneg hden_nonneg
  have hle : Real.sqrt (q : Real) ≤ (a + 1 : Real) / (b : Real) :=
    (Real.sqrt_le_iff).2 ⟨hnonneg, hsq⟩
  have hup :
      (a + 1 : Real) / (b : Real) ≤ (sqrtUpperBase q : Real) := by
    have hup' :
        (a + 1 : Real) / (b : Real) ≤
          dyadicToReal (dyadicOfRatUp ((a + 1 : Rat) / b)) := by
      simpa using real_le_dyadicOfRatUp ((a + 1 : Rat) / b)
    simpa [sqrtUpperBase, num, den, a, b] using hup'
  exact le_trans hle hup

/-- Alternate square-root lower bound in reals. -/
theorem sqrtLowerAlt_le_real_sqrt {q : Dyadic} (hq : 0 ≤ q) :
    (sqrtLowerAlt q : Real) ≤ Real.sqrt (q : Real) := by
  classical
  set num : Nat := q.toRat.num.natAbs
  set den : Nat := q.toRat.den
  set a : Nat := Nat.sqrt (num * den)
  have hden_pos : 0 < (den : Real) := by
    exact_mod_cast q.toRat.den_pos
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
    exact_mod_cast q.toRat.den_pos.ne'
  have hq_cast : (q : Real) = (num : Real) * den / (den : Real) ^ 2 := by
    have hnum_nonneg : 0 ≤ q.toRat.num := by
      have hq' : (0 : Rat) ≤ q.toRat :=
        (Dyadic.toRat_le_toRat_iff (x := 0) (y := q)).2 hq
      exact (Rat.num_nonneg (q := q.toRat)).2 hq'
    have hnum_eq : (num : Int) = q.toRat.num := by
      simpa [num] using (Int.natAbs_of_nonneg hnum_nonneg)
    have hnum_cast : (q.toRat.num : Real) = (num : Real) := by
      exact_mod_cast hnum_eq.symm
    have hq_rat : (q : Real) = (q.toRat.num : Real) / q.toRat.den := by
      simp [dyadicToReal, Rat.cast_def]
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
  have hq_nonneg : 0 ≤ (q : Real) := by
    exact dyadicToReal_nonneg_of_nonneg hq
  have hle : (a : Real) / (den : Real) ≤ Real.sqrt (q : Real) :=
    (Real.le_sqrt hnonneg hq_nonneg).2 hsq
  have hdown :
      (sqrtLowerAlt q : Real) ≤ (a : Real) / (den : Real) := by
    have hdown' :
        dyadicToReal (dyadicOfRatDown ((a : Rat) / den)) ≤
          (a : Real) / (den : Real) := by
      simpa using dyadicOfRatDown_le_real ((a : Rat) / den)
    simpa [sqrtLowerAlt, num, den, a] using hdown'
  exact le_trans hdown hle

/-- Alternate square-root upper bound in reals. -/
theorem real_sqrt_le_sqrtUpperAlt {q : Dyadic} (hq : 0 ≤ q) :
    Real.sqrt (q : Real) ≤ (sqrtUpperAlt q : Real) := by
  classical
  set num : Nat := q.toRat.num.natAbs
  set den : Nat := q.toRat.den
  set a : Nat := Nat.sqrt (num * den)
  have hden_pos : 0 < (den : Real) := by
    exact_mod_cast q.toRat.den_pos
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
    exact_mod_cast q.toRat.den_pos.ne'
  have hq_cast : (q : Real) = (num : Real) * den / (den : Real) ^ 2 := by
    have hnum_nonneg : 0 ≤ q.toRat.num := by
      have hq' : (0 : Rat) ≤ q.toRat :=
        (Dyadic.toRat_le_toRat_iff (x := 0) (y := q)).2 hq
      exact (Rat.num_nonneg (q := q.toRat)).2 hq'
    have hnum_eq : (num : Int) = q.toRat.num := by
      simpa [num] using (Int.natAbs_of_nonneg hnum_nonneg)
    have hnum_cast : (q.toRat.num : Real) = (num : Real) := by
      exact_mod_cast hnum_eq.symm
    have hq_rat : (q : Real) = (q.toRat.num : Real) / q.toRat.den := by
      simp [dyadicToReal, Rat.cast_def]
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
  have hup :
      (a + 1 : Real) / (den : Real) ≤ (sqrtUpperAlt q : Real) := by
    have hup' :
        (a + 1 : Real) / (den : Real) ≤
          dyadicToReal (dyadicOfRatUp ((a + 1 : Rat) / den)) := by
      simpa using real_le_dyadicOfRatUp ((a + 1 : Rat) / den)
    simpa [sqrtUpperAlt, num, den, a] using hup'
  exact le_trans hle hup

/-- Square-root lower bound in reals (tighter of two bounds). -/
theorem sqrtLower_le_real_sqrt {q : Dyadic} (hq : 0 ≤ q) :
    (sqrtLower q : Real) ≤ Real.sqrt (q : Real) := by
  have hbase := sqrtLowerBase_le_real_sqrt (q := q) hq
  have halt := sqrtLowerAlt_le_real_sqrt (q := q) hq
  simpa [sqrtLower] using (max_le_iff).2 ⟨hbase, halt⟩

/-- Square-root upper bound in reals (tighter of two bounds). -/
theorem real_sqrt_le_sqrtUpper {q : Dyadic} (hq : 0 ≤ q) :
    Real.sqrt (q : Real) ≤ (sqrtUpper q : Real) := by
  have hbase := real_sqrt_le_sqrtUpperBase (q := q) hq
  have halt := real_sqrt_le_sqrtUpperAlt (q := q) hq
  simpa [sqrtUpper] using (le_min_iff).2 ⟨hbase, halt⟩

/-- Bounds for multiplying a scalar by a bounded value. -/
def scaleInterval (x lo hi : Dyadic) : Dyadic × Dyadic :=
  if 0 ≤ x then
    (x * lo, x * hi)
  else
    (x * hi, x * lo)

/-- `scaleInterval` bounds a product. -/
theorem scaleInterval_bounds {x lo hi y : Dyadic}
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
theorem scaleInterval_bounds_real {x lo hi : Dyadic} {y : Real}
    (hlo : (lo : Real) ≤ y) (hhi : y ≤ (hi : Real)) :
    let bounds := scaleInterval x lo hi
    (bounds.1 : Real) ≤ (x : Real) * y ∧ (x : Real) * y ≤ (bounds.2 : Real) := by
  by_cases hx : 0 ≤ x
  · have h1 : (x : Real) * (lo : Real) ≤ (x : Real) * y := by
      have hx' : 0 ≤ (x : Real) := dyadicToReal_nonneg_of_nonneg hx
      exact mul_le_mul_of_nonneg_left hlo hx'
    have h2 : (x : Real) * y ≤ (x : Real) * (hi : Real) := by
      have hx' : 0 ≤ (x : Real) := dyadicToReal_nonneg_of_nonneg hx
      exact mul_le_mul_of_nonneg_left hhi hx'
    simp [scaleInterval, hx, h1, h2]
  · have hx' : x ≤ 0 := le_of_lt (lt_of_not_ge hx)
    have h1 : (x : Real) * (hi : Real) ≤ (x : Real) * y := by
      have hx'' : (x : Real) ≤ 0 := (dyadicToReal_nonpos_iff (x := x)).2 hx'
      exact mul_le_mul_of_nonpos_left hhi hx''
    have h2 : (x : Real) * y ≤ (x : Real) * (lo : Real) := by
      have hx'' : (x : Real) ≤ 0 := (dyadicToReal_nonpos_iff (x := x)).2 hx'
      exact mul_le_mul_of_nonpos_left hlo hx''
    simp [scaleInterval, hx, h1, h2]

/-- Real-valued LayerNorm output for a vector. -/
noncomputable def layerNormReal {n : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic) (x : Fin n → Dyadic) : Fin n → Real :=
  if n = 0 then
    fun _ => 0
  else
    let μ : Real := meanRat x
    let varEps : Real := (varianceRat x : Real) + (eps : Real)
    let invStd : Real := (Real.sqrt varEps)⁻¹
    fun i => (gamma i : Real) * ((x i : Real) - μ) * invStd + (beta i : Real)

/-- Real-valued LayerNorm output for a real vector. -/
noncomputable def layerNormRealOfReal {n : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic) (x : Fin n → Real) : Fin n → Real :=
  if n = 0 then
    fun _ => 0
  else
    let μ : Real := meanReal x
    let varEps : Real := varianceReal x + (eps : Real)
    let invStd : Real := (Real.sqrt varEps)⁻¹
    fun i => (gamma i : Real) * (x i - μ) * invStd + (beta i : Real)

/-- Interval bounds for LayerNorm outputs. -/
def layerNormBounds {n : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic) (x : Fin n → Dyadic) :
    (Fin n → Dyadic) × (Fin n → Dyadic) :=
  if n = 0 then
    (fun _ => 0, fun _ => 0)
  else
    let μLo := mean x
    let μHi := meanUpper x
    let centeredBound : Fin n → Dyadic := fun i =>
      max |x i - μHi| |x i - μLo|
    let invStdBound : Dyadic := dyadicDivUp 1 (sqrtLower eps)
    let radius : Fin n → Dyadic := fun i => |gamma i| * centeredBound i * invStdBound
    (fun i => beta i - radius i, fun i => beta i + radius i)

/-- `layerNormBounds` soundness for real LayerNorm outputs. -/
theorem layerNormBounds_spec {n : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic) (x : Fin n → Dyadic)
    (hne : n ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps) :
    let bounds := layerNormBounds eps gamma beta x
    ∀ i,
      (bounds.1 i : Real) ≤ layerNormReal eps gamma beta x i ∧
        layerNormReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds i
  let μLo : Dyadic := mean x
  let μHi : Dyadic := meanUpper x
  let centeredBound : Fin n → Dyadic := fun j => max |x j - μHi| |x j - μLo|
  let invStdBound : Dyadic := dyadicDivUp 1 (sqrtLower eps)
  let varEps : Real := (varianceRat x : Real) + (eps : Real)
  let μ : Real := meanRat x
  let invStd : Real := (Real.sqrt varEps)⁻¹
  have hcentered_nonneg : 0 ≤ (centeredBound i : Real) := by
    have h0 : 0 ≤ centeredBound i := by
      dsimp [centeredBound]
      exact le_trans (abs_nonneg _) (le_max_left _ _)
    exact dyadicToReal_nonneg_of_nonneg h0
  have hmean_lo_real : (μLo : Real) ≤ μ := by
    have h := dyadicOfRatDown_le_real (meanRat x)
    simpa [μLo, μ, mean_def x hne] using h
  have hmean_hi_real : μ ≤ (μHi : Real) := by
    have h := real_le_dyadicOfRatUp (meanRat x)
    simpa [μHi, μ, meanUpper_def x hne] using h
  have hcentered_abs : |(x i : Real) - μ| ≤ (centeredBound i : Real) := by
    have hlo : (x i : Real) - (μHi : Real) ≤ (x i : Real) - μ := by
      exact sub_le_sub_left hmean_hi_real (x i : Real)
    have hhi : (x i : Real) - μ ≤ (x i : Real) - (μLo : Real) := by
      exact sub_le_sub_left hmean_lo_real (x i : Real)
    have hbound := abs_le_max_of_bounds hlo hhi
    simpa [centeredBound, μLo, μHi, dyadicToReal_abs, dyadicToReal_sub,
      dyadicToReal_max] using hbound
  have hvar_nonneg : 0 ≤ (varianceRat x : Real) := varianceRat_nonneg_real x hne
  have hsqrt_lower :
      (sqrtLower eps : Real) ≤ Real.sqrt varEps := by
    have hsqrt_eps :
        (sqrtLower eps : Real) ≤ Real.sqrt (eps : Real) := by
      have h := sqrtLower_le_real_sqrt (q := eps) (by exact le_of_lt heps)
      simpa using h
    have hle : (eps : Real) ≤ varEps := by
      have hle' : (eps : Real) ≤ (varianceRat x : Real) + (eps : Real) :=
        le_add_of_nonneg_left hvar_nonneg
      simpa [varEps] using hle'
    have hsqrt_eps' : Real.sqrt (eps : Real) ≤ Real.sqrt varEps := by
      exact Real.sqrt_le_sqrt hle
    exact le_trans hsqrt_eps hsqrt_eps'
  have hsqrt_lower_pos : 0 < (sqrtLower eps : Real) := by
    simpa [dyadicToReal_zero] using
      (dyadicToReal_lt_iff (x := 0) (y := sqrtLower eps)).2 hsqrt
  have hinv_sqrt : invStd ≤ (sqrtLower eps : Real)⁻¹ := by
    have h := inv_anti₀ hsqrt_lower_pos hsqrt_lower
    simpa [invStd] using h
  have hinv_bound : (sqrtLower eps : Real)⁻¹ ≤ (invStdBound : Real) := by
    have hy : sqrtLower eps ≠ 0 := ne_of_gt hsqrt
    have hdiv := dyadicDivUp_ge_real (x := 1) (y := sqrtLower eps) hy
    simpa [invStdBound, one_div] using hdiv
  have hinv : invStd ≤ (invStdBound : Real) := by
    exact le_trans hinv_sqrt hinv_bound
  have hinv_nonneg : 0 ≤ invStd := by
    have hsqrt_nonneg : 0 ≤ Real.sqrt varEps := by
      exact Real.sqrt_nonneg _
    exact inv_nonneg.2 hsqrt_nonneg
  have hmul1 : |(x i : Real) - μ| * invStd ≤
      (centeredBound i : Real) * (invStdBound : Real) := by
    have hleft :
        |(x i : Real) - μ| * invStd ≤ (centeredBound i : Real) * invStd := by
      exact mul_le_mul_of_nonneg_right hcentered_abs hinv_nonneg
    have hright :
        (centeredBound i : Real) * invStd ≤ (centeredBound i : Real) * (invStdBound : Real) := by
      exact mul_le_mul_of_nonneg_left hinv hcentered_nonneg
    exact le_trans hleft hright
  have hmul2 : |(gamma i : Real)| * |(x i : Real) - μ| * invStd ≤
      |(gamma i : Real)| * (centeredBound i : Real) * (invStdBound : Real) := by
    have hgamma_nonneg : 0 ≤ |(gamma i : Real)| := abs_nonneg _
    have hmul2' : |(gamma i : Real)| * (|(x i : Real) - μ| * invStd) ≤
        |(gamma i : Real)| * ((centeredBound i : Real) * (invStdBound : Real)) := by
      exact mul_le_mul_of_nonneg_left hmul1 hgamma_nonneg
    simpa [mul_assoc] using hmul2'
  let t : Real := (gamma i : Real) * ((x i : Real) - μ) * invStd
  have ht_abs :
      |t| ≤ |(gamma i : Real)| * (centeredBound i : Real) * (invStdBound : Real) := by
    have ht : |t| = |(gamma i : Real)| * |(x i : Real) - μ| * invStd := by
      have hinv_abs : |invStd| = invStd := abs_of_nonneg hinv_nonneg
      simp [t, abs_mul, hinv_abs, mul_assoc]
    simpa [ht] using hmul2
  let radius : Fin n → Dyadic := fun j => |gamma j| * centeredBound j * invStdBound
  have ht_abs' : |t| ≤ (radius i : Real) := by
    simpa [radius, centeredBound, invStdBound] using ht_abs
  have hbounds : -(radius i : Real) ≤ t ∧ t ≤ (radius i : Real) := by
    exact abs_le.mp ht_abs'
  have hlow :
      (beta i : Real) - (radius i : Real) ≤ t + (beta i : Real) := by
    have h := add_le_add_left hbounds.1 (beta i : Real)
    simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using h
  have hhigh :
      t + (beta i : Real) ≤ (beta i : Real) + (radius i : Real) := by
    have h := add_le_add_left hbounds.2 (beta i : Real)
    simpa [add_comm, add_left_comm, add_assoc] using h
  have hreal :
      layerNormReal eps gamma beta x i = t + (beta i : Real) := by
    simp [layerNormReal, hne, μ, invStd, varEps, t, add_comm]
  have hlo : (bounds.1 i : Real) ≤ layerNormReal eps gamma beta x i := by
    simpa [bounds, layerNormBounds, hne, radius, centeredBound, invStdBound, μLo, μHi,
      hreal] using hlow
  have hhi : layerNormReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
    simpa [bounds, layerNormBounds, hne, radius, centeredBound, invStdBound, μLo, μHi,
      hreal] using hhigh
  exact And.intro hlo hhi

/-- Interval bounds for LayerNorm outputs from per-coordinate intervals. -/
def layerNormIntervalBounds {n : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic) (lo hi : Fin n → Dyadic) :
    (Fin n → Dyadic) × (Fin n → Dyadic) :=
  if n = 0 then
    (fun _ => 0, fun _ => 0)
  else
    let μLo := mean lo
    let μHi := meanUpper hi
    let centeredBound : Fin n → Dyadic := fun i =>
      max |lo i - μHi| |hi i - μLo|
    let invStdBound : Dyadic := dyadicDivUp 1 (sqrtLower eps)
    let radius : Fin n → Dyadic := fun i => |gamma i| * centeredBound i * invStdBound
    (fun i => beta i - radius i, fun i => beta i + radius i)

/-- `layerNormIntervalBounds` soundness for real LayerNorm outputs. -/
theorem layerNormIntervalBounds_spec {n : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic) (lo hi : Fin n → Dyadic) (x : Fin n → Dyadic)
    (hne : n ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (hlo : ∀ i, lo i ≤ x i) (hhi : ∀ i, x i ≤ hi i) :
    let bounds := layerNormIntervalBounds eps gamma beta lo hi
    ∀ i,
      (bounds.1 i : Real) ≤ layerNormReal eps gamma beta x i ∧
        layerNormReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds i
  let μLo : Dyadic := mean lo
  let μHi : Dyadic := meanUpper hi
  let centeredBound : Fin n → Dyadic := fun j => max |lo j - μHi| |hi j - μLo|
  let invStdBound : Dyadic := dyadicDivUp 1 (sqrtLower eps)
  let varEps : Real := (varianceRat x : Real) + (eps : Real)
  let μ : Real := meanRat x
  let invStd : Real := (Real.sqrt varEps)⁻¹
  have hcentered_nonneg : 0 ≤ (centeredBound i : Real) := by
    have h0 : 0 ≤ centeredBound i := by
      dsimp [centeredBound]
      exact le_trans (abs_nonneg _) (le_max_left _ _)
    exact dyadicToReal_nonneg_of_nonneg h0
  have hcentered_abs : |(x i : Real) - μ| ≤ (centeredBound i : Real) := by
    have hmean_lo_real : (μLo : Real) ≤ μ := by
      have hmean_rat : (meanRat lo : Real) ≤ (meanRat x : Real) :=
        meanRat_le_meanRat_real lo x hne hlo
      have hdown : (μLo : Real) ≤ (meanRat lo : Real) := by
        simpa [μLo, mean_def lo hne] using dyadicOfRatDown_le_real (meanRat lo)
      exact le_trans hdown hmean_rat
    have hmean_hi_real : μ ≤ (μHi : Real) := by
      have hmean_rat : (meanRat x : Real) ≤ (meanRat hi : Real) :=
        meanRat_le_meanRat_real x hi hne hhi
      have hup : (meanRat hi : Real) ≤ (μHi : Real) := by
        simpa [μHi, meanUpper_def hi hne] using real_le_dyadicOfRatUp (meanRat hi)
      exact le_trans hmean_rat hup
    have hlo' : (lo i : Real) - (μHi : Real) ≤ (x i : Real) - μ := by
      have h1 : (lo i : Real) - (μHi : Real) ≤ (lo i : Real) - μ := by
        exact sub_le_sub_left hmean_hi_real (lo i : Real)
      have h2 : (lo i : Real) - μ ≤ (x i : Real) - μ := by
        exact sub_le_sub_right
          (by
            simpa using dyadicToReal_le_of_le (hlo i))
          μ
      exact le_trans h1 h2
    have hhi' : (x i : Real) - μ ≤ (hi i : Real) - (μLo : Real) := by
      have h1 : (x i : Real) - μ ≤ (hi i : Real) - μ := by
        exact sub_le_sub_right
          (by
            simpa using dyadicToReal_le_of_le (hhi i))
          μ
      have h2 : (hi i : Real) - μ ≤ (hi i : Real) - (μLo : Real) := by
        exact sub_le_sub_left hmean_lo_real (hi i : Real)
      exact le_trans h1 h2
    have hbound := abs_le_max_of_bounds hlo' hhi'
    simpa [centeredBound, μLo, μHi, dyadicToReal_abs, dyadicToReal_sub,
      dyadicToReal_max] using hbound
  have hsqrt_lower :
      (sqrtLower eps : Real) ≤ Real.sqrt varEps := by
    have hsqrt_eps :
        (sqrtLower eps : Real) ≤ Real.sqrt (eps : Real) := by
      have h := sqrtLower_le_real_sqrt (q := eps) (by exact le_of_lt heps)
      simpa using h
    have hvar_nonneg : 0 ≤ (varianceRat x : Real) := varianceRat_nonneg_real x hne
    have hle : (eps : Real) ≤ varEps := by
      have hle' : (eps : Real) ≤ (varianceRat x : Real) + (eps : Real) :=
        le_add_of_nonneg_left hvar_nonneg
      simpa [varEps] using hle'
    have hsqrt_eps' : Real.sqrt (eps : Real) ≤ Real.sqrt varEps := by
      exact Real.sqrt_le_sqrt hle
    exact le_trans hsqrt_eps hsqrt_eps'
  have hsqrt_lower_pos : 0 < (sqrtLower eps : Real) := by
    simpa [dyadicToReal_zero] using
      (dyadicToReal_lt_iff (x := 0) (y := sqrtLower eps)).2 hsqrt
  have hinv_sqrt : invStd ≤ (sqrtLower eps : Real)⁻¹ := by
    have h := inv_anti₀ hsqrt_lower_pos hsqrt_lower
    simpa [invStd] using h
  have hinv_bound : (sqrtLower eps : Real)⁻¹ ≤ (invStdBound : Real) := by
    have hy : sqrtLower eps ≠ 0 := ne_of_gt hsqrt
    have hdiv := dyadicDivUp_ge_real (x := 1) (y := sqrtLower eps) hy
    simpa [invStdBound, one_div] using hdiv
  have hinv : invStd ≤ (invStdBound : Real) := by
    exact le_trans hinv_sqrt hinv_bound
  have hinv_nonneg : 0 ≤ invStd := by
    have hsqrt_nonneg : 0 ≤ Real.sqrt varEps := by
      exact Real.sqrt_nonneg _
    exact inv_nonneg.2 hsqrt_nonneg
  have hmul1 : |(x i : Real) - μ| * invStd ≤
      (centeredBound i : Real) * (invStdBound : Real) := by
    have hleft :
        |(x i : Real) - μ| * invStd ≤ (centeredBound i : Real) * invStd := by
      exact mul_le_mul_of_nonneg_right hcentered_abs hinv_nonneg
    have hright :
        (centeredBound i : Real) * invStd ≤ (centeredBound i : Real) * (invStdBound : Real) := by
      exact mul_le_mul_of_nonneg_left hinv hcentered_nonneg
    exact le_trans hleft hright
  have hmul2 : |(gamma i : Real)| * |(x i : Real) - μ| * invStd ≤
      |(gamma i : Real)| * (centeredBound i : Real) * (invStdBound : Real) := by
    have hgamma_nonneg : 0 ≤ |(gamma i : Real)| := abs_nonneg _
    have hmul2' : |(gamma i : Real)| * (|(x i : Real) - μ| * invStd) ≤
        |(gamma i : Real)| * ((centeredBound i : Real) * (invStdBound : Real)) := by
      exact mul_le_mul_of_nonneg_left hmul1 hgamma_nonneg
    simpa [mul_assoc] using hmul2'
  let t : Real := (gamma i : Real) * ((x i : Real) - μ) * invStd
  have ht_abs :
      |t| ≤ |(gamma i : Real)| * (centeredBound i : Real) * (invStdBound : Real) := by
    have ht : |t| = |(gamma i : Real)| * |(x i : Real) - μ| * invStd := by
      have hinv_abs : |invStd| = invStd := abs_of_nonneg hinv_nonneg
      simp [t, abs_mul, hinv_abs, mul_assoc]
    simpa [ht] using hmul2
  let radius : Fin n → Dyadic := fun j => |gamma j| * centeredBound j * invStdBound
  have ht_abs' : |t| ≤ (radius i : Real) := by
    simpa [radius, centeredBound, invStdBound] using ht_abs
  have hbounds : -(radius i : Real) ≤ t ∧ t ≤ (radius i : Real) := by
    exact abs_le.mp ht_abs'
  have hlow :
      (beta i : Real) - (radius i : Real) ≤ t + (beta i : Real) := by
    have h := add_le_add_left hbounds.1 (beta i : Real)
    simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using h
  have hhigh :
      t + (beta i : Real) ≤ (beta i : Real) + (radius i : Real) := by
    have h := add_le_add_left hbounds.2 (beta i : Real)
    simpa [add_comm, add_left_comm, add_assoc] using h
  have hreal :
      layerNormReal eps gamma beta x i = t + (beta i : Real) := by
    simp [layerNormReal, hne, μ, invStd, varEps, t, add_comm]
  have hlo : (bounds.1 i : Real) ≤ layerNormReal eps gamma beta x i := by
    simpa [bounds, layerNormIntervalBounds, hne, radius, centeredBound, invStdBound, μLo, μHi,
      hreal] using hlow
  have hhi : layerNormReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
    simpa [bounds, layerNormIntervalBounds, hne, radius, centeredBound, invStdBound, μLo, μHi,
      hreal] using hhigh
  exact And.intro hlo hhi

/-- Interval bounds for LayerNorm outputs from an absolute input bound. -/
def layerNormAbsBounds {n : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic) (absBound : Dyadic) :
    (Fin n → Dyadic) × (Fin n → Dyadic) :=
  let centeredBound : Dyadic := 2 * absBound
  let invStdBound : Dyadic := dyadicDivUp 1 (sqrtLower eps)
  let radius : Fin n → Dyadic := fun i => |gamma i| * centeredBound * invStdBound
  (fun i => beta i - radius i, fun i => beta i + radius i)

/-- `layerNormAbsBounds` soundness for real LayerNorm outputs under absolute input bounds. -/
theorem layerNormAbsBounds_spec {n : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic) (absBound : Dyadic) (x : Fin n → Dyadic)
    (hne : n ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (habs : ∀ i, |x i| ≤ absBound) :
    let bounds := layerNormAbsBounds eps gamma beta absBound
    ∀ i,
      (bounds.1 i : Real) ≤ layerNormReal eps gamma beta x i ∧
        layerNormReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds i
  have hmean_abs_real : |(meanRat x : Real)| ≤ (absBound : Real) := by
    have h :=
      meanReal_abs_le_bound (x := fun j => (x j : Real)) (bound := absBound) hne
        (by
          intro j
          exact dyadicToReal_abs_le_of_le (habs j))
    simpa [meanReal_eq_meanRat] using h
  have hbound_nonneg : 0 ≤ absBound := by
    have hposn : 0 < n := Nat.pos_of_ne_zero hne
    let i0 : Fin n := ⟨0, hposn⟩
    have h0 : 0 ≤ |x i0| := abs_nonneg _
    exact le_trans h0 (habs i0)
  let centeredBound : Dyadic := 2 * absBound
  let invStdBound : Dyadic := dyadicDivUp 1 (sqrtLower eps)
  let varEps : Real := (varianceRat x : Real) + (eps : Real)
  let μ : Real := meanRat x
  let invStd : Real := (Real.sqrt varEps)⁻¹
  have hcentered_abs : |(x i : Real) - μ| ≤ (centeredBound : Real) := by
    have h1 : |(x i : Real) - μ| ≤ |(x i : Real)| + |μ| := by
      simpa [sub_eq_add_neg, abs_neg] using abs_add_le (x i : Real) (-μ)
    have hx : |(x i : Real)| ≤ (absBound : Real) := by
      exact dyadicToReal_abs_le_of_le (habs i)
    have hmu : |μ| ≤ (absBound : Real) := by
      simpa [μ] using hmean_abs_real
    have h2 : |(x i : Real)| + |μ| ≤ (absBound : Real) + (absBound : Real) :=
      add_le_add hx hmu
    have h12 : |(x i : Real) - μ| ≤ (absBound : Real) + (absBound : Real) :=
      le_trans h1 h2
    simpa [centeredBound, two_mul] using h12
  have hbound_nonneg_real : 0 ≤ (absBound : Real) := by
    exact dyadicToReal_nonneg_of_nonneg hbound_nonneg
  have hcentered_nonneg : 0 ≤ (centeredBound : Real) := by
    have hsum := add_nonneg hbound_nonneg_real hbound_nonneg_real
    simpa [centeredBound, two_mul] using hsum
  have hsqrt_lower :
      (sqrtLower eps : Real) ≤ Real.sqrt varEps := by
    have hsqrt_eps :
        (sqrtLower eps : Real) ≤ Real.sqrt (eps : Real) := by
      have h := sqrtLower_le_real_sqrt (q := eps) (by exact le_of_lt heps)
      simpa using h
    have hvar_nonneg : 0 ≤ (varianceRat x : Real) := varianceRat_nonneg_real x hne
    have hle : (eps : Real) ≤ varEps := by
      have hle' : (eps : Real) ≤ (varianceRat x : Real) + (eps : Real) :=
        le_add_of_nonneg_left hvar_nonneg
      simpa [varEps] using hle'
    have hsqrt_eps' : Real.sqrt (eps : Real) ≤ Real.sqrt varEps := by
      exact Real.sqrt_le_sqrt hle
    exact le_trans hsqrt_eps hsqrt_eps'
  have hsqrt_lower_pos : 0 < (sqrtLower eps : Real) := by
    simpa [dyadicToReal_zero] using
      (dyadicToReal_lt_iff (x := 0) (y := sqrtLower eps)).2 hsqrt
  have hinv_sqrt : invStd ≤ (sqrtLower eps : Real)⁻¹ := by
    have h := inv_anti₀ hsqrt_lower_pos hsqrt_lower
    simpa [invStd] using h
  have hinv_bound : (sqrtLower eps : Real)⁻¹ ≤ (invStdBound : Real) := by
    have hy : sqrtLower eps ≠ 0 := ne_of_gt hsqrt
    have hdiv := dyadicDivUp_ge_real (x := 1) (y := sqrtLower eps) hy
    simpa [invStdBound, one_div] using hdiv
  have hinv : invStd ≤ (invStdBound : Real) := by
    exact le_trans hinv_sqrt hinv_bound
  have hinv_nonneg : 0 ≤ invStd := by
    have hsqrt_nonneg : 0 ≤ Real.sqrt varEps := by
      exact Real.sqrt_nonneg _
    exact inv_nonneg.2 hsqrt_nonneg
  have hmul1 : |(x i : Real) - μ| * invStd ≤
      (centeredBound : Real) * (invStdBound : Real) := by
    have hleft :
        |(x i : Real) - μ| * invStd ≤ (centeredBound : Real) * invStd := by
      exact mul_le_mul_of_nonneg_right hcentered_abs hinv_nonneg
    have hright :
        (centeredBound : Real) * invStd ≤ (centeredBound : Real) * (invStdBound : Real) := by
      exact mul_le_mul_of_nonneg_left hinv hcentered_nonneg
    exact le_trans hleft hright
  have hmul2 : |(gamma i : Real)| * |(x i : Real) - μ| * invStd ≤
      |(gamma i : Real)| * (centeredBound : Real) * (invStdBound : Real) := by
    have hgamma_nonneg : 0 ≤ |(gamma i : Real)| := abs_nonneg _
    have hmul2' : |(gamma i : Real)| * (|(x i : Real) - μ| * invStd) ≤
        |(gamma i : Real)| * ((centeredBound : Real) * (invStdBound : Real)) := by
      exact mul_le_mul_of_nonneg_left hmul1 hgamma_nonneg
    simpa [mul_assoc] using hmul2'
  let t : Real := (gamma i : Real) * ((x i : Real) - μ) * invStd
  have ht_abs :
      |t| ≤ |(gamma i : Real)| * (centeredBound : Real) * (invStdBound : Real) := by
    have ht : |t| = |(gamma i : Real)| * |(x i : Real) - μ| * invStd := by
      have hinv_abs : |invStd| = invStd := abs_of_nonneg hinv_nonneg
      simp [t, abs_mul, hinv_abs, mul_assoc]
    simpa [ht] using hmul2
  let radius : Fin n → Dyadic := fun j => |gamma j| * centeredBound * invStdBound
  have ht_abs' : |t| ≤ (radius i : Real) := by
    simpa [radius, centeredBound, invStdBound] using ht_abs
  have hbounds : -(radius i : Real) ≤ t ∧ t ≤ (radius i : Real) := by
    exact abs_le.mp ht_abs'
  have hlow :
      (beta i : Real) - (radius i : Real) ≤ t + (beta i : Real) := by
    have h := add_le_add_left hbounds.1 (beta i : Real)
    simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using h
  have hhigh :
      t + (beta i : Real) ≤ (beta i : Real) + (radius i : Real) := by
    have h := add_le_add_left hbounds.2 (beta i : Real)
    simpa [add_comm, add_left_comm, add_assoc] using h
  have hreal :
      layerNormReal eps gamma beta x i = t + (beta i : Real) := by
    simp [layerNormReal, hne, μ, invStd, varEps, t, add_comm]
  have hlo : (bounds.1 i : Real) ≤ layerNormReal eps gamma beta x i := by
    simpa [bounds, layerNormAbsBounds, radius, centeredBound, invStdBound, hreal] using hlow
  have hhi : layerNormReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
    simpa [bounds, layerNormAbsBounds, radius, centeredBound, invStdBound, hreal] using hhigh
  exact And.intro hlo hhi

/-- `layerNormAbsBounds` soundness for real LayerNorm outputs on real inputs. -/
theorem layerNormAbsBounds_spec_real {n : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic) (absBound : Dyadic) (x : Fin n → Real)
    (hne : n ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (habs : ∀ i, |x i| ≤ (absBound : Real)) :
    let bounds := layerNormAbsBounds eps gamma beta absBound
    ∀ i,
      (bounds.1 i : Real) ≤ layerNormRealOfReal eps gamma beta x i ∧
        layerNormRealOfReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds i
  have hmean_abs : |meanReal x| ≤ (absBound : Real) :=
    meanReal_abs_le_bound x absBound hne habs
  have hbound_nonneg_real : 0 ≤ (absBound : Real) := by
    have hposn : 0 < n := Nat.pos_of_ne_zero hne
    let i0 : Fin n := ⟨0, hposn⟩
    have h0 : 0 ≤ |x i0| := abs_nonneg _
    exact le_trans h0 (habs i0)
  let centeredBound : Dyadic := 2 * absBound
  let invStdBound : Dyadic := dyadicDivUp 1 (sqrtLower eps)
  let varEps : Real := varianceReal x + (eps : Real)
  let μ : Real := meanReal x
  let invStd : Real := (Real.sqrt varEps)⁻¹
  have hcentered_abs : |x i - μ| ≤ (centeredBound : Real) := by
    have h1 : |x i - μ| ≤ |x i| + |μ| := by
      simpa [sub_eq_add_neg, abs_neg] using abs_add_le (x i) (-μ)
    have hx : |x i| ≤ (absBound : Real) := habs i
    have hmu : |μ| ≤ (absBound : Real) := by
      simpa using hmean_abs
    have h2 : |x i| + |μ| ≤ (absBound : Real) + (absBound : Real) :=
      add_le_add hx hmu
    have h12 : |x i - μ| ≤ (absBound : Real) + (absBound : Real) :=
      le_trans h1 h2
    simpa [centeredBound, two_mul] using h12
  have hcentered_nonneg : 0 ≤ (centeredBound : Real) := by
    have hsum := add_nonneg hbound_nonneg_real hbound_nonneg_real
    simpa [centeredBound, two_mul] using hsum
  have hvar_nonneg : 0 ≤ varianceReal x := varianceReal_nonneg x hne
  have hsqrt_lower :
      (sqrtLower eps : Real) ≤ Real.sqrt varEps := by
    have hsqrt_eps :
        (sqrtLower eps : Real) ≤ Real.sqrt (eps : Real) := by
      have h := sqrtLower_le_real_sqrt (q := eps) (by exact le_of_lt heps)
      simpa using h
    have hle : (eps : Real) ≤ varEps := by
      have hle' : (eps : Real) ≤ varianceReal x + (eps : Real) := by
        exact le_add_of_nonneg_left hvar_nonneg
      simpa [varEps] using hle'
    have hsqrt_eps' : Real.sqrt (eps : Real) ≤ Real.sqrt varEps := by
      exact Real.sqrt_le_sqrt hle
    exact le_trans hsqrt_eps hsqrt_eps'
  have hsqrt_lower_pos : 0 < (sqrtLower eps : Real) := by
    simpa [dyadicToReal_zero] using
      (dyadicToReal_lt_iff (x := 0) (y := sqrtLower eps)).2 hsqrt
  have hinv_sqrt : invStd ≤ (sqrtLower eps : Real)⁻¹ := by
    have h := inv_anti₀ hsqrt_lower_pos hsqrt_lower
    simpa [invStd] using h
  have hinv_bound : (sqrtLower eps : Real)⁻¹ ≤ (invStdBound : Real) := by
    have hy : sqrtLower eps ≠ 0 := ne_of_gt hsqrt
    have hdiv := dyadicDivUp_ge_real (x := 1) (y := sqrtLower eps) hy
    simpa [invStdBound, one_div] using hdiv
  have hinv : invStd ≤ (invStdBound : Real) := by
    exact le_trans hinv_sqrt hinv_bound
  have hinv_nonneg : 0 ≤ invStd := by
    have hsqrt_nonneg : 0 ≤ Real.sqrt varEps := by
      exact Real.sqrt_nonneg _
    exact inv_nonneg.2 hsqrt_nonneg
  have hmul1 : |x i - μ| * invStd ≤
      (centeredBound : Real) * (invStdBound : Real) := by
    have hleft :
        |x i - μ| * invStd ≤ (centeredBound : Real) * invStd := by
      exact mul_le_mul_of_nonneg_right hcentered_abs hinv_nonneg
    have hright :
        (centeredBound : Real) * invStd ≤ (centeredBound : Real) * (invStdBound : Real) := by
      exact mul_le_mul_of_nonneg_left hinv hcentered_nonneg
    exact le_trans hleft hright
  have hmul2 : |(gamma i : Real)| * |x i - μ| * invStd ≤
      |(gamma i : Real)| * (centeredBound : Real) * (invStdBound : Real) := by
    have hgamma_nonneg : 0 ≤ |(gamma i : Real)| := abs_nonneg _
    have hmul2' : |(gamma i : Real)| * (|x i - μ| * invStd) ≤
        |(gamma i : Real)| * ((centeredBound : Real) * (invStdBound : Real)) := by
      exact mul_le_mul_of_nonneg_left hmul1 hgamma_nonneg
    simpa [mul_assoc] using hmul2'
  let t : Real := (gamma i : Real) * (x i - μ) * invStd
  have ht_abs :
      |t| ≤ |(gamma i : Real)| * (centeredBound : Real) * (invStdBound : Real) := by
    have ht : |t| = |(gamma i : Real)| * |x i - μ| * invStd := by
      have hinv_abs : |invStd| = invStd := abs_of_nonneg hinv_nonneg
      simp [t, abs_mul, hinv_abs, mul_assoc]
    simpa [ht] using hmul2
  let radius : Fin n → Dyadic := fun j => |gamma j| * centeredBound * invStdBound
  have ht_abs' : |t| ≤ (radius i : Real) := by
    simpa [radius, centeredBound, invStdBound] using ht_abs
  have hbounds : -(radius i : Real) ≤ t ∧ t ≤ (radius i : Real) := by
    exact abs_le.mp ht_abs'
  have hlow :
      (beta i : Real) - (radius i : Real) ≤ t + (beta i : Real) := by
    have h := add_le_add_left hbounds.1 (beta i : Real)
    simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using h
  have hhigh :
      t + (beta i : Real) ≤ (beta i : Real) + (radius i : Real) := by
    have h := add_le_add_left hbounds.2 (beta i : Real)
    simpa [add_comm, add_left_comm, add_assoc] using h
  have hreal :
      layerNormRealOfReal eps gamma beta x i = t + (beta i : Real) := by
    simp [layerNormRealOfReal, hne, μ, invStd, varEps, t, add_comm]
  have hlo : (bounds.1 i : Real) ≤ layerNormRealOfReal eps gamma beta x i := by
    simpa [bounds, layerNormAbsBounds, radius, centeredBound, invStdBound, hreal] using hlow
  have hhi : layerNormRealOfReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
    simpa [bounds, layerNormAbsBounds, radius, centeredBound, invStdBound, hreal] using hhigh
  exact And.intro hlo hhi

/-- `layerNormIntervalBounds` soundness for real LayerNorm outputs on real inputs. -/
theorem layerNormIntervalBounds_spec_real {n : Nat}
    (eps : Dyadic) (gamma beta : Fin n → Dyadic) (lo hi : Fin n → Dyadic) (x : Fin n → Real)
    (hne : n ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (hlo : ∀ i, (lo i : Real) ≤ x i) (hhi : ∀ i, x i ≤ (hi i : Real)) :
    let bounds := layerNormIntervalBounds eps gamma beta lo hi
    ∀ i,
      (bounds.1 i : Real) ≤ layerNormRealOfReal eps gamma beta x i ∧
        layerNormRealOfReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds i
  have hmean_lo : (mean lo : Real) ≤ meanReal x := by
    have h :=
      meanReal_le_meanReal (x := fun j => (lo j : Real)) (y := x) hne
        (fun j => hlo j)
    have hrat : (meanRat lo : Real) ≤ meanReal x := by
      simpa [meanReal_eq_meanRat] using h
    have hdown : (mean lo : Real) ≤ (meanRat lo : Real) := by
      simpa [mean_def lo hne] using dyadicOfRatDown_le_real (meanRat lo)
    exact le_trans hdown hrat
  have hmean_hi : meanReal x ≤ (meanUpper hi : Real) := by
    have h :=
      meanReal_le_meanReal (x := x) (y := fun j => (hi j : Real)) hne
        (fun j => hhi j)
    have hrat : meanReal x ≤ (meanRat hi : Real) := by
      simpa [meanReal_eq_meanRat] using h
    have hup : (meanRat hi : Real) ≤ (meanUpper hi : Real) := by
      simpa [meanUpper_def hi hne] using real_le_dyadicOfRatUp (meanRat hi)
    exact le_trans hrat hup
  let μLo : Dyadic := mean lo
  let μHi : Dyadic := meanUpper hi
  let centeredBound : Fin n → Dyadic := fun j => max |lo j - μHi| |hi j - μLo|
  let invStdBound : Dyadic := dyadicDivUp 1 (sqrtLower eps)
  let varEps : Real := varianceReal x + (eps : Real)
  let μ : Real := meanReal x
  let invStd : Real := (Real.sqrt varEps)⁻¹
  have hcentered_nonneg : 0 ≤ (centeredBound i : Real) := by
    have h0 : 0 ≤ centeredBound i := by
      dsimp [centeredBound]
      exact le_trans (abs_nonneg _) (le_max_left _ _)
    exact dyadicToReal_nonneg_of_nonneg h0
  have hcentered_abs : |x i - μ| ≤ (centeredBound i : Real) := by
    have hmean_lo_real : (μLo : Real) ≤ μ := by
      simpa [μLo, μ] using hmean_lo
    have hmean_hi_real : μ ≤ (μHi : Real) := by
      simpa [μHi, μ] using hmean_hi
    have hlo' : (lo i : Real) - (μHi : Real) ≤ x i - μ := by
      have h1 : (lo i : Real) - (μHi : Real) ≤ (lo i : Real) - μ := by
        exact sub_le_sub_left hmean_hi_real (lo i : Real)
      have h2 : (lo i : Real) - μ ≤ x i - μ := by
        exact sub_le_sub_right (hlo i) μ
      exact le_trans h1 h2
    have hhi' : x i - μ ≤ (hi i : Real) - (μLo : Real) := by
      have h1 : x i - μ ≤ (hi i : Real) - μ := by
        exact sub_le_sub_right (hhi i) μ
      have h2 : (hi i : Real) - μ ≤ (hi i : Real) - (μLo : Real) := by
        exact sub_le_sub_left hmean_lo_real (hi i : Real)
      exact le_trans h1 h2
    have hbound := abs_le_max_of_bounds hlo' hhi'
    simpa [centeredBound, μLo, μHi, dyadicToReal_abs, dyadicToReal_sub,
      dyadicToReal_max] using hbound
  have hvar_nonneg : 0 ≤ varianceReal x := varianceReal_nonneg x hne
  have hsqrt_lower :
      (sqrtLower eps : Real) ≤ Real.sqrt varEps := by
    have hsqrt_eps :
        (sqrtLower eps : Real) ≤ Real.sqrt (eps : Real) := by
      have h := sqrtLower_le_real_sqrt (q := eps) (by exact le_of_lt heps)
      simpa using h
    have hle : (eps : Real) ≤ varEps := by
      have hle' : (eps : Real) ≤ varianceReal x + (eps : Real) := by
        exact le_add_of_nonneg_left hvar_nonneg
      simpa [varEps] using hle'
    have hsqrt_eps' : Real.sqrt (eps : Real) ≤ Real.sqrt varEps := by
      exact Real.sqrt_le_sqrt hle
    exact le_trans hsqrt_eps hsqrt_eps'
  have hsqrt_lower_pos : 0 < (sqrtLower eps : Real) := by
    simpa [dyadicToReal_zero] using
      (dyadicToReal_lt_iff (x := 0) (y := sqrtLower eps)).2 hsqrt
  have hinv_sqrt : invStd ≤ (sqrtLower eps : Real)⁻¹ := by
    have h := inv_anti₀ hsqrt_lower_pos hsqrt_lower
    simpa [invStd] using h
  have hinv_bound : (sqrtLower eps : Real)⁻¹ ≤ (invStdBound : Real) := by
    have hy : sqrtLower eps ≠ 0 := ne_of_gt hsqrt
    have hdiv := dyadicDivUp_ge_real (x := 1) (y := sqrtLower eps) hy
    simpa [invStdBound, one_div] using hdiv
  have hinv : invStd ≤ (invStdBound : Real) := by
    exact le_trans hinv_sqrt hinv_bound
  have hinv_nonneg : 0 ≤ invStd := by
    have hsqrt_nonneg : 0 ≤ Real.sqrt varEps := by
      exact Real.sqrt_nonneg _
    exact inv_nonneg.2 hsqrt_nonneg
  have hmul1 : |x i - μ| * invStd ≤
      (centeredBound i : Real) * (invStdBound : Real) := by
    have hleft : |x i - μ| * invStd ≤ (centeredBound i : Real) * invStd := by
      exact mul_le_mul_of_nonneg_right hcentered_abs hinv_nonneg
    have hright :
        (centeredBound i : Real) * invStd ≤ (centeredBound i : Real) * (invStdBound : Real) := by
      exact mul_le_mul_of_nonneg_left hinv hcentered_nonneg
    exact le_trans hleft hright
  have hmul2 : |(gamma i : Real)| * |x i - μ| * invStd ≤
      |(gamma i : Real)| * (centeredBound i : Real) * (invStdBound : Real) := by
    have hgamma_nonneg : 0 ≤ |(gamma i : Real)| := abs_nonneg _
    have hmul2' : |(gamma i : Real)| * (|x i - μ| * invStd) ≤
        |(gamma i : Real)| * ((centeredBound i : Real) * (invStdBound : Real)) := by
      exact mul_le_mul_of_nonneg_left hmul1 hgamma_nonneg
    simpa [mul_assoc] using hmul2'
  let t : Real := (gamma i : Real) * (x i - μ) * invStd
  have ht_abs :
      |t| ≤ |(gamma i : Real)| * (centeredBound i : Real) * (invStdBound : Real) := by
    have ht : |t| = |(gamma i : Real)| * |x i - μ| * invStd := by
      have hinv_abs : |invStd| = invStd := abs_of_nonneg hinv_nonneg
      simp [t, abs_mul, hinv_abs, mul_assoc]
    simpa [ht] using hmul2
  let radius : Fin n → Dyadic := fun j => |gamma j| * centeredBound j * invStdBound
  have ht_abs' : |t| ≤ (radius i : Real) := by
    simpa [radius, centeredBound, invStdBound] using ht_abs
  have hbounds : -(radius i : Real) ≤ t ∧ t ≤ (radius i : Real) := by
    exact abs_le.mp ht_abs'
  have hlow :
      (beta i : Real) - (radius i : Real) ≤ t + (beta i : Real) := by
    have h := add_le_add_left hbounds.1 (beta i : Real)
    simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using h
  have hhigh :
      t + (beta i : Real) ≤ (beta i : Real) + (radius i : Real) := by
    have h := add_le_add_left hbounds.2 (beta i : Real)
    simpa [add_comm, add_left_comm, add_assoc] using h
  have hreal :
      layerNormRealOfReal eps gamma beta x i = t + (beta i : Real) := by
    simp [layerNormRealOfReal, hne, μ, invStd, varEps, t, add_comm]
  have hlo : (bounds.1 i : Real) ≤ layerNormRealOfReal eps gamma beta x i := by
    simpa [bounds, layerNormIntervalBounds, hne, radius, centeredBound, invStdBound, μLo, μHi,
      hreal] using hlow
  have hhi : layerNormRealOfReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
    simpa [bounds, layerNormIntervalBounds, hne, radius, centeredBound, invStdBound, μLo, μHi,
      hreal] using hhigh
  exact And.intro hlo hhi

end Bounds

end Sound

end Nfp
