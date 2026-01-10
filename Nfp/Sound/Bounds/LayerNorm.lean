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
LayerNorm interval bounds for rational inputs.

This module computes rational interval bounds for LayerNorm outputs and proves
those bounds sound for real-valued LayerNorm semantics.
-/

namespace Nfp

namespace Sound

namespace Bounds

open scoped BigOperators

/-! Square-root bounds. -/

lemma rat_nat_cast_nonneg (n : Nat) : (0 : Rat) ≤ (n : Rat) := by
  simp

lemma rat_nat_cast_pos {n : Nat} (h : 0 < n) : (0 : Rat) < (n : Rat) := by
  exact (Nat.cast_pos (α := Rat)).2 h

/-- Base rational lower bound for a square root. -/
def sqrtLowerBase (q : Rat) : Rat :=
  let num := q.num.natAbs
  let den := q.den
  let a := Nat.sqrt num
  let b := Nat.sqrt den
  ratRoundDown ((a : Rat) / (b + 1))

/-- Base rational upper bound for a square root. -/
def sqrtUpperBase (q : Rat) : Rat :=
  let num := q.num.natAbs
  let den := q.den
  let a := Nat.sqrt num
  let b := Nat.sqrt den
  ratRoundUp ((a + 1 : Rat) / b)

/-- Alternate rational lower bound for a square root. -/
def sqrtLowerAlt (q : Rat) : Rat :=
  let num := q.num.natAbs
  let den := q.den
  let a := Nat.sqrt (num * den)
  ratRoundDown ((a : Rat) / den)

/-- Alternate rational upper bound for a square root. -/
def sqrtUpperAlt (q : Rat) : Rat :=
  let num := q.num.natAbs
  let den := q.den
  let a := Nat.sqrt (num * den)
  ratRoundUp ((a + 1 : Rat) / den)

/-- Extra precision scale for `sqrtLowerScaled`. -/
def sqrtLowerScale : Nat := 65536

/-- Scaled rational lower bound for a square root (extra precision). -/
def sqrtLowerScaled (q : Rat) : Rat :=
  let num := q.num.natAbs
  let den := q.den
  let scale := sqrtLowerScale
  let a := Nat.sqrt (num * den * scale * scale)
  ratRoundDown ((a : Rat) / (den * scale))

/-- Scaled rational upper bound for a square root (extra precision). -/
def sqrtUpperScaled (q : Rat) : Rat :=
  let num := q.num.natAbs
  let den := q.den
  let scale := sqrtLowerScale
  let a := Nat.sqrt (num * den * scale * scale)
  ratRoundUp ((a + 1 : Rat) / (den * scale))

/-- Rational lower bound for a square root (tighter of three bounds). -/
def sqrtLower (q : Rat) : Rat :=
  max (max (sqrtLowerBase q) (sqrtLowerAlt q)) (sqrtLowerScaled q)

/-- Rational upper bound for a square root (tighter of three bounds). -/
def sqrtUpper (q : Rat) : Rat :=
  min (min (sqrtUpperBase q) (sqrtUpperAlt q)) (sqrtUpperScaled q)

/-- `sqrtLowerBase` is nonnegative. -/
theorem sqrtLowerBase_nonneg (q : Rat) : 0 ≤ sqrtLowerBase q := by
  classical
  unfold sqrtLowerBase
  have hnum : 0 ≤ (Nat.sqrt q.num.natAbs : Rat) := by
    exact_mod_cast (Nat.zero_le (Nat.sqrt q.num.natAbs))
  have hden : 0 ≤ (Nat.sqrt q.den + 1 : Rat) := by
    exact_mod_cast (Nat.zero_le (Nat.sqrt q.den + 1))
  have hrat : 0 ≤ (Nat.sqrt q.num.natAbs : Rat) / (Nat.sqrt q.den + 1) := by
    exact div_nonneg hnum hden
  exact ratRoundDown_nonneg hrat

/-! Strict positivity helpers. -/

/-! Base bounds. -/


/-- `sqrtUpperBase` is nonnegative. -/
theorem sqrtUpperBase_nonneg (q : Rat) : 0 ≤ sqrtUpperBase q := by
  classical
  unfold sqrtUpperBase
  have hnum : 0 ≤ (Nat.sqrt q.num.natAbs + 1 : Rat) := by
    exact_mod_cast (Nat.zero_le (Nat.sqrt q.num.natAbs + 1))
  have hden : 0 ≤ (Nat.sqrt q.den : Rat) := by
    exact_mod_cast (Nat.zero_le (Nat.sqrt q.den))
  have hrat :
      0 ≤ (Nat.sqrt q.num.natAbs + 1 : Rat) / (Nat.sqrt q.den) := by
    exact div_nonneg hnum hden
  exact ratRoundUp_nonneg hrat

/-- `sqrtUpperBase` is always positive. -/
theorem sqrtUpperBase_pos (q : Rat) : 0 < sqrtUpperBase q := by
  classical
  unfold sqrtUpperBase
  have hnum_pos : (0 : Rat) < (Nat.sqrt q.num.natAbs + 1 : Rat) := by
    exact_mod_cast (Nat.succ_pos (Nat.sqrt q.num.natAbs))
  have hden_pos : (0 : Rat) < (Nat.sqrt q.den : Rat) := by
    have hden : 0 < q.den := q.den_pos
    exact_mod_cast (Nat.sqrt_pos.2 hden)
  have hrat_pos :
      (0 : Rat) < (Nat.sqrt q.num.natAbs + 1 : Rat) / (Nat.sqrt q.den) := by
    exact div_pos hnum_pos hden_pos
  exact ratRoundUp_pos hrat_pos

/-! Alternate bounds. -/

/-- `sqrtLowerAlt` is nonnegative. -/
theorem sqrtLowerAlt_nonneg (q : Rat) : 0 ≤ sqrtLowerAlt q := by
  classical
  unfold sqrtLowerAlt
  have hnum : 0 ≤ (Nat.sqrt (q.num.natAbs * q.den) : Rat) := by
    exact_mod_cast (Nat.zero_le (Nat.sqrt (q.num.natAbs * q.den)))
  have hden : 0 ≤ (q.den : Rat) := by
    exact_mod_cast (Nat.zero_le q.den)
  have hrat :
      0 ≤ (Nat.sqrt (q.num.natAbs * q.den) : Rat) / q.den := by
    exact div_nonneg hnum hden
  exact ratRoundDown_nonneg hrat


/-- `sqrtUpperAlt` is nonnegative. -/
theorem sqrtUpperAlt_nonneg (q : Rat) : 0 ≤ sqrtUpperAlt q := by
  classical
  unfold sqrtUpperAlt
  have hnum : 0 ≤ (Nat.sqrt (q.num.natAbs * q.den) + 1 : Rat) := by
    exact_mod_cast (Nat.zero_le (Nat.sqrt (q.num.natAbs * q.den) + 1))
  have hden : 0 ≤ (q.den : Rat) := by
    exact_mod_cast (Nat.zero_le q.den)
  have hrat :
      0 ≤ (Nat.sqrt (q.num.natAbs * q.den) + 1 : Rat) / q.den := by
    exact div_nonneg hnum hden
  exact ratRoundUp_nonneg hrat

/-- `sqrtUpperAlt` is always positive. -/
theorem sqrtUpperAlt_pos (q : Rat) : 0 < sqrtUpperAlt q := by
  classical
  unfold sqrtUpperAlt
  have hnum_pos :
      (0 : Rat) < (Nat.sqrt (q.num.natAbs * q.den) + 1 : Rat) := by
    exact_mod_cast (Nat.succ_pos (Nat.sqrt (q.num.natAbs * q.den)))
  have hden_pos : (0 : Rat) < (q.den : Rat) := by
    exact_mod_cast q.den_pos
  have hrat_pos :
      (0 : Rat) <
        (Nat.sqrt (q.num.natAbs * q.den) + 1 : Rat) / q.den := by
    exact div_pos hnum_pos hden_pos
  exact ratRoundUp_pos hrat_pos

/-- `sqrtUpperScaled` is nonnegative. -/
theorem sqrtUpperScaled_nonneg (q : Rat) : 0 ≤ sqrtUpperScaled q := by
  classical
  unfold sqrtUpperScaled
  have hnum :
      0 ≤ (Nat.sqrt (q.num.natAbs * q.den * sqrtLowerScale * sqrtLowerScale) + 1 : Rat) := by
    exact_mod_cast
      (Nat.zero_le (Nat.sqrt (q.num.natAbs * q.den * sqrtLowerScale * sqrtLowerScale) + 1))
  have hden : 0 ≤ (q.den * sqrtLowerScale : Rat) := by
    exact_mod_cast (Nat.zero_le (q.den * sqrtLowerScale))
  have hrat :
      0 ≤ (Nat.sqrt (q.num.natAbs * q.den * sqrtLowerScale * sqrtLowerScale) + 1 : Rat) /
        (q.den * sqrtLowerScale) := by
    exact div_nonneg hnum hden
  exact ratRoundUp_nonneg hrat

/-- `sqrtUpperScaled` is always positive. -/
theorem sqrtUpperScaled_pos (q : Rat) : 0 < sqrtUpperScaled q := by
  classical
  unfold sqrtUpperScaled
  have hnum_pos :
      (0 : Rat) <
        (Nat.sqrt (q.num.natAbs * q.den * sqrtLowerScale * sqrtLowerScale) + 1 : Rat) := by
    exact_mod_cast
      (Nat.succ_pos (Nat.sqrt (q.num.natAbs * q.den * sqrtLowerScale * sqrtLowerScale)))
  have hden_pos : (0 : Rat) < (q.den * sqrtLowerScale : Rat) := by
    have hden : 0 < q.den := q.den_pos
    have hscale : 0 < sqrtLowerScale := by
      simp [sqrtLowerScale]
    exact_mod_cast (Nat.mul_pos hden hscale)
  have hrat_pos :
      (0 : Rat) <
        (Nat.sqrt (q.num.natAbs * q.den * sqrtLowerScale * sqrtLowerScale) + 1 : Rat) /
          (q.den * sqrtLowerScale) := by
    exact div_pos hnum_pos hden_pos
  exact ratRoundUp_pos hrat_pos

/-! Combined bounds. -/

/-- `sqrtLower` is nonnegative. -/
theorem sqrtLower_nonneg (q : Rat) : 0 ≤ sqrtLower q := by
  have hbase : 0 ≤ sqrtLowerBase q := sqrtLowerBase_nonneg q
  have hmax : 0 ≤ max (sqrtLowerBase q) (sqrtLowerAlt q) :=
    le_trans hbase (le_max_left _ _)
  exact le_trans hmax (le_max_left _ _)


/-- `sqrtUpper` is nonnegative. -/
theorem sqrtUpper_nonneg (q : Rat) : 0 ≤ sqrtUpper q := by
  have hbase : 0 ≤ sqrtUpperBase q := sqrtUpperBase_nonneg q
  have halt : 0 ≤ sqrtUpperAlt q := sqrtUpperAlt_nonneg q
  have hscaled : 0 ≤ sqrtUpperScaled q := sqrtUpperScaled_nonneg q
  have hmin1 : 0 ≤ min (sqrtUpperBase q) (sqrtUpperAlt q) := by
    exact le_min hbase halt
  exact le_min hmin1 hscaled

/-- `sqrtUpper` is always positive. -/
theorem sqrtUpper_pos (q : Rat) : 0 < sqrtUpper q := by
  have hbase : 0 < sqrtUpperBase q := sqrtUpperBase_pos q
  have halt : 0 < sqrtUpperAlt q := sqrtUpperAlt_pos q
  have hscaled : 0 < sqrtUpperScaled q := sqrtUpperScaled_pos q
  have hmin1 : 0 < min (sqrtUpperBase q) (sqrtUpperAlt q) := by
    exact lt_min hbase halt
  exact lt_min hmin1 hscaled

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
  have hq_nonneg : 0 ≤ (q : Real) := by
    exact ratToReal_nonneg_of_nonneg hq
  have hle : (a : Real) / (b + 1 : Real) ≤ Real.sqrt (q : Real) :=
    (Real.le_sqrt hnonneg hq_nonneg).2 hsq
  have hdown :
      (sqrtLowerBase q : Real) ≤ (a : Real) / (b + 1 : Real) := by
    have hdown' :
        ratToReal (ratRoundDown ((a : Rat) / (b + 1))) ≤
          (a : Real) / (b + 1 : Real) := by
      simpa using ratRoundDown_le_real ((a : Rat) / (b + 1))
    simpa [sqrtLowerBase, num, den, a, b] using hdown'
  exact le_trans hdown hle

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
  have hup :
      (a + 1 : Real) / (b : Real) ≤ (sqrtUpperBase q : Real) := by
    have hup' :
        (a + 1 : Real) / (b : Real) ≤
          ratToReal (ratRoundUp ((a + 1 : Rat) / b)) := by
      simpa using real_le_ratRoundUp ((a + 1 : Rat) / b)
    simpa [sqrtUpperBase, num, den, a, b] using hup'
  exact le_trans hle hup

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
  have hq_nonneg : 0 ≤ (q : Real) := by
    exact ratToReal_nonneg_of_nonneg hq
  have hle : (a : Real) / (den : Real) ≤ Real.sqrt (q : Real) :=
    (Real.le_sqrt hnonneg hq_nonneg).2 hsq
  have hdown :
      (sqrtLowerAlt q : Real) ≤ (a : Real) / (den : Real) := by
    have hdown' :
        ratToReal (ratRoundDown ((a : Rat) / den)) ≤
          (a : Real) / (den : Real) := by
      simpa using ratRoundDown_le_real ((a : Rat) / den)
    simpa [sqrtLowerAlt, num, den, a] using hdown'
  exact le_trans hdown hle

/-- Scaled square-root lower bound in reals. -/
theorem sqrtLowerScaled_le_real_sqrt {q : Rat} (hq : 0 ≤ q) :
    (sqrtLowerScaled q : Real) ≤ Real.sqrt (q : Real) := by
  classical
  set num : Nat := q.num.natAbs
  set den : Nat := q.den
  set scale : Nat := sqrtLowerScale
  set a : Nat := Nat.sqrt (num * den * scale * scale)
  have hden_pos : 0 < (den : Real) := by
    exact_mod_cast q.den_pos
  have hscale_pos : 0 < (scale : Real) := by
    have hscale_pos_nat : 0 < scale := by
      simp [scale, sqrtLowerScale]
    exact_mod_cast hscale_pos_nat
  have hnumden_le : (a ^ 2 : Real) ≤ (num * den * scale * scale : Nat) := by
    exact_mod_cast (Nat.sqrt_le' (num * den * scale * scale))
  have hmul :
      (a ^ 2 : Real) ≤ (num : Real) * den * (scale : Real) * (scale : Real) := by
    simpa [num, den, scale, Nat.cast_mul, mul_assoc, mul_left_comm, mul_comm] using hnumden_le
  have hdenScale_pos : 0 < (den : Real) * (scale : Real) :=
    mul_pos hden_pos hscale_pos
  have hdenScale_pos2 : 0 < ((den : Real) * (scale : Real)) ^ 2 := by
    exact pow_pos hdenScale_pos 2
  have hmul' :
      (a ^ 2 : Real) * ((den : Real) * (scale : Real)) ^ 2 ≤
        ((num : Real) * den * (scale : Real) * (scale : Real)) *
          ((den : Real) * (scale : Real)) ^ 2 := by
    have hnonneg : 0 ≤ ((den : Real) * (scale : Real)) ^ 2 := by
      exact sq_nonneg _
    exact mul_le_mul_of_nonneg_right hmul hnonneg
  have hdiv :
      (a ^ 2 : Real) / ((den : Real) * (scale : Real)) ^ 2 ≤
        ((num : Real) * den * (scale : Real) * (scale : Real)) /
          ((den : Real) * (scale : Real)) ^ 2 := by
    exact (div_le_div_iff₀ hdenScale_pos2 hdenScale_pos2).2 hmul'
  have hdenScale_ne : ((den : Real) * (scale : Real)) ≠ 0 :=
    ne_of_gt hdenScale_pos
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
  have hq_eq :
      ((num : Real) * den * (scale : Real) * (scale : Real)) /
          ((den : Real) * (scale : Real)) ^ 2 = (num : Real) / den := by
    field_simp [hdenScale_ne]
  have hpow :
      ((a : Real) / ((den : Real) * (scale : Real))) ^ 2 =
        (a ^ 2 : Real) / ((den : Real) * (scale : Real)) ^ 2 := by
    simp [pow_two, div_mul_div_comm]
  have hsq :
      ((a : Real) / ((den : Real) * (scale : Real))) ^ 2 ≤ (q : Real) := by
    calc
      ((a : Real) / ((den : Real) * (scale : Real))) ^ 2
          = (a ^ 2 : Real) / ((den : Real) * (scale : Real)) ^ 2 := hpow
      _ ≤ ((num : Real) * den * (scale : Real) * (scale : Real)) /
            ((den : Real) * (scale : Real)) ^ 2 := hdiv
      _ = (num : Real) / den := hq_eq
      _ = (q : Real) := by simp [hq_cast]
  have hnonneg : 0 ≤ (a : Real) / ((den : Real) * (scale : Real)) := by
    have hnum_nonneg : 0 ≤ (a : Real) := by exact_mod_cast (Nat.zero_le a)
    have hden_nonneg : 0 ≤ (den : Real) * (scale : Real) := by
      nlinarith [hden_pos, hscale_pos]
    exact div_nonneg hnum_nonneg hden_nonneg
  have hq_nonneg : 0 ≤ (q : Real) := by
    exact ratToReal_nonneg_of_nonneg hq
  have hle :
      (a : Real) / ((den : Real) * (scale : Real)) ≤ Real.sqrt (q : Real) :=
    (Real.le_sqrt hnonneg hq_nonneg).2 hsq
  have hdown :
      (sqrtLowerScaled q : Real) ≤ (a : Real) / ((den : Real) * (scale : Real)) := by
    have hdown' :
        ratToReal (ratRoundDown ((a : Rat) / (den * scale))) ≤
          (a : Real) / ((den : Real) * (scale : Real)) := by
      simpa using ratRoundDown_le_real ((a : Rat) / (den * scale))
    simpa [sqrtLowerScaled, num, den, scale, a] using hdown'
  exact le_trans hdown hle

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
  have hup :
      (a + 1 : Real) / (den : Real) ≤ (sqrtUpperAlt q : Real) := by
    have hup' :
        (a + 1 : Real) / (den : Real) ≤
          ratToReal (ratRoundUp ((a + 1 : Rat) / den)) := by
      simpa using real_le_ratRoundUp ((a + 1 : Rat) / den)
    simpa [sqrtUpperAlt, num, den, a] using hup'
  exact le_trans hle hup

/-- Scaled square-root upper bound in reals. -/
theorem real_sqrt_le_sqrtUpperScaled {q : Rat} (hq : 0 ≤ q) :
    Real.sqrt (q : Real) ≤ (sqrtUpperScaled q : Real) := by
  classical
  set num : Nat := q.num.natAbs
  set den : Nat := q.den
  set scale : Nat := sqrtLowerScale
  set a : Nat := Nat.sqrt (num * den * scale * scale)
  have hden_pos : 0 < (den : Real) := by
    exact_mod_cast q.den_pos
  have hscale_pos : 0 < (scale : Real) := by
    have hscale_pos_nat : 0 < scale := by
      simp [scale, sqrtLowerScale]
    exact_mod_cast hscale_pos_nat
  have hnumden_lt : (num * den * scale * scale : Real) < (a + 1) ^ 2 := by
    exact_mod_cast (Nat.lt_succ_sqrt' (num * den * scale * scale))
  have hmul :
      (num : Real) * den * (scale : Real) * (scale : Real) ≤ (a + 1 : Real) ^ 2 := by
    exact le_of_lt hnumden_lt
  have hdenScale_pos : 0 < (den : Real) * (scale : Real) := by
    exact mul_pos hden_pos hscale_pos
  have hdenScale_pos2 : 0 < ((den : Real) * (scale : Real)) ^ 2 := by
    exact pow_pos hdenScale_pos 2
  have hdiv :
      (num : Real) * den * (scale : Real) * (scale : Real) /
          ((den : Real) * (scale : Real)) ^ 2 ≤
        (a + 1 : Real) ^ 2 / ((den : Real) * (scale : Real)) ^ 2 := by
    have hmul' :
        (num : Real) * den * (scale : Real) * (scale : Real) *
            ((den : Real) * (scale : Real)) ^ 2 ≤
          (a + 1 : Real) ^ 2 * ((den : Real) * (scale : Real)) ^ 2 := by
      have hden_sq_nonneg : 0 ≤ ((den : Real) * (scale : Real)) ^ 2 := by
        exact sq_nonneg _
      exact mul_le_mul_of_nonneg_right hmul hden_sq_nonneg
    exact (div_le_div_iff₀ hdenScale_pos2 hdenScale_pos2).2 hmul'
  have hdenScale_ne : ((den : Real) * (scale : Real)) ≠ 0 := by
    exact ne_of_gt hdenScale_pos
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
  have hq_eq :
      ((num : Real) * den * (scale : Real) * (scale : Real)) /
          ((den : Real) * (scale : Real)) ^ 2 = (num : Real) / den := by
    field_simp [hdenScale_ne]
  have hq_cast' :
      (q : Real) =
        ((num : Real) * den * (scale : Real) * (scale : Real)) /
          ((den : Real) * (scale : Real)) ^ 2 := by
    calc
      (q : Real) = (num : Real) / den := hq_cast
      _ = ((num : Real) * den * (scale : Real) * (scale : Real)) /
            ((den : Real) * (scale : Real)) ^ 2 := hq_eq.symm
  have hpow :
      ((a + 1 : Real) / ((den : Real) * (scale : Real))) ^ 2 =
        (a + 1 : Real) ^ 2 / ((den : Real) * (scale : Real)) ^ 2 := by
    simp [pow_two, div_mul_div_comm]
  have hsq :
      (q : Real) ≤ ((a + 1 : Real) / ((den : Real) * (scale : Real))) ^ 2 := by
    simpa [hq_cast', hpow] using hdiv
  have hnonneg : 0 ≤ ((a + 1 : Real) / ((den : Real) * (scale : Real))) := by
    have hnum_nonneg : 0 ≤ (a + 1 : Real) := by
      exact_mod_cast (Nat.zero_le (a + 1))
    have hden_nonneg : 0 ≤ (den : Real) * (scale : Real) := by
      nlinarith [hden_pos, hscale_pos]
    exact div_nonneg hnum_nonneg hden_nonneg
  have hle :
      Real.sqrt (q : Real) ≤ (a + 1 : Real) / ((den : Real) * (scale : Real)) :=
    (Real.sqrt_le_iff).2 ⟨hnonneg, hsq⟩
  have hup :
      (a + 1 : Real) / ((den : Real) * (scale : Real)) ≤ (sqrtUpperScaled q : Real) := by
    have hup' :
        (a + 1 : Real) / ((den : Real) * (scale : Real)) ≤
          ratToReal (ratRoundUp ((a + 1 : Rat) / (den * scale))) := by
      simpa using real_le_ratRoundUp ((a + 1 : Rat) / (den * scale))
    simpa [sqrtUpperScaled, num, den, scale, a] using hup'
  exact le_trans hle hup

/-- Square-root lower bound in reals (tighter of three bounds). -/
theorem sqrtLower_le_real_sqrt {q : Rat} (hq : 0 ≤ q) :
    (sqrtLower q : Real) ≤ Real.sqrt (q : Real) := by
  have hbase := sqrtLowerBase_le_real_sqrt (q := q) hq
  have halt := sqrtLowerAlt_le_real_sqrt (q := q) hq
  have hscaled := sqrtLowerScaled_le_real_sqrt (q := q) hq
  have hmax1 :
      (max (sqrtLowerBase q) (sqrtLowerAlt q) : Real) ≤ Real.sqrt (q : Real) := by
    simpa [ratToReal_max] using (max_le_iff).2 ⟨hbase, halt⟩
  have hmax2 :
      (max (max (sqrtLowerBase q) (sqrtLowerAlt q)) (sqrtLowerScaled q) : Real) ≤
        Real.sqrt (q : Real) := by
    simpa [ratToReal_max] using (max_le_iff).2 ⟨hmax1, hscaled⟩
  simpa [sqrtLower] using hmax2

/-- Square-root upper bound in reals (tighter of three bounds). -/
theorem real_sqrt_le_sqrtUpper {q : Rat} (hq : 0 ≤ q) :
    Real.sqrt (q : Real) ≤ (sqrtUpper q : Real) := by
  have hbase := real_sqrt_le_sqrtUpperBase (q := q) hq
  have halt := real_sqrt_le_sqrtUpperAlt (q := q) hq
  have hscaled := real_sqrt_le_sqrtUpperScaled (q := q) hq
  have hmin1 :
      Real.sqrt (q : Real) ≤ min (sqrtUpperBase q : Real) (sqrtUpperAlt q : Real) := by
    exact (le_min_iff).2 ⟨hbase, halt⟩
  have hmin2 :
      Real.sqrt (q : Real) ≤
        min (min (sqrtUpperBase q : Real) (sqrtUpperAlt q : Real)) (sqrtUpperScaled q : Real) := by
    exact (le_min_iff).2 ⟨hmin1, hscaled⟩
  simpa [sqrtUpper, ratToReal_min] using hmin2

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
      have hx' : 0 ≤ (x : Real) := ratToReal_nonneg_of_nonneg hx
      exact mul_le_mul_of_nonneg_left hlo hx'
    have h2 : (x : Real) * y ≤ (x : Real) * (hi : Real) := by
      have hx' : 0 ≤ (x : Real) := ratToReal_nonneg_of_nonneg hx
      exact mul_le_mul_of_nonneg_left hhi hx'
    simp [scaleInterval, hx, h1, h2]
  · have hx' : x ≤ 0 := le_of_lt (lt_of_not_ge hx)
    have h1 : (x : Real) * (hi : Real) ≤ (x : Real) * y := by
      have hx'' : (x : Real) ≤ 0 := (ratToReal_nonpos_iff (x := x)).2 hx'
      exact mul_le_mul_of_nonpos_left hhi hx''
    have h2 : (x : Real) * y ≤ (x : Real) * (lo : Real) := by
      have hx'' : (x : Real) ≤ 0 := (ratToReal_nonpos_iff (x := x)).2 hx'
      exact mul_le_mul_of_nonpos_left hlo hx''
    simp [scaleInterval, hx, h1, h2]

/-- Real-valued LayerNorm output for a vector. -/
noncomputable def layerNormReal {n : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat) (x : Fin n → Rat) : Fin n → Real :=
  if n = 0 then
    fun _ => 0
  else
    let μ : Real := meanRat x
    let varEps : Real := (varianceRat x : Real) + (eps : Real)
    let invStd : Real := (Real.sqrt varEps)⁻¹
    fun i => (gamma i : Real) * ((x i : Real) - μ) * invStd + (beta i : Real)

/-- Real-valued LayerNorm output for a real vector. -/
noncomputable def layerNormRealOfReal {n : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat) (x : Fin n → Real) : Fin n → Real :=
  if n = 0 then
    fun _ => 0
  else
    let μ : Real := meanReal x
    let varEps : Real := varianceReal x + (eps : Real)
    let invStd : Real := (Real.sqrt varEps)⁻¹
    fun i => (gamma i : Real) * (x i - μ) * invStd + (beta i : Real)

/-- Interval bounds for LayerNorm outputs. -/
def layerNormBounds {n : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat) (x : Fin n → Rat) :
    (Fin n → Rat) × (Fin n → Rat) :=
  if n = 0 then
    (fun _ => 0, fun _ => 0)
  else
    let μ : Rat := mean x
    let centered : Fin n → Rat := fun i => x i - μ
    let var : Rat := variance x
    let varEps : Rat := var + eps
    let sqrtLowerBound : Rat := max (sqrtLower eps) (sqrtLower varEps)
    let sqrtUpperBound : Rat := sqrtUpper varEps
    let invStdLower : Rat := ratDivDown 1 sqrtUpperBound
    let invStdUpper : Rat := ratDivUp 1 sqrtLowerBound
    let coeff : Fin n → Rat := fun i => gamma i * centered i
    let lo : Fin n → Rat := fun i =>
      if 0 ≤ coeff i then
        beta i + coeff i * invStdLower
      else
        beta i + coeff i * invStdUpper
    let hi : Fin n → Rat := fun i =>
      if 0 ≤ coeff i then
        beta i + coeff i * invStdUpper
      else
        beta i + coeff i * invStdLower
    (lo, hi)

/-- `layerNormBounds` soundness for real LayerNorm outputs. -/
theorem layerNormBounds_spec {n : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat) (x : Fin n → Rat)
    (hne : n ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps) :
    let bounds := layerNormBounds eps gamma beta x
    ∀ i,
      (bounds.1 i : Real) ≤ layerNormReal eps gamma beta x i ∧
        layerNormReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds i
  let μRat : Rat := mean x
  let varRat : Rat := variance x
  let varEpsRat : Rat := varRat + eps
  let sqrtLowerBound : Rat := max (sqrtLower eps) (sqrtLower varEpsRat)
  let sqrtUpperBound : Rat := sqrtUpper varEpsRat
  let invStdLower : Rat := ratDivDown 1 sqrtUpperBound
  let invStdUpper : Rat := ratDivUp 1 sqrtLowerBound
  let centered : Rat := x i - μRat
  let coeff : Rat := gamma i * centered
  let μ : Real := meanRat x
  let varEps : Real := (varianceRat x : Real) + (eps : Real)
  let invStd : Real := (Real.sqrt varEps)⁻¹
  have hmu : (μRat : Real) = μ := by
    simp [μRat, μ, mean_def, hne, ratRoundDown]
  have hvar : (varRat : Real) = (varianceRat x : Real) := by
    simp [varRat, variance_def, hne, ratRoundDown]
  have hvarEps : (varEpsRat : Real) = varEps := by
    simp [varEpsRat, varEps, hvar]
  have hvar_nonneg : 0 ≤ (varianceRat x : Real) := varianceRat_nonneg_real x hne
  have hvar_nonneg_rat : 0 ≤ varianceRat x := by
    exact (ratToReal_nonneg_iff (x := varianceRat x)).1 hvar_nonneg
  have hvarRat_nonneg : 0 ≤ varRat := by
    have h := ratRoundDown_nonneg (q := varianceRat x) hvar_nonneg_rat
    simpa [varRat, variance_def x hne] using h
  have hvarEps_nonneg : 0 ≤ varEpsRat := by
    exact add_nonneg hvarRat_nonneg (le_of_lt heps)
  have hsqrt_lower :
      (sqrtLowerBound : Real) ≤ Real.sqrt varEps := by
    have hsqrt_eps : (sqrtLower eps : Real) ≤ Real.sqrt varEps := by
      have hsqrt_eps' : (sqrtLower eps : Real) ≤ Real.sqrt (eps : Real) := by
        have h := sqrtLower_le_real_sqrt (q := eps) (by exact le_of_lt heps)
        simpa using h
      have hle : (eps : Real) ≤ varEps := by
        have hle' : (eps : Real) ≤ (varianceRat x : Real) + (eps : Real) :=
          le_add_of_nonneg_left hvar_nonneg
        simpa [varEps] using hle'
      exact le_trans hsqrt_eps' (Real.sqrt_le_sqrt hle)
    have hsqrt_var : (sqrtLower varEpsRat : Real) ≤ Real.sqrt varEps := by
      have hsqrt_var' :
          (sqrtLower varEpsRat : Real) ≤ Real.sqrt (varEpsRat : Real) := by
        have h := sqrtLower_le_real_sqrt (q := varEpsRat) hvarEps_nonneg
        simpa using h
      have hle : (varEpsRat : Real) ≤ varEps := by
        simp [hvarEps]
      exact le_trans hsqrt_var' (Real.sqrt_le_sqrt hle)
    have hmax :
        max (sqrtLower eps : Real) (sqrtLower varEpsRat : Real) ≤ Real.sqrt varEps :=
      (max_le_iff).2 ⟨hsqrt_eps, hsqrt_var⟩
    simpa [sqrtLowerBound, ratToReal_max] using hmax
  have hsqrt_upper :
      Real.sqrt varEps ≤ (sqrtUpperBound : Real) := by
    have h := real_sqrt_le_sqrtUpper (q := varEpsRat) hvarEps_nonneg
    simpa [sqrtUpperBound, hvarEps] using h
  have hsqrt_lower_pos_rat : 0 < sqrtLowerBound := by
    have hpos : 0 < sqrtLower eps := hsqrt
    have hpos' : 0 < max (sqrtLower eps) (sqrtLower varEpsRat) :=
      lt_of_lt_of_le hpos (le_max_left _ _)
    simpa [sqrtLowerBound] using hpos'
  have hsqrt_lower_pos : 0 < (sqrtLowerBound : Real) := by
    exact (Rat.cast_pos (K := Real) (q := sqrtLowerBound)).2 hsqrt_lower_pos_rat
  have hsqrt_upper_pos_rat : 0 < sqrtUpperBound := by
    simpa [sqrtUpperBound] using sqrtUpper_pos varEpsRat
  have hsqrt_upper_pos : 0 < (sqrtUpperBound : Real) := by
    exact (Rat.cast_pos (K := Real) (q := sqrtUpperBound)).2 hsqrt_upper_pos_rat
  have hvarEps_pos : 0 < varEps := by
    have heps_real : 0 < (eps : Real) := by
      exact_mod_cast heps
    have hpos := add_pos_of_nonneg_of_pos hvar_nonneg heps_real
    simpa [varEps] using hpos
  have hsqrt_pos : 0 < Real.sqrt varEps := Real.sqrt_pos.2 hvarEps_pos
  have hinv_lower_real :
      (sqrtUpperBound : Real)⁻¹ ≤ invStd := by
    have hle := inv_anti₀ hsqrt_pos hsqrt_upper
    simpa [invStd] using hle
  have hinv_upper_real :
      invStd ≤ (sqrtLowerBound : Real)⁻¹ := by
    have hle := inv_anti₀ hsqrt_lower_pos hsqrt_lower
    simpa [invStd] using hle
  have hupper_ne : sqrtUpperBound ≠ 0 := ne_of_gt hsqrt_upper_pos_rat
  have hlower_ne : sqrtLowerBound ≠ 0 := ne_of_gt hsqrt_lower_pos_rat
  have hinv_lower : (invStdLower : Real) ≤ invStd := by
    simpa [invStdLower, ratDivDown, hupper_ne, one_div] using hinv_lower_real
  have hinv_upper : invStd ≤ (invStdUpper : Real) := by
    simpa [invStdUpper, ratDivUp, hlower_ne, one_div] using hinv_upper_real
  have hlayer :
      layerNormReal eps gamma beta x i =
        (beta i : Real) + (coeff : Real) * invStd := by
    simp [layerNormReal, hne, coeff, centered, μ, hmu, invStd, varEps, add_comm, mul_assoc]
  by_cases hcoeff : 0 ≤ coeff
  · have hcoeff_real : 0 ≤ (coeff : Real) :=
      ratToReal_nonneg_of_nonneg hcoeff
    have hlow_raw :
        (beta i : Real) + (coeff : Real) * (invStdLower : Real) ≤
          (beta i : Real) + (coeff : Real) * invStd := by
      have hmul := mul_le_mul_of_nonneg_left hinv_lower hcoeff_real
      exact add_le_add_right hmul (beta i : Real)
    have hhigh_raw :
        (beta i : Real) + (coeff : Real) * invStd ≤
          (beta i : Real) + (coeff : Real) * (invStdUpper : Real) := by
      have hmul := mul_le_mul_of_nonneg_left hinv_upper hcoeff_real
      exact add_le_add_right hmul (beta i : Real)
    have hlo : (bounds.1 i : Real) ≤ layerNormReal eps gamma beta x i := by
      simpa [bounds, layerNormBounds, hne, μRat, centered, varRat, varEpsRat,
        sqrtLowerBound, sqrtUpperBound, invStdLower, invStdUpper, coeff, hcoeff, hlayer]
        using hlow_raw
    have hhi : layerNormReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
      simpa [bounds, layerNormBounds, hne, μRat, centered, varRat, varEpsRat,
        sqrtLowerBound, sqrtUpperBound, invStdLower, invStdUpper, coeff, hcoeff, hlayer]
        using hhigh_raw
    exact And.intro hlo hhi
  · have hcoeff_lt : coeff < 0 := lt_of_not_ge hcoeff
    have hcoeff_real : (coeff : Real) ≤ 0 := by
      exact_mod_cast (le_of_lt hcoeff_lt)
    have hlow_raw :
        (beta i : Real) + (coeff : Real) * (invStdUpper : Real) ≤
          (beta i : Real) + (coeff : Real) * invStd := by
      have hmul := mul_le_mul_of_nonpos_left hinv_upper hcoeff_real
      exact add_le_add_right hmul (beta i : Real)
    have hhigh_raw :
        (beta i : Real) + (coeff : Real) * invStd ≤
          (beta i : Real) + (coeff : Real) * (invStdLower : Real) := by
      have hmul := mul_le_mul_of_nonpos_left hinv_lower hcoeff_real
      exact add_le_add_right hmul (beta i : Real)
    have hlo : (bounds.1 i : Real) ≤ layerNormReal eps gamma beta x i := by
      simpa [bounds, layerNormBounds, hne, μRat, centered, varRat, varEpsRat,
        sqrtLowerBound, sqrtUpperBound, invStdLower, invStdUpper, coeff, hcoeff, hlayer]
        using hlow_raw
    have hhi : layerNormReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
      simpa [bounds, layerNormBounds, hne, μRat, centered, varRat, varEpsRat,
        sqrtLowerBound, sqrtUpperBound, invStdLower, invStdUpper, coeff, hcoeff, hlayer]
        using hhigh_raw
    exact And.intro hlo hhi

/-- Interval bounds for LayerNorm outputs from per-coordinate intervals. -/
def layerNormIntervalBounds {n : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat) (lo hi : Fin n → Rat) :
    (Fin n → Rat) × (Fin n → Rat) :=
  if n = 0 then
    (fun _ => 0, fun _ => 0)
  else
    let μLo := mean lo
    let μHi := meanUpper hi
    let centeredBound : Fin n → Rat := fun i =>
      max |lo i - μHi| |hi i - μLo|
    let invStdBound : Rat := ratDivUp 1 (sqrtLower eps)
    let radius : Fin n → Rat := fun i => |gamma i| * centeredBound i * invStdBound
    (fun i => beta i - radius i, fun i => beta i + radius i)

/-- `layerNormIntervalBounds` soundness for real LayerNorm outputs. -/
theorem layerNormIntervalBounds_spec {n : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat) (lo hi : Fin n → Rat) (x : Fin n → Rat)
    (hne : n ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps)
    (hlo : ∀ i, lo i ≤ x i) (hhi : ∀ i, x i ≤ hi i) :
    let bounds := layerNormIntervalBounds eps gamma beta lo hi
    ∀ i,
      (bounds.1 i : Real) ≤ layerNormReal eps gamma beta x i ∧
        layerNormReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
  classical
  intro bounds i
  let μLo : Rat := mean lo
  let μHi : Rat := meanUpper hi
  let centeredBound : Fin n → Rat := fun j => max |lo j - μHi| |hi j - μLo|
  let invStdBound : Rat := ratDivUp 1 (sqrtLower eps)
  let varEps : Real := (varianceRat x : Real) + (eps : Real)
  let μ : Real := meanRat x
  let invStd : Real := (Real.sqrt varEps)⁻¹
  have hcentered_nonneg : 0 ≤ (centeredBound i : Real) := by
    have h0 : 0 ≤ centeredBound i := by
      dsimp [centeredBound]
      exact le_trans (abs_nonneg _) (le_max_left _ _)
    exact ratToReal_nonneg_of_nonneg h0
  have hcentered_abs : |(x i : Real) - μ| ≤ (centeredBound i : Real) := by
    have hmean_lo_real : (μLo : Real) ≤ μ := by
      have hmean_rat : (meanRat lo : Real) ≤ (meanRat x : Real) :=
        meanRat_le_meanRat_real lo x hne hlo
      have hdown : (μLo : Real) ≤ (meanRat lo : Real) := by
        simpa [μLo, mean_def lo hne] using ratRoundDown_le_real (meanRat lo)
      exact le_trans hdown hmean_rat
    have hmean_hi_real : μ ≤ (μHi : Real) := by
      have hmean_rat : (meanRat x : Real) ≤ (meanRat hi : Real) :=
        meanRat_le_meanRat_real x hi hne hhi
      have hup : (meanRat hi : Real) ≤ (μHi : Real) := by
        simpa [μHi, meanUpper_def hi hne] using real_le_ratRoundUp (meanRat hi)
      exact le_trans hmean_rat hup
    have hlo' : (lo i : Real) - (μHi : Real) ≤ (x i : Real) - μ := by
      have h1 : (lo i : Real) - (μHi : Real) ≤ (lo i : Real) - μ := by
        exact sub_le_sub_left hmean_hi_real (lo i : Real)
      have h2 : (lo i : Real) - μ ≤ (x i : Real) - μ := by
        exact sub_le_sub_right
          (by
            exact ratToReal_le_of_le (hlo i))
          μ
      exact le_trans h1 h2
    have hhi' : (x i : Real) - μ ≤ (hi i : Real) - (μLo : Real) := by
      have h1 : (x i : Real) - μ ≤ (hi i : Real) - μ := by
        exact sub_le_sub_right
          (by
            exact ratToReal_le_of_le (hhi i))
          μ
      have h2 : (hi i : Real) - μ ≤ (hi i : Real) - (μLo : Real) := by
        exact sub_le_sub_left hmean_lo_real (hi i : Real)
      exact le_trans h1 h2
    have hbound := abs_le_max_of_bounds hlo' hhi'
    simpa [centeredBound, μLo, μHi, ratToReal_abs, ratToReal_sub,
      ratToReal_max] using hbound
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
    exact (Rat.cast_pos (K := Real) (q := sqrtLower eps)).2 hsqrt
  have hinv_sqrt : invStd ≤ (sqrtLower eps : Real)⁻¹ := by
    have h := inv_anti₀ hsqrt_lower_pos hsqrt_lower
    simpa [invStd] using h
  have hinv_bound : (sqrtLower eps : Real)⁻¹ ≤ (invStdBound : Real) := by
    have hy : sqrtLower eps ≠ 0 := ne_of_gt hsqrt
    have hdiv := ratDivUp_ge_real (x := 1) (y := sqrtLower eps) hy
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
  let radius : Fin n → Rat := fun j => |gamma j| * centeredBound j * invStdBound
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
    (eps : Rat) (gamma beta : Fin n → Rat) (absBound : Rat) :
    (Fin n → Rat) × (Fin n → Rat) :=
  let centeredBound : Rat := 2 * absBound
  let invStdBound : Rat := ratDivUp 1 (sqrtLower eps)
  let radius : Fin n → Rat := fun i => |gamma i| * centeredBound * invStdBound
  (fun i => beta i - radius i, fun i => beta i + radius i)

/-- `layerNormAbsBounds` soundness for real LayerNorm outputs under absolute input bounds. -/
theorem layerNormAbsBounds_spec {n : Nat}
    (eps : Rat) (gamma beta : Fin n → Rat) (absBound : Rat) (x : Fin n → Rat)
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
          exact ratToReal_abs_le_of_le (habs j))
    simpa [meanReal_eq_meanRat] using h
  have hbound_nonneg : 0 ≤ absBound := by
    have hposn : 0 < n := Nat.pos_of_ne_zero hne
    let i0 : Fin n := ⟨0, hposn⟩
    have h0 : 0 ≤ |x i0| := abs_nonneg _
    exact le_trans h0 (habs i0)
  let centeredBound : Rat := 2 * absBound
  let invStdBound : Rat := ratDivUp 1 (sqrtLower eps)
  let varEps : Real := (varianceRat x : Real) + (eps : Real)
  let μ : Real := meanRat x
  let invStd : Real := (Real.sqrt varEps)⁻¹
  have hcentered_abs : |(x i : Real) - μ| ≤ (centeredBound : Real) := by
    have h1 : |(x i : Real) - μ| ≤ |(x i : Real)| + |μ| := by
      simpa [sub_eq_add_neg, abs_neg] using abs_add_le (x i : Real) (-μ)
    have hx : |(x i : Real)| ≤ (absBound : Real) := by
      exact ratToReal_abs_le_of_le (habs i)
    have hmu : |μ| ≤ (absBound : Real) := by
      simpa [μ] using hmean_abs_real
    have h2 : |(x i : Real)| + |μ| ≤ (absBound : Real) + (absBound : Real) :=
      add_le_add hx hmu
    have h12 : |(x i : Real) - μ| ≤ (absBound : Real) + (absBound : Real) :=
      le_trans h1 h2
    simpa [centeredBound, two_mul] using h12
  have hbound_nonneg_real : 0 ≤ (absBound : Real) := by
    exact ratToReal_nonneg_of_nonneg hbound_nonneg
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
    exact (Rat.cast_pos (K := Real) (q := sqrtLower eps)).2 hsqrt
  have hinv_sqrt : invStd ≤ (sqrtLower eps : Real)⁻¹ := by
    have h := inv_anti₀ hsqrt_lower_pos hsqrt_lower
    simpa [invStd] using h
  have hinv_bound : (sqrtLower eps : Real)⁻¹ ≤ (invStdBound : Real) := by
    have hy : sqrtLower eps ≠ 0 := ne_of_gt hsqrt
    have hdiv := ratDivUp_ge_real (x := 1) (y := sqrtLower eps) hy
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
  let radius : Fin n → Rat := fun j => |gamma j| * centeredBound * invStdBound
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
    (eps : Rat) (gamma beta : Fin n → Rat) (absBound : Rat) (x : Fin n → Real)
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
  let centeredBound : Rat := 2 * absBound
  let invStdBound : Rat := ratDivUp 1 (sqrtLower eps)
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
    exact (Rat.cast_pos (K := Real) (q := sqrtLower eps)).2 hsqrt
  have hinv_sqrt : invStd ≤ (sqrtLower eps : Real)⁻¹ := by
    have h := inv_anti₀ hsqrt_lower_pos hsqrt_lower
    simpa [invStd] using h
  have hinv_bound : (sqrtLower eps : Real)⁻¹ ≤ (invStdBound : Real) := by
    have hy : sqrtLower eps ≠ 0 := ne_of_gt hsqrt
    have hdiv := ratDivUp_ge_real (x := 1) (y := sqrtLower eps) hy
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
  let radius : Fin n → Rat := fun j => |gamma j| * centeredBound * invStdBound
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
    (eps : Rat) (gamma beta : Fin n → Rat) (lo hi : Fin n → Rat) (x : Fin n → Real)
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
      simpa [mean_def lo hne] using ratRoundDown_le_real (meanRat lo)
    exact le_trans hdown hrat
  have hmean_hi : meanReal x ≤ (meanUpper hi : Real) := by
    have h :=
      meanReal_le_meanReal (x := x) (y := fun j => (hi j : Real)) hne
        (fun j => hhi j)
    have hrat : meanReal x ≤ (meanRat hi : Real) := by
      simpa [meanReal_eq_meanRat] using h
    have hup : (meanRat hi : Real) ≤ (meanUpper hi : Real) := by
      simpa [meanUpper_def hi hne] using real_le_ratRoundUp (meanRat hi)
    exact le_trans hrat hup
  let μLo : Rat := mean lo
  let μHi : Rat := meanUpper hi
  let centeredBound : Fin n → Rat := fun j => max |lo j - μHi| |hi j - μLo|
  let invStdBound : Rat := ratDivUp 1 (sqrtLower eps)
  let varEps : Real := varianceReal x + (eps : Real)
  let μ : Real := meanReal x
  let invStd : Real := (Real.sqrt varEps)⁻¹
  have hcentered_nonneg : 0 ≤ (centeredBound i : Real) := by
    have h0 : 0 ≤ centeredBound i := by
      dsimp [centeredBound]
      exact le_trans (abs_nonneg _) (le_max_left _ _)
    exact ratToReal_nonneg_of_nonneg h0
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
    simpa [centeredBound, μLo, μHi, ratToReal_abs, ratToReal_sub,
      ratToReal_max] using hbound
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
    exact (Rat.cast_pos (K := Real) (q := sqrtLower eps)).2 hsqrt
  have hinv_sqrt : invStd ≤ (sqrtLower eps : Real)⁻¹ := by
    have h := inv_anti₀ hsqrt_lower_pos hsqrt_lower
    simpa [invStd] using h
  have hinv_bound : (sqrtLower eps : Real)⁻¹ ≤ (invStdBound : Real) := by
    have hy : sqrtLower eps ≠ 0 := ne_of_gt hsqrt
    have hdiv := ratDivUp_ge_real (x := 1) (y := sqrtLower eps) hy
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
  let radius : Fin n → Rat := fun j => |gamma j| * centeredBound j * invStdBound
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
