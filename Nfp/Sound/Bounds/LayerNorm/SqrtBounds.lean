-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Algebra.Order.Ring.Basic
import Mathlib.Data.Nat.Sqrt
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Rat.Cast.Order
import Nfp.Core.Basic

/-!
Square-root bounds for LayerNorm intervals.

This module isolates the rational sqrt lower/upper bounds and their basic
nonnegativity/positivity lemmas so the main LayerNorm bounds stay focused.
-/

namespace Nfp

namespace Sound

namespace Bounds

/-! Square-root bounds. -/

lemma rat_nat_cast_nonneg (n : Nat) : (0 : Rat) ≤ (n : Rat) := by
  simp

lemma rat_nat_cast_pos {n : Nat} (h : 0 < n) : (0 : Rat) < (n : Rat) := by
  exact (Nat.cast_pos (α := Rat)).2 h

/-- `ratRoundDown` preserves nonnegativity for nonnegative divisions. -/
theorem ratRoundDown_nonneg_div {a b : Rat} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    0 ≤ ratRoundDown (a / b) := by
  exact ratRoundDown_nonneg (q := a / b) (by exact div_nonneg ha hb)

/-- `ratRoundUp` preserves nonnegativity for nonnegative divisions. -/
theorem ratRoundUp_nonneg_div {a b : Rat} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    0 ≤ ratRoundUp (a / b) := by
  exact ratRoundUp_nonneg (q := a / b) (by exact div_nonneg ha hb)

/-- `ratRoundUp` preserves positivity for positive divisions. -/
theorem ratRoundUp_pos_div {a b : Rat} (ha : 0 < a) (hb : 0 < b) :
    0 < ratRoundUp (a / b) := by
  exact ratRoundUp_pos (q := a / b) (by exact div_pos ha hb)

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
def sqrtLowerScale : Nat := 1048576

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
  have hnum : 0 ≤ (Nat.sqrt q.num.natAbs : Rat) :=
    rat_nat_cast_nonneg (Nat.sqrt q.num.natAbs)
  have hden : 0 ≤ (Nat.sqrt q.den : Rat) + 1 := by
    simpa using rat_nat_cast_nonneg (Nat.sqrt q.den + 1)
  exact ratRoundDown_nonneg_div hnum hden

/-- `sqrtUpperBase` is nonnegative. -/
theorem sqrtUpperBase_nonneg (q : Rat) : 0 ≤ sqrtUpperBase q := by
  classical
  unfold sqrtUpperBase
  have hnum : 0 ≤ (Nat.sqrt q.num.natAbs : Rat) + 1 := by
    simpa using rat_nat_cast_nonneg (Nat.sqrt q.num.natAbs + 1)
  have hden : 0 ≤ (Nat.sqrt q.den : Rat) :=
    rat_nat_cast_nonneg (Nat.sqrt q.den)
  exact ratRoundUp_nonneg_div hnum hden

/-- `sqrtUpperBase` is always positive. -/
theorem sqrtUpperBase_pos (q : Rat) : 0 < sqrtUpperBase q := by
  classical
  unfold sqrtUpperBase
  have hnum_pos : (0 : Rat) < (Nat.sqrt q.num.natAbs : Rat) + 1 := by
    simpa using rat_nat_cast_pos (Nat.succ_pos (Nat.sqrt q.num.natAbs))
  have hden_pos : (0 : Rat) < (Nat.sqrt q.den : Rat) := by
    have hden : 0 < q.den := q.den_pos
    exact rat_nat_cast_pos (Nat.sqrt_pos.2 hden)
  exact ratRoundUp_pos_div hnum_pos hden_pos

/-- `sqrtLowerAlt` is nonnegative. -/
theorem sqrtLowerAlt_nonneg (q : Rat) : 0 ≤ sqrtLowerAlt q := by
  classical
  unfold sqrtLowerAlt
  have hnum : 0 ≤ (Nat.sqrt (q.num.natAbs * q.den) : Rat) :=
    rat_nat_cast_nonneg (Nat.sqrt (q.num.natAbs * q.den))
  have hden : 0 ≤ (q.den : Rat) :=
    rat_nat_cast_nonneg q.den
  exact ratRoundDown_nonneg_div hnum hden

/-- `sqrtUpperAlt` is nonnegative. -/
theorem sqrtUpperAlt_nonneg (q : Rat) : 0 ≤ sqrtUpperAlt q := by
  classical
  unfold sqrtUpperAlt
  have hnum : 0 ≤ (Nat.sqrt (q.num.natAbs * q.den) : Rat) + 1 := by
    simpa using rat_nat_cast_nonneg (Nat.sqrt (q.num.natAbs * q.den) + 1)
  have hden : 0 ≤ (q.den : Rat) :=
    rat_nat_cast_nonneg q.den
  exact ratRoundUp_nonneg_div hnum hden

/-- `sqrtUpperAlt` is always positive. -/
theorem sqrtUpperAlt_pos (q : Rat) : 0 < sqrtUpperAlt q := by
  classical
  unfold sqrtUpperAlt
  have hnum_pos :
      (0 : Rat) < (Nat.sqrt (q.num.natAbs * q.den) : Rat) + 1 := by
    simpa using rat_nat_cast_pos (Nat.succ_pos (Nat.sqrt (q.num.natAbs * q.den)))
  have hden_pos : (0 : Rat) < (q.den : Rat) :=
    rat_nat_cast_pos q.den_pos
  exact ratRoundUp_pos_div hnum_pos hden_pos

/-- `sqrtUpperScaled` is nonnegative. -/
theorem sqrtUpperScaled_nonneg (q : Rat) : 0 ≤ sqrtUpperScaled q := by
  classical
  unfold sqrtUpperScaled
  have hnum :
      0 ≤ (Nat.sqrt (q.num.natAbs * q.den * sqrtLowerScale * sqrtLowerScale) : Rat) + 1 := by
    simpa using rat_nat_cast_nonneg
      (Nat.sqrt (q.num.natAbs * q.den * sqrtLowerScale * sqrtLowerScale) + 1)
  have hden : 0 ≤ (q.den : Rat) * (sqrtLowerScale : Rat) := by
    simpa [Nat.cast_mul] using rat_nat_cast_nonneg (q.den * sqrtLowerScale)
  exact ratRoundUp_nonneg_div hnum hden

/-- `sqrtUpperScaled` is always positive. -/
theorem sqrtUpperScaled_pos (q : Rat) : 0 < sqrtUpperScaled q := by
  classical
  unfold sqrtUpperScaled
  have hnum_pos :
      (0 : Rat) <
        (Nat.sqrt (q.num.natAbs * q.den * sqrtLowerScale * sqrtLowerScale) : Rat) + 1 := by
    simpa using rat_nat_cast_pos
      (Nat.succ_pos (Nat.sqrt (q.num.natAbs * q.den * sqrtLowerScale * sqrtLowerScale)))
  have hden_pos : (0 : Rat) < (q.den : Rat) * (sqrtLowerScale : Rat) := by
    have hden : 0 < q.den := q.den_pos
    have hscale : 0 < sqrtLowerScale := by
      simp [sqrtLowerScale]
    simpa [Nat.cast_mul] using rat_nat_cast_pos (Nat.mul_pos hden hscale)
  exact ratRoundUp_pos_div hnum_pos hden_pos

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

/-! Real-valued bounds. -/

/-- Cast a nonnegative rational as `num.natAbs / den`. -/
theorem rat_cast_eq_num_den {q : Rat} (hq : 0 ≤ q) :
    (q : Real) = (q.num.natAbs : Real) / q.den := by
  have hnum_nonneg : 0 ≤ q.num := (Rat.num_nonneg (q := q)).2 hq
  have hnum_eq : (q.num.natAbs : Int) = q.num := by
    exact (Int.natAbs_of_nonneg hnum_nonneg)
  have hnum_cast : (q.num : Real) = (q.num.natAbs : Real) := by
    exact (congrArg (fun z : Int => (z : Real)) hnum_eq).symm
  have hq_rat : (q : Real) = (q.num : Real) / q.den := by
    simp [Rat.cast_def]
  calc
    (q : Real) = (q.num : Real) / q.den := hq_rat
    _ = (q.num.natAbs : Real) / q.den := by
      rw [hnum_cast]

/-- Cast a nonnegative rational as `num.natAbs * den / den^2`. -/
theorem rat_cast_eq_num_den_mul {q : Rat} (hq : 0 ≤ q) :
    (q : Real) = (q.num.natAbs : Real) * q.den / (q.den : Real) ^ 2 := by
  have hq_cast : (q : Real) = (q.num.natAbs : Real) / q.den :=
    rat_cast_eq_num_den (q := q) hq
  have hden_ne : (q.den : Real) ≠ 0 := by
    exact_mod_cast q.den_pos.ne'
  have hq_eq :
      (q.num.natAbs : Real) / q.den =
        (q.num.natAbs : Real) * q.den / (q.den : Real) ^ 2 := by
    field_simp [hden_ne]
  exact hq_cast.trans hq_eq

/-- Cast a nonnegative rational as `num.natAbs * den * scale^2 / (den * scale)^2`. -/
theorem rat_cast_eq_num_den_scale {q : Rat} (hq : 0 ≤ q) {scale : Nat} (hscale : 0 < scale) :
    (q : Real) =
      (q.num.natAbs : Real) * q.den * (scale : Real) * (scale : Real) /
        ((q.den : Real) * (scale : Real)) ^ 2 := by
  have hq_cast : (q : Real) = (q.num.natAbs : Real) / q.den :=
    rat_cast_eq_num_den (q := q) hq
  have hden_pos : 0 < (q.den : Real) := by
    exact_mod_cast q.den_pos
  have hscale_pos : 0 < (scale : Real) := by
    exact_mod_cast hscale
  have hden_scale_ne : ((q.den : Real) * (scale : Real)) ≠ 0 := by
    exact ne_of_gt (mul_pos hden_pos hscale_pos)
  have hq_eq :
      (q.num.natAbs : Real) / q.den =
        (q.num.natAbs : Real) * q.den * (scale : Real) * (scale : Real) /
          ((q.den : Real) * (scale : Real)) ^ 2 := by
    field_simp [hden_scale_ne]
  exact hq_cast.trans hq_eq

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
    simpa [pow_two] using (mul_pos hbpos hbpos)
  have hdiv : (a ^ 2 : Real) / (b + 1) ^ 2 ≤ (num : Real) / den := by
    exact (div_le_div_iff₀ hbpos2 hden_pos).2 hmul
  have hpow : ((a : Real) / (b + 1 : Real)) ^ 2 = (a ^ 2 : Real) / (b + 1) ^ 2 := by
    simp [pow_two, div_mul_div_comm]
  have hq_cast : (q : Real) = (num : Real) / den := by
    simpa [num, den] using (rat_cast_eq_num_den (q := q) hq)
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
    simpa [pow_two] using (mul_pos hbpos hbpos)
  have hdiv : (num : Real) / den ≤ (a + 1) ^ 2 / (b : Real) ^ 2 := by
    exact (div_le_div_iff₀ hden_pos hbpos2).2 hmul
  have hpow : ((a + 1 : Real) / (b : Real)) ^ 2 = (a + 1) ^ 2 / (b : Real) ^ 2 := by
    simp [pow_two, div_mul_div_comm]
  have hq_cast : (q : Real) = (num : Real) / den := by
    simpa [num, den] using (rat_cast_eq_num_den (q := q) hq)
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

/-
  Local automation for monotone scaling steps in real-valued bounds.
-/
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
    exact pow_pos hden_pos 2
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
    simpa [num, den] using (rat_cast_eq_num_den_mul (q := q) hq)
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
  have hscale_pos_nat : 0 < scale := by
    simp [scale, sqrtLowerScale]
  have hscale_pos : 0 < (scale : Real) := by
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
  have hq_cast :
      (q : Real) =
        (num : Real) * den * (scale : Real) * (scale : Real) /
          ((den : Real) * (scale : Real)) ^ 2 := by
    simpa [num, den, scale] using
      (rat_cast_eq_num_den_scale (q := q) hq (scale := scale) hscale_pos_nat)
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
      _ = (q : Real) := by simp [hq_cast]
  have hnonneg : 0 ≤ (a : Real) / ((den : Real) * (scale : Real)) := by
    have hnum_nonneg : 0 ≤ (a : Real) := by exact_mod_cast (Nat.zero_le a)
    have hden_nonneg : 0 ≤ (den : Real) * (scale : Real) := by
      exact mul_nonneg (le_of_lt hden_pos) (le_of_lt hscale_pos)
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
    exact pow_pos hden_pos 2
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
    simpa [num, den] using (rat_cast_eq_num_den_mul (q := q) hq)
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
  have hscale_pos_nat : 0 < scale := by
    simp [scale, sqrtLowerScale]
  have hscale_pos : 0 < (scale : Real) := by
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
      exact mul_le_mul hmul (le_refl _) hden_sq_nonneg (sq_nonneg _)
    exact (div_le_div_iff₀ hdenScale_pos2 hdenScale_pos2).2 hmul'
  have hq_cast' :
      (q : Real) =
        ((num : Real) * den * (scale : Real) * (scale : Real)) /
          ((den : Real) * (scale : Real)) ^ 2 := by
    simpa [num, den, scale] using
      (rat_cast_eq_num_den_scale (q := q) hq (scale := scale) hscale_pos_nat)
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
      exact mul_nonneg (le_of_lt hden_pos) (le_of_lt hscale_pos)
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

end Bounds

end Sound

end Nfp
