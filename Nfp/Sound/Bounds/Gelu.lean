-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Abs
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Nfp.Core.Basic

/-!
Tanh-based GELU bounds for GPT-2 style MLPs.
These bounds are used to propagate interval constraints through nonlinear gates.
-/

namespace Nfp

namespace Sound

namespace Bounds

/-- Tanh-based GELU activation used by GPT-2 (approximate form). -/
noncomputable def geluTanh (x : Real) : Real :=
  let k : Real := Real.sqrt (2 / Real.pi)
  let c : Real := (44715 : Real) / 1000000
  x * ((1 + Real.tanh (k * (x + c * x ^ 3))) / 2)

/-- The hyperbolic tangent is bounded in absolute value by `1`. -/
theorem abs_tanh_le_one (x : Real) : |Real.tanh x| ≤ 1 := by
  have hpos_exp : 0 < Real.exp x := Real.exp_pos x
  have hpos_exp_neg : 0 < Real.exp (-x) := Real.exp_pos (-x)
  have hsum_pos : 0 < Real.exp x + Real.exp (-x) :=
    add_pos hpos_exp hpos_exp_neg
  have hsum_nonneg : 0 ≤ Real.exp x + Real.exp (-x) := le_of_lt hsum_pos
  have habs : |Real.exp x - Real.exp (-x)| ≤ Real.exp x + Real.exp (-x) := by
    have h := abs_add_le (Real.exp x) (-Real.exp (-x))
    simpa [sub_eq_add_neg, abs_neg, abs_of_nonneg (le_of_lt hpos_exp),
      abs_of_nonneg (le_of_lt hpos_exp_neg)] using h
  calc
    |Real.tanh x| =
        |(Real.exp x - Real.exp (-x)) / (Real.exp x + Real.exp (-x))| := by
          simp [Real.tanh_eq]
    _ = |Real.exp x - Real.exp (-x)| / (Real.exp x + Real.exp (-x)) := by
          simp [abs_div, abs_of_nonneg hsum_nonneg]
    _ ≤ (Real.exp x + Real.exp (-x)) / (Real.exp x + Real.exp (-x)) := by
          exact div_le_div_of_nonneg_right habs hsum_nonneg
    _ = 1 := by
          have hne : Real.exp x + Real.exp (-x) ≠ 0 := ne_of_gt hsum_pos
          simp [hne]

/-- The tanh coefficient in `geluTanh` lies in `[0, 1]`. -/
theorem geluTanh_coeff_bounds (x : Real) :
    0 ≤
        (1 +
            Real.tanh
              (Real.sqrt (2 / Real.pi) *
                (x + (44715 : Real) / 1000000 * x ^ 3))) /
          2 ∧
      (1 +
            Real.tanh
              (Real.sqrt (2 / Real.pi) *
                (x + (44715 : Real) / 1000000 * x ^ 3))) /
          2 ≤ 1 := by
  have habs :=
    abs_tanh_le_one
      (Real.sqrt (2 / Real.pi) * (x + (44715 : Real) / 1000000 * x ^ 3))
  have hbounds := abs_le.mp habs
  constructor <;> nlinarith

/-- `geluTanh` outputs stay between `min x 0` and `max x 0`. -/
theorem geluTanh_bounds (x : Real) :
    min x 0 ≤ geluTanh x ∧ geluTanh x ≤ max x 0 := by
  by_cases hx : 0 ≤ x
  · have hcoeff := geluTanh_coeff_bounds x
    have hnonneg :
        0 ≤ x *
          ((1 +
                Real.tanh
                  (Real.sqrt (2 / Real.pi) *
                    (x + (44715 : Real) / 1000000 * x ^ 3))) /
            2) := by
      exact mul_nonneg hx hcoeff.1
    have hle :
        x *
          ((1 +
                Real.tanh
                  (Real.sqrt (2 / Real.pi) *
                    (x + (44715 : Real) / 1000000 * x ^ 3))) /
            2) ≤ x := by
      have h := mul_le_mul_of_nonneg_left hcoeff.2 hx
      simpa [mul_one] using h
    have h0 : 0 ≤ geluTanh x := by
      simpa [geluTanh] using hnonneg
    have h1 : geluTanh x ≤ x := by
      simpa [geluTanh] using hle
    simpa [min_eq_right hx, max_eq_left hx] using And.intro h0 h1
  · have hx' : x ≤ 0 := le_of_lt (lt_of_not_ge hx)
    have hcoeff := geluTanh_coeff_bounds x
    have hle0 :
        x *
          ((1 +
                Real.tanh
                  (Real.sqrt (2 / Real.pi) *
                    (x + (44715 : Real) / 1000000 * x ^ 3))) /
            2) ≤ 0 := by
      exact mul_nonpos_of_nonpos_of_nonneg hx' hcoeff.1
    have hxle :
        x ≤
          x *
            ((1 +
                  Real.tanh
                    (Real.sqrt (2 / Real.pi) *
                      (x + (44715 : Real) / 1000000 * x ^ 3))) /
              2) := by
      have h := mul_le_mul_of_nonpos_left hcoeff.2 hx'
      simpa [mul_one] using h
    have h0 : geluTanh x ≤ 0 := by
      simpa [geluTanh] using hle0
    have h1 : x ≤ geluTanh x := by
      simpa [geluTanh] using hxle
    simpa [min_eq_left hx', max_eq_right hx'] using And.intro h1 h0

/-- Interval bounds for GELU given input bounds. -/
def geluInterval (lo hi : Rat) : Rat × Rat :=
  (if lo ≤ 0 then lo else 0, if 0 ≤ hi then hi else 0)

/-- `geluInterval` soundly bounds `geluTanh` on a real interval. -/
theorem geluInterval_bounds {lo hi : Rat} {x : Real}
    (hlo : (lo : Real) ≤ x) (hhi : x ≤ (hi : Real)) :
    (geluInterval lo hi).1 ≤ (geluTanh x : Real) ∧
      (geluTanh x : Real) ≤ (geluInterval lo hi).2 := by
  have hgelu := geluTanh_bounds x
  by_cases hlo0 : lo ≤ 0
  · have hlo0r : (lo : Real) ≤ 0 := by
      exact (ratToReal_nonpos_iff (x := lo)).2 hlo0
    have hmin : min (lo : Real) 0 ≤ min x 0 := min_le_min hlo le_rfl
    have hlo' : (lo : Real) ≤ geluTanh x := by
      have hmin' : (lo : Real) ≤ min x 0 := by
        simpa [min_eq_left hlo0r] using hmin
      exact le_trans hmin' hgelu.1
    have hmax : max x 0 ≤ max (hi : Real) 0 := max_le_max hhi le_rfl
    have hhi' : geluTanh x ≤ max (hi : Real) 0 := le_trans hgelu.2 hmax
    constructor
    · simpa [geluInterval, hlo0] using hlo'
    · by_cases hhi0 : 0 ≤ hi
      · have hhi0r : 0 ≤ (hi : Real) := by
          exact ratToReal_nonneg_of_nonneg hhi0
        have hmax' : max (hi : Real) 0 = (hi : Real) := max_eq_left hhi0r
        simpa [geluInterval, hhi0, hmax'] using hhi'
      · have hhi0r : (hi : Real) ≤ 0 := by
          exact (ratToReal_nonpos_iff (x := hi)).2 (le_of_not_ge hhi0)
        have hx0 : x ≤ 0 := le_trans hhi hhi0r
        have hmax' : max x 0 = 0 := max_eq_right hx0
        have hhi'' : geluTanh x ≤ (0 : Real) := by
          simpa [hmax'] using hgelu.2
        simpa [geluInterval, hhi0, ratToReal_zero] using hhi''
  · have hlo0r : 0 ≤ (lo : Real) := by
      exact ratToReal_nonneg_of_nonneg (le_of_not_ge hlo0)
    have hx0 : 0 ≤ x := le_trans hlo0r hlo
    have hmin' : min x 0 = 0 := min_eq_right hx0
    have hlo' : (0 : Real) ≤ geluTanh x := by
      simpa [hmin'] using hgelu.1
    have hmax : max x 0 ≤ max (hi : Real) 0 := max_le_max hhi le_rfl
    have hhi' : geluTanh x ≤ max (hi : Real) 0 := le_trans hgelu.2 hmax
    constructor
    · simpa [geluInterval, hlo0, ratToReal_zero] using hlo'
    · by_cases hhi0 : 0 ≤ hi
      · have hhi0r : 0 ≤ (hi : Real) := by
          exact ratToReal_nonneg_of_nonneg hhi0
        have hmax' : max (hi : Real) 0 = (hi : Real) := max_eq_left hhi0r
        simpa [geluInterval, hhi0, hmax'] using hhi'
      · have hhi0r : (hi : Real) ≤ 0 := by
          exact (ratToReal_nonpos_iff (x := hi)).2 (le_of_not_ge hhi0)
        have hx0' : x ≤ 0 := le_trans hhi hhi0r
        have hmax' : max x 0 = 0 := max_eq_right hx0'
        have hhi'' : geluTanh x ≤ (0 : Real) := by
          simpa [hmax'] using hgelu.2
        simpa [geluInterval, hhi0, ratToReal_zero] using hhi''

end Bounds

end Sound

end Nfp
