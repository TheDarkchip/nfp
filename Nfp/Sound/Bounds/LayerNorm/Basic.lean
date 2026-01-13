-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Algebra.Order.Field.Basic
public import Mathlib.Algebra.Order.Ring.Basic
public import Mathlib.Data.Real.Sqrt
public import Mathlib.Data.Rat.BigOperators
public import Mathlib.Data.Rat.Cast.Order
public import Nfp.Core.Basic
public import Nfp.Sound.Bounds.LayerNorm.MeanVariance
public import Nfp.Sound.Bounds.LayerNorm.SqrtBounds
public import Nfp.Sound.Linear.FinFold

/-!
LayerNorm interval bounds for rational inputs.

This module computes rational interval bounds for LayerNorm outputs and proves
those bounds sound for real-valued LayerNorm semantics.
-/

@[expose] public section

namespace Nfp

namespace Sound

namespace Bounds

open scoped BigOperators

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
  · have hbounds : x * lo ≤ x * y ∧ x * y ≤ x * hi := by
      exact ⟨mul_le_mul_of_nonneg_left hlo hx, mul_le_mul_of_nonneg_left hhi hx⟩
    simpa [scaleInterval, hx] using hbounds
  · have hx' : x ≤ 0 := le_of_lt (lt_of_not_ge hx)
    have hbounds : x * hi ≤ x * y ∧ x * y ≤ x * lo := by
      exact ⟨mul_le_mul_of_nonpos_left hhi hx', mul_le_mul_of_nonpos_left hlo hx'⟩
    simpa [scaleInterval, hx] using hbounds

/-- `scaleInterval` bounds interpreted in the reals. -/
theorem scaleInterval_bounds_real {x lo hi : Rat} {y : Real}
    (hlo : (lo : Real) ≤ y) (hhi : y ≤ (hi : Real)) :
    let bounds := scaleInterval x lo hi
    (bounds.1 : Real) ≤ (x : Real) * y ∧ (x : Real) * y ≤ (bounds.2 : Real) := by
  by_cases hx : 0 ≤ x
  · have hx' : 0 ≤ (x : Real) := ratToReal_nonneg_of_nonneg hx
    have hbounds : (x : Real) * (lo : Real) ≤ (x : Real) * y ∧
        (x : Real) * y ≤ (x : Real) * (hi : Real) := by
      exact ⟨mul_le_mul_of_nonneg_left hlo hx', mul_le_mul_of_nonneg_left hhi hx'⟩
    simpa [scaleInterval, hx] using hbounds
  · have hx' : x ≤ 0 := le_of_lt (lt_of_not_ge hx)
    have hx'' : (x : Real) ≤ 0 := (ratToReal_nonpos_iff (x := x)).2 hx'
    have hbounds : (x : Real) * (hi : Real) ≤ (x : Real) * y ∧
        (x : Real) * y ≤ (x : Real) * (lo : Real) := by
      exact ⟨mul_le_mul_of_nonpos_left hhi hx'', mul_le_mul_of_nonpos_left hlo hx''⟩
    simpa [scaleInterval, hx] using hbounds

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
      simpa only [add_comm] using add_le_add_left hmul (beta i : Real)
    have hhigh_raw :
        (beta i : Real) + (coeff : Real) * invStd ≤
          (beta i : Real) + (coeff : Real) * (invStdUpper : Real) := by
      have hmul := mul_le_mul_of_nonneg_left hinv_upper hcoeff_real
      simpa only [add_comm] using add_le_add_left hmul (beta i : Real)
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
      simpa only [add_comm] using add_le_add_left hmul (beta i : Real)
    have hhigh_raw :
        (beta i : Real) + (coeff : Real) * invStd ≤
          (beta i : Real) + (coeff : Real) * (invStdLower : Real) := by
      have hmul := mul_le_mul_of_nonpos_left hinv_lower hcoeff_real
      simpa only [add_comm] using add_le_add_left hmul (beta i : Real)
    have hlo : (bounds.1 i : Real) ≤ layerNormReal eps gamma beta x i := by
      simpa [bounds, layerNormBounds, hne, μRat, centered, varRat, varEpsRat,
        sqrtLowerBound, sqrtUpperBound, invStdLower, invStdUpper, coeff, hcoeff, hlayer]
        using hlow_raw
    have hhi : layerNormReal eps gamma beta x i ≤ (bounds.2 i : Real) := by
      simpa [bounds, layerNormBounds, hne, μRat, centered, varRat, varEpsRat,
        sqrtLowerBound, sqrtUpperBound, invStdLower, invStdUpper, coeff, hcoeff, hlayer]
        using hhigh_raw
    exact And.intro hlo hhi

/-!
Local bounds for monotone multiplication in real-valued bounds.
-/

/-- Lower sqrt bound against the variance-plus-eps term. -/
theorem sqrtLower_le_real_sqrt_varEps {n : Nat} (eps : Rat) (x : Fin n → Rat)
    (hne : n ≠ 0) (heps : 0 < eps) :
    let varEps : Real := (varianceRat x : Real) + (eps : Real)
    (sqrtLower eps : Real) ≤ Real.sqrt varEps := by
  intro varEps
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

/-- Inverse-std upper bound from the lower sqrt bound. -/
theorem invStd_le_invStdBound {n : Nat} (eps : Rat) (x : Fin n → Rat)
    (hne : n ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps) :
    let varEps : Real := (varianceRat x : Real) + (eps : Real)
    let invStd : Real := (Real.sqrt varEps)⁻¹
    let invStdBound : Rat := ratDivUp 1 (sqrtLower eps)
    invStd ≤ (invStdBound : Real) := by
  intro varEps invStd invStdBound
  have hsqrt_lower : (sqrtLower eps : Real) ≤ Real.sqrt varEps := by
    simpa [varEps] using
      (sqrtLower_le_real_sqrt_varEps (eps := eps) (x := x) hne heps)
  have hsqrt_lower_pos : 0 < (sqrtLower eps : Real) := by
    exact (Rat.cast_pos (K := Real) (q := sqrtLower eps)).2 hsqrt
  have hinv_sqrt : invStd ≤ (sqrtLower eps : Real)⁻¹ := by
    have h := inv_anti₀ hsqrt_lower_pos hsqrt_lower
    simpa [invStd] using h
  have hinv_bound : (sqrtLower eps : Real)⁻¹ ≤ (invStdBound : Real) := by
    have hy : sqrtLower eps ≠ 0 := ne_of_gt hsqrt
    have hdiv := ratDivUp_ge_real (x := 1) (y := sqrtLower eps) hy
    simpa [invStdBound, one_div] using hdiv
  exact le_trans hinv_sqrt hinv_bound

/-- Inverse-std is nonnegative. -/
theorem invStd_nonneg {n : Nat} (eps : Rat) (x : Fin n → Rat) :
    let varEps : Real := (varianceRat x : Real) + (eps : Real)
    let invStd : Real := (Real.sqrt varEps)⁻¹
    0 ≤ invStd := by
  intro varEps invStd
  have hsqrt_nonneg : 0 ≤ Real.sqrt varEps := by
    exact Real.sqrt_nonneg _
  exact inv_nonneg.2 hsqrt_nonneg

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
  have hinv : invStd ≤ (invStdBound : Real) := by
    simpa [varEps, invStd, invStdBound] using
      (invStd_le_invStdBound (eps := eps) (x := x) hne heps hsqrt)
  have hinv_nonneg : 0 ≤ invStd := by
    simp [varEps, invStd]
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
    simpa only [mul_assoc] using hmul2'
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
    have h : (beta i : Real) + -(radius i : Real) ≤ (beta i : Real) + t := by
      simpa only [add_comm, add_left_comm, add_assoc] using
        add_le_add_left hbounds.1 (beta i : Real)
    simpa only [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using h
  have hhigh :
      t + (beta i : Real) ≤ (beta i : Real) + (radius i : Real) := by
    have h : (beta i : Real) + t ≤ (beta i : Real) + (radius i : Real) := by
      simpa only [add_comm, add_left_comm, add_assoc] using
        add_le_add_left hbounds.2 (beta i : Real)
    simpa only [add_comm, add_left_comm, add_assoc] using h
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

/-- Bound a centered value by double the absolute bound. -/
private theorem abs_sub_le_double_bound {a b bound : Real}
    (ha : |a| ≤ bound) (hb : |b| ≤ bound) :
    |a - b| ≤ bound + bound := by
  have h1 : |a - b| ≤ |a| + |b| := by
    simpa [sub_eq_add_neg, abs_neg] using abs_add_le a (-b)
  exact le_trans h1 (add_le_add ha hb)

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
    have hx : |(x i : Real)| ≤ (absBound : Real) := by
      exact ratToReal_abs_le_of_le (habs i)
    have hmu : |μ| ≤ (absBound : Real) := by
      simpa [μ] using hmean_abs_real
    have h12 : |(x i : Real) - μ| ≤ (absBound : Real) + (absBound : Real) :=
      abs_sub_le_double_bound hx hmu
    simpa [centeredBound, two_mul] using h12
  have hbound_nonneg_real : 0 ≤ (absBound : Real) := by
    exact ratToReal_nonneg_of_nonneg hbound_nonneg
  have hcentered_nonneg : 0 ≤ (centeredBound : Real) := by
    have hsum := add_nonneg hbound_nonneg_real hbound_nonneg_real
    simpa [centeredBound, two_mul] using hsum
  have hinv : invStd ≤ (invStdBound : Real) := by
    simpa [varEps, invStd, invStdBound] using
      (invStd_le_invStdBound (eps := eps) (x := x) hne heps hsqrt)
  have hinv_nonneg : 0 ≤ invStd := by
    simp [varEps, invStd]
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
    simpa only [mul_assoc] using hmul2'
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
    have h : (beta i : Real) + -(radius i : Real) ≤ (beta i : Real) + t := by
      simpa only [add_comm, add_left_comm, add_assoc] using
        add_le_add_left hbounds.1 (beta i : Real)
    simpa only [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using h
  have hhigh :
      t + (beta i : Real) ≤ (beta i : Real) + (radius i : Real) := by
    have h : (beta i : Real) + t ≤ (beta i : Real) + (radius i : Real) := by
      simpa only [add_comm, add_left_comm, add_assoc] using
        add_le_add_left hbounds.2 (beta i : Real)
    simpa only [add_comm, add_left_comm, add_assoc] using h
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
    have hx : |x i| ≤ (absBound : Real) := habs i
    have hmu : |μ| ≤ (absBound : Real) := by
      simpa using hmean_abs
    have h12 : |x i - μ| ≤ (absBound : Real) + (absBound : Real) :=
      abs_sub_le_double_bound hx hmu
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
    simpa only [mul_assoc] using hmul2'
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
    have h : (beta i : Real) + -(radius i : Real) ≤ (beta i : Real) + t := by
      simpa only [add_comm, add_left_comm, add_assoc] using
        add_le_add_left hbounds.1 (beta i : Real)
    simpa only [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using h
  have hhigh :
      t + (beta i : Real) ≤ (beta i : Real) + (radius i : Real) := by
    have h : (beta i : Real) + t ≤ (beta i : Real) + (radius i : Real) := by
      simpa only [add_comm, add_left_comm, add_assoc] using
        add_le_add_left hbounds.2 (beta i : Real)
    simpa only [add_comm, add_left_comm, add_assoc] using h
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
    simpa only [mul_assoc] using hmul2'
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
    have h : (beta i : Real) + -(radius i : Real) ≤ (beta i : Real) + t := by
      simpa only [add_comm, add_left_comm, add_assoc] using
        add_le_add_left hbounds.1 (beta i : Real)
    simpa only [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using h
  have hhigh :
      t + (beta i : Real) ≤ (beta i : Real) + (radius i : Real) := by
    have h : (beta i : Real) + t ≤ (beta i : Real) + (radius i : Real) := by
      simpa only [add_comm, add_left_comm, add_assoc] using
        add_le_add_left hbounds.2 (beta i : Real)
    simpa only [add_comm, add_left_comm, add_assoc] using h
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
