-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Bounds.LayerNorm.MeanVariance
public import Nfp.Bounds.LayerNorm.SqrtBounds

/-!
Inverse-standard-deviation bounds for LayerNorm.

This module isolates invStd bounds and their soundness proof to keep
`LayerNorm/Basic.lean` below the style linter's file-length limit.
-/

public section

namespace Nfp


namespace Bounds

/-- Bounds for the LayerNorm inverse standard deviation term. -/
def invStdBounds {n : Nat} (eps : Rat) (x : Fin n → Rat) : Rat × Rat :=
  if n = 0 then
    (0, 0)
  else
    let var : Rat := variance x
    let varEps : Rat := var + eps
    let sqrtLowerBound : Rat := max (sqrtLower eps) (sqrtLower varEps)
    let sqrtUpperBound : Rat := sqrtUpper varEps
    (ratDivDown 1 sqrtUpperBound, ratDivUp 1 sqrtLowerBound)

/-- Unfolding lemma for `invStdBounds`. -/
theorem invStdBounds_def {n : Nat} (eps : Rat) (x : Fin n → Rat) :
    invStdBounds eps x =
      if n = 0 then
        (0, 0)
      else
        let var : Rat := variance x
        let varEps : Rat := var + eps
        let sqrtLowerBound : Rat := max (sqrtLower eps) (sqrtLower varEps)
        let sqrtUpperBound : Rat := sqrtUpper varEps
        (ratDivDown 1 sqrtUpperBound, ratDivUp 1 sqrtLowerBound) := by
  simp [invStdBounds]

/-- `invStdBounds` soundness for real inverse-std terms. -/
theorem invStdBounds_spec {n : Nat} (eps : Rat) (x : Fin n → Rat)
    (hne : n ≠ 0) (heps : 0 < eps) (hsqrt : 0 < sqrtLower eps) :
    let bounds := invStdBounds eps x
    let invStd : Real := (Real.sqrt ((varianceRat x : Real) + (eps : Real)))⁻¹
    (bounds.1 : Real) ≤ invStd ∧ invStd ≤ (bounds.2 : Real) := by
  classical
  intro bounds invStd
  let varRat : Rat := variance x
  let varEpsRat : Rat := varRat + eps
  let sqrtLowerBound : Rat := max (sqrtLower eps) (sqrtLower varEpsRat)
  let sqrtUpperBound : Rat := sqrtUpper varEpsRat
  let invStdLower : Rat := ratDivDown 1 sqrtUpperBound
  let invStdUpper : Rat := ratDivUp 1 sqrtLowerBound
  let varEps : Real := (varianceRat x : Real) + (eps : Real)
  have hvar : (varRat : Real) = (varianceRat x : Real) := by
    simp [varRat, variance_def, hne, ratRoundDown_def]
  have hvarEps : (varEpsRat : Real) = varEps := by
    simp [varEpsRat, varEps, hvar]
  have hvar_nonneg : 0 ≤ (varianceRat x : Real) := varianceRat_nonneg_real x hne
  have hvar_nonneg_rat : 0 ≤ varianceRat x := by
    have hvar_nonneg_real : 0 ≤ ratToReal (varianceRat x) := by
      simpa [ratToReal_def] using hvar_nonneg
    exact (ratToReal_nonneg_iff (x := varianceRat x)).1 hvar_nonneg_real
  have hvarRat_nonneg : 0 ≤ varRat := by
    have h := ratRoundDown_nonneg (q := varianceRat x) hvar_nonneg_rat
    simpa [varRat, variance_def x hne] using h
  have hvarEps_nonneg : 0 ≤ varEpsRat := by
    exact add_nonneg hvarRat_nonneg (le_of_lt heps)
  have hsqrt_var : (sqrtLower varEpsRat : Real) ≤ Real.sqrt varEps := by
    have h := sqrtLower_le_real_sqrt (q := varEpsRat) hvarEps_nonneg
    simpa [hvarEps] using h
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
    simpa [invStd, varEps] using hle
  have hinv_upper_real :
      invStd ≤ (sqrtLowerBound : Real)⁻¹ := by
    have hle := inv_anti₀ hsqrt_lower_pos hsqrt_lower
    simpa [invStd, varEps] using hle
  have hupper_ne : sqrtUpperBound ≠ 0 := ne_of_gt hsqrt_upper_pos_rat
  have hlower_ne : sqrtLowerBound ≠ 0 := ne_of_gt hsqrt_lower_pos_rat
  have hinv_lower : (invStdLower : Real) ≤ invStd := by
    simpa [invStdLower, ratDivDown_def, hupper_ne, one_div] using hinv_lower_real
  have hinv_upper : invStd ≤ (invStdUpper : Real) := by
    simpa [invStdUpper, ratDivUp_def, hlower_ne, one_div] using hinv_upper_real
  have hbounds : bounds = (invStdLower, invStdUpper) := by
    simp [bounds, invStdBounds, hne, varRat, varEpsRat, sqrtLowerBound, sqrtUpperBound,
      invStdLower, invStdUpper]
  constructor
  · simpa [hbounds] using hinv_lower
  · simpa [hbounds] using hinv_upper

end Bounds


end Nfp
