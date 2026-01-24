-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.Pure.InductionHead.ModelInputs
public import Nfp.Sound.Induction.ScoreIntervals
public import Nfp.Sound.Induction.ValueBounds

/-!
Soundness bridge for LN-derived score interval bounds.
-/

public section

namespace Nfp

namespace IO

namespace Pure

namespace InductionHeadCert

open Nfp.Bounds
open Nfp.Sound

noncomputable section

/-- Slice-derived score bounds are sound for `scoresRealOfInputs`. -/
theorem scoresRealOfInputs_bounds_from_slices {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (modelSlice : Nfp.IO.InductionHeadCert.ModelSlice seq)
    (hLn : lnSlice.dModel = modelSlice.dModel)
    (hModel : modelSlice.dModel ≠ 0)
    (hEps : 0 < lnSlice.lnEps)
    (hSlack : 0 ≤ lnSlice.lnSlack)
    (hScalePos : ∀ scale, lnSlice.lnScale? = some scale → 0 < scale)
    (hSqrt :
      match lnSlice.lnScale? with
      | some scale => 0 < sqrtLowerWithScale scale lnSlice.lnEps
      | none => 0 < sqrtLower lnSlice.lnEps)
    (hScaleNonneg : 0 ≤ modelSlice.scoreScale) :
    let inputs := inputsOfScoreSlices lnSlice modelSlice hLn
    let lnBounds : Fin seq → (Fin modelSlice.dModel → Rat) × (Fin modelSlice.dModel → Rat) :=
      fun q =>
        match lnSlice.lnScale? with
        | some scale =>
            layerNormBoundsWithScale scale lnSlice.lnEps inputs.ln1Gamma inputs.ln1Beta
              (inputs.embed q)
        | none =>
            layerNormBounds lnSlice.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q)
    let lnLo : Fin seq → Fin modelSlice.dModel → Rat :=
      fun q i => (lnBounds q).1 i - lnSlice.lnSlack
    let lnHi : Fin seq → Fin modelSlice.dModel → Rat :=
      fun q i => (lnBounds q).2 i + lnSlice.lnSlack
    let qLo : Fin seq → Fin modelSlice.headDim → Rat := fun q d =>
      dotIntervalLower (fun j => inputs.wq j d) (lnLo q) (lnHi q) + inputs.bq d
    let qHi : Fin seq → Fin modelSlice.headDim → Rat := fun q d =>
      dotIntervalUpper (fun j => inputs.wq j d) (lnLo q) (lnHi q) + inputs.bq d
    let kLo : Fin seq → Fin modelSlice.headDim → Rat := fun k d =>
      dotIntervalLower (fun j => inputs.wk j d) (lnLo k) (lnHi k) + inputs.bk d
    let kHi : Fin seq → Fin modelSlice.headDim → Rat := fun k d =>
      dotIntervalUpper (fun j => inputs.wk j d) (lnLo k) (lnHi k) + inputs.bk d
    let baseLo : Fin seq → Fin seq → Rat := fun q k =>
      inputs.scale * dotIntervalMulLower (qLo q) (qHi q) (kLo k) (kHi k)
    let baseHi : Fin seq → Fin seq → Rat := fun q k =>
      inputs.scale * dotIntervalMulUpper (qLo q) (qHi q) (kLo k) (kHi k)
    let scoreLo : Fin seq → Fin seq → Rat := fun q k =>
      if inputs.maskCausal then
        if k ≤ q then baseLo q k else inputs.maskValue
      else
        baseLo q k
    let scoreHi : Fin seq → Fin seq → Rat := fun q k =>
      if inputs.maskCausal then
        if k ≤ q then baseHi q k else inputs.maskValue
      else
        baseHi q k
    ∀ q k,
      (scoreLo q k : Real) ≤ scoresRealOfInputs inputs q k ∧
        scoresRealOfInputs inputs q k ≤ (scoreHi q k : Real) := by
  classical
  intro inputs lnBounds lnLo lnHi qLo qHi kLo kHi baseLo baseHi scoreLo scoreHi q k
  have hinputs : inputs = inputsOfScoreSlices lnSlice modelSlice hLn := by
    rfl
  cases hscale : lnSlice.lnScale? with
  | none =>
      have hEps' : 0 < inputs.lnEps := by
        simpa [hinputs, inputsOfScoreSlices_lnEps] using hEps
      have hSqrt' : 0 < sqrtLower inputs.lnEps := by
        have hSqrt'' : 0 < sqrtLower lnSlice.lnEps := by
          simpa [hscale] using hSqrt
        simpa [hinputs, inputsOfScoreSlices_lnEps] using hSqrt''
      have hln :=
        lnRealOfInputs_bounds_with_slack inputs lnSlice.lnSlack hSlack hModel hEps' hSqrt'
      have hln' : ∀ q i,
          (lnLo q i : Real) ≤ lnRealOfInputs inputs q i ∧
            lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
        simpa [lnBounds, lnLo, lnHi, hscale, hinputs, inputsOfScoreSlices_lnEps] using hln
      have hScaleNonneg' : 0 ≤ inputs.scale := by
        simpa [hinputs, inputsOfScoreSlices_scale] using hScaleNonneg
      have hscore :=
        scoresRealOfInputs_bounds_of_lnBounds inputs lnLo lnHi hln' hScaleNonneg'
      have hscore' := hscore q k
      simpa [qLo, qHi, kLo, kHi, baseLo, baseHi, scoreLo, scoreHi, hscale] using hscore'
  | some scale =>
      have hEps' : 0 < inputs.lnEps := by
        simpa [hinputs, inputsOfScoreSlices_lnEps] using hEps
      have hSqrt' : 0 < sqrtLowerWithScale scale inputs.lnEps := by
        have hSqrt'' : 0 < sqrtLowerWithScale scale lnSlice.lnEps := by
          simpa [hscale] using hSqrt
        simpa [hinputs, inputsOfScoreSlices_lnEps] using hSqrt''
      have hScalePos' : 0 < scale := hScalePos scale hscale
      have hln :=
        lnRealOfInputs_bounds_with_scale_slack inputs scale hScalePos' lnSlice.lnSlack hSlack
          hModel hEps' hSqrt'
      have hln' : ∀ q i,
          (lnLo q i : Real) ≤ lnRealOfInputs inputs q i ∧
            lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
        simpa [lnBounds, lnLo, lnHi, hscale, hinputs, inputsOfScoreSlices_lnEps] using hln
      have hScaleNonneg' : 0 ≤ inputs.scale := by
        simpa [hinputs, inputsOfScoreSlices_scale] using hScaleNonneg
      have hscore :=
        scoresRealOfInputs_bounds_of_lnBounds inputs lnLo lnHi hln' hScaleNonneg'
      have hscore' := hscore q k
      simpa [qLo, qHi, kLo, kHi, baseLo, baseHi, scoreLo, scoreHi, hscale] using hscore'

end

end InductionHeadCert

end Pure

end IO

end Nfp
