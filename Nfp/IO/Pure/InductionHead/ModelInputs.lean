-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Bounds.LayerNorm
public import Nfp.IO.InductionHead.ModelDirectionSlice
public import Nfp.IO.InductionHead.ModelLnSlice
public import Nfp.IO.InductionHead.ModelSlice
public import Nfp.IO.InductionHead.ModelValueSlice
public import Nfp.Linear.FinFold
public import Nfp.Model.InductionHead
public import Nfp.Sound.Induction.ValueBounds
public import Nfp.Sound.Induction.HeadOutput
public import Nfp.Sound.Induction.LogitDiff

/-!
Soundness bridge from model slices to value-path bounds.
-/

public section

namespace Nfp

namespace IO

namespace Pure

namespace InductionHeadCert

open Nfp.Bounds
open Nfp.Sound

/-- Assemble induction-head inputs from model slices and direction metadata.
    Uses the pre-LN residual stream from the LayerNorm slice. -/
def inputsOfSlices {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    Model.InductionHeadInputs seq valueSlice.dModel valueSlice.headDim :=
  Model.InductionHeadInputs.ofPreLnResidual
    (scale := 0)
    (active := ∅)
    (prev := fun q => q)
    (preLn := by simpa [hLn] using lnSlice.embed)
    (lnEps := lnSlice.lnEps)
    (ln1Gamma := by simpa [hLn] using lnSlice.lnGamma)
    (ln1Beta := by simpa [hLn] using lnSlice.lnBeta)
    (wq := fun _ _ => 0)
    (bq := fun _ => 0)
    (wk := fun _ _ => 0)
    (bk := fun _ => 0)
    (wv := valueSlice.wv)
    (bv := valueSlice.bv)
    (wo := valueSlice.wo)
    (attnBias := valueSlice.attnBias)
    (maskCausal := false)
    (maskValue := 0)
    (directionSpec := { target := dirSlice.target, negative := dirSlice.negative })
    (direction := by simpa [hDir] using dirSlice.direction)

/-- `inputsOfSlices` preserves the value-projection weights. -/
theorem inputsOfSlices_wv {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    (inputsOfSlices lnSlice valueSlice dirSlice hLn hDir).wv = valueSlice.wv := by
  rfl

/-- `inputsOfSlices` preserves the value-projection bias. -/
theorem inputsOfSlices_bv {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    (inputsOfSlices lnSlice valueSlice dirSlice hLn hDir).bv = valueSlice.bv := by
  rfl

/-- `inputsOfSlices` preserves the output-projection weights. -/
theorem inputsOfSlices_wo {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    (inputsOfSlices lnSlice valueSlice dirSlice hLn hDir).wo = valueSlice.wo := by
  rfl

/-- `inputsOfSlices` preserves the attention output bias. -/
theorem inputsOfSlices_attnBias {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    (inputsOfSlices lnSlice valueSlice dirSlice hLn hDir).attnBias = valueSlice.attnBias := by
  rfl

/-! `inputsOfSlices` preserves LayerNorm inputs. -/

/-- `inputsOfSlices` preserves embeddings. -/
theorem inputsOfSlices_embed {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    (inputsOfSlices lnSlice valueSlice dirSlice hLn hDir).embed =
      (by simpa [hLn] using lnSlice.embed) := by
  rfl

/-- `inputsOfSlices` preserves LayerNorm gamma. -/
theorem inputsOfSlices_ln1Gamma {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    (inputsOfSlices lnSlice valueSlice dirSlice hLn hDir).ln1Gamma =
      (by simpa [hLn] using lnSlice.lnGamma) := by
  rfl

/-- `inputsOfSlices` preserves LayerNorm beta. -/
theorem inputsOfSlices_ln1Beta {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    (inputsOfSlices lnSlice valueSlice dirSlice hLn hDir).ln1Beta =
      (by simpa [hLn] using lnSlice.lnBeta) := by
  rfl

/-- `inputsOfSlices` preserves the direction vector. -/
theorem inputsOfSlices_direction {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    (inputsOfSlices lnSlice valueSlice dirSlice hLn hDir).direction =
      (by simpa [hDir] using dirSlice.direction) := by
  rfl

/-! Score-focused inputs from LayerNorm and model score slices. -/

/-- Assemble induction-head inputs for score bounds from LayerNorm and model slices. -/
def inputsOfScoreSlices {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (modelSlice : Nfp.IO.InductionHeadCert.ModelSlice seq)
    (hLn : lnSlice.dModel = modelSlice.dModel) :
    Model.InductionHeadInputs seq modelSlice.dModel modelSlice.headDim :=
  Model.InductionHeadInputs.ofPreLnResidual
    (scale := modelSlice.scoreScale)
    (active := ∅)
    (prev := fun q => q)
    (preLn := by simpa [hLn] using lnSlice.embed)
    (lnEps := lnSlice.lnEps)
    (ln1Gamma := by simpa [hLn] using lnSlice.lnGamma)
    (ln1Beta := by simpa [hLn] using lnSlice.lnBeta)
    (wq := modelSlice.wq)
    (bq := modelSlice.bq)
    (wk := modelSlice.wk)
    (bk := modelSlice.bk)
    (wv := fun _ _ => 0)
    (bv := fun _ => 0)
    (wo := fun _ _ => 0)
    (attnBias := fun _ => 0)
    (maskCausal := modelSlice.maskCausal)
    (maskValue := modelSlice.scoreMask)
    (directionSpec := { target := 0, negative := 0 })
    (direction := fun _ => 0)

/-- `inputsOfScoreSlices` preserves embeddings. -/
theorem inputsOfScoreSlices_embed {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (modelSlice : Nfp.IO.InductionHeadCert.ModelSlice seq)
    (hLn : lnSlice.dModel = modelSlice.dModel) :
    (inputsOfScoreSlices lnSlice modelSlice hLn).embed =
      (by simpa [hLn] using lnSlice.embed) := by
  rfl

/-- `inputsOfScoreSlices` preserves LayerNorm gamma. -/
theorem inputsOfScoreSlices_ln1Gamma {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (modelSlice : Nfp.IO.InductionHeadCert.ModelSlice seq)
    (hLn : lnSlice.dModel = modelSlice.dModel) :
    (inputsOfScoreSlices lnSlice modelSlice hLn).ln1Gamma =
      (by simpa [hLn] using lnSlice.lnGamma) := by
  rfl

/-- `inputsOfScoreSlices` preserves LayerNorm epsilon. -/
theorem inputsOfScoreSlices_lnEps {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (modelSlice : Nfp.IO.InductionHeadCert.ModelSlice seq)
    (hLn : lnSlice.dModel = modelSlice.dModel) :
    (inputsOfScoreSlices lnSlice modelSlice hLn).lnEps = lnSlice.lnEps := by
  rfl

/-- `inputsOfScoreSlices` preserves LayerNorm beta. -/
theorem inputsOfScoreSlices_ln1Beta {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (modelSlice : Nfp.IO.InductionHeadCert.ModelSlice seq)
    (hLn : lnSlice.dModel = modelSlice.dModel) :
    (inputsOfScoreSlices lnSlice modelSlice hLn).ln1Beta =
      (by simpa [hLn] using lnSlice.lnBeta) := by
  rfl

/-- `inputsOfScoreSlices` preserves the score scale. -/
theorem inputsOfScoreSlices_scale {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (modelSlice : Nfp.IO.InductionHeadCert.ModelSlice seq)
    (hLn : lnSlice.dModel = modelSlice.dModel) :
    (inputsOfScoreSlices lnSlice modelSlice hLn).scale = modelSlice.scoreScale := by
  rfl

/-- `inputsOfScoreSlices` preserves the score mask settings. -/
theorem inputsOfScoreSlices_mask {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (modelSlice : Nfp.IO.InductionHeadCert.ModelSlice seq)
    (hLn : lnSlice.dModel = modelSlice.dModel) :
    (inputsOfScoreSlices lnSlice modelSlice hLn).maskCausal = modelSlice.maskCausal ∧
      (inputsOfScoreSlices lnSlice modelSlice hLn).maskValue = modelSlice.scoreMask := by
  constructor <;> rfl

/-- `inputsOfScoreSlices` preserves Q/K projections. -/
theorem inputsOfScoreSlices_qk {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (modelSlice : Nfp.IO.InductionHeadCert.ModelSlice seq)
    (hLn : lnSlice.dModel = modelSlice.dModel) :
    (inputsOfScoreSlices lnSlice modelSlice hLn).wq = modelSlice.wq ∧
      (inputsOfScoreSlices lnSlice modelSlice hLn).wk = modelSlice.wk ∧
      (inputsOfScoreSlices lnSlice modelSlice hLn).bq = modelSlice.bq ∧
      (inputsOfScoreSlices lnSlice modelSlice hLn).bk = modelSlice.bk := by
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  · rfl

/--
Slice-derived value bounds are sound for `valsRealOfInputs`.

This uses `ValueBounds` to justify the same interval construction as the
model-slice checker.
-/
theorem valsRealOfInputs_bounds_from_slices {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel)
    (hModel : valueSlice.dModel ≠ 0)
    (hEps : 0 < lnSlice.lnEps)
    (hSlack : 0 ≤ lnSlice.lnSlack)
    (hScalePos : ∀ scale, lnSlice.lnScale? = some scale → 0 < scale)
    (hSqrt :
      match lnSlice.lnScale? with
      | some scale => 0 < sqrtLowerWithScale scale lnSlice.lnEps
      | none => 0 < sqrtLower lnSlice.lnEps) :
    let inputs := inputsOfSlices lnSlice valueSlice dirSlice hLn hDir
    let lnBounds : Fin seq → (Fin valueSlice.dModel → Rat) × (Fin valueSlice.dModel → Rat) :=
      fun q =>
        match lnSlice.lnScale? with
        | some scale =>
            layerNormBoundsWithScale scale lnSlice.lnEps inputs.ln1Gamma inputs.ln1Beta
              (inputs.embed q)
        | none =>
            layerNormBounds lnSlice.lnEps inputs.ln1Gamma inputs.ln1Beta
              (inputs.embed q)
    let lnLo : Fin seq → Fin valueSlice.dModel → Rat :=
      fun q i => (lnBounds q).1 i - lnSlice.lnSlack
    let lnHi : Fin seq → Fin valueSlice.dModel → Rat :=
      fun q i => (lnBounds q).2 i + lnSlice.lnSlack
    let dirHead : Fin valueSlice.headDim → Rat :=
      fun d => (dirHeadVecOfInputs inputs).get d
    let bias : Rat := attnBiasDot inputs
    let vLo : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
      dotIntervalLower (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
    let vHi : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
      dotIntervalUpper (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
    let valLo : Fin seq → Rat := fun k => dotIntervalLower dirHead (vLo k) (vHi k) + bias
    let valHi : Fin seq → Rat := fun k => dotIntervalUpper dirHead (vLo k) (vHi k) + bias
    ∀ k, (valLo k : Real) ≤ valsRealOfInputs inputs k ∧
      valsRealOfInputs inputs k ≤ (valHi k : Real) := by
  classical
  intro inputs lnBounds lnLo lnHi dirHead bias vLo vHi valLo valHi k
  cases hscale : lnSlice.lnScale? with
  | none =>
      have hSqrt' : 0 < sqrtLower lnSlice.lnEps := by
        simpa [hscale] using hSqrt
      have hvals :=
        valsRealOfInputs_bounds_with_lnSlack inputs lnSlice.lnSlack hSlack
          hModel hEps hSqrt'
      have hvals' := hvals k
      simpa [lnBounds, lnLo, lnHi, vLo, vHi, valLo, valHi, dirHead, bias, hscale] using hvals'
  | some scale =>
      have hSqrt' : 0 < sqrtLowerWithScale scale lnSlice.lnEps := by
        simpa [hscale] using hSqrt
      have hScalePos' : 0 < scale := hScalePos scale hscale
      have hvals :=
        valsRealOfInputs_bounds_with_scale_lnSlack inputs scale hScalePos' lnSlice.lnSlack hSlack
          hModel hEps hSqrt'
      have hvals' := hvals k
      simpa [lnBounds, lnLo, lnHi, vLo, vHi, valLo, valHi, dirHead, bias, hscale] using hvals'

/--
Slice-derived weighted head-output bounds are sound for `headOutputWithWeights`.
-/
theorem headOutputWithWeights_bounds_from_slices {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (weights : Fin seq → Fin seq → Rat)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel)
    (hModel : valueSlice.dModel ≠ 0)
    (hEps : 0 < lnSlice.lnEps)
    (hSlack : 0 ≤ lnSlice.lnSlack)
    (hScalePos : ∀ scale, lnSlice.lnScale? = some scale → 0 < scale)
    (hSqrt :
      match lnSlice.lnScale? with
      | some scale => 0 < sqrtLowerWithScale scale lnSlice.lnEps
      | none => 0 < sqrtLower lnSlice.lnEps) :
    let inputs := inputsOfSlices lnSlice valueSlice dirSlice hLn hDir
    let lnBounds : Fin seq → (Fin valueSlice.dModel → Rat) × (Fin valueSlice.dModel → Rat) :=
      fun q =>
        match lnSlice.lnScale? with
        | some scale =>
            layerNormBoundsWithScale scale lnSlice.lnEps inputs.ln1Gamma inputs.ln1Beta
              (inputs.embed q)
        | none =>
            layerNormBounds lnSlice.lnEps inputs.ln1Gamma inputs.ln1Beta
              (inputs.embed q)
    let lnLo : Fin seq → Fin valueSlice.dModel → Rat :=
      fun q i => (lnBounds q).1 i - lnSlice.lnSlack
    let lnHi : Fin seq → Fin valueSlice.dModel → Rat :=
      fun q i => (lnBounds q).2 i + lnSlice.lnSlack
    let vLo : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
      dotIntervalLower (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
    let vHi : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
      dotIntervalUpper (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
    let headLo : Fin seq → Fin valueSlice.dModel → Rat := fun k i =>
      dotIntervalLower (fun d => inputs.wo i d) (vLo k) (vHi k) + inputs.attnBias i
    let headHi : Fin seq → Fin valueSlice.dModel → Rat := fun k i =>
      dotIntervalUpper (fun d => inputs.wo i d) (vLo k) (vHi k) + inputs.attnBias i
    let outLo : Fin seq → Fin valueSlice.dModel → Rat := fun q i =>
      dotIntervalLower (fun k => weights q k) (fun k => headLo k i) (fun k => headHi k i)
    let outHi : Fin seq → Fin valueSlice.dModel → Rat := fun q i =>
      dotIntervalUpper (fun k => weights q k) (fun k => headLo k i) (fun k => headHi k i)
    ∀ q i,
      (outLo q i : Real) ≤ Sound.headOutputWithWeights weights inputs q i ∧
        Sound.headOutputWithWeights weights inputs q i ≤ (outHi q i : Real) := by
  classical
  intro inputs lnBounds lnLo lnHi vLo vHi headLo headHi outLo outHi q i
  cases hscale : lnSlice.lnScale? with
  | none =>
      have hSqrt' : 0 < sqrtLower lnSlice.lnEps := by
        simpa [hscale] using hSqrt
      have hln :=
        Sound.lnRealOfInputs_bounds_with_slack inputs lnSlice.lnSlack hSlack hModel hEps hSqrt'
      have hln' : ∀ q i,
          (lnLo q i : Real) ≤ Sound.lnRealOfInputs inputs q i ∧
            Sound.lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
        simpa [lnBounds, lnLo, lnHi, hscale] using hln
      have hhead :=
        Sound.headOutputWithWeights_bounds_of_lnBounds weights inputs lnLo lnHi hln'
      have hhead' := hhead q i
      simpa [outLo, outHi, vLo, vHi, headLo, headHi, hscale] using hhead'
  | some scale =>
      have hSqrt' : 0 < sqrtLowerWithScale scale lnSlice.lnEps := by
        simpa [hscale] using hSqrt
      have hScalePos' : 0 < scale := hScalePos scale hscale
      have hln :=
        Sound.lnRealOfInputs_bounds_with_scale_slack inputs scale hScalePos' lnSlice.lnSlack
          hSlack hModel hEps hSqrt'
      have hln' : ∀ q i,
          (lnLo q i : Real) ≤ Sound.lnRealOfInputs inputs q i ∧
            Sound.lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
        simpa [lnBounds, lnLo, lnHi, hscale] using hln
      have hhead :=
        Sound.headOutputWithWeights_bounds_of_lnBounds weights inputs lnLo lnHi hln'
      have hhead' := hhead q i
      simpa [outLo, outHi, vLo, vHi, headLo, headHi, hscale] using hhead'

/--
Compute weighted head-output bounds from model slices.
-/
def headOutputBoundsWeightedFromSlices {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (weights : Fin seq → Fin seq → Rat)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    (Fin seq → Fin valueSlice.dModel → Rat) × (Fin seq → Fin valueSlice.dModel → Rat) :=
  let inputs := inputsOfSlices lnSlice valueSlice dirSlice hLn hDir
  let lnBounds : Fin seq → (Fin valueSlice.dModel → Rat) × (Fin valueSlice.dModel → Rat) :=
    fun q =>
      match lnSlice.lnScale? with
      | some scale =>
          layerNormBoundsWithScale scale lnSlice.lnEps inputs.ln1Gamma inputs.ln1Beta
            (inputs.embed q)
      | none =>
          layerNormBounds lnSlice.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q)
  let lnLo : Fin seq → Fin valueSlice.dModel → Rat :=
    fun q i => (lnBounds q).1 i - lnSlice.lnSlack
  let lnHi : Fin seq → Fin valueSlice.dModel → Rat :=
    fun q i => (lnBounds q).2 i + lnSlice.lnSlack
  let vLo : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
    dotIntervalLower (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
  let vHi : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
    dotIntervalUpper (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
  let headLo : Fin seq → Fin valueSlice.dModel → Rat := fun k i =>
    dotIntervalLower (fun d => inputs.wo i d) (vLo k) (vHi k) + inputs.attnBias i
  let headHi : Fin seq → Fin valueSlice.dModel → Rat := fun k i =>
    dotIntervalUpper (fun d => inputs.wo i d) (vLo k) (vHi k) + inputs.attnBias i
  let outLo : Fin seq → Fin valueSlice.dModel → Rat := fun q i =>
    dotIntervalLower (fun k => weights q k) (fun k => headLo k i) (fun k => headHi k i)
  let outHi : Fin seq → Fin valueSlice.dModel → Rat := fun q i =>
    dotIntervalUpper (fun k => weights q k) (fun k => headLo k i) (fun k => headHi k i)
  (outLo, outHi)

/--
Slice-derived weighted head-output bounds are sound for `headOutputWithWeights`.
-/
theorem headOutputBoundsWeightedFromSlices_spec {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (weights : Fin seq → Fin seq → Rat)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel)
    (hModel : valueSlice.dModel ≠ 0)
    (hEps : 0 < lnSlice.lnEps)
    (hSlack : 0 ≤ lnSlice.lnSlack)
    (hScalePos : ∀ scale, lnSlice.lnScale? = some scale → 0 < scale)
    (hSqrt :
      match lnSlice.lnScale? with
      | some scale => 0 < sqrtLowerWithScale scale lnSlice.lnEps
      | none => 0 < sqrtLower lnSlice.lnEps) :
    let bounds := headOutputBoundsWeightedFromSlices lnSlice valueSlice dirSlice weights hLn hDir
    let inputs := inputsOfSlices lnSlice valueSlice dirSlice hLn hDir
    ∀ q i,
      (bounds.1 q i : Real) ≤ Sound.headOutputWithWeights weights inputs q i ∧
        Sound.headOutputWithWeights weights inputs q i ≤ (bounds.2 q i : Real) := by
  classical
  intro bounds inputs q i
  have h :=
    headOutputWithWeights_bounds_from_slices lnSlice valueSlice dirSlice weights
      hLn hDir hModel hEps hSlack hScalePos hSqrt q i
  simpa [headOutputBoundsWeightedFromSlices, bounds, inputs] using h

/--
Compute weighted residual bounds from model slices and explicit weights.
-/
def residualBoundsWeightedFromSlices {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (weights : Fin seq → Fin seq → Rat)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    (Fin seq → Fin valueSlice.dModel → Rat) × (Fin seq → Fin valueSlice.dModel → Rat) :=
  let inputs := inputsOfSlices lnSlice valueSlice dirSlice hLn hDir
  let outBounds := headOutputBoundsWeightedFromSlices lnSlice valueSlice dirSlice weights hLn hDir
  let resLo : Fin seq → Fin valueSlice.dModel → Rat := fun q i => inputs.embed q i + outBounds.1 q i
  let resHi : Fin seq → Fin valueSlice.dModel → Rat := fun q i => inputs.embed q i + outBounds.2 q i
  (resLo, resHi)

/--
Slice-derived weighted residual bounds are sound for `embed + headOutputWithWeights`.
-/
theorem residualBoundsWeightedFromSlices_spec {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (weights : Fin seq → Fin seq → Rat)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel)
    (hModel : valueSlice.dModel ≠ 0)
    (hEps : 0 < lnSlice.lnEps)
    (hSlack : 0 ≤ lnSlice.lnSlack)
    (hScalePos : ∀ scale, lnSlice.lnScale? = some scale → 0 < scale)
    (hSqrt :
      match lnSlice.lnScale? with
      | some scale => 0 < sqrtLowerWithScale scale lnSlice.lnEps
      | none => 0 < sqrtLower lnSlice.lnEps) :
    let bounds := residualBoundsWeightedFromSlices lnSlice valueSlice dirSlice weights hLn hDir
    let inputs := inputsOfSlices lnSlice valueSlice dirSlice hLn hDir
    ∀ q i,
      (bounds.1 q i : Real) ≤
          inputs.embed q i + Sound.headOutputWithWeights weights inputs q i ∧
        inputs.embed q i + Sound.headOutputWithWeights weights inputs q i ≤
          (bounds.2 q i : Real) := by
  classical
  intro bounds inputs q i
  let outBounds :=
    headOutputBoundsWeightedFromSlices lnSlice valueSlice dirSlice weights hLn hDir
  have hout :=
    headOutputBoundsWeightedFromSlices_spec lnSlice valueSlice dirSlice weights
      hLn hDir hModel hEps hSlack hScalePos hSqrt
  have hout' : ∀ q i,
      (outBounds.1 q i : Real) ≤ Sound.headOutputWithWeights weights inputs q i ∧
        Sound.headOutputWithWeights weights inputs q i ≤ (outBounds.2 q i : Real) := by
    simpa [outBounds, inputs] using hout
  have hlo : ∀ q i, (inputs.embed q i : Real) ≤ inputs.embed q i := by
    intro q i
    rfl
  have hhi : ∀ q i, inputs.embed q i ≤ (inputs.embed q i : Real) := by
    intro q i
    rfl
  have hres :=
    Sound.residualWithHeadOutputWithWeights_bounds_at weights inputs outBounds.1 outBounds.2
      hout' (lo := inputs.embed) (hi := inputs.embed) hlo hhi q i
  simpa [bounds, residualBoundsWeightedFromSlices, outBounds, inputs] using hres

/--
Compute weighted residual direction bounds from model slices and explicit weights.
-/
def residualDirectionBoundsWeightedFromSlices {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (weights : Fin seq → Fin seq → Rat)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    (Fin seq → Rat) × (Fin seq → Rat) :=
  let inputs := inputsOfSlices lnSlice valueSlice dirSlice hLn hDir
  let resBounds := residualBoundsWeightedFromSlices lnSlice valueSlice dirSlice weights hLn hDir
  let dir : Fin valueSlice.dModel → Rat := inputs.direction
  (fun q => dotIntervalLower dir (resBounds.1 q) (resBounds.2 q),
    fun q => dotIntervalUpper dir (resBounds.1 q) (resBounds.2 q))

/--
Slice-derived weighted residual direction bounds are sound for `embed + headOutputWithWeights`.
-/
theorem residualDirectionBoundsWeightedFromSlices_spec {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (weights : Fin seq → Fin seq → Rat)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel)
    (hModel : valueSlice.dModel ≠ 0)
    (hEps : 0 < lnSlice.lnEps)
    (hSlack : 0 ≤ lnSlice.lnSlack)
    (hScalePos : ∀ scale, lnSlice.lnScale? = some scale → 0 < scale)
    (hSqrt :
      match lnSlice.lnScale? with
      | some scale => 0 < sqrtLowerWithScale scale lnSlice.lnEps
      | none => 0 < sqrtLower lnSlice.lnEps) :
    let bounds :=
      residualDirectionBoundsWeightedFromSlices lnSlice valueSlice dirSlice weights hLn hDir
    let inputs := inputsOfSlices lnSlice valueSlice dirSlice hLn hDir
    ∀ q,
      (bounds.1 q : Real) ≤
          dotProduct (fun i => (inputs.direction i : Real))
            (fun i => inputs.embed q i + Sound.headOutputWithWeights weights inputs q i) ∧
        dotProduct (fun i => (inputs.direction i : Real))
            (fun i => inputs.embed q i + Sound.headOutputWithWeights weights inputs q i) ≤
          (bounds.2 q : Real) := by
  classical
  intro bounds inputs q
  let resBounds :=
    residualBoundsWeightedFromSlices lnSlice valueSlice dirSlice weights hLn hDir
  have hres :=
    residualBoundsWeightedFromSlices_spec lnSlice valueSlice dirSlice weights
      hLn hDir hModel hEps hSlack hScalePos hSqrt
  have hlo : ∀ i,
      (resBounds.1 q i : Real) ≤
        inputs.embed q i + Sound.headOutputWithWeights weights inputs q i := by
    intro i
    have h := (hres q i).1
    simpa [resBounds] using h
  have hhi : ∀ i,
      inputs.embed q i + Sound.headOutputWithWeights weights inputs q i ≤
        (resBounds.2 q i : Real) := by
    intro i
    have h := (hres q i).2
    simpa [resBounds] using h
  have hlow :=
    dotIntervalLower_le_dotProduct_real (v := inputs.direction)
      (lo := resBounds.1 q) (hi := resBounds.2 q)
      (x := fun i => inputs.embed q i + Sound.headOutputWithWeights weights inputs q i)
      hlo hhi
  have hhigh :=
    dotProduct_le_dotIntervalUpper_real (v := inputs.direction)
      (lo := resBounds.1 q) (hi := resBounds.2 q)
      (x := fun i => inputs.embed q i + Sound.headOutputWithWeights weights inputs q i)
      hlo hhi
  constructor
  · simpa [bounds, residualDirectionBoundsWeightedFromSlices, resBounds] using hlow
  · simpa [bounds, residualDirectionBoundsWeightedFromSlices, resBounds] using hhigh

/--
Compute weighted residual direction bounds from model slices and explicit weights
using direction-space value bounds.
-/
def residualDirectionBoundsWeightedFastFromValBounds {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (weights : Fin seq → Fin seq → Rat)
    (valBounds : Fin seq → Rat × Rat)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    (Fin seq → Rat) × (Fin seq → Rat) :=
  let inputs := inputsOfSlices lnSlice valueSlice dirSlice hLn hDir
  let valLo : Fin seq → Rat := fun k => (valBounds k).1
  let valHi : Fin seq → Rat := fun k => (valBounds k).2
  let embedDot : Fin seq → Rat := fun q => dotProduct inputs.direction (inputs.embed q)
  let outLo : Fin seq → Rat := fun q =>
    embedDot q + dotIntervalLower (weights q) valLo valHi
  let outHi : Fin seq → Rat := fun q =>
    embedDot q + dotIntervalUpper (weights q) valLo valHi
  (outLo, outHi)

/--
Fast weighted residual direction bounds are sound when given sound per-key value bounds.
-/
theorem residualDirectionBoundsWeightedFastFromValBounds_spec {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (weights : Fin seq → Fin seq → Rat)
    (valBounds : Fin seq → Rat × Rat)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel)
    (hvals :
      let inputs := inputsOfSlices lnSlice valueSlice dirSlice hLn hDir
      ∀ k, (valBounds k).1 ≤ valsRealOfInputs inputs k ∧
        valsRealOfInputs inputs k ≤ (valBounds k).2) :
    let bounds :=
      residualDirectionBoundsWeightedFastFromValBounds
        lnSlice valueSlice dirSlice weights valBounds hLn hDir
    let inputs := inputsOfSlices lnSlice valueSlice dirSlice hLn hDir
    ∀ q,
      (bounds.1 q : Real) ≤
          dotProduct (fun i => (inputs.direction i : Real))
            (fun i => inputs.embed q i + Sound.headOutputWithWeights weights inputs q i) ∧
        dotProduct (fun i => (inputs.direction i : Real))
            (fun i => inputs.embed q i + Sound.headOutputWithWeights weights inputs q i) ≤
          (bounds.2 q : Real) := by
  classical
  intro bounds inputs q
  let valLo : Fin seq → Rat := fun k => (valBounds k).1
  let valHi : Fin seq → Rat := fun k => (valBounds k).2
  let embedDot : Fin seq → Rat := fun q => dotProduct inputs.direction (inputs.embed q)
  have hvals' : ∀ k, (valLo k : Real) ≤ valsRealOfInputs inputs k ∧
      valsRealOfInputs inputs k ≤ (valHi k : Real) := by
    simpa [valLo, valHi, inputs] using hvals
  have hlow :=
    dotIntervalLower_le_dotProduct_real (v := weights q)
      (lo := valLo) (hi := valHi) (x := fun k => valsRealOfInputs inputs k)
      (fun k => (hvals' k).1) (fun k => (hvals' k).2)
  have hhigh :=
    dotProduct_le_dotIntervalUpper_real (v := weights q)
      (lo := valLo) (hi := valHi) (x := fun k => valsRealOfInputs inputs k)
      (fun k => (hvals' k).1) (fun k => (hvals' k).2)
  let dir : Fin valueSlice.dModel → Real := fun i => (inputs.direction i : Real)
  have hswap :
      dotProduct dir (fun i => Sound.headOutputWithWeights weights inputs q i) =
        dotProduct (fun k => (weights q k : Real)) (fun k => valsRealOfInputs inputs k) := by
    have hswap' :
        dotProduct dir (fun i => Sound.headOutputWithWeights weights inputs q i) =
          ∑ k, (weights q k : Real) * dotProduct dir (fun i =>
            headValueRealOfInputs inputs k i) := by
      calc
        dotProduct dir (fun i => Sound.headOutputWithWeights weights inputs q i)
            = ∑ i, dir i * ∑ k, (weights q k : Real) * headValueRealOfInputs inputs k i := by
                simp [dotProduct, Sound.headOutputWithWeights_def, dir]
        _ = ∑ i, ∑ k, dir i * ((weights q k : Real) * headValueRealOfInputs inputs k i) := by
                simp [Finset.mul_sum]
        _ = ∑ k, ∑ i, dir i * ((weights q k : Real) * headValueRealOfInputs inputs k i) := by
                simpa using
                  (Finset.sum_comm (s := (Finset.univ : Finset (Fin valueSlice.dModel)))
                    (t := (Finset.univ : Finset (Fin seq)))
                    (f := fun i k =>
                      dir i * ((weights q k : Real) * headValueRealOfInputs inputs k i)))
        _ = ∑ k, (weights q k : Real) * ∑ i, dir i * headValueRealOfInputs inputs k i := by
                refine Finset.sum_congr rfl ?_
                intro k _
                calc
                  ∑ i, dir i * ((weights q k : Real) * headValueRealOfInputs inputs k i)
                      = ∑ i, (weights q k : Real) *
                          (dir i * headValueRealOfInputs inputs k i) := by
                          refine Finset.sum_congr rfl ?_
                          intro i _
                          simp [mul_assoc, mul_comm]
                  _ = (weights q k : Real) *
                      ∑ i, dir i * headValueRealOfInputs inputs k i := by
                          simp [Finset.mul_sum]
    have hvalsReal :
        ∀ k, dotProduct dir (fun i => headValueRealOfInputs inputs k i) =
          valsRealOfInputs inputs k := by
      intro k
      simpa [dir] using
        (Sound.direction_dot_headValue_eq_valsReal (inputs := inputs) (k := k))
    have hsum :
        dotProduct dir (fun i => Sound.headOutputWithWeights weights inputs q i) =
          ∑ k, (weights q k : Real) * valsRealOfInputs inputs k := by
      calc
        dotProduct dir (fun i => Sound.headOutputWithWeights weights inputs q i)
            = ∑ k, (weights q k : Real) *
                dotProduct dir (fun i => headValueRealOfInputs inputs k i) := hswap'
        _ = ∑ k, (weights q k : Real) * valsRealOfInputs inputs k := by
            refine Finset.sum_congr rfl ?_
            intro k _
            exact congrArg (fun x => (weights q k : Real) * x) (hvalsReal k)
    simpa [dotProduct] using hsum
  have hEmbed :
      (embedDot q : Real) =
        dotProduct dir (fun i => (inputs.embed q i : Real)) := by
    classical
    simp [embedDot, dir, dotProduct]
  have hsplit :
      dotProduct dir (fun i => inputs.embed q i + Sound.headOutputWithWeights weights inputs q i) =
        dotProduct dir (fun i => (inputs.embed q i : Real)) +
          dotProduct dir (fun i => Sound.headOutputWithWeights weights inputs q i) := by
    simpa [dir] using
      (Nfp.Linear.dotProduct_add_right (x := dir)
        (y := fun i => (inputs.embed q i : Real))
        (z := fun i => Sound.headOutputWithWeights weights inputs q i))
  have hsum :
      dotProduct dir (fun i => inputs.embed q i + Sound.headOutputWithWeights weights inputs q i) =
        (embedDot q : Real) +
          dotProduct (fun k => (weights q k : Real)) (fun k => valsRealOfInputs inputs k) := by
    calc
      dotProduct dir (fun i =>
          inputs.embed q i + Sound.headOutputWithWeights weights inputs q i)
          = dotProduct dir (fun i => (inputs.embed q i : Real)) +
              dotProduct dir (fun i => Sound.headOutputWithWeights weights inputs q i) := hsplit
      _ = (embedDot q : Real) +
            dotProduct (fun k => (weights q k : Real)) (fun k => valsRealOfInputs inputs k) := by
            simp [hEmbed, hswap]
  have hsum' :
      dotProduct (fun i => (inputs.direction i : Real))
          (fun i => inputs.embed q i + Sound.headOutputWithWeights weights inputs q i) =
        (embedDot q : Real) +
          dotProduct (fun k => (weights q k : Real)) (fun k => valsRealOfInputs inputs k) := by
    simpa [dir] using hsum
  constructor
  · have hlow' := add_le_add_left hlow (embedDot q : Real)
    simpa [bounds, residualDirectionBoundsWeightedFastFromValBounds,
      valLo, valHi, embedDot, hsum', inputs] using hlow'
  · have hhigh' := add_le_add_left hhigh (embedDot q : Real)
    simpa [bounds, residualDirectionBoundsWeightedFastFromValBounds,
      valLo, valHi, embedDot, hsum', inputs] using hhigh'

/--
Compute weighted residual direction bounds directly from model slices.
-/
def residualDirectionBoundsWeightedFastFromSlices {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (weights : Fin seq → Fin seq → Rat)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    (Fin seq → Rat) × (Fin seq → Rat) :=
  let inputs := inputsOfSlices lnSlice valueSlice dirSlice hLn hDir
  let lnBounds : Fin seq → (Fin valueSlice.dModel → Rat) × (Fin valueSlice.dModel → Rat) :=
    fun q =>
      match lnSlice.lnScale? with
      | some scale =>
          layerNormBoundsWithScale scale lnSlice.lnEps inputs.ln1Gamma inputs.ln1Beta
            (inputs.embed q)
      | none =>
          layerNormBounds lnSlice.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q)
  let lnLo : Fin seq → Fin valueSlice.dModel → Rat :=
    fun q i => (lnBounds q).1 i - lnSlice.lnSlack
  let lnHi : Fin seq → Fin valueSlice.dModel → Rat :=
    fun q i => (lnBounds q).2 i + lnSlice.lnSlack
  let dirHead : Fin valueSlice.headDim → Rat :=
    fun d => (dirHeadVecOfInputs inputs).get d
  let bias : Rat := attnBiasDot inputs
  let vLo : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
    dotIntervalLower (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
  let vHi : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
    dotIntervalUpper (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
  let valLo : Fin seq → Rat := fun k => dotIntervalLower dirHead (vLo k) (vHi k) + bias
  let valHi : Fin seq → Rat := fun k => dotIntervalUpper dirHead (vLo k) (vHi k) + bias
  let embedDot : Fin seq → Rat := fun q => dotProduct inputs.direction (inputs.embed q)
  let outLo : Fin seq → Rat := fun q =>
    embedDot q + dotIntervalLower (weights q) valLo valHi
  let outHi : Fin seq → Rat := fun q =>
    embedDot q + dotIntervalUpper (weights q) valLo valHi
  (outLo, outHi)

/--
Fast slice-derived weighted residual direction bounds are sound for
`embed + headOutputWithWeights`.
-/
theorem residualDirectionBoundsWeightedFastFromSlices_spec {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (weights : Fin seq → Fin seq → Rat)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel)
    (hModel : valueSlice.dModel ≠ 0)
    (hEps : 0 < lnSlice.lnEps)
    (hSlack : 0 ≤ lnSlice.lnSlack)
    (hScalePos : ∀ scale, lnSlice.lnScale? = some scale → 0 < scale)
    (hSqrt :
      match lnSlice.lnScale? with
      | some scale => 0 < sqrtLowerWithScale scale lnSlice.lnEps
      | none => 0 < sqrtLower lnSlice.lnEps) :
    let bounds :=
      residualDirectionBoundsWeightedFastFromSlices
        lnSlice valueSlice dirSlice weights hLn hDir
    let inputs := inputsOfSlices lnSlice valueSlice dirSlice hLn hDir
    ∀ q,
      (bounds.1 q : Real) ≤
          dotProduct (fun i => (inputs.direction i : Real))
            (fun i => inputs.embed q i + Sound.headOutputWithWeights weights inputs q i) ∧
        dotProduct (fun i => (inputs.direction i : Real))
            (fun i => inputs.embed q i + Sound.headOutputWithWeights weights inputs q i) ≤
          (bounds.2 q : Real) := by
  classical
  intro bounds inputs q
  let lnBounds : Fin seq → (Fin valueSlice.dModel → Rat) × (Fin valueSlice.dModel → Rat) :=
    fun q =>
      match lnSlice.lnScale? with
      | some scale =>
          layerNormBoundsWithScale scale lnSlice.lnEps inputs.ln1Gamma inputs.ln1Beta
            (inputs.embed q)
      | none =>
          layerNormBounds lnSlice.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q)
  let lnLo : Fin seq → Fin valueSlice.dModel → Rat :=
    fun q i => (lnBounds q).1 i - lnSlice.lnSlack
  let lnHi : Fin seq → Fin valueSlice.dModel → Rat :=
    fun q i => (lnBounds q).2 i + lnSlice.lnSlack
  let dirHead : Fin valueSlice.headDim → Rat :=
    fun d => (dirHeadVecOfInputs inputs).get d
  let bias : Rat := attnBiasDot inputs
  let vLo : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
    dotIntervalLower (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
  let vHi : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
    dotIntervalUpper (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
  let valLo : Fin seq → Rat := fun k => dotIntervalLower dirHead (vLo k) (vHi k) + bias
  let valHi : Fin seq → Rat := fun k => dotIntervalUpper dirHead (vLo k) (vHi k) + bias
  let embedDot : Fin seq → Rat := fun q => dotProduct inputs.direction (inputs.embed q)
  have hvals :=
    valsRealOfInputs_bounds_from_slices lnSlice valueSlice dirSlice
      hLn hDir hModel hEps hSlack hScalePos hSqrt
  have hvals' : ∀ k, (valLo k : Real) ≤ valsRealOfInputs inputs k ∧
      valsRealOfInputs inputs k ≤ (valHi k : Real) := by
    simpa [lnBounds, lnLo, lnHi, dirHead, bias, vLo, vHi, valLo, valHi, inputs] using hvals
  have hlow :=
    dotIntervalLower_le_dotProduct_real (v := weights q)
      (lo := valLo) (hi := valHi) (x := fun k => valsRealOfInputs inputs k)
      (fun k => (hvals' k).1) (fun k => (hvals' k).2)
  have hhigh :=
    dotProduct_le_dotIntervalUpper_real (v := weights q)
      (lo := valLo) (hi := valHi) (x := fun k => valsRealOfInputs inputs k)
      (fun k => (hvals' k).1) (fun k => (hvals' k).2)
  let dir : Fin valueSlice.dModel → Real := fun i => (inputs.direction i : Real)
  have hswap :
      dotProduct dir (fun i => Sound.headOutputWithWeights weights inputs q i) =
        dotProduct (fun k => (weights q k : Real)) (fun k => valsRealOfInputs inputs k) := by
    have hswap' :
        dotProduct dir (fun i => Sound.headOutputWithWeights weights inputs q i) =
          ∑ k, (weights q k : Real) * dotProduct dir (fun i =>
            headValueRealOfInputs inputs k i) := by
      calc
        dotProduct dir (fun i => Sound.headOutputWithWeights weights inputs q i)
            = ∑ i, dir i * ∑ k, (weights q k : Real) * headValueRealOfInputs inputs k i := by
                simp [dotProduct, Sound.headOutputWithWeights_def, dir]
        _ = ∑ i, ∑ k, dir i * ((weights q k : Real) * headValueRealOfInputs inputs k i) := by
                simp [Finset.mul_sum]
        _ = ∑ k, ∑ i, dir i * ((weights q k : Real) * headValueRealOfInputs inputs k i) := by
                simpa using
                  (Finset.sum_comm (s := (Finset.univ : Finset (Fin valueSlice.dModel)))
                    (t := (Finset.univ : Finset (Fin seq)))
                    (f := fun i k =>
                      dir i * ((weights q k : Real) * headValueRealOfInputs inputs k i)))
        _ = ∑ k, (weights q k : Real) * ∑ i, dir i * headValueRealOfInputs inputs k i := by
                refine Finset.sum_congr rfl ?_
                intro k _
                calc
                  ∑ i, dir i * ((weights q k : Real) * headValueRealOfInputs inputs k i)
                      = ∑ i, (weights q k : Real) *
                          (dir i * headValueRealOfInputs inputs k i) := by
                          refine Finset.sum_congr rfl ?_
                          intro i _
                          simp [mul_assoc, mul_comm]
                  _ = (weights q k : Real) *
                      ∑ i, dir i * headValueRealOfInputs inputs k i := by
                          simp [Finset.mul_sum]
    have hvalsReal :
        ∀ k, dotProduct dir (fun i => headValueRealOfInputs inputs k i) =
          valsRealOfInputs inputs k := by
      intro k
      simpa [dir] using
        (Sound.direction_dot_headValue_eq_valsReal (inputs := inputs) (k := k))
    have hsum :
        dotProduct dir (fun i => Sound.headOutputWithWeights weights inputs q i) =
          ∑ k, (weights q k : Real) * valsRealOfInputs inputs k := by
      calc
        dotProduct dir (fun i => Sound.headOutputWithWeights weights inputs q i)
            = ∑ k, (weights q k : Real) *
                dotProduct dir (fun i => headValueRealOfInputs inputs k i) := hswap'
        _ = ∑ k, (weights q k : Real) * valsRealOfInputs inputs k := by
            refine Finset.sum_congr rfl ?_
            intro k _
            exact congrArg (fun x => (weights q k : Real) * x) (hvalsReal k)
    simpa [dotProduct] using hsum
  have hEmbed :
      (embedDot q : Real) =
        dotProduct dir (fun i => (inputs.embed q i : Real)) := by
    classical
    simp [embedDot, dir, dotProduct]
  have hsplit :
      dotProduct dir (fun i => inputs.embed q i + Sound.headOutputWithWeights weights inputs q i) =
        dotProduct dir (fun i => (inputs.embed q i : Real)) +
          dotProduct dir (fun i => Sound.headOutputWithWeights weights inputs q i) := by
    simpa [dir] using
      (Nfp.Linear.dotProduct_add_right (x := dir)
        (y := fun i => (inputs.embed q i : Real))
        (z := fun i => Sound.headOutputWithWeights weights inputs q i))
  have hsum :
      dotProduct dir (fun i => inputs.embed q i + Sound.headOutputWithWeights weights inputs q i) =
        (embedDot q : Real) +
          dotProduct (fun k => (weights q k : Real)) (fun k => valsRealOfInputs inputs k) := by
    calc
      dotProduct dir (fun i =>
          inputs.embed q i + Sound.headOutputWithWeights weights inputs q i)
          = dotProduct dir (fun i => (inputs.embed q i : Real)) +
              dotProduct dir (fun i => Sound.headOutputWithWeights weights inputs q i) := hsplit
      _ = (embedDot q : Real) +
            dotProduct (fun k => (weights q k : Real)) (fun k => valsRealOfInputs inputs k) := by
            simp [hEmbed, hswap]
  have hsum' :
      dotProduct (fun i => (inputs.direction i : Real))
          (fun i => inputs.embed q i + Sound.headOutputWithWeights weights inputs q i) =
        (embedDot q : Real) +
          dotProduct (fun k => (weights q k : Real)) (fun k => valsRealOfInputs inputs k) := by
    simpa [dir] using hsum
  constructor
  · have hlow' := add_le_add_left hlow (embedDot q : Real)
    simpa [bounds, residualDirectionBoundsWeightedFastFromSlices,
      lnBounds, lnLo, lnHi, dirHead, bias, vLo, vHi, valLo, valHi, embedDot, hsum', inputs] using
      hlow'
  · have hhigh' := add_le_add_left hhigh (embedDot q : Real)
    simpa [bounds, residualDirectionBoundsWeightedFastFromSlices,
      lnBounds, lnLo, lnHi, dirHead, bias, vLo, vHi, valLo, valHi, embedDot, hsum', inputs] using
      hhigh'

end InductionHeadCert

end Pure

end IO

end Nfp
