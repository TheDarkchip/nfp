-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Bounds.LayerNorm
public import Nfp.IO.InductionHead.ModelDirectionSlice
public import Nfp.IO.InductionHead.ModelLnSlice
public import Nfp.IO.InductionHead.ModelValueSlice
public import Nfp.Model.InductionHead
public import Nfp.Sound.Induction.ValueBounds

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

/-- Assemble induction-head inputs from model slices and direction metadata. -/
noncomputable def inputsOfSlices {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel) :
    Model.InductionHeadInputs seq valueSlice.dModel valueSlice.headDim :=
  { scale := 0
    active := ∅
    prev := fun q => q
    embed := by simpa [hLn] using lnSlice.embed
    lnEps := lnSlice.lnEps
    ln1Gamma := by simpa [hLn] using lnSlice.lnGamma
    ln1Beta := by simpa [hLn] using lnSlice.lnBeta
    wq := fun _ _ => 0
    bq := fun _ => 0
    wk := fun _ _ => 0
    bk := fun _ => 0
    wv := valueSlice.wv
    bv := valueSlice.bv
    wo := valueSlice.wo
    attnBias := valueSlice.attnBias
    maskCausal := false
    maskValue := 0
    directionSpec := { target := dirSlice.target, negative := dirSlice.negative }
    direction := by simpa [hDir] using dirSlice.direction }

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
    let vLo : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
      dotIntervalLower (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
    let vHi : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
      dotIntervalUpper (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
    let valLo : Fin seq → Rat := fun k => dotIntervalLower dirHead (vLo k) (vHi k)
    let valHi : Fin seq → Rat := fun k => dotIntervalUpper dirHead (vLo k) (vHi k)
    ∀ k, (valLo k : Real) ≤ valsRealOfInputs inputs k ∧
      valsRealOfInputs inputs k ≤ (valHi k : Real) := by
  classical
  intro inputs lnBounds lnLo lnHi dirHead vLo vHi valLo valHi k
  cases hscale : lnSlice.lnScale? with
  | none =>
      have hSqrt' : 0 < sqrtLower lnSlice.lnEps := by
        simpa [hscale] using hSqrt
      have hvals :=
        valsRealOfInputs_bounds_with_lnSlack inputs lnSlice.lnSlack hSlack
          hModel hEps hSqrt'
      have hvals' := hvals k
      simpa [lnBounds, lnLo, lnHi, vLo, vHi, valLo, valHi, dirHead, hscale] using hvals'
  | some scale =>
      have hSqrt' : 0 < sqrtLowerWithScale scale lnSlice.lnEps := by
        simpa [hscale] using hSqrt
      have hScalePos' : 0 < scale := hScalePos scale hscale
      have hvals :=
        valsRealOfInputs_bounds_with_scale_lnSlack inputs scale hScalePos' lnSlice.lnSlack hSlack
          hModel hEps hSqrt'
      have hvals' := hvals k
      simpa [lnBounds, lnLo, lnHi, vLo, vHi, valLo, valHi, dirHead, hscale] using hvals'

end InductionHeadCert

end Pure

end IO

end Nfp
