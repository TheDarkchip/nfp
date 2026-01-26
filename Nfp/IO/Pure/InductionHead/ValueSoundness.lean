-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.Pure.InductionHead.ValueCheck
public import Nfp.IO.Pure.InductionHead.ModelInputs

/-!
Soundness bridge for model-anchored value certificates.
-/

public section

namespace Nfp

namespace IO

namespace Pure

namespace InductionHeadCert

open Nfp.Bounds
open Nfp.Sound

/--
If the model-slice check passes, certified values lie within the slice-derived interval
and the true model values (computed from the slices) lie within the same interval.
-/
theorem valuesWithinModelBounds_interval_sound {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel)
    (values : Circuit.ValueIntervalCert seq)
    (hcheck : valuesWithinModelBounds lnSlice valueSlice dirSlice hLn hDir values = true)
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
    ∀ k,
      valLo k ≤ values.vals k ∧ values.vals k ≤ valHi k ∧
        (valLo k : Real) ≤ valsRealOfInputs inputs k ∧
          valsRealOfInputs inputs k ≤ (valHi k : Real) := by
  classical
  intro inputs lnBounds lnLo lnHi dirHead bias vLo vHi valLo valHi k
  set embed : Fin seq → Fin valueSlice.dModel → Rat := by
    simpa [hLn] using lnSlice.embed
  set lnGamma : Fin valueSlice.dModel → Rat := by
    simpa [hLn] using lnSlice.lnGamma
  set lnBeta : Fin valueSlice.dModel → Rat := by
    simpa [hLn] using lnSlice.lnBeta
  set direction : Fin valueSlice.dModel → Rat := by
    simpa [hDir] using dirSlice.direction
  set biasSlice : Rat :=
    Linear.dotFin valueSlice.dModel (fun j => valueSlice.attnBias j) direction
  set dirHeadSlice : Fin valueSlice.headDim → Rat :=
    fun d => Linear.dotFin valueSlice.dModel (fun j => valueSlice.wo j d) direction
  set lnBoundsSlice : Fin seq → (Fin valueSlice.dModel → Rat) × (Fin valueSlice.dModel → Rat) :=
    fun q =>
      match lnSlice.lnScale? with
      | some scale => Bounds.layerNormBoundsWithScale scale lnSlice.lnEps lnGamma lnBeta (embed q)
      | none => Bounds.layerNormBounds lnSlice.lnEps lnGamma lnBeta (embed q)
  set lnLoSlice : Fin seq → Fin valueSlice.dModel → Rat :=
    fun q i => (lnBoundsSlice q).1 i - lnSlice.lnSlack
  set lnHiSlice : Fin seq → Fin valueSlice.dModel → Rat :=
    fun q i => (lnBoundsSlice q).2 i + lnSlice.lnSlack
  set vLoSlice : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
    dotIntervalLower (fun j => valueSlice.wv j d) (lnLoSlice k) (lnHiSlice k) + valueSlice.bv d
  set vHiSlice : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
    dotIntervalUpper (fun j => valueSlice.wv j d) (lnLoSlice k) (lnHiSlice k) + valueSlice.bv d
  set valLoSlice : Fin seq → Rat := fun k =>
    dotIntervalLower dirHeadSlice (vLoSlice k) (vHiSlice k) + biasSlice
  set valHiSlice : Fin seq → Rat := fun k =>
    dotIntervalUpper dirHeadSlice (vLoSlice k) (vHiSlice k) + biasSlice
  set vLoFastSlice : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
    (dotIntervalBoundsFast (fun j => valueSlice.wv j d) (lnLoSlice k) (lnHiSlice k)).1 +
      valueSlice.bv d
  set vHiFastSlice : Fin seq → Fin valueSlice.headDim → Rat := fun k d =>
    (dotIntervalBoundsFast (fun j => valueSlice.wv j d) (lnLoSlice k) (lnHiSlice k)).2 +
      valueSlice.bv d
  set valLoFastSlice : Fin seq → Rat := fun k =>
    (dotIntervalBoundsFast dirHeadSlice (vLoFastSlice k) (vHiFastSlice k)).1 + biasSlice
  set valHiFastSlice : Fin seq → Rat := fun k =>
    (dotIntervalBoundsFast dirHeadSlice (vLoFastSlice k) (vHiFastSlice k)).2 + biasSlice
  have hcert :=
    valuesWithinModelBounds_sound lnSlice valueSlice dirSlice hLn hDir values hcheck
  have htrue :=
    valsRealOfInputs_bounds_from_slices lnSlice valueSlice dirSlice hLn hDir
      hModel hEps hSlack hScalePos hSqrt
  have hcert' : valLoFastSlice k ≤ values.vals k ∧ values.vals k ≤ valHiFastSlice k := by
    simpa [valLoFastSlice, valHiFastSlice] using hcert k
  have hvLoFast_eq : vLoFastSlice = vLoSlice := by
    funext k d
    simp [vLoFastSlice, vLoSlice, dotIntervalBoundsFast_fst]
  have hvHiFast_eq : vHiFastSlice = vHiSlice := by
    funext k d
    simp [vHiFastSlice, vHiSlice, dotIntervalBoundsFast_snd]
  have hvalLoFast_eq : valLoFastSlice = valLoSlice := by
    funext k
    simp [valLoFastSlice, valLoSlice, hvLoFast_eq, hvHiFast_eq, dotIntervalBoundsFast_fst,
      biasSlice]
  have hvalHiFast_eq : valHiFastSlice = valHiSlice := by
    funext k
    simp [valHiFastSlice, valHiSlice, hvLoFast_eq, hvHiFast_eq, dotIntervalBoundsFast_snd,
      biasSlice]
  have hcertSlice : valLoSlice k ≤ values.vals k ∧ values.vals k ≤ valHiSlice k := by
    simpa [hvalLoFast_eq, hvalHiFast_eq] using hcert'
  have htrue' : (valLo k : Real) ≤ valsRealOfInputs inputs k ∧
      valsRealOfInputs inputs k ≤ (valHi k : Real) := by
    simpa [lnBounds, lnLo, lnHi, vLo, vHi, valLo, valHi, dirHead, bias] using htrue k
  have hcert'' : valLo k ≤ values.vals k ∧ values.vals k ≤ valHi k := by
    have hinputs : inputs = inputsOfSlices lnSlice valueSlice dirSlice hLn hDir := by
      rfl
    have hwo : inputs.wo = valueSlice.wo := by
      simpa [hinputs] using inputsOfSlices_wo lnSlice valueSlice dirSlice hLn hDir
    have hwv : inputs.wv = valueSlice.wv := by
      simpa [hinputs] using inputsOfSlices_wv lnSlice valueSlice dirSlice hLn hDir
    have hbv : inputs.bv = valueSlice.bv := by
      simpa [hinputs] using inputsOfSlices_bv lnSlice valueSlice dirSlice hLn hDir
    have hdir : inputs.direction = direction := by
      simpa [hinputs, direction] using
        inputsOfSlices_direction lnSlice valueSlice dirSlice hLn hDir
    have hembed : inputs.embed = embed := by
      simpa [hinputs, embed] using
        inputsOfSlices_embed lnSlice valueSlice dirSlice hLn hDir
    have hlnGamma : inputs.ln1Gamma = lnGamma := by
      simpa [hinputs, lnGamma] using
        inputsOfSlices_ln1Gamma lnSlice valueSlice dirSlice hLn hDir
    have hlnBeta : inputs.ln1Beta = lnBeta := by
      simpa [hinputs, lnBeta] using
        inputsOfSlices_ln1Beta lnSlice valueSlice dirSlice hLn hDir
    have hlnBounds_eq : lnBounds = lnBoundsSlice := by
      funext q
      cases hscale : lnSlice.lnScale? with
      | none =>
          simp [lnBounds, lnBoundsSlice, hscale, hembed, hlnGamma, hlnBeta]
      | some scale =>
          simp [lnBounds, lnBoundsSlice, hscale, hembed, hlnGamma, hlnBeta]
    have hlnLo_eq : lnLo = lnLoSlice := by
      funext q i
      simp [lnLo, lnLoSlice, hlnBounds_eq]
    have hlnHi_eq : lnHi = lnHiSlice := by
      funext q i
      simp [lnHi, lnHiSlice, hlnBounds_eq]
    have hdirHead_eq : dirHead = dirHeadSlice := by
      funext d
      simp [dirHead, dirHeadSlice, dirHeadVecOfInputs_get, hwo, hdir]
    have hbias_eq : bias = biasSlice := by
      simp [bias, biasSlice, hdir, inputsOfSlices_attnBias, attnBiasDot_def]
    have hvLo_eq : vLo = vLoSlice := by
      funext k d
      simp [vLo, vLoSlice, hwv, hbv, hlnLo_eq, hlnHi_eq]
    have hvHi_eq : vHi = vHiSlice := by
      funext k d
      simp [vHi, vHiSlice, hwv, hbv, hlnLo_eq, hlnHi_eq]
    have hvalLo_eq : valLo = valLoSlice := by
      simp [valLo, valLoSlice, hdirHead_eq, hvLo_eq, hvHi_eq, hbias_eq]
    have hvalHi_eq : valHi = valHiSlice := by
      simp [valHi, valHiSlice, hdirHead_eq, hvLo_eq, hvHi_eq, hbias_eq]
    simpa [hvalLo_eq, hvalHi_eq] using hcertSlice
  exact ⟨hcert''.1, hcert''.2, htrue'.1, htrue'.2⟩

/--
If both the certified values and the true values lie in the same interval, their
difference is bounded by the interval width.
-/
theorem valuesWithinModelBounds_diff_bound {seq : Nat}
    (lnSlice : Nfp.IO.InductionHeadCert.ModelLnSlice seq)
    (valueSlice : Nfp.IO.InductionHeadCert.ModelValueSlice)
    (dirSlice : Nfp.IO.InductionHeadCert.ModelDirectionSlice)
    (hLn : lnSlice.dModel = valueSlice.dModel)
    (hDir : dirSlice.dModel = valueSlice.dModel)
    (values : Circuit.ValueIntervalCert seq)
    (hcheck : valuesWithinModelBounds lnSlice valueSlice dirSlice hLn hDir values = true)
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
    ∀ k,
      |valsRealOfInputs inputs k - (values.vals k : Real)| ≤
        (valHi k - valLo k : Rat) := by
  classical
  intro inputs lnBounds lnLo lnHi dirHead bias vLo vHi valLo valHi k
  have hbounds :=
    valuesWithinModelBounds_interval_sound lnSlice valueSlice dirSlice hLn hDir values hcheck
      hModel hEps hSlack hScalePos hSqrt
  have hk := hbounds k
  have hlo_vals : (valLo k : Real) ≤ valsRealOfInputs inputs k := hk.2.2.1
  have hhi_vals : valsRealOfInputs inputs k ≤ (valHi k : Real) := hk.2.2.2
  have hlo_cert : (valLo k : Real) ≤ (values.vals k : Real) := by
    simpa [ratToReal_def] using ratToReal_le_of_le hk.1
  have hhi_cert : (values.vals k : Real) ≤ (valHi k : Real) := by
    simpa [ratToReal_def] using ratToReal_le_of_le hk.2.1
  have hdiff1 :
      valsRealOfInputs inputs k - (values.vals k : Real) ≤
        (valHi k - valLo k : Real) := by
    have h1 : valsRealOfInputs inputs k ≤ (valHi k : Real) := hhi_vals
    have h2 : (valLo k : Real) ≤ (values.vals k : Real) := hlo_cert
    linarith
  have hdiff2 :
      (values.vals k : Real) - valsRealOfInputs inputs k ≤
        (valHi k - valLo k : Real) := by
    have h1 : (values.vals k : Real) ≤ (valHi k : Real) := hhi_cert
    have h2 : (valLo k : Real) ≤ valsRealOfInputs inputs k := hlo_vals
    linarith
  have hdiff :
      |valsRealOfInputs inputs k - (values.vals k : Real)| ≤
        (valHi k - valLo k : Real) := by
    have h := abs_le.mpr ⟨by
      have : -(valHi k - valLo k : Real) ≤ valsRealOfInputs inputs k - (values.vals k : Real) := by
        linarith
      simpa using this, hdiff1⟩
    exact h
  simpa [ratToReal_sub, ratToReal_def] using hdiff

end InductionHeadCert

end Pure

end IO

end Nfp
