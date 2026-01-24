-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Sound.Induction.CoreDefs
public import Nfp.Bounds.Interval

/-!
Value-path bounds for induction-head inputs.
-/

public section

namespace Nfp

namespace Sound

open Nfp.Bounds

variable {seq dModel dHead : Nat}

noncomputable section

/-- LayerNorm outputs stay within `layerNormBounds` expanded by `lnSlack`. -/
theorem lnRealOfInputs_bounds_with_slack
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (lnSlack : Rat) (hSlack : 0 ≤ lnSlack)
    (hModel : dModel ≠ 0) (hEps : 0 < inputs.lnEps) (hSqrt : 0 < sqrtLower inputs.lnEps) :
    let lnBounds : Fin seq → (Fin dModel → Rat) × (Fin dModel → Rat) := fun q =>
      layerNormBounds inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q)
    let lnLo : Fin seq → Fin dModel → Rat := fun q i => (lnBounds q).1 i - lnSlack
    let lnHi : Fin seq → Fin dModel → Rat := fun q i => (lnBounds q).2 i + lnSlack
    ∀ q i,
      (lnLo q i : Real) ≤ lnRealOfInputs inputs q i ∧
        lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
  classical
  intro lnBounds lnLo lnHi q i
  have hln :=
    layerNormBounds_spec (eps := inputs.lnEps)
      (gamma := inputs.ln1Gamma) (beta := inputs.ln1Beta)
      (x := inputs.embed q) hModel hEps hSqrt
  have hln' : (lnBounds q).1 i ≤ lnRealOfInputs inputs q i ∧
      lnRealOfInputs inputs q i ≤ (lnBounds q).2 i := by
    simpa [lnBounds, lnRealOfInputs_def] using hln i
  have hlow : (lnLo q i : Real) ≤ lnRealOfInputs inputs q i := by
    have hlow' : (lnBounds q).1 i - lnSlack ≤ (lnBounds q).1 i := by
      have hlow'' : (lnBounds q).1 i - lnSlack ≤ (lnBounds q).1 i :=
        sub_le_self _ hSlack
      simpa using hlow''
    have hlow'_real : ((lnBounds q).1 i - lnSlack : Real) ≤ ((lnBounds q).1 i : Real) := by
      simpa [ratToReal_def] using ratToReal_le_of_le hlow'
    have hlow'' : ((lnBounds q).1 i - lnSlack : Real) ≤ lnRealOfInputs inputs q i :=
      le_trans hlow'_real hln'.1
    simpa [lnLo] using hlow''
  have hhigh : lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
    have hhigh' : (lnBounds q).2 i ≤ (lnBounds q).2 i + lnSlack := by
      have hhigh'' : (lnBounds q).2 i ≤ (lnBounds q).2 i + lnSlack :=
        le_add_of_nonneg_right hSlack
      simpa using hhigh''
    have hhigh'_real : ((lnBounds q).2 i : Real) ≤ ((lnBounds q).2 i + lnSlack : Real) := by
      simpa [ratToReal_def] using ratToReal_le_of_le hhigh'
    have hhigh'' : lnRealOfInputs inputs q i ≤ ((lnBounds q).2 i + lnSlack : Real) :=
      le_trans hln'.2 hhigh'_real
    simpa [lnHi] using hhigh''
  exact ⟨hlow, hhigh⟩

/-- LayerNorm outputs stay within scaled bounds expanded by `lnSlack`. -/
theorem lnRealOfInputs_bounds_with_scale_slack
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (scale : Nat) (hScale : 0 < scale)
    (lnSlack : Rat) (hSlack : 0 ≤ lnSlack)
    (hModel : dModel ≠ 0)
    (hEps : 0 < inputs.lnEps)
    (hSqrt : 0 < sqrtLowerWithScale scale inputs.lnEps) :
    let lnBounds : Fin seq → (Fin dModel → Rat) × (Fin dModel → Rat) := fun q =>
      layerNormBoundsWithScale scale inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q)
    let lnLo : Fin seq → Fin dModel → Rat := fun q i => (lnBounds q).1 i - lnSlack
    let lnHi : Fin seq → Fin dModel → Rat := fun q i => (lnBounds q).2 i + lnSlack
    ∀ q i,
      (lnLo q i : Real) ≤ lnRealOfInputs inputs q i ∧
        lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
  classical
  intro lnBounds lnLo lnHi q i
  have hln :=
    layerNormBoundsWithScale_spec (scale := scale)
      (eps := inputs.lnEps)
      (gamma := inputs.ln1Gamma) (beta := inputs.ln1Beta)
      (x := inputs.embed q) hModel hEps hSqrt hScale
  have hln' : (lnBounds q).1 i ≤ lnRealOfInputs inputs q i ∧
      lnRealOfInputs inputs q i ≤ (lnBounds q).2 i := by
    simpa [lnBounds, lnRealOfInputs_def] using hln i
  have hlow : (lnLo q i : Real) ≤ lnRealOfInputs inputs q i := by
    have hlow' : (lnBounds q).1 i - lnSlack ≤ (lnBounds q).1 i := by
      have hlow'' : (lnBounds q).1 i - lnSlack ≤ (lnBounds q).1 i :=
        sub_le_self _ hSlack
      simpa using hlow''
    have hlow'_real : ((lnBounds q).1 i - lnSlack : Real) ≤ ((lnBounds q).1 i : Real) := by
      simpa [ratToReal_def] using ratToReal_le_of_le hlow'
    have hlow'' : ((lnBounds q).1 i - lnSlack : Real) ≤ lnRealOfInputs inputs q i :=
      le_trans hlow'_real hln'.1
    simpa [lnLo] using hlow''
  have hhigh : lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
    have hhigh' : (lnBounds q).2 i ≤ (lnBounds q).2 i + lnSlack := by
      have hhigh'' : (lnBounds q).2 i ≤ (lnBounds q).2 i + lnSlack :=
        le_add_of_nonneg_right hSlack
      simpa using hhigh''
    have hhigh'_real : ((lnBounds q).2 i : Real) ≤ ((lnBounds q).2 i + lnSlack : Real) := by
      simpa [ratToReal_def] using ratToReal_le_of_le hhigh'
    have hhigh'' : lnRealOfInputs inputs q i ≤ ((lnBounds q).2 i + lnSlack : Real) :=
      le_trans hln'.2 hhigh'_real
    simpa [lnHi] using hhigh''
  exact ⟨hlow, hhigh⟩

/-- Value outputs are bounded by any LayerNorm interval bounds. -/
theorem valsRealOfInputs_bounds_of_lnBounds
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (lnLo lnHi : Fin seq → Fin dModel → Rat)
    (hln : ∀ q i,
      (lnLo q i : Real) ≤ lnRealOfInputs inputs q i ∧
        lnRealOfInputs inputs q i ≤ (lnHi q i : Real)) :
    let vLo : Fin seq → Fin dHead → Rat := fun k d =>
      dotIntervalLower (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
    let vHi : Fin seq → Fin dHead → Rat := fun k d =>
      dotIntervalUpper (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
    let dirHead : Fin dHead → Rat := fun d => (dirHeadVecOfInputs inputs).get d
    let valLo : Fin seq → Rat := fun k => dotIntervalLower dirHead (vLo k) (vHi k)
    let valHi : Fin seq → Rat := fun k => dotIntervalUpper dirHead (vLo k) (vHi k)
    ∀ k,
      (valLo k : Real) ≤ valsRealOfInputs inputs k ∧
        valsRealOfInputs inputs k ≤ (valHi k : Real) := by
  classical
  intro vLo vHi dirHead valLo valHi k
  have hv_bounds : ∀ d,
      (vLo k d : Real) ≤ vRealOfInputs inputs k d ∧
        vRealOfInputs inputs k d ≤ (vHi k d : Real) := by
    intro d
    have hlow :=
      dotIntervalLower_le_dotProduct_real_add
        (v := fun j => inputs.wv j d) (lo := lnLo k) (hi := lnHi k)
        (x := fun j => lnRealOfInputs inputs k j) (b := (inputs.bv d : Real))
        (hlo := fun j => (hln k j).1)
        (hhi := fun j => (hln k j).2)
    have hhigh :=
      dotProduct_le_dotIntervalUpper_real_add
        (v := fun j => inputs.wv j d) (lo := lnLo k) (hi := lnHi k)
        (x := fun j => lnRealOfInputs inputs k j) (b := (inputs.bv d : Real))
        (hlo := fun j => (hln k j).1)
        (hhi := fun j => (hln k j).2)
    constructor
    · simpa [vLo, vRealOfInputs_def] using hlow
    · simpa [vHi, vRealOfInputs_def] using hhigh
  have hlow :=
    dotIntervalLower_le_dotProduct_real
      (v := dirHead) (lo := vLo k) (hi := vHi k)
      (x := fun d => vRealOfInputs inputs k d)
      (hlo := fun d => (hv_bounds d).1)
      (hhi := fun d => (hv_bounds d).2)
  have hhigh :=
    dotProduct_le_dotIntervalUpper_real
      (v := dirHead) (lo := vLo k) (hi := vHi k)
      (x := fun d => vRealOfInputs inputs k d)
      (hlo := fun d => (hv_bounds d).1)
      (hhi := fun d => (hv_bounds d).2)
  constructor
  · simpa [valLo, valsRealOfInputs_def, dirHead] using hlow
  · simpa [valHi, valsRealOfInputs_def, dirHead] using hhigh

/-- Value outputs are bounded by LayerNorm bounds expanded by `lnSlack`. -/
theorem valsRealOfInputs_bounds_with_lnSlack
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (lnSlack : Rat) (hSlack : 0 ≤ lnSlack)
    (hModel : dModel ≠ 0) (hEps : 0 < inputs.lnEps) (hSqrt : 0 < sqrtLower inputs.lnEps) :
    let lnBounds : Fin seq → (Fin dModel → Rat) × (Fin dModel → Rat) := fun q =>
      layerNormBounds inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q)
    let lnLo : Fin seq → Fin dModel → Rat := fun q i => (lnBounds q).1 i - lnSlack
    let lnHi : Fin seq → Fin dModel → Rat := fun q i => (lnBounds q).2 i + lnSlack
    let vLo : Fin seq → Fin dHead → Rat := fun k d =>
      dotIntervalLower (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
    let vHi : Fin seq → Fin dHead → Rat := fun k d =>
      dotIntervalUpper (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
    let dirHead : Fin dHead → Rat := fun d => (dirHeadVecOfInputs inputs).get d
    let valLo : Fin seq → Rat := fun k => dotIntervalLower dirHead (vLo k) (vHi k)
    let valHi : Fin seq → Rat := fun k => dotIntervalUpper dirHead (vLo k) (vHi k)
    ∀ k,
      (valLo k : Real) ≤ valsRealOfInputs inputs k ∧
        valsRealOfInputs inputs k ≤ (valHi k : Real) := by
  intro lnBounds lnLo lnHi vLo vHi dirHead valLo valHi k
  have hln :=
    lnRealOfInputs_bounds_with_slack inputs lnSlack hSlack hModel hEps hSqrt
  have hln' : ∀ q i,
      (lnLo q i : Real) ≤ lnRealOfInputs inputs q i ∧
        lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
    simpa [lnBounds, lnLo, lnHi] using hln
  have hvals := valsRealOfInputs_bounds_of_lnBounds inputs lnLo lnHi hln'
  have hvals' := hvals k
  simpa [vLo, vHi, dirHead, valLo, valHi] using hvals'

/-- Value outputs are bounded by scaled LayerNorm bounds expanded by `lnSlack`. -/
theorem valsRealOfInputs_bounds_with_scale_lnSlack
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (scale : Nat) (hScale : 0 < scale)
    (lnSlack : Rat) (hSlack : 0 ≤ lnSlack)
    (hModel : dModel ≠ 0)
    (hEps : 0 < inputs.lnEps)
    (hSqrt : 0 < sqrtLowerWithScale scale inputs.lnEps) :
    let lnBounds : Fin seq → (Fin dModel → Rat) × (Fin dModel → Rat) := fun q =>
      layerNormBoundsWithScale scale inputs.lnEps inputs.ln1Gamma inputs.ln1Beta (inputs.embed q)
    let lnLo : Fin seq → Fin dModel → Rat := fun q i => (lnBounds q).1 i - lnSlack
    let lnHi : Fin seq → Fin dModel → Rat := fun q i => (lnBounds q).2 i + lnSlack
    let vLo : Fin seq → Fin dHead → Rat := fun k d =>
      dotIntervalLower (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
    let vHi : Fin seq → Fin dHead → Rat := fun k d =>
      dotIntervalUpper (fun j => inputs.wv j d) (lnLo k) (lnHi k) + inputs.bv d
    let dirHead : Fin dHead → Rat := fun d => (dirHeadVecOfInputs inputs).get d
    let valLo : Fin seq → Rat := fun k => dotIntervalLower dirHead (vLo k) (vHi k)
    let valHi : Fin seq → Rat := fun k => dotIntervalUpper dirHead (vLo k) (vHi k)
    ∀ k,
      (valLo k : Real) ≤ valsRealOfInputs inputs k ∧
        valsRealOfInputs inputs k ≤ (valHi k : Real) := by
  intro lnBounds lnLo lnHi vLo vHi dirHead valLo valHi k
  have hln :=
    lnRealOfInputs_bounds_with_scale_slack inputs scale hScale lnSlack hSlack hModel hEps hSqrt
  have hln' : ∀ q i,
      (lnLo q i : Real) ≤ lnRealOfInputs inputs q i ∧
        lnRealOfInputs inputs q i ≤ (lnHi q i : Real) := by
    simpa [lnBounds, lnLo, lnHi] using hln
  have hvals := valsRealOfInputs_bounds_of_lnBounds inputs lnLo lnHi hln'
  have hvals' := hvals k
  simpa [vLo, vHi, dirHead, valLo, valHi] using hvals'

end

end Sound

end Nfp
