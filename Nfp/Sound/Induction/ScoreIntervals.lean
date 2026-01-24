-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Sound.Induction.CoreDefs
public import Nfp.Bounds.Interval

/-!
Score interval bounds derived from LayerNorm interval bounds.
-/

public section

namespace Nfp

namespace Sound

open Nfp.Bounds

variable {seq dModel dHead : Nat}

noncomputable section

/-- Score bounds from any LayerNorm interval bounds, assuming nonnegative score scale. -/
theorem scoresRealOfInputs_bounds_of_lnBounds
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (lnLo lnHi : Fin seq → Fin dModel → Rat)
    (hln : ∀ q i,
      (lnLo q i : Real) ≤ lnRealOfInputs inputs q i ∧
        lnRealOfInputs inputs q i ≤ (lnHi q i : Real))
    (hscale : 0 ≤ inputs.scale) :
    let qLo : Fin seq → Fin dHead → Rat := fun q d =>
      dotIntervalLower (fun j => inputs.wq j d) (lnLo q) (lnHi q) + inputs.bq d
    let qHi : Fin seq → Fin dHead → Rat := fun q d =>
      dotIntervalUpper (fun j => inputs.wq j d) (lnLo q) (lnHi q) + inputs.bq d
    let kLo : Fin seq → Fin dHead → Rat := fun k d =>
      dotIntervalLower (fun j => inputs.wk j d) (lnLo k) (lnHi k) + inputs.bk d
    let kHi : Fin seq → Fin dHead → Rat := fun k d =>
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
  intro qLo qHi kLo kHi baseLo baseHi scoreLo scoreHi q k
  have hscale_real : 0 ≤ (inputs.scale : Real) := by
    simpa [ratToReal_def] using ratToReal_nonneg_of_nonneg hscale
  have hq_bounds : ∀ q d,
      (qLo q d : Real) ≤ qRealOfInputs inputs q d ∧
        qRealOfInputs inputs q d ≤ (qHi q d : Real) := by
    intro q d
    have hlow :=
      dotIntervalLower_le_dotProduct_real_add
        (v := fun j => inputs.wq j d) (lo := lnLo q) (hi := lnHi q)
        (x := fun j => lnRealOfInputs inputs q j) (b := (inputs.bq d : Real))
        (hlo := fun j => (hln q j).1)
        (hhi := fun j => (hln q j).2)
    have hhigh :=
      dotProduct_le_dotIntervalUpper_real_add
        (v := fun j => inputs.wq j d) (lo := lnLo q) (hi := lnHi q)
        (x := fun j => lnRealOfInputs inputs q j) (b := (inputs.bq d : Real))
        (hlo := fun j => (hln q j).1)
        (hhi := fun j => (hln q j).2)
    constructor
    · simpa [qLo, qRealOfInputs_def] using hlow
    · simpa [qHi, qRealOfInputs_def] using hhigh
  have hk_bounds : ∀ k d,
      (kLo k d : Real) ≤ kRealOfInputs inputs k d ∧
        kRealOfInputs inputs k d ≤ (kHi k d : Real) := by
    intro k d
    have hlow :=
      dotIntervalLower_le_dotProduct_real_add
        (v := fun j => inputs.wk j d) (lo := lnLo k) (hi := lnHi k)
        (x := fun j => lnRealOfInputs inputs k j) (b := (inputs.bk d : Real))
        (hlo := fun j => (hln k j).1)
        (hhi := fun j => (hln k j).2)
    have hhigh :=
      dotProduct_le_dotIntervalUpper_real_add
        (v := fun j => inputs.wk j d) (lo := lnLo k) (hi := lnHi k)
        (x := fun j => lnRealOfInputs inputs k j) (b := (inputs.bk d : Real))
        (hlo := fun j => (hln k j).1)
        (hhi := fun j => (hln k j).2)
    constructor
    · simpa [kLo, kRealOfInputs_def] using hlow
    · simpa [kHi, kRealOfInputs_def] using hhigh
  have hdot_lo :
      (dotIntervalMulLower (qLo q) (qHi q) (kLo k) (kHi k) : Real) ≤
        dotProduct (fun d => qRealOfInputs inputs q d) (fun d => kRealOfInputs inputs k d) := by
    exact
      dotIntervalMulLower_le_dotProduct_real
        (lo₁ := qLo q) (hi₁ := qHi q) (lo₂ := kLo k) (hi₂ := kHi k)
        (x := fun d => qRealOfInputs inputs q d) (y := fun d => kRealOfInputs inputs k d)
        (hlo₁ := fun d => (hq_bounds q d).1)
        (hhi₁ := fun d => (hq_bounds q d).2)
        (hlo₂ := fun d => (hk_bounds k d).1)
        (hhi₂ := fun d => (hk_bounds k d).2)
  have hdot_hi :
      dotProduct (fun d => qRealOfInputs inputs q d) (fun d => kRealOfInputs inputs k d) ≤
        (dotIntervalMulUpper (qLo q) (qHi q) (kLo k) (kHi k) : Real) := by
    exact
      dotProduct_le_dotIntervalMulUpper_real
        (lo₁ := qLo q) (hi₁ := qHi q) (lo₂ := kLo k) (hi₂ := kHi k)
        (x := fun d => qRealOfInputs inputs q d) (y := fun d => kRealOfInputs inputs k d)
        (hlo₁ := fun d => (hq_bounds q d).1)
        (hhi₁ := fun d => (hq_bounds q d).2)
        (hlo₂ := fun d => (hk_bounds k d).1)
        (hhi₂ := fun d => (hk_bounds k d).2)
  have hbase_lo :
      (baseLo q k : Real) ≤
        (inputs.scale : Real) *
          dotProduct (fun d => qRealOfInputs inputs q d) (fun d => kRealOfInputs inputs k d) := by
    have hmul := mul_le_mul_of_nonneg_left hdot_lo hscale_real
    simpa [baseLo, ratToReal_mul, ratToReal_def] using hmul
  have hbase_hi :
      (inputs.scale : Real) *
          dotProduct (fun d => qRealOfInputs inputs q d) (fun d => kRealOfInputs inputs k d) ≤
        (baseHi q k : Real) := by
    have hmul := mul_le_mul_of_nonneg_left hdot_hi hscale_real
    simpa [baseHi, ratToReal_mul, ratToReal_def] using hmul
  by_cases hmask : inputs.maskCausal
  · by_cases hk : k ≤ q
    · have hbase : (baseLo q k : Real) ≤ scoresRealOfInputs inputs q k ∧
          scoresRealOfInputs inputs q k ≤ (baseHi q k : Real) := by
        have hscore :
            scoresRealOfInputs inputs q k =
              (inputs.scale : Real) *
                dotProduct (fun d => qRealOfInputs inputs q d)
                  (fun d => kRealOfInputs inputs k d) := by
          simp [scoresRealOfInputs_def, hmask, hk]
        constructor
        · simpa [hscore] using hbase_lo
        · simpa [hscore] using hbase_hi
      simpa [scoreLo, scoreHi, hmask, hk] using hbase
    · have hscore :
          scoresRealOfInputs inputs q k = (inputs.maskValue : Real) := by
        simp [scoresRealOfInputs_def, hmask, hk]
      have hscore' :
          (scoreLo q k : Real) ≤ scoresRealOfInputs inputs q k ∧
            scoresRealOfInputs inputs q k ≤ (scoreHi q k : Real) := by
        simp [scoreLo, scoreHi, hmask, hk, hscore]
      exact hscore'
  · have hbase : (baseLo q k : Real) ≤ scoresRealOfInputs inputs q k ∧
        scoresRealOfInputs inputs q k ≤ (baseHi q k : Real) := by
      have hscore :
          scoresRealOfInputs inputs q k =
            (inputs.scale : Real) *
              dotProduct (fun d => qRealOfInputs inputs q d)
                (fun d => kRealOfInputs inputs k d) := by
        simp [scoresRealOfInputs_def, hmask]
      constructor
      · simpa [hscore] using hbase_lo
      · simpa [hscore] using hbase_hi
    simpa [scoreLo, scoreHi, hmask] using hbase

end

end Sound

end Nfp
