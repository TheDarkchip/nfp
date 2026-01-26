-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Sound.Induction.CoreDefs
public import Nfp.Bounds.Interval

/-!
Head-output definitions for induction heads.
-/

public section

namespace Nfp

namespace Sound

open Nfp.Circuit
open Nfp.Bounds

variable {seq dModel dHead : Nat}

noncomputable section

/-- Real-valued head output using explicit score inputs (includes attention output bias). -/
def headOutputWithScores (scores : Fin seq → Fin seq → Real)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) : Real :=
  let weights : Fin seq → Fin seq → Real :=
    weightsRealOfInputsWithScores scores inputs
  let vals : Fin seq → Real := fun k => headValueRealOfInputs inputs k i
  dotProduct (weights q) vals

/-- Unfolding lemma for `headOutputWithScores`. -/
theorem headOutputWithScores_def (scores : Fin seq → Fin seq → Real)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) :
    headOutputWithScores scores inputs q i =
      let weights : Fin seq → Fin seq → Real :=
        weightsRealOfInputsWithScores scores inputs
      let vals : Fin seq → Real := fun k => headValueRealOfInputs inputs k i
      dotProduct (weights q) vals := by
  simp [headOutputWithScores]

/-- Real-valued head output for a query and model dimension (includes attention output bias). -/
def headOutput (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) : Real :=
  headOutputWithScores (scoresRealOfInputs inputs) inputs q i

/-- Unfolding lemma for `headOutput`. -/
theorem headOutput_def (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) :
    headOutput inputs q i =
      headOutputWithScores (scoresRealOfInputs inputs) inputs q i := by
  simp [headOutput]

/-!
Weighted head-output bounds from explicit per-key bounds.
-/

/-- Real-valued head output using explicit weight entries (Rat-valued weights). -/
def headOutputWithWeights (weights : Fin seq → Fin seq → Rat)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) : Real :=
  dotProduct (fun k => (weights q k : Real)) (fun k => headValueRealOfInputs inputs k i)

/-- Unfolding lemma for `headOutputWithWeights`. -/
theorem headOutputWithWeights_def (weights : Fin seq → Fin seq → Rat)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) :
    headOutputWithWeights weights inputs q i =
      dotProduct (fun k => (weights q k : Real)) (fun k => headValueRealOfInputs inputs k i) := by
  simp [headOutputWithWeights]

/-- Head outputs are bounded by per-key head-value intervals. -/
theorem headOutputWithWeights_bounds
    (weights : Fin seq → Fin seq → Rat)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (valsLo valsHi : Fin seq → Fin dModel → Rat)
    (hvals : ∀ k i,
      (valsLo k i : Real) ≤ headValueRealOfInputs inputs k i ∧
        headValueRealOfInputs inputs k i ≤ (valsHi k i : Real)) :
    let outLo : Fin seq → Fin dModel → Rat := fun q i =>
      dotIntervalLower (fun k => weights q k) (fun k => valsLo k i) (fun k => valsHi k i)
    let outHi : Fin seq → Fin dModel → Rat := fun q i =>
      dotIntervalUpper (fun k => weights q k) (fun k => valsLo k i) (fun k => valsHi k i)
    ∀ q i,
      (outLo q i : Real) ≤ headOutputWithWeights weights inputs q i ∧
        headOutputWithWeights weights inputs q i ≤ (outHi q i : Real) := by
  classical
  intro outLo outHi q i
  have hlow :=
    dotIntervalLower_le_dotProduct_real
      (v := fun k => weights q k)
      (lo := fun k => valsLo k i)
      (hi := fun k => valsHi k i)
      (x := fun k => headValueRealOfInputs inputs k i)
      (hlo := fun k => (hvals k i).1)
      (hhi := fun k => (hvals k i).2)
  have hhigh :=
    dotProduct_le_dotIntervalUpper_real
      (v := fun k => weights q k)
      (lo := fun k => valsLo k i)
      (hi := fun k => valsHi k i)
      (x := fun k => headValueRealOfInputs inputs k i)
      (hlo := fun k => (hvals k i).1)
      (hhi := fun k => (hvals k i).2)
  constructor
  · simpa [outLo, headOutputWithWeights] using hlow
  · simpa [outHi, headOutputWithWeights] using hhigh

end

end Sound

end Nfp
