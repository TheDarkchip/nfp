-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Sound.Induction.CoreDefs

/-!
Head-output definitions for induction heads.
-/

public section

namespace Nfp

namespace Sound

open Nfp.Circuit

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

end

end Sound

end Nfp
