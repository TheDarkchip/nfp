-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Cert.ResidualInterval
public import Nfp.Sound.Induction.CoreDefs

/-!
Head-output interval certificates for induction heads.
-/

public section

namespace Nfp

namespace Sound

open Nfp.Circuit

variable {seq dModel dHead : Nat}

noncomputable section

/-- Real-valued head output using explicit score inputs. -/
def headOutputWithScores (scores : Fin seq → Fin seq → Real)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) : Real :=
  let weights : Fin seq → Fin seq → Real := fun q k =>
    Circuit.softmax (scores q) k
  let vals : Fin seq → Real := fun k => headValueRealOfInputs inputs k i
  dotProduct (weights q) vals

/-- Unfolding lemma for `headOutputWithScores`. -/
theorem headOutputWithScores_def (scores : Fin seq → Fin seq → Real)
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) :
    headOutputWithScores scores inputs q i =
      let weights : Fin seq → Fin seq → Real := fun q k =>
        Circuit.softmax (scores q) k
      let vals : Fin seq → Real := fun k => headValueRealOfInputs inputs k i
      dotProduct (weights q) vals := by
  simp [headOutputWithScores]

/-- Real-valued head output for a query and model dimension. -/
def headOutput (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) : Real :=
  headOutputWithScores (scoresRealOfInputs inputs) inputs q i

/-- Unfolding lemma for `headOutput`. -/
theorem headOutput_def (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) (i : Fin dModel) :
    headOutput inputs q i =
      headOutputWithScores (scoresRealOfInputs inputs) inputs q i := by
  simp [headOutput]

/-- Soundness predicate for head-output interval bounds. -/
structure HeadOutputIntervalSound [NeZero seq]
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (active : Finset (Fin seq))
    (c : Circuit.ResidualIntervalCert dModel) : Prop where
  /-- Interval bounds are ordered coordinatewise. -/
  bounds : Circuit.ResidualIntervalBounds c
  /-- Active-query outputs lie inside the interval bounds. -/
  output_mem :
    ∀ q, q ∈ active → ∀ i,
      (c.lo i : Real) ≤ headOutput inputs q i ∧
        headOutput inputs q i ≤ (c.hi i : Real)

end

end Sound

end Nfp
