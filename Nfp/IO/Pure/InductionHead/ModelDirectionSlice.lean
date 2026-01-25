-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.InductionHead.ModelDirectionSlice

/-!
Pure validation for unembedding-based direction inputs.

This module is in the trusted IO.Pure boundary. It validates raw arrays and
produces a checked direction slice with explicit size/positivity proofs.
-/

public section

namespace Nfp

namespace IO

namespace Pure

namespace InductionHeadCert

/-- Raw direction-slice arrays from parsing. -/
structure ModelDirectionSliceRaw where
  /-- Hidden dimension. -/
  dModel : Nat
  /-- Target token index (metadata). -/
  target : Nat
  /-- Negative token index (metadata). -/
  negative : Nat
  /-- Unembedding row for the target token. -/
  unembedTarget : Array Rat
  /-- Unembedding row for the negative token. -/
  unembedNegative : Array Rat

/-- Checked direction slice with size/positivity proofs. -/
structure ModelDirectionSliceChecked where
  /-- Raw inputs. -/
  raw : ModelDirectionSliceRaw
  /-- Positive hidden dimension. -/
  dModelPos : 0 < raw.dModel
  /-- Target row length matches hidden dimension. -/
  targetLen : raw.unembedTarget.size = raw.dModel
  /-- Negative row length matches hidden dimension. -/
  negativeLen : raw.unembedNegative.size = raw.dModel

/-- Extract the checked direction slice. -/
def ModelDirectionSliceChecked.toSlice (checked : ModelDirectionSliceChecked) :
    Nfp.IO.InductionHeadCert.ModelDirectionSlice :=
  let raw := checked.raw
  let targetFun : Fin raw.dModel → Rat := fun i =>
    let hi : i.1 < raw.unembedTarget.size :=
      Nat.lt_of_lt_of_eq i.isLt checked.targetLen.symm
    raw.unembedTarget[i.1]'hi
  let negativeFun : Fin raw.dModel → Rat := fun i =>
    let hi : i.1 < raw.unembedNegative.size :=
      Nat.lt_of_lt_of_eq i.isLt checked.negativeLen.symm
    raw.unembedNegative[i.1]'hi
  let directionFun : Fin raw.dModel → Rat := fun i =>
    targetFun i - negativeFun i
  { dModel := raw.dModel
    target := raw.target
    negative := raw.negative
    unembedTarget := targetFun
    unembedNegative := negativeFun
    direction := directionFun }

/-- Unfolding lemma for `ModelDirectionSliceChecked.toSlice`. -/
theorem ModelDirectionSliceChecked.toSlice_def (checked : ModelDirectionSliceChecked) :
    checked.toSlice = checked.toSlice := by
  rfl

/-- Validate raw direction-slice arrays. -/
def checkModelDirectionSliceRaw (raw : ModelDirectionSliceRaw) :
    Except String ModelDirectionSliceChecked := do
  if hModel : raw.dModel = 0 then
    throw "model-d-model must be positive"
  else
    let hModelPos : 0 < raw.dModel := Nat.pos_of_ne_zero hModel
    if hTargetLen : raw.unembedTarget.size = raw.dModel then
      if hNegLen : raw.unembedNegative.size = raw.dModel then
        pure
          { raw := raw
            dModelPos := hModelPos
            targetLen := hTargetLen
            negativeLen := hNegLen }
      else
        throw "model-unembed-negative has unexpected length"
    else
      throw "model-unembed-target has unexpected length"

end InductionHeadCert

end Pure

end IO

end Nfp
