-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.InductionHead.ModelSlice

/-!
Pure validation for model-slice inputs.

This module is in the trusted IO.Pure boundary. It validates raw arrays and
produces a checked model slice with explicit size/positivity proofs.
-/

public section

namespace Nfp

namespace IO

namespace Pure

namespace InductionHeadCert

/-- Raw model-slice arrays from parsing. -/
structure ModelSliceRaw (seq : Nat) where
  /-- Hidden dimension. -/
  dModel : Nat
  /-- Head dimension. -/
  headDim : Nat
  /-- Layer index (metadata). -/
  layer : Nat
  /-- Head index (metadata). -/
  head : Nat
  /-- Scale factor for attention scores. -/
  scoreScale : Rat
  /-- Mask value for causal attention (k > q). -/
  scoreMask : Rat
  /-- Whether to apply a causal mask to attention scores. -/
  maskCausal : Bool
  /-- Optional decimal precision for fast integer score checks. -/
  decimals? : Option Nat
  /-- Residual inputs (post-layernorm). -/
  resid : Array (Array Rat)
  /-- Q projection slice (d_model × head_dim). -/
  wq : Array (Array Rat)
  /-- K projection slice (d_model × head_dim). -/
  wk : Array (Array Rat)
  /-- Q bias slice (head_dim). -/
  bq : Array Rat
  /-- K bias slice (head_dim). -/
  bk : Array Rat

/-- Checked model-slice with size/positivity proofs. -/
structure ModelSliceChecked (seq : Nat) where
  /-- Raw inputs. -/
  raw : ModelSliceRaw seq
  /-- Positive hidden dimension. -/
  dModelPos : 0 < raw.dModel
  /-- Positive head dimension. -/
  headDimPos : 0 < raw.headDim
  /-- Residual row count matches sequence length. -/
  residRows : raw.resid.size = seq
  /-- Residual column count matches hidden dimension. -/
  residCols : raw.resid.all (fun row => row.size = raw.dModel) = true
  /-- Wq row count matches hidden dimension. -/
  wqRows : raw.wq.size = raw.dModel
  /-- Wq column count matches head dimension. -/
  wqCols : raw.wq.all (fun row => row.size = raw.headDim) = true
  /-- Wk row count matches hidden dimension. -/
  wkRows : raw.wk.size = raw.dModel
  /-- Wk column count matches head dimension. -/
  wkCols : raw.wk.all (fun row => row.size = raw.headDim) = true
  /-- Bq length matches head dimension. -/
  bqLen : raw.bq.size = raw.headDim
  /-- Bk length matches head dimension. -/
  bkLen : raw.bk.size = raw.headDim

/-- Extract the checked model slice. -/
def ModelSliceChecked.toSlice {seq : Nat} (checked : ModelSliceChecked seq) :
    Nfp.IO.InductionHeadCert.ModelSlice seq :=
  let raw := checked.raw
  let residFun : Fin seq → Fin raw.dModel → Rat := fun q i =>
    let row := (raw.resid[q.1]? ).getD #[]
    (row[i.1]? ).getD 0
  let wqFun : Fin raw.dModel → Fin raw.headDim → Rat := fun i j =>
    let row := (raw.wq[i.1]? ).getD #[]
    (row[j.1]? ).getD 0
  let wkFun : Fin raw.dModel → Fin raw.headDim → Rat := fun i j =>
    let row := (raw.wk[i.1]? ).getD #[]
    (row[j.1]? ).getD 0
  let bqFun : Fin raw.headDim → Rat := fun j =>
    (raw.bq[j.1]? ).getD 0
  let bkFun : Fin raw.headDim → Rat := fun j =>
    (raw.bk[j.1]? ).getD 0
  { dModel := raw.dModel
    headDim := raw.headDim
    layer := raw.layer
    head := raw.head
    scoreScale := raw.scoreScale
    scoreMask := raw.scoreMask
    maskCausal := raw.maskCausal
    decimals? := raw.decimals?
    resid := residFun
    wq := wqFun
    wk := wkFun
    bq := bqFun
    bk := bkFun }

/-- Unfolding lemma for `ModelSliceChecked.toSlice`. -/
theorem ModelSliceChecked.toSlice_def {seq : Nat} (checked : ModelSliceChecked seq) :
    checked.toSlice = checked.toSlice := by
  rfl

/-- Validate raw model-slice arrays. -/
def checkModelSliceRaw {seq : Nat} (raw : ModelSliceRaw seq) :
    Except String (ModelSliceChecked seq) := do
  if hModel : raw.dModel = 0 then
    throw "model-d-model must be positive"
  else
    let hModelPos : 0 < raw.dModel := Nat.pos_of_ne_zero hModel
    if hHead : raw.headDim = 0 then
      throw "model-head-dim must be positive"
    else
      let hHeadPos : 0 < raw.headDim := Nat.pos_of_ne_zero hHead
      if hResidRows : raw.resid.size = seq then
        if hResidCols : raw.resid.all (fun row => row.size = raw.dModel) = true then
          if hWqRows : raw.wq.size = raw.dModel then
            if hWqCols : raw.wq.all (fun row => row.size = raw.headDim) = true then
              if hWkRows : raw.wk.size = raw.dModel then
                if hWkCols : raw.wk.all (fun row => row.size = raw.headDim) = true then
                  if hBqLen : raw.bq.size = raw.headDim then
                    if hBkLen : raw.bk.size = raw.headDim then
                      pure
                        { raw := raw
                          dModelPos := hModelPos
                          headDimPos := hHeadPos
                          residRows := hResidRows
                          residCols := hResidCols
                          wqRows := hWqRows
                          wqCols := hWqCols
                          wkRows := hWkRows
                          wkCols := hWkCols
                          bqLen := hBqLen
                          bkLen := hBkLen }
                    else
                      throw "model-bk has unexpected length"
                  else
                    throw "model-bq/bk has unexpected length"
                else
                  throw "model-wk has unexpected column count"
              else
                throw "model-wq/wk has unexpected row count"
            else
              throw "model-wq has unexpected column count"
          else
            throw "model-wq/wk has unexpected row count"
        else
          throw "model-resid has unexpected column count"
      else
        throw "model-resid has unexpected row count"

end InductionHeadCert

end Pure

end IO

end Nfp
