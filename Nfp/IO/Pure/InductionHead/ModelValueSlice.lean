-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.InductionHead.ModelValueSlice

/-!
Pure validation for value-path model-slice inputs.

This module is in the trusted IO.Pure boundary. It validates raw arrays and
produces a checked value slice with explicit size/positivity proofs.
-/-

public section

namespace Nfp

namespace IO

namespace Pure

namespace InductionHeadCert

/-- Raw value-slice arrays from parsing. -/
structure ModelValueSliceRaw where
  /-- Hidden dimension. -/
  dModel : Nat
  /-- Head dimension. -/
  headDim : Nat
  /-- Layer index (metadata). -/
  layer : Nat
  /-- Head index (metadata). -/
  head : Nat
  /-- Value projection slice (d_model × head_dim). -/
  wv : Array (Array Rat)
  /-- Value projection bias (head_dim). -/
  bv : Array Rat
  /-- Output projection slice (d_model × head_dim). -/
  wo : Array (Array Rat)
  /-- Attention output bias (d_model). -/
  attnBias : Array Rat

/-- Checked value slice with size/positivity proofs. -/
structure ModelValueSliceChecked where
  /-- Raw inputs. -/
  raw : ModelValueSliceRaw
  /-- Positive hidden dimension. -/
  dModelPos : 0 < raw.dModel
  /-- Positive head dimension. -/
  headDimPos : 0 < raw.headDim
  /-- Wv row count matches hidden dimension. -/
  wvRows : raw.wv.size = raw.dModel
  /-- Wv column count matches head dimension. -/
  wvCols : raw.wv.all (fun row => row.size = raw.headDim) = true
  /-- Wo row count matches hidden dimension. -/
  woRows : raw.wo.size = raw.dModel
  /-- Wo column count matches head dimension. -/
  woCols : raw.wo.all (fun row => row.size = raw.headDim) = true
  /-- Bv length matches head dimension. -/
  bvLen : raw.bv.size = raw.headDim
  /-- Attn-bias length matches hidden dimension. -/
  attnBiasLen : raw.attnBias.size = raw.dModel

/-- Extract the checked value slice. -/
def ModelValueSliceChecked.toSlice (checked : ModelValueSliceChecked) :
    Nfp.IO.InductionHeadCert.ModelValueSlice :=
  let raw := checked.raw
  let wvFun : Fin raw.dModel → Fin raw.headDim → Rat := fun i j =>
    let row := (raw.wv[i.1]? ).getD #[]
    (row[j.1]? ).getD 0
  let woFun : Fin raw.dModel → Fin raw.headDim → Rat := fun i j =>
    let row := (raw.wo[i.1]? ).getD #[]
    (row[j.1]? ).getD 0
  let bvFun : Fin raw.headDim → Rat := fun j =>
    (raw.bv[j.1]? ).getD 0
  let attnBiasFun : Fin raw.dModel → Rat := fun i =>
    (raw.attnBias[i.1]? ).getD 0
  { dModel := raw.dModel
    headDim := raw.headDim
    layer := raw.layer
    head := raw.head
    wv := wvFun
    bv := bvFun
    wo := woFun
    attnBias := attnBiasFun }

/-- Unfolding lemma for `ModelValueSliceChecked.toSlice`. -/
theorem ModelValueSliceChecked.toSlice_def (checked : ModelValueSliceChecked) :
    checked.toSlice = checked.toSlice := by
  rfl

/-- Validate raw value-slice arrays. -/
def checkModelValueSliceRaw (raw : ModelValueSliceRaw) :
    Except String ModelValueSliceChecked := do
  if hModel : raw.dModel = 0 then
    throw "model-d-model must be positive"
  else
    let hModelPos : 0 < raw.dModel := Nat.pos_of_ne_zero hModel
    if hHead : raw.headDim = 0 then
      throw "model-head-dim must be positive"
    else
      let hHeadPos : 0 < raw.headDim := Nat.pos_of_ne_zero hHead
      if hWvRows : raw.wv.size = raw.dModel then
        if hWvCols : raw.wv.all (fun row => row.size = raw.headDim) = true then
          if hWoRows : raw.wo.size = raw.dModel then
            if hWoCols : raw.wo.all (fun row => row.size = raw.headDim) = true then
              if hBvLen : raw.bv.size = raw.headDim then
                if hAttnBiasLen : raw.attnBias.size = raw.dModel then
                  pure
                    { raw := raw
                      dModelPos := hModelPos
                      headDimPos := hHeadPos
                      wvRows := hWvRows
                      wvCols := hWvCols
                      woRows := hWoRows
                      woCols := hWoCols
                      bvLen := hBvLen
                      attnBiasLen := hAttnBiasLen }
                else
                  throw "model-attn-bias has unexpected length"
              else
                throw "model-bv has unexpected length"
            else
              throw "model-wo has unexpected column count"
          else
            throw "model-wo has unexpected row count"
        else
          throw "model-wv has unexpected column count"
      else
        throw "model-wv has unexpected row count"

end InductionHeadCert

end Pure

end IO

end Nfp
