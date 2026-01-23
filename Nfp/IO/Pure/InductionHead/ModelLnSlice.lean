-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.InductionHead.ModelLnSlice

/-!
Pure validation for LayerNorm model-slice inputs.

This module is in the trusted IO.Pure boundary. It validates raw arrays and
produces a checked LayerNorm slice with explicit size/positivity proofs.
-/

public section

namespace Nfp

namespace IO

namespace Pure

namespace InductionHeadCert

/-- Raw LayerNorm slice arrays from parsing. -/
structure ModelLnSliceRaw (seq : Nat) where
  /-- Hidden dimension. -/
  dModel : Nat
  /-- LayerNorm epsilon. -/
  lnEps : Rat
  /-- Nonnegative slack for LayerNorm bounds. -/
  lnSlack : Rat
  /-- Optional LayerNorm sqrt bound scale (positive when present). -/
  lnScale? : Option Nat
  /-- Optional fast-path toggle for fixed-denominator checks. -/
  lnFast? : Option Bool
  /-- Optional decimal precision for fixed-denominator checks. -/
  lnDecimals? : Option Nat
  /-- LayerNorm gamma. -/
  lnGamma : Array Rat
  /-- LayerNorm beta. -/
  lnBeta : Array Rat
  /-- Pre-LN embeddings/residual stream. -/
  embed : Array (Array Rat)

/-- Checked LayerNorm slice with size/positivity proofs. -/
structure ModelLnSliceChecked (seq : Nat) where
  /-- Raw inputs. -/
  raw : ModelLnSliceRaw seq
  /-- Positive hidden dimension. -/
  dModelPos : 0 < raw.dModel
  /-- Positive LayerNorm epsilon. -/
  lnEpsPos : 0 < raw.lnEps
  /-- Nonnegative LayerNorm slack. -/
  lnSlackNonneg : 0 ≤ raw.lnSlack
  /-- Embedding row count matches sequence length. -/
  embedRows : raw.embed.size = seq
  /-- Embedding column count matches hidden dimension. -/
  embedCols : raw.embed.all (fun row => row.size = raw.dModel) = true
  /-- Gamma length matches hidden dimension. -/
  gammaLen : raw.lnGamma.size = raw.dModel
  /-- Beta length matches hidden dimension. -/
  betaLen : raw.lnBeta.size = raw.dModel

/-- Extract the checked LayerNorm slice. -/
def ModelLnSliceChecked.toSlice {seq : Nat} (checked : ModelLnSliceChecked seq) :
    Nfp.IO.InductionHeadCert.ModelLnSlice seq :=
  let raw := checked.raw
  let embedFun : Fin seq → Fin raw.dModel → Rat := fun q i =>
    let row := (raw.embed[q.1]? ).getD #[]
    (row[i.1]? ).getD 0
  let gammaFun : Fin raw.dModel → Rat := fun i =>
    (raw.lnGamma[i.1]? ).getD 0
  let betaFun : Fin raw.dModel → Rat := fun i =>
    (raw.lnBeta[i.1]? ).getD 0
  { dModel := raw.dModel
    lnEps := raw.lnEps
    lnSlack := raw.lnSlack
    lnScale? := raw.lnScale?
    lnFast? := raw.lnFast?
    lnDecimals? := raw.lnDecimals?
    lnGamma := gammaFun
    lnBeta := betaFun
    embed := embedFun }

/-- Unfolding lemma for `ModelLnSliceChecked.toSlice`. -/
theorem ModelLnSliceChecked.toSlice_def {seq : Nat} (checked : ModelLnSliceChecked seq) :
    checked.toSlice = checked.toSlice := by
  rfl

/-- Validate raw LayerNorm slice arrays. -/
def checkModelLnSliceRaw {seq : Nat} (raw : ModelLnSliceRaw seq) :
    Except String (ModelLnSliceChecked seq) := do
  if hModel : raw.dModel = 0 then
    throw "model-d-model must be positive"
  else
    let hModelPos : 0 < raw.dModel := Nat.pos_of_ne_zero hModel
    if hEps : 0 < raw.lnEps then
      let hEpsPos : 0 < raw.lnEps := hEps
      if hSlack : 0 ≤ raw.lnSlack then
        let hSlackNonneg : 0 ≤ raw.lnSlack := hSlack
        if let some scale := raw.lnScale? then
          if scale = 0 then
            throw "model-ln-scale must be positive"
          else
            pure ()
        if hEmbedRows : raw.embed.size = seq then
          if hEmbedCols : raw.embed.all (fun row => row.size = raw.dModel) = true then
            if hGammaLen : raw.lnGamma.size = raw.dModel then
              if hBetaLen : raw.lnBeta.size = raw.dModel then
                pure
                  { raw := raw
                    dModelPos := hModelPos
                    lnEpsPos := hEpsPos
                    lnSlackNonneg := hSlackNonneg
                    embedRows := hEmbedRows
                    embedCols := hEmbedCols
                    gammaLen := hGammaLen
                    betaLen := hBetaLen }
              else
                throw "model-ln-gamma/beta has unexpected length"
            else
              throw "model-ln-gamma/beta has unexpected length"
          else
            throw "model-embed has unexpected column count"
        else
          throw "model-embed has unexpected row count"
      else
        throw "model-ln-slack must be nonnegative"
    else
      throw "model-ln-eps must be positive"

end InductionHeadCert

end Pure

end IO

end Nfp
