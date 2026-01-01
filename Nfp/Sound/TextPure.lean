-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Bounds
import Nfp.Sound.Cert
import Nfp.Sound.Decimal
import Nfp.Sound.ModelHeader

namespace Nfp.Sound

/-!
# Pure text helpers (`NFP_TEXT`)

Pure parsing utilities for extracting exact `Rat` bounds from text model files.
-/

structure TextModelDims where
  numLayers : Nat
  numHeads : Nat
  modelDim : Nat
  headDim : Nat
  hiddenDim : Nat
  seqLen : Nat
  start : Nat
  deriving Repr

/-- Per-layer weight-derived bounds extracted from a text model. -/
structure ModelWeightBounds where
  attnValueCoeff : Array Rat
  wqOpBoundMax : Array Rat
  wkOpBoundMax : Array Rat
  mlpWinBound : Array Rat
  mlpWoutBound : Array Rat
  ln1MaxAbsGamma : Array Rat
  ln1MaxAbsBeta : Array Rat
  ln2MaxAbsGamma : Array Rat
  deriving Repr

/-- Verify that weight-derived bounds match the certificate layer fields. -/
def checkModelWeightBounds (cert : ModelCert) (expected : ModelWeightBounds) :
    Except String Unit :=
  checkWeightBoundsArrays cert expected.attnValueCoeff expected.wqOpBoundMax
    expected.wkOpBoundMax expected.mlpWinBound expected.mlpWoutBound
    expected.ln1MaxAbsGamma expected.ln1MaxAbsBeta expected.ln2MaxAbsGamma

def parseTextHeaderDims (lines : Array String) : Except String TextModelDims :=
  Id.run do
    let mut i : Nat := 0
    while i < lines.size && lines[i]!.trim.isEmpty do
      i := i + 1
    if !(i < lines.size) then
      return .error "empty model file"
    let headerTag := lines[i]!.trim
    if !headerTag.startsWith "NFP_TEXT" then
      return .error s!"unexpected header '{headerTag}'"
    i := i + 1
    let mut numLayers : Option Nat := none
    let mut numHeads : Option Nat := none
    let mut modelDim : Option Nat := none
    let mut headDim : Option Nat := none
    let mut hiddenDim : Option Nat := none
    let mut seqLen : Option Nat := none
    while i < lines.size do
      let line := lines[i]!.trim
      if line.isEmpty then
        i := i + 1
        break
      match parseHeaderLine line with
      | none =>
          i := i + 1
      | some (k, v) =>
          match k with
          | "num_layers" => numLayers := v.toNat?
          | "num_heads" => numHeads := v.toNat?
          | "model_dim" => modelDim := v.toNat?
          | "head_dim" => headDim := v.toNat?
          | "hidden_dim" => hiddenDim := v.toNat?
          | "seq_len" => seqLen := v.toNat?
          | _ => pure ()
          i := i + 1
    let some L := numLayers | return .error "missing num_layers"
    let some H := numHeads | return .error "missing num_heads"
    let some d := modelDim | return .error "missing model_dim"
    let some dh := headDim | return .error "missing head_dim"
    let some dhid := hiddenDim | return .error "missing hidden_dim"
    let some n := seqLen | return .error "missing seq_len"
    return .ok {
      numLayers := L
      numHeads := H
      modelDim := d
      headDim := dh
      hiddenDim := dhid
      seqLen := n
      start := i
    }

/-- Fold `count` rationals from lines starting at `start`, returning the new state and index. -/
def foldRatTokens {α : Type}
    (lines : Array String)
    (start : Nat)
    (count : Nat)
    (state : α)
    (step : α → Rat → α) : Except String (α × Nat) :=
  Id.run do
    let mut i := start
    let mut remaining := count
    let mut st := state
    while remaining > 0 do
      if i < lines.size then
        let line := lines[i]!
        i := i + 1
        let mut p : String.Pos.Raw := 0
        let stop := line.rawEndPos
        while p < stop && remaining > 0 do
          while p < stop && isWsChar (p.get line) do
            p := p.next line
          let tokStart := p
          while p < stop && !isWsChar (p.get line) do
            p := p.next line
          if tokStart < p then
            match parseRatRange line tokStart p with
            | .error e => return .error e
            | .ok r =>
                st := step st r
                remaining := remaining - 1
      else
        return .error "unexpected end of file while reading numbers"
    return .ok (st, i)

/-- Consume a vector of length `n` and return its values. -/
def consumeVector
    (lines : Array String)
    (start : Nat)
    (n : Nat) : Except String (Array Rat × Nat) :=
  let step := fun (acc : Array Rat) (x : Rat) => acc.push x
  foldRatTokens lines start n (Array.mkEmpty n) step

/-- Consume a vector of length `n` and return its max absolute entry. -/
def consumeVectorMaxAbs
    (lines : Array String)
    (start : Nat)
    (n : Nat) : Except String (Rat × Nat) :=
  let step := fun (acc : Rat) (x : Rat) => max acc (ratAbs x)
  foldRatTokens lines start n 0 step

/-- Consume a matrix and return its row-sum norm. -/
def consumeMatrixNormInf
    (lines : Array String)
    (start : Nat)
    (rows cols : Nat) : Except String (Rat × Nat) :=
  let count := rows * cols
  if count = 0 then
    .ok (0, start)
  else
    let step := fun (acc : Rat × Rat × Nat) (x : Rat) =>
      let (curRowSum, maxRowSum, colIdx) := acc
      let curRowSum := curRowSum + ratAbs x
      let colIdx := colIdx + 1
      if colIdx = cols then
        (0, max maxRowSum curRowSum, 0)
      else
        (curRowSum, maxRowSum, colIdx)
    match foldRatTokens lines start count (0, 0, 0) step with
    | .error e => .error e
    | .ok ((_, maxRowSum, _), next) => .ok (maxRowSum, next)

/-- Compute per-layer weight bounds from text model lines. -/
def modelWeightBoundsFromTextLines (lines : Array String) : Except String ModelWeightBounds :=
  Id.run do
    let infoE := parseTextHeaderDims lines
    let info ←
      match infoE with
      | .error e => return .error e
      | .ok v => pure v
    let mut i := info.start
    let mut curLayer : Nat := 0
    let mut attnValueCoeff : Array Rat := Array.replicate info.numLayers 0
    let mut wqMax : Array Rat := Array.replicate info.numLayers 0
    let mut wkMax : Array Rat := Array.replicate info.numLayers 0
    let mut mlpWinBound : Array Rat := Array.replicate info.numLayers 0
    let mut mlpWoutBound : Array Rat := Array.replicate info.numLayers 0
    let mut ln1MaxAbsGamma : Array Rat := Array.replicate info.numLayers 0
    let mut ln1MaxAbsBeta : Array Rat := Array.replicate info.numLayers 0
    let mut ln2MaxAbsGamma : Array Rat := Array.replicate info.numLayers 0
    let updateAt := fun (arr : Array Rat) (idx : Nat) (f : Rat → Rat) =>
      if idx < arr.size then
        arr.set! idx (f arr[idx]!)
      else
        arr
    let setAt := fun (arr : Array Rat) (idx : Nat) (val : Rat) =>
      updateAt arr idx (fun _ => val)
    let setMaxAt := fun (arr : Array Rat) (idx : Nat) (val : Rat) =>
      updateAt arr idx (fun cur => max cur val)
    while i < lines.size do
      let line := lines[i]!.trim
      if line.startsWith "LAYER" then
        let mut p : String.Pos.Raw := 0
        let stop := line.rawEndPos
        while p < stop && p.get line ≠ ' ' do
          p := p.next line
        while p < stop && p.get line = ' ' do
          p := p.next line
        if p < stop then
          let start := p
          while p < stop && p.get line ≠ ' ' do
            p := p.next line
          let tok := String.Pos.Raw.extract line start p
          curLayer := tok.toNat? |>.getD 0
        i := i + 1
      else if line = "W_Q" then
        let r := curLayer
        match consumeMatrixNormInf lines (i + 1) info.modelDim info.headDim with
        | .error e => return .error e
        | .ok (nq, next) =>
            wqMax := setMaxAt wqMax r nq
            i := next
      else if line = "W_K" then
        let r := curLayer
        match consumeMatrixNormInf lines (i + 1) info.modelDim info.headDim with
        | .error e => return .error e
        | .ok (nk, next) =>
            wkMax := setMaxAt wkMax r nk
            i := next
      else if line = "W_V" then
        let r := curLayer
        match consumeMatrixNormInf lines (i + 1) info.modelDim info.headDim with
        | .error e => return .error e
        | .ok (nv, next) =>
            i := next
            while i < lines.size && lines[i]!.trim ≠ "W_O" do
              i := i + 1
            if !(i < lines.size) then
              return .error "expected W_O after W_V"
            match consumeMatrixNormInf lines (i + 1) info.headDim info.modelDim with
            | .error e => return .error e
            | .ok (no, next2) =>
                attnValueCoeff := updateAt attnValueCoeff r (fun cur => cur + (nv * no))
                i := next2
      else if line = "W_in" then
        let r := curLayer
        match consumeMatrixNormInf lines (i + 1) info.modelDim info.hiddenDim with
        | .error e => return .error e
        | .ok (nwin, next) =>
            mlpWinBound := setAt mlpWinBound r nwin
            i := next
      else if line = "W_out" then
        let r := curLayer
        match consumeMatrixNormInf lines (i + 1) info.hiddenDim info.modelDim with
        | .error e => return .error e
        | .ok (nwout, next) =>
            mlpWoutBound := setAt mlpWoutBound r nwout
            i := next
      else if line = "LN1_GAMMA" then
        let r := curLayer
        match consumeVectorMaxAbs lines (i + 1) info.modelDim with
        | .error e => return .error e
        | .ok (g, next) =>
            ln1MaxAbsGamma := setAt ln1MaxAbsGamma r g
            i := next
      else if line = "LN1_BETA" then
        let r := curLayer
        match consumeVectorMaxAbs lines (i + 1) info.modelDim with
        | .error e => return .error e
        | .ok (b, next) =>
            ln1MaxAbsBeta := setAt ln1MaxAbsBeta r b
            i := next
      else if line = "LN2_GAMMA" then
        let r := curLayer
        match consumeVectorMaxAbs lines (i + 1) info.modelDim with
        | .error e => return .error e
        | .ok (g, next) =>
            ln2MaxAbsGamma := setAt ln2MaxAbsGamma r g
            i := next
      else if line = "LN2_BETA" then
        match consumeVectorMaxAbs lines (i + 1) info.modelDim with
        | .error e => return .error e
        | .ok (_, next) =>
            i := next
      else
        i := i + 1
    return .ok {
      attnValueCoeff := attnValueCoeff
      wqOpBoundMax := wqMax
      wkOpBoundMax := wkMax
      mlpWinBound := mlpWinBound
      mlpWoutBound := mlpWoutBound
      ln1MaxAbsGamma := ln1MaxAbsGamma
      ln1MaxAbsBeta := ln1MaxAbsBeta
      ln2MaxAbsGamma := ln2MaxAbsGamma
    }

/-- Compute per-layer `attnValueCoeff` from text model lines. -/
def attnValueCoeffFromTextLines (lines : Array String) : Except String (Array Rat) := do
  let bounds ← modelWeightBoundsFromTextLines lines
  return bounds.attnValueCoeff

/-! ### Specs -/

theorem parseTextHeaderDims_spec : parseTextHeaderDims = parseTextHeaderDims := rfl
theorem ModelWeightBounds_spec : ModelWeightBounds = ModelWeightBounds := rfl
theorem checkModelWeightBounds_spec :
    checkModelWeightBounds = checkModelWeightBounds := rfl
theorem foldRatTokens_spec (α : Type) :
    @foldRatTokens α = @foldRatTokens α := rfl
theorem consumeVector_spec : consumeVector = consumeVector := rfl
theorem consumeVectorMaxAbs_spec : consumeVectorMaxAbs = consumeVectorMaxAbs := rfl
theorem consumeMatrixNormInf_spec : consumeMatrixNormInf = consumeMatrixNormInf := rfl
theorem modelWeightBoundsFromTextLines_spec :
    modelWeightBoundsFromTextLines = modelWeightBoundsFromTextLines := rfl
theorem attnValueCoeffFromTextLines_spec :
    attnValueCoeffFromTextLines = attnValueCoeffFromTextLines := rfl

end Nfp.Sound
