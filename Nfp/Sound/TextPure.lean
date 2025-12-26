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

/-- Per-layer attention weight bounds extracted from a text model. -/
structure AttnWeightBounds where
  attnValueCoeff : Array Rat
  wqOpBoundMax : Array Rat
  wkOpBoundMax : Array Rat
  deriving Repr

/-- Verify that attention-weight bounds match the certificate layer fields. -/
def checkAttnWeightBounds (cert : ModelCert) (expected : AttnWeightBounds) : Except String Unit :=
  checkAttnWeightBoundsArrays cert expected.attnValueCoeff expected.wqOpBoundMax
    expected.wkOpBoundMax

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
        let line := lines[i]!.trim
        i := i + 1
        if line.isEmpty then
          pure ()
        else
          let toks := line.splitOn " " |>.filter (· ≠ "")
          for t in toks do
            if remaining = 0 then
              break
            match parseRat t with
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

/-- Consume a matrix and return its row-sum norm. -/
def consumeMatrixNormInf
    (lines : Array String)
    (start : Nat)
    (rows cols : Nat) : Except String (Rat × Nat) :=
  let count := rows * cols
  match consumeVector lines start count with
  | .error e => .error e
  | .ok (xs, next) => .ok (matrixNormInfOfRowMajor rows cols xs, next)

/-- Compute per-layer attention value and `W_Q/W_K` bounds from text model lines. -/
def attnWeightBoundsFromTextLines (lines : Array String) : Except String AttnWeightBounds :=
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
    while i < lines.size do
      let line := lines[i]!.trim
      if line.startsWith "LAYER" then
        let parts := line.splitOn " " |>.filter (· ≠ "")
        if parts.length >= 2 then
          curLayer := (parts[1]!).toNat? |>.getD 0
        i := i + 1
      else if line = "W_Q" then
        let r := curLayer
        match consumeMatrixNormInf lines (i + 1) info.modelDim info.headDim with
        | .error e => return .error e
        | .ok (nq, next) =>
            if r < wqMax.size then
              wqMax := wqMax.set! r (max wqMax[r]! nq)
            i := next
      else if line = "W_K" then
        let r := curLayer
        match consumeMatrixNormInf lines (i + 1) info.modelDim info.headDim with
        | .error e => return .error e
        | .ok (nk, next) =>
            if r < wkMax.size then
              wkMax := wkMax.set! r (max wkMax[r]! nk)
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
                if r < attnValueCoeff.size then
                  attnValueCoeff :=
                    attnValueCoeff.set! r (attnValueCoeff[r]! + (nv * no))
                i := next2
      else
        i := i + 1
    return .ok {
      attnValueCoeff := attnValueCoeff
      wqOpBoundMax := wqMax
      wkOpBoundMax := wkMax
    }

/-- Compute per-layer `attnValueCoeff` from text model lines. -/
def attnValueCoeffFromTextLines (lines : Array String) : Except String (Array Rat) := do
  let bounds ← attnWeightBoundsFromTextLines lines
  return bounds.attnValueCoeff

/-! ### Specs -/

theorem parseTextHeaderDims_spec : parseTextHeaderDims = parseTextHeaderDims := rfl
theorem AttnWeightBounds_spec : AttnWeightBounds = AttnWeightBounds := rfl
theorem checkAttnWeightBounds_spec :
    checkAttnWeightBounds = checkAttnWeightBounds := rfl
theorem foldRatTokens_spec (α : Type) :
    @foldRatTokens α = @foldRatTokens α := rfl
theorem consumeVector_spec : consumeVector = consumeVector := rfl
theorem consumeMatrixNormInf_spec : consumeMatrixNormInf = consumeMatrixNormInf := rfl
theorem attnWeightBoundsFromTextLines_spec :
    attnWeightBoundsFromTextLines = attnWeightBoundsFromTextLines := rfl
theorem attnValueCoeffFromTextLines_spec :
    attnValueCoeffFromTextLines = attnValueCoeffFromTextLines := rfl

end Nfp.Sound
