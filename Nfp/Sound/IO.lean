import Std
import Nfp.Sound.Cert

namespace Nfp.Sound

open IO

/-!
# Sound `.nfpt` loader (exact Rat parsing)

This is a minimal, *sound* loader intended for certification.

It does **not** construct the full `ConcreteModel` (Float-based). Instead it parses only the
weights needed for conservative amplification constants `Cᵢ`, using exact `Rat` arithmetic.

Trusted base:
- Parsing from text to `Rat` via `Nfp.Sound.parseRat`.
- Exact accumulation of row-sum norms and max-abs values.

No `Float` arithmetic is used as an input to certification.
-/

/-- Parse `key=value` header lines. -/
def parseHeaderLine (line : String) : Option (String × String) :=
  let line := line.trim
  if line.isEmpty then none
  else
    match line.splitOn "=" with
    | [k, v] => some (k.trim, v.trim)
    | _ => none

/-- Read `count` rationals from lines starting at `start`, folding into `state`.

Returns `(state, nextLineIndex)`.
-/
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

/-- Consume a matrix in row-major order and return its exact `‖·‖∞` row-sum norm.

Returns `(normInf, nextLineIndex)`.
-/
def consumeMatrixNormInf
    (lines : Array String)
    (start : Nat)
    (rows cols : Nat) : Except String (Rat × Nat) :=
  let count := rows * cols
  let init : RowSumAcc := { rows := rows, cols := cols }
  let step := fun (acc : RowSumAcc) (x : Rat) => acc.feed x
  match foldRatTokens lines start count init step with
  | .error e => .error e
  | .ok (acc, next) => .ok (acc.finish, next)

/-- Consume a vector of length `n` and return `max |xᵢ|`.

Returns `(maxAbs, nextLineIndex)`.
-/
def consumeMaxAbs
    (lines : Array String)
    (start : Nat)
    (n : Nat) : Except String (Rat × Nat) :=
  let step := fun (m : Rat) (x : Rat) => max m (ratAbs x)
  foldRatTokens lines start n 0 step

/-- Soundly compute conservative per-layer amplification constants from a `.nfpt` file. -/
def certifyModelFile
    (path : System.FilePath)
    (eps : Rat := defaultEps)
    (actDerivBound : Rat := defaultActDerivBound) : IO (Except String ModelCert) := do
  let contents ← IO.FS.readFile path
  let lines : Array String := (contents.splitOn "\n").toArray

  -- Header
  let mut i : Nat := 0
  while i < lines.size && lines[i]!.trim.isEmpty do
    i := i + 1

  if !(i < lines.size) then
    return .error "empty file"

  let headerTag := lines[i]!.trim
  if !headerTag.startsWith "NFP_TEXT" then
    return .error s!"unexpected header '{headerTag}'"
  i := i + 1

  let mut numLayers : Option Nat := none
  let mut numHeads : Option Nat := none
  let mut modelDim : Option Nat := none
  let mut headDim : Option Nat := none
  let mut hiddenDim : Option Nat := none

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
      | _ => pure ()
      i := i + 1

  let some L := numLayers | return .error "missing num_layers"
  let some _ := numHeads | return .error "missing num_heads"
  let some d := modelDim | return .error "missing model_dim"
  let some dh := headDim | return .error "missing head_dim"
  let some dhid := hiddenDim | return .error "missing hidden_dim"

  -- Accumulators
  let mut ln1GammaMax : Array Rat := Array.replicate L 1
  let mut ln2GammaMax : Array Rat := Array.replicate L 1
  let mut attnSum : Array Rat := Array.replicate L 0
  let mut mlpWin : Array Rat := Array.replicate L 0
  let mut mlpWout : Array Rat := Array.replicate L 0

  let mut curLayer : Nat := 0

  -- Scan remaining sections
  while i < lines.size do
    let line := lines[i]!.trim

    if line.startsWith "LAYER" then
      let parts := line.splitOn " " |>.filter (· ≠ "")
      if parts.length >= 2 then
        curLayer := (parts[1]!).toNat? |>.getD 0
      i := i + 1
    else if line = "W_V" then
      -- Expect: modelDim × headDim
      let r := curLayer
      match consumeMatrixNormInf lines (i + 1) d dh with
      | .error e => return .error e
      | .ok (nv, next) =>
        -- Find W_O next by scanning forward (format guarantee: W_O follows eventually)
        i := next
        while i < lines.size && lines[i]!.trim ≠ "W_O" do
          i := i + 1
        if !(i < lines.size) then
          return .error "expected W_O after W_V"
        match consumeMatrixNormInf lines (i + 1) dh d with
        | .error e => return .error e
        | .ok (no, next2) =>
          if r < attnSum.size then
            attnSum := attnSum.set! r (attnSum[r]! + (nv * no))
          i := next2
    else if line = "W_in" then
      match consumeMatrixNormInf lines (i + 1) d dhid with
      | .error e => return .error e
      | .ok (n, next) =>
        if curLayer < mlpWin.size then
          mlpWin := mlpWin.set! curLayer n
        i := next
    else if line = "W_out" then
      match consumeMatrixNormInf lines (i + 1) dhid d with
      | .error e => return .error e
      | .ok (n, next) =>
        if curLayer < mlpWout.size then
          mlpWout := mlpWout.set! curLayer n
        i := next
    else if line = "LN1_GAMMA" then
      match consumeMaxAbs lines (i + 1) d with
      | .error e => return .error e
      | .ok (m, next) =>
        if curLayer < ln1GammaMax.size then
          ln1GammaMax := ln1GammaMax.set! curLayer m
        i := next
    else if line = "LN2_GAMMA" then
      match consumeMaxAbs lines (i + 1) d with
      | .error e => return .error e
      | .ok (m, next) =>
        if curLayer < ln2GammaMax.size then
          ln2GammaMax := ln2GammaMax.set! curLayer m
        i := next
    else
      -- default: advance
      i := i + 1

  -- Build layer reports
  let mut layers : Array LayerAmplificationCert := Array.mkEmpty L
  let mut totalAmp : Rat := 1
  for l in [:L] do
    let ln1Max := ln1GammaMax[l]!
    let ln2Max := ln2GammaMax[l]!
    let ln1Bound := layerNormOpBoundConservative ln1Max eps
    let ln2Bound := layerNormOpBoundConservative ln2Max eps
    let attnW := ln1Bound * softmaxJacobianNormInfWorst * attnSum[l]!
    let mlpW := ln2Bound * (mlpWin[l]! * actDerivBound * mlpWout[l]!)
    let C := attnW + mlpW
    layers := layers.push {
      layerIdx := l
      ln1MaxAbsGamma := ln1Max
      ln2MaxAbsGamma := ln2Max
      ln1Bound := ln1Bound
      ln2Bound := ln2Bound
      attnWeightContribution := attnW
      mlpWeightContribution := mlpW
      C := C
    }
    totalAmp := totalAmp * (1 + C)

  return .ok {
    modelPath := path.toString
    eps := eps
    actDerivBound := actDerivBound
    softmaxJacobianNormInfWorst := softmaxJacobianNormInfWorst
    layers := layers
    totalAmplificationFactor := totalAmp
  }

end Nfp.Sound
