-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Cert
import Nfp.Sound.Interval
import Nfp.Sound.Cache
import Nfp.Sound.Fixed

namespace Nfp.Sound

open IO

/-!
# Sound `.nfpt` loader (exact Rat parsing, legacy text format)

This is a minimal, *sound* loader intended for certification on the legacy text format.

It does **not** construct the full `ConcreteModel` (Float-based). Instead it parses only the
weights needed for conservative amplification constants `Cᵢ`, using exact `Rat` arithmetic.

It can optionally consume an input `.nfpt` file (for `EMBEDDINGS`) to enable **local**
LayerNorm certification on a bounded region around that input.

Current limitation: this parser only accepts the legacy `NFP_TEXT_V1/V2` format.
Binary (`NFP_BINARY_V1`) support is pending.

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

/-- Consume a vector of length `n` into an array. Returns `(xs, nextLineIndex)`. -/
def consumeVector
    (lines : Array String)
    (start : Nat)
    (n : Nat) : Except String (Array Rat × Nat) :=
  let step := fun (acc : Array Rat) (x : Rat) => acc.push x
  foldRatTokens lines start n (Array.mkEmpty n) step

/-- Consume a matrix in row-major order and return its exact `‖·‖∞` row-sum norm.

Returns `(normInf, nextLineIndex)`.
-/
def consumeMatrixNormInf
    (lines : Array String)
    (start : Nat)
    (rows cols : Nat) : Except String (Rat × Nat) :=
  let count := rows * cols
  match consumeVector lines start count with
  | .error e => .error e
  | .ok (xs, next) => .ok (matrixNormInfOfRowMajor rows cols xs, next)

/-- Parsed matrix norm is definitionally the spec-level bound on the parsed data. -/
theorem consumeMatrixNormInf_spec
    (lines : Array String) (start rows cols : Nat) (xs : Array Rat) (next : Nat)
    (h : consumeVector lines start (rows * cols) = .ok (xs, next)) :
    consumeMatrixNormInf lines start rows cols =
      .ok (matrixNormInfOfRowMajor rows cols xs, next) := by
  simp [consumeMatrixNormInf, h]

/-- Consume a vector of length `n` and return `max |xᵢ|`.

Returns `(maxAbs, nextLineIndex)`.
-/
def consumeMaxAbs
    (lines : Array String)
    (start : Nat)
    (n : Nat) : Except String (Rat × Nat) :=
  let step := fun (m : Rat) (x : Rat) => max m (ratAbs x)
  foldRatTokens lines start n 0 step

private def maxAbsOfVector (xs : Array Rat) : Rat :=
  xs.foldl (fun acc x => max acc (ratAbs x)) 0

private def addVecIntervals (a b : Array RatInterval) : Array RatInterval :=
  Id.run do
    if a.size ≠ b.size then
      return a
    let mut out : Array RatInterval := Array.mkEmpty a.size
    for i in [:a.size] do
      out := out.push (RatInterval.add a[i]! b[i]!)
    return out

private def addConstVec (a : Array RatInterval) (b : Array Rat) : Array RatInterval :=
  Id.run do
    if a.size ≠ b.size then
      return a
    let mut out : Array RatInterval := Array.mkEmpty a.size
    for i in [:a.size] do
      out := out.push (RatInterval.add a[i]! (RatInterval.const b[i]!))
    return out

private def unionVecIntervals (a b : Array RatInterval) : Array RatInterval :=
  Id.run do
    if a.size ≠ b.size then
      return a
    let mut out : Array RatInterval := Array.mkEmpty a.size
    for i in [:a.size] do
      out := out.push (RatInterval.union a[i]! b[i]!)
    return out

private def zeroIntervals (n : Nat) : Array RatInterval :=
  Array.replicate n (RatInterval.const 0)

private def unionRows (rows : Array (Array RatInterval)) (dim : Nat) : Array RatInterval :=
  Id.run do
    if rows.isEmpty then
      return zeroIntervals dim
    let mut out : Array RatInterval := zeroIntervals dim
    let r0 := rows[0]!
    if r0.size = dim then
      out := r0
    for r in rows do
      if r.size = dim then
        out := unionVecIntervals out r
    return out

private def layerNormRowApprox (row : Array RatInterval) (gamma beta : Array Rat) (eps : Rat) :
    (Array RatInterval × Rat) :=
  if row.size = 0 || gamma.size ≠ row.size || beta.size ≠ row.size then
    (row, 0)
  else
    Id.run do
      let μ := RatInterval.mean row
      let varLB := RatInterval.varianceLowerBound row
      let invσUpper : Rat :=
        if varLB ≤ 0 then
          -- Sound fallback for IBP propagation: `1/σ ≤ 1/eps` (conservative, but rigorous).
          layerNormOpBoundConservative 1 eps
        else
          layerNormOpBoundLocal 1 varLB eps
      let mut out : Array RatInterval := Array.mkEmpty row.size
      for i in [:row.size] do
        let centered := RatInterval.sub row[i]! μ
        let scaled := RatInterval.scale (gamma[i]! * invσUpper) centered
        out := out.push (RatInterval.add scaled (RatInterval.const beta[i]!))
      return (out, varLB)

private def minVarAcrossRows (rows : Array (Array RatInterval)) : Rat :=
  Id.run do
    let mut best : Option Rat := none
    for r in rows do
      let v := RatInterval.varianceLowerBound r
      best := some (match best with | none => v | some b => min b v)
    best.getD 0

private def findLineIdxFrom (lines : Array String) (start : Nat) (p : String → Bool) : Option Nat :=
  Id.run do
    let mut i := start
    while i < lines.size do
      if p (lines[i]!.trim) then
        return some i
      i := i + 1
    return none

private def skipUntil (lines : Array String) (start : Nat) (p : String → Bool) : Nat :=
  match findLineIdxFrom lines start p with
  | some i => i
  | none => lines.size

private def skipBlankLines (lines : Array String) (start : Nat) : Nat :=
  Id.run do
    let mut i := start
    while i < lines.size && lines[i]!.trim.isEmpty do
      i := i + 1
    return i

/-!
### Fast skipping without parsing

For local SOUND certification we do not need `W_Q`, `W_K`, `b_Q`, or `b_K` numerically
(they don't affect the Jacobian bounds we certify in this streaming-only pass).

Parsing decimals into `Rat` is expensive, so we skip these sections by **counting tokens**
instead of calling `parseRat`.
-/

private def countWsTokens (s : String) : Nat :=
  Id.run do
    let bytes := s.toUTF8
    let mut i : Nat := 0
    let mut inTok : Bool := false
    let mut cnt : Nat := 0
    while i < bytes.size do
      let b := bytes[i]!
      let isWs : Bool := b = 32 || b = 9  -- ' ' or '\t'
      if isWs then
        inTok := false
      else if !inTok then
        inTok := true
        cnt := cnt + 1
      i := i + 1
    return cnt

private def consumeTokensSkipFast
    (lines : Array String) (start : Nat) (numTokens : Nat) : Except String Nat :=
  Id.run do
    let mut iLine := start
    let mut remaining := numTokens
    while remaining > 0 do
      if iLine ≥ lines.size then
        return .error "unexpected end of file while skipping tokens"
      let line := lines[iLine]!.trim
      iLine := iLine + 1
      if line.isEmpty then
        pure ()
      else
        let c := countWsTokens line
        if c ≥ remaining then
          remaining := 0
        else
          remaining := remaining - c
    return .ok iLine

private def consumeMatrixSkip
    (lines : Array String)
    (start : Nat)
    (rows cols : Nat) : Except String Nat :=
  match foldRatTokens lines start (rows * cols) () (fun _ _ => ()) with
  | .error e => .error e
  | .ok (_, next) => .ok next

private def consumeMatrixSkipFast
    (lines : Array String)
    (start : Nat)
    (rows cols : Nat) : Except String Nat :=
  consumeTokensSkipFast lines start (rows * cols)

private def consumeVectorSkipFast
    (lines : Array String)
    (start : Nat)
    (n : Nat) : Except String Nat :=
  consumeTokensSkipFast lines start n

/-!
Streaming multiplication for row-major stored matrices.

The `.nfpt` format stores matrices row-major with `rows` = input dimension and `cols` = output
dimension in the repo's row-vector convention: `y = x · W` where `W : rows×cols`.

We compute `y` in a single pass over weights by accumulating contributions row-by-row:
for each input index `i`, parse the `i`-th row `w_{i,*}` and add `w_{i,j} * x[i]` into `y[j]`.
This never stores the matrix.
-/
private def consumeMatrixMulAndNormInf
    (lines : Array String)
    (start : Nat)
    (rows cols : Nat)
    (input : Array RatInterval) : Except String (Array RatInterval × Rat × Nat) :=
  Id.run do
    if input.size ≠ rows then
      return .error "input interval dimension mismatch"
    let mut out : Array RatInterval := zeroIntervals cols
    let mut iLine := start
    let mut remaining := rows * cols
    let mut idx : Nat := 0
    let mut curRowAbs : Rat := 0
    let mut maxRowAbs : Rat := 0

    while remaining > 0 do
      if iLine ≥ lines.size then
        return .error "unexpected end of file while reading matrix"
      let line := lines[iLine]!.trim
      iLine := iLine + 1
      if line.isEmpty then
        pure ()
      else
        let toks := line.splitOn " " |>.filter (· ≠ "")
        for t in toks do
          if remaining = 0 then
            break
          match parseRat t with
          | .error e => return .error e
          | .ok w =>
              let r := idx / cols
              let c := idx % cols
              curRowAbs := curRowAbs + ratAbs w
              -- out[c] += w * input[r]
              let term := RatInterval.scale w (input[r]!)
              out := out.set! c (RatInterval.add (out[c]!) term)
              idx := idx + 1
              remaining := remaining - 1
              if c + 1 = cols then
                maxRowAbs := max maxRowAbs curRowAbs
                curRowAbs := 0
    -- Account for a partial last row (should not happen if rows*cols consumed).
    maxRowAbs := max maxRowAbs curRowAbs
    return .ok (out, maxRowAbs, iLine)

/-- Soundly compute conservative per-layer amplification constants from a `.nfpt` file. -/
def certifyModelFileGlobal
    (path : System.FilePath)
    (eps : Rat := defaultEps)
    (actDerivBound : Rat := defaultActDerivBound)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0) : IO (Except String ModelCert) := do
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
  let some _ := numHeads | return .error "missing num_heads"
  let some d := modelDim | return .error "missing model_dim"
  let some dh := headDim | return .error "missing head_dim"
  let some dhid := hiddenDim | return .error "missing hidden_dim"

  let inputVarLowerMin? : Option Rat := none

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
    let ln1Var? : Option Rat := if l = 0 then inputVarLowerMin? else none
    let ln2Var? : Option Rat := none
    let ln1Bound :=
      match ln1Var? with
      | some v => layerNormOpBoundLocal ln1Max v eps
      | none => layerNormOpBoundConservative ln1Max eps
    let ln2Bound :=
      match ln2Var? with
      | some v => layerNormOpBoundLocal ln2Max v eps
      | none => layerNormOpBoundConservative ln2Max eps
    let attnW := ln1Bound * softmaxJacobianNormInfWorst * attnSum[l]!
    let mlpW := ln2Bound * (mlpWin[l]! * actDerivBound * mlpWout[l]!)
    let C := attnW + mlpW
    layers := layers.push {
      layerIdx := l
      ln1MaxAbsGamma := ln1Max
      ln2MaxAbsGamma := ln2Max
      ln1VarianceLowerBound? := ln1Var?
      ln2VarianceLowerBound? := ln2Var?
      ln1Bound := ln1Bound
      ln2Bound := ln2Bound
      attnWeightContribution := attnW
      mlpWeightContribution := mlpW
      C := C
    }
    totalAmp := totalAmp * (1 + C)

  return .ok {
    modelPath := path.toString
    inputPath? := inputPath?.map (·.toString)
    inputDelta := inputDelta
    eps := eps
    actDerivBound := actDerivBound
    softmaxJacobianNormInfWorst := softmaxJacobianNormInfWorst
    layers := layers
    totalAmplificationFactor := totalAmp
  }

/-- Parse input `EMBEDDINGS` from an `.nfpt` file and return intervals `xᵢ ∈ [xᵢ-δ, xᵢ+δ]`
as an array of rows (`seqLen` rows, each of length `modelDim`). -/
private def loadEmbeddingsIntervals
    (path : System.FilePath) (seqLen modelDim : Nat) (delta : Rat) :
    IO (Except String (Array (Array RatInterval))) := do
  let contents ← IO.FS.readFile path
  let lines : Array String := (contents.splitOn "\n").toArray
  let mut i : Nat := 0
  while i < lines.size && lines[i]!.trim.isEmpty do
    i := i + 1
  if !(i < lines.size) then
    return .error "empty input file"
  let headerTag := lines[i]!.trim
  if !headerTag.startsWith "NFP_TEXT" then
    return .error s!"unexpected input header '{headerTag}'"
  i := i + 1
  -- Scan to EMBEDDINGS (optionally skipping TOKENS).
  i := skipUntil lines i (fun s => s = "EMBEDDINGS")
  if !(i < lines.size) then
    return .error "Missing EMBEDDINGS section in input file"
  i := i + 1

  let step :=
    fun (st : (Array (Array RatInterval) × Array RatInterval)) (x : Rat) =>
      let (rows, cur) := st
      let cur := cur.push { lo := x - delta, hi := x + delta }
      if cur.size = modelDim then
        (rows.push cur, #[])
      else
        (rows, cur)
  match foldRatTokens lines i (seqLen * modelDim) (#[], #[]) step with
  | .error e => return .error e
  | .ok ((rows, cur), _) =>
      if cur.size ≠ 0 then
        return .error "EMBEDDINGS parse ended mid-row"
      if rows.size ≠ seqLen then
        return .error s!"EMBEDDINGS length mismatch: expected {seqLen} rows, got {rows.size}"
      return .ok rows

private structure LayerNormParams where
  gamma : Array Rat
  beta : Array Rat

private def collectLayerNormParams
    (lines : Array String) (L d : Nat) :
    Except String (Array LayerNormParams × Array LayerNormParams) :=
  Id.run do
    let defP : LayerNormParams := { gamma := Array.replicate d 1, beta := Array.replicate d 0 }
    let mut ln1 : Array LayerNormParams :=
      Array.replicate L defP
    let mut ln2 : Array LayerNormParams :=
      Array.replicate L defP
    let mut i : Nat := 0
    let mut curLayer : Nat := 0
    while i < lines.size do
      let s := lines[i]!.trim
      if s.startsWith "LAYER" then
        let parts := s.splitOn " " |>.filter (· ≠ "")
        if parts.length >= 2 then
          curLayer := (parts[1]!).toNat? |>.getD curLayer
        i := i + 1
      else if s = "LN1_GAMMA" then
        match consumeVector lines (i + 1) d with
        | .error e => return .error e
        | .ok (xs, next) =>
            if curLayer < L then
              let old := ln1.getD curLayer defP
              ln1 := ln1.set! curLayer { old with gamma := xs }
            i := next
      else if s = "LN1_BETA" then
        match consumeVector lines (i + 1) d with
        | .error e => return .error e
        | .ok (xs, next) =>
            if curLayer < L then
              let old := ln1.getD curLayer defP
              ln1 := ln1.set! curLayer { old with beta := xs }
            i := next
      else if s = "LN2_GAMMA" then
        match consumeVector lines (i + 1) d with
        | .error e => return .error e
        | .ok (xs, next) =>
            if curLayer < L then
              let old := ln2.getD curLayer defP
              ln2 := ln2.set! curLayer { old with gamma := xs }
            i := next
      else if s = "LN2_BETA" then
        match consumeVector lines (i + 1) d with
        | .error e => return .error e
        | .ok (xs, next) =>
            if curLayer < L then
              let old := ln2.getD curLayer defP
              ln2 := ln2.set! curLayer { old with beta := xs }
            i := next
      else
        i := i + 1
    return .ok (ln1, ln2)

/-!
## Cached fixed-point local certification (fast path)

The original local path (`RatInterval` + `parseRat`) is mathematically rigorous but too slow for
large models because it performs gcd-based normalization on the hot path.

We therefore prefer a cached fixed-point representation (`sound_cache/*.nfpc`) and run local IBP
in scaled-`Int` arithmetic with conservative outward rounding.
-/

private def defaultFixedScalePow10 : Nat := 9
private def fixedUlpSlack : Int := 1

private def scaleCfgOfPow10 (p : Nat) : Fixed10Cfg := { scalePow10 := p }

private def ratCeilMulNat (x : Rat) (k : Nat) : Int :=
  if x ≤ 0 then
    0
  else
    let num : Int := x.num
    let den : Nat := x.den
    let numK : Int := num * (Int.ofNat k)
    let q := numK.ediv (Int.ofNat den)
    let r := numK.emod (Int.ofNat den)
    if r = 0 then q else q + 1

private def fixedMeanInterval (xs : Array Fixed10Interval) : Fixed10Interval :=
  if xs.isEmpty then
    { lo := 0, hi := 0 }
  else
    Id.run do
      let n : Nat := xs.size
      let mut loSum : Int := 0
      let mut hiSum : Int := 0
      for x in xs do
        loSum := loSum + x.lo
        hiSum := hiSum + x.hi
      let loμ := loSum.ediv (Int.ofNat n)
      let hiμ :=
        let q := hiSum.ediv (Int.ofNat n)
        let r := hiSum.emod (Int.ofNat n)
        if r = 0 then q else q + 1
      { lo := loμ, hi := hiμ }

private def fixedVarianceLowerBoundRange (cfg : Fixed10Cfg) (xs : Array Fixed10Interval) : Rat :=
  if xs.size < 2 then
    0
  else
    Id.run do
      let n : Nat := xs.size
      let nRat : Rat := (n : Nat)
      let mut loMax : Int := xs[0]!.lo
      let mut hiMin : Int := xs[0]!.hi
      for x in xs do
        loMax := max loMax x.lo
        hiMin := min hiMin x.hi
      let δInt : Int := max 0 (loMax - hiMin)
      if δInt = 0 then
        return 0
      let δRat : Rat :=
        Rat.normalize δInt cfg.scaleNat (den_nz := by
          have h10pos : (0 : Nat) < 10 := by decide
          exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
      let δSq : Rat := δRat * δRat
      let nMinus1 : Rat := ((n - 1 : Nat) : Rat)
      return (nMinus1 * δSq) / (nRat * nRat)

private def fixedLayerNormRowApprox
    (cfg : Fixed10Cfg)
    (row : Array Fixed10Interval)
    (gamma beta : Array Fixed10Interval)
    (eps : Rat) :
    (Array Fixed10Interval × Rat) :=
  if row.size = 0 || gamma.size ≠ row.size || beta.size ≠ row.size then
    (row, 0)
  else
    Id.run do
      let μ := fixedMeanInterval row
      let varLB := fixedVarianceLowerBoundRange cfg row
      let invσUpper : Rat :=
        if varLB ≤ 0 then
          layerNormOpBoundConservative 1 eps
        else
          layerNormOpBoundLocal 1 varLB eps
      let invσUpperInt : Int := ratCeilMulNat invσUpper cfg.scaleNat
      let invσFix : Fixed10Interval := { lo := invσUpperInt, hi := invσUpperInt }
      let mut out : Array Fixed10Interval := Array.mkEmpty row.size
      for i in [:row.size] do
        let centered := Fixed10Interval.sub row[i]! μ
        let coeff := Fixed10Interval.mul cfg gamma[i]! invσFix
        let scaled := Fixed10Interval.mul cfg coeff centered
        out := out.push (Fixed10Interval.add scaled beta[i]!)
      return (out, varLB)

private def readVecIntervals
    (r : SoundCache.I32Reader) (n : Nat) (slack : Int) :
    IO (Array Fixed10Interval × SoundCache.I32Reader) := do
  let mut rr := r
  let mut out : Array Fixed10Interval := Array.mkEmpty n
  for _ in [:n] do
    let (x, rr2) ← SoundCache.I32Reader.readI32 rr
    rr := rr2
    out := out.push { lo := x - slack, hi := x + slack }
  return (out, rr)

private def maxAbsVecFixed (xs : Array Fixed10Interval) : Int :=
  xs.foldl (fun acc x => max acc (Fixed10Interval.absUpper x)) 0

private def addVecFixed (a b : Array Fixed10Interval) : Array Fixed10Interval :=
  Id.run do
    if a.size ≠ b.size then
      return a
    let mut out := Array.mkEmpty a.size
    for i in [:a.size] do
      out := out.push (Fixed10Interval.add a[i]! b[i]!)
    return out

private def consumeMatrixMulAndNormInfFixed
    (cfg : Fixed10Cfg)
    (slack : Int)
    (r : SoundCache.I32Reader)
    (rows cols : Nat)
    (input : Array Fixed10Interval) :
    IO (Array Fixed10Interval × Rat × SoundCache.I32Reader) := do
  if input.size ≠ rows then
    return (Array.replicate cols { lo := 0, hi := 0 }, 0, r)
  let mut rr := r
  let mut out : Array Fixed10Interval := Array.replicate cols { lo := 0, hi := 0 }
  let mut curRowAbs : Int := 0
  let mut maxRowAbs : Int := 0
  for rowIdx in [:rows] do
    let xi := input[rowIdx]!
    for colIdx in [:cols] do
      let (w, rr2) ← SoundCache.I32Reader.readI32 rr
      rr := rr2
      let wAbsBound : Int := (if w < 0 then -w else w) + slack
      curRowAbs := curRowAbs + wAbsBound
      let wI : Fixed10Interval := { lo := w - slack, hi := w + slack }
      let term := Fixed10Interval.mul cfg wI xi
      out := out.set! colIdx (Fixed10Interval.add (out[colIdx]!) term)
    maxRowAbs := max maxRowAbs curRowAbs
    curRowAbs := 0
  let normInf : Rat :=
    Rat.normalize maxRowAbs cfg.scaleNat (den_nz := by
      have h10pos : (0 : Nat) < 10 := by decide
      exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
  return (out, normInf, rr)

private def loadEmbeddingsUnionFixed
    (cfg : Fixed10Cfg)
    (path : System.FilePath)
    (expectedModelDim : Nat)
    (delta : Rat) : IO (Except String (Array Fixed10Interval)) := do
  let deltaInt : Int := ratCeilMulNat delta cfg.scaleNat
  let mut out : Array Fixed10Interval := Array.replicate expectedModelDim { lo := 0, hi := 0 }
  let mut iCol : Nat := 0
  let mut remaining : Nat := 0

  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  -- Header: read until blank line.
  let mut seqLen : Option Nat := none
  let mut modelDim : Option Nat := none
  let mut seenHeader : Bool := false
  while true do
    let line ← h.getLine
    if line.isEmpty then
      return .error "unexpected EOF while reading input header"
    let s := line.trim
    if !seenHeader then
      if s.startsWith "NFP_TEXT" then
        seenHeader := true
      continue
    if s.isEmpty then
      break
    match parseHeaderLine s with
    | none => pure ()
    | some (k, v) =>
        if k = "seq_len" then
          seqLen := v.toNat?
        else if k = "model_dim" then
          modelDim := v.toNat?
        else
          pure ()

  let some n := seqLen | return .error "missing seq_len in input file"
  let some d := modelDim | return .error "missing model_dim in input file"
  if d ≠ expectedModelDim then
    return .error s!"input model_dim mismatch (expected {expectedModelDim}, got {d})"
  remaining := n * d

  -- Scan to EMBEDDINGS marker.
  let mut found : Bool := false
  while !found do
    let line ← h.getLine
    if line.isEmpty then
      return .error "unexpected EOF while scanning for EMBEDDINGS"
    if line.trim = "EMBEDDINGS" then
      found := true

  while remaining > 0 do
    let line ← h.getLine
    if line.isEmpty then
      return .error "unexpected EOF while reading EMBEDDINGS"
    let s := line.trim
    if s.isEmpty then
      continue
    let bytes := s.toUTF8
    let mut j : Nat := 0
    while j < bytes.size && remaining > 0 do
      while j < bytes.size && (bytes[j]! = 32 || bytes[j]! = 9) do
        j := j + 1
      if j ≥ bytes.size then
        break
      let tokStart := j
      while j < bytes.size && (bytes[j]! ≠ 32 && bytes[j]! ≠ 9) do
        j := j + 1
      let tokStop := j
      match parseFixed10Rounded cfg.scalePow10 bytes tokStart tokStop with
      | .error e => return .error e
      | .ok x =>
          let lo := x - fixedUlpSlack - deltaInt
          let hi := x + fixedUlpSlack + deltaInt
          let cur := out[iCol]!
          out := out.set! iCol { lo := min cur.lo lo, hi := max cur.hi hi }
          iCol := (iCol + 1) % expectedModelDim
          remaining := remaining - 1
  return .ok out

private def ensureSoundCache
    (modelPath : System.FilePath)
    (scalePow10 : Nat := defaultFixedScalePow10) :
    IO (Except String (System.FilePath × SoundCache.Header)) := do
  SoundCache.ensureCacheDir
  let modelHash ← SoundCache.fnv1a64File modelPath
  let mdata ← modelPath.metadata
  let modelSize : UInt64 := mdata.byteSize
  let cpath := SoundCache.cachePath modelPath modelHash scalePow10
  if !(← cpath.pathExists) then
    match (← SoundCache.buildCacheFile modelPath cpath scalePow10) with
    | .error e => return .error e
    | .ok _ => pure ()
  let h ← IO.FS.Handle.mk cpath IO.FS.Mode.read
  let hdr ← SoundCache.readHeader h
  if hdr.modelHash ≠ modelHash then
    return .error "sound cache hash mismatch"
  if hdr.modelSize ≠ modelSize then
    return .error "sound cache size mismatch"
  return .ok (cpath, hdr)

/-- Local (input-dependent) certificate path using streaming interval propagation.

This is conservative in two key ways to remain streaming/memory-safe:
- it uses a **union box** over tokens throughout (so we never hold `seqLen×modelDim` intervals),
  which is sound (a superset) but can be looser than per-token tracking,
- it uses union boxes for attention/MLP linear maps to avoid `seqLen×hiddenDim` blowups.
-/
private def certifyModelFileLocalText
    (path : System.FilePath)
    (eps : Rat)
    (actDerivBound : Rat)
    (inputPath : System.FilePath)
    (inputDelta : Rat) : IO (Except String ModelCert) := do
  let contents ← IO.FS.readFile path
  let lines : Array String := (contents.splitOn "\n").toArray

  -- Header
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
    | none => i := i + 1
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

  -- Prepass: collect LN parameters.
  let (ln1Params, ln2Params) ←
    match collectLayerNormParams lines L d with
    | .error e => return .error e
    | .ok x => pure x

  let defLn : LayerNormParams := { gamma := Array.replicate d 1, beta := Array.replicate d 0 }

  -- Input: per-token residual intervals.
  let residual0 ← loadEmbeddingsIntervals inputPath n d inputDelta
  match residual0 with
  | .error e => return .error e
  | .ok residualRows0 =>
      -- Use a single union box for all tokens (sound superset, much faster than `seqLen×modelDim`).
      let mut residualUnion := unionRows residualRows0 d

      -- Start scanning at first layer marker.
      let mut pos : Nat := skipUntil lines 0 (fun s => s.startsWith "LAYER")

      let mut layers : Array LayerAmplificationCert := Array.mkEmpty L
      let mut totalAmp : Rat := 1

      for l in [:L] do
        -- Ensure we're at the next layer.
        pos := skipUntil lines pos (fun s => s.startsWith "LAYER")
        if pos ≥ lines.size then
          return .error s!"unexpected end of file while scanning layer {l}"
        pos := pos + 1

        -- LN1: compute per-row outputs (for union) and min variance LB (for Jacobian bound).
        let p1 := ln1Params.getD l defLn
        let (ln1Out, ln1VarLB) := layerNormRowApprox residualUnion p1.gamma p1.beta eps
        let ln1MaxAbsGamma := maxAbsOfVector p1.gamma
        let ln1Bound :=
          if ln1VarLB > 0 then
            layerNormOpBoundLocal ln1MaxAbsGamma ln1VarLB eps
          else
            layerNormOpBoundConservative ln1MaxAbsGamma eps
        let ln1Union := ln1Out

        -- Attention (streaming): use union input box.
        let mut attnUnion : Array RatInterval := zeroIntervals d
        let mut attnCoeff : Rat := 0
        for _h in [:H] do
          pos := skipBlankLines lines pos
          if !(pos < lines.size) then
            return .error "unexpected end of file while scanning HEAD"
          if !(lines[pos]!.trim.startsWith "HEAD") then
            return .error "expected HEAD marker before per-head matrices"
          pos := pos + 1

          pos := skipBlankLines lines pos
          if !(pos < lines.size && lines[pos]!.trim = "W_Q") then
            return .error "missing W_Q"
          match consumeMatrixSkipFast lines (pos + 1) d dh with
          | .error e => return .error e
          | .ok next => pos := next
          -- Optional per-head Q bias (does not affect Jacobian,
          -- but must be parsed to stay in sync).
          pos := skipBlankLines lines pos
          if pos < lines.size && lines[pos]!.trim = "b_Q" then
            match consumeVectorSkipFast lines (pos + 1) dh with
            | .error e => return .error e
            | .ok next => pos := next

          pos := skipBlankLines lines pos
          if !(pos < lines.size && lines[pos]!.trim = "W_K") then
            return .error "missing W_K"
          match consumeMatrixSkipFast lines (pos + 1) d dh with
          | .error e => return .error e
          | .ok next => pos := next
          -- Optional per-head K bias (does not affect Jacobian,
          -- but must be parsed to stay in sync).
          pos := skipBlankLines lines pos
          if pos < lines.size && lines[pos]!.trim = "b_K" then
            match consumeVectorSkipFast lines (pos + 1) dh with
            | .error e => return .error e
            | .ok next => pos := next

          pos := skipBlankLines lines pos
          if !(pos < lines.size && lines[pos]!.trim = "W_V") then
            return .error "missing W_V"
          match consumeMatrixMulAndNormInf lines (pos + 1) d dh ln1Union with
          | .error e => return .error e
          | .ok (vHidden, nv, nextV) =>
              pos := nextV
              -- Optional per-head V bias (affects forward activations / variance,
              -- so we include it).
              pos := skipBlankLines lines pos
              let mut vHidden := vHidden
              if pos < lines.size && lines[pos]!.trim = "b_V" then
                match consumeVector lines (pos + 1) dh with
                | .error e => return .error e
                | .ok (bv, nextBv) =>
                    pos := nextBv
                    vHidden := addConstVec vHidden bv

              pos := skipBlankLines lines pos
              if !(pos < lines.size && lines[pos]!.trim = "W_O") then
                return .error "missing W_O"
              match consumeMatrixMulAndNormInf lines (pos + 1) dh d vHidden with
              | .error e => return .error e
                  | .ok (vOut, no, nextO) =>
                      pos := nextO
                      attnUnion := addVecIntervals attnUnion vOut
                      attnCoeff := attnCoeff + nv * no

        -- Shared attention projection bias (affects forward activations / variance,
        -- so we include it).
        pos := skipBlankLines lines pos
        if pos < lines.size && lines[pos]!.trim = "ATTN_BIAS" then
          match consumeVector lines (pos + 1) d with
          | .error e => return .error e
          | .ok (bAttn, nextB) =>
              pos := nextB
              attnUnion := addConstVec attnUnion bAttn

        residualUnion := addVecIntervals residualUnion attnUnion

        -- LN2: compute per-row outputs and min variance LB.
        let p2 := ln2Params.getD l defLn
        let (ln2Out, ln2VarLB) := layerNormRowApprox residualUnion p2.gamma p2.beta eps
        let ln2MaxAbsGamma := maxAbsOfVector p2.gamma
        let ln2Bound :=
          if ln2VarLB > 0 then
            layerNormOpBoundLocal ln2MaxAbsGamma ln2VarLB eps
          else
            layerNormOpBoundConservative ln2MaxAbsGamma eps
        let ln2Union := ln2Out

        -- MLP (streaming): W_in, b_in, W_out, b_out.
        pos := skipBlankLines lines pos
        if !(pos < lines.size && lines[pos]!.trim = "MLP") then
          return .error "missing MLP section"
        pos := pos + 1

        pos := skipBlankLines lines pos
        if !(pos < lines.size && lines[pos]!.trim = "W_in") then
          return .error "missing W_in"
        match consumeMatrixMulAndNormInf lines (pos + 1) d dhid ln2Union with
        | .error e => return .error e
        | .ok (hidden, nWin, nextWin) =>
            pos := nextWin

            pos := skipBlankLines lines pos
            if !(pos < lines.size && lines[pos]!.trim = "b_in") then
              return .error "missing b_in"
            match consumeVector lines (pos + 1) dhid with
            | .error e => return .error e
            | .ok (bin, nextBin) =>
                pos := nextBin

                let hiddenB := addConstVec hidden bin
                let actHidden := hiddenB.map RatInterval.geluOverapprox

                pos := skipBlankLines lines pos
                if !(pos < lines.size && lines[pos]!.trim = "W_out") then
                  return .error "missing W_out"
                match consumeMatrixMulAndNormInf lines (pos + 1) dhid d actHidden with
                | .error e => return .error e
                | .ok (mlpOut0, nWout, nextWout) =>
                    pos := nextWout
                    pos := skipBlankLines lines pos
                    if !(pos < lines.size && lines[pos]!.trim = "b_out") then
                      return .error "missing b_out"
                    match consumeVector lines (pos + 1) d with
                    | .error e => return .error e
                    | .ok (bout, nextBout) =>
                        pos := nextBout
                        let mlpOut := addConstVec mlpOut0 bout
                        residualUnion := addVecIntervals residualUnion mlpOut

                        let attnW := ln1Bound * softmaxJacobianNormInfWorst * attnCoeff
                        let mlpCoeff := nWin * actDerivBound * nWout
                        let mlpW := ln2Bound * mlpCoeff
                        let C := attnW + mlpW

                        layers := layers.push {
                          layerIdx := l
                          ln1MaxAbsGamma := ln1MaxAbsGamma
                          ln2MaxAbsGamma := ln2MaxAbsGamma
                          ln1VarianceLowerBound? := some ln1VarLB
                          ln2VarianceLowerBound? := some ln2VarLB
                          ln1Bound := ln1Bound
                          ln2Bound := ln2Bound
                          attnWeightContribution := attnW
                          mlpWeightContribution := mlpW
                          C := C
                        }
                        totalAmp := totalAmp * (1 + C)
                        pos := skipUntil lines pos (fun s => s.startsWith "LAYER")

      return .ok {
        modelPath := path.toString
        inputPath? := some inputPath.toString
        inputDelta := inputDelta
        eps := eps
        actDerivBound := actDerivBound
        softmaxJacobianNormInfWorst := softmaxJacobianNormInfWorst
        layers := layers
        totalAmplificationFactor := totalAmp
      }

private def certifyModelFileLocal
    (path : System.FilePath)
    (eps : Rat)
    (actDerivBound : Rat)
    (inputPath : System.FilePath)
    (inputDelta : Rat) : IO (Except String ModelCert) := do
  -- Prefer cached fixed-point path; fall back to the (slow) Rat-based path on any cache error.
  match (← ensureSoundCache path) with
  | .error _ =>
      certifyModelFileLocalText path eps actDerivBound inputPath inputDelta
  | .ok (cpath, hdr) =>
      let cfg : Fixed10Cfg := scaleCfgOfPow10 hdr.scalePow10.toNat
      let slack : Int := fixedUlpSlack
      let modelDim := hdr.modelDim.toNat
      let headDim := hdr.headDim.toNat
      let hiddenDim := hdr.hiddenDim.toNat
      let L := hdr.numLayers.toNat
      let H := hdr.numHeads.toNat
      -- For now we read embeddings from the input `.nfpt` file and use a union box.
      let residualUnionE ← loadEmbeddingsUnionFixed cfg inputPath modelDim inputDelta
      match residualUnionE with
      | .error e => return .error e
      | .ok residualUnion0 =>
          let mut residualUnion := residualUnion0
          -- Open cache and position reader after header.
          let ch ← IO.FS.Handle.mk cpath IO.FS.Mode.read
          let _ ← SoundCache.readHeader ch
          let mut rr ← SoundCache.I32Reader.init ch

          let mut layers : Array LayerAmplificationCert := Array.mkEmpty L
          let mut totalAmp : Rat := 1

          for l in [:L] do
            -- LN params from cache
            let (ln1Gamma, rr1) ← readVecIntervals rr modelDim slack
            let (ln1Beta, rr2) ← readVecIntervals rr1 modelDim slack
            let (ln2Gamma, rr3) ← readVecIntervals rr2 modelDim slack
            let (ln2Beta, rr4) ← readVecIntervals rr3 modelDim slack
            rr := rr4

            -- LN1
            let (ln1Out, ln1VarLB) := fixedLayerNormRowApprox cfg residualUnion ln1Gamma ln1Beta eps
            let ln1MaxAbsGamma : Rat :=
              Rat.normalize (maxAbsVecFixed ln1Gamma) cfg.scaleNat (den_nz := by
                have h10pos : (0 : Nat) < 10 := by decide
                exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
            let ln1Bound :=
              if ln1VarLB > 0 then
                layerNormOpBoundLocal ln1MaxAbsGamma ln1VarLB eps
              else
                layerNormOpBoundConservative ln1MaxAbsGamma eps

            -- Attention (streaming from cache)
            let mut attnUnion : Array Fixed10Interval :=
              Array.replicate modelDim { lo := 0, hi := 0 }
            let mut attnCoeff : Rat := 0
            for _h in [:H] do
              let (vHidden0, nWv, rrV) ←
                consumeMatrixMulAndNormInfFixed cfg slack rr modelDim headDim ln1Out
              rr := rrV
              let (bV, rrBv) ← readVecIntervals rr headDim slack
              rr := rrBv
              let vHidden := addVecFixed vHidden0 bV
              let (vOut, nWo, rrO) ←
                consumeMatrixMulAndNormInfFixed cfg slack rr headDim modelDim vHidden
              rr := rrO
              attnUnion := addVecFixed attnUnion vOut
              attnCoeff := attnCoeff + nWv * nWo

            let (attnBias, rrB) ← readVecIntervals rr modelDim slack
            rr := rrB
            attnUnion := addVecFixed attnUnion attnBias
            residualUnion := addVecFixed residualUnion attnUnion

            -- LN2
            let (ln2Out, ln2VarLB) := fixedLayerNormRowApprox cfg residualUnion ln2Gamma ln2Beta eps
            let ln2MaxAbsGamma : Rat :=
              Rat.normalize (maxAbsVecFixed ln2Gamma) cfg.scaleNat (den_nz := by
                have h10pos : (0 : Nat) < 10 := by decide
                exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
            let ln2Bound :=
              if ln2VarLB > 0 then
                layerNormOpBoundLocal ln2MaxAbsGamma ln2VarLB eps
              else
                layerNormOpBoundConservative ln2MaxAbsGamma eps

            -- MLP
            let (hidden0, nWin, rrWin) ←
              consumeMatrixMulAndNormInfFixed cfg slack rr modelDim hiddenDim ln2Out
            rr := rrWin
            let (bIn, rrBin) ← readVecIntervals rr hiddenDim slack
            rr := rrBin
            let hiddenB := addVecFixed hidden0 bIn
            let actHidden := hiddenB.map Fixed10Interval.geluOverapprox
            let (mlpOut0, nWout, rrWout) ←
              consumeMatrixMulAndNormInfFixed cfg slack rr hiddenDim modelDim actHidden
            rr := rrWout
            let (bOut, rrBout) ← readVecIntervals rr modelDim slack
            rr := rrBout
            let mlpOut := addVecFixed mlpOut0 bOut
            residualUnion := addVecFixed residualUnion mlpOut

            let attnW := ln1Bound * softmaxJacobianNormInfWorst * attnCoeff
            let mlpCoeff := nWin * actDerivBound * nWout
            let mlpW := ln2Bound * mlpCoeff
            let C := attnW + mlpW

            layers := layers.push {
              layerIdx := l
              ln1MaxAbsGamma := ln1MaxAbsGamma
              ln2MaxAbsGamma := ln2MaxAbsGamma
              ln1VarianceLowerBound? := some ln1VarLB
              ln2VarianceLowerBound? := some ln2VarLB
              ln1Bound := ln1Bound
              ln2Bound := ln2Bound
              attnWeightContribution := attnW
              mlpWeightContribution := mlpW
              C := C
            }
            totalAmp := totalAmp * (1 + C)

          return .ok {
            modelPath := path.toString
            inputPath? := some inputPath.toString
            inputDelta := inputDelta
            eps := eps
            actDerivBound := actDerivBound
            softmaxJacobianNormInfWorst := softmaxJacobianNormInfWorst
            layers := layers
            totalAmplificationFactor := totalAmp
          }

/-- Soundly compute certification bounds from a `.nfpt` model file.

If an input is provided via `inputPath?`, the certificate uses streaming rational IBP to obtain
local (input-dependent) LayerNorm variance lower bounds at every layer.
Otherwise it falls back to the weight-only global certificate.
-/
def certifyModelFile
    (path : System.FilePath)
    (eps : Rat := defaultEps)
    (actDerivBound : Rat := defaultActDerivBound)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0) : IO (Except String ModelCert) := do
  match inputPath? with
  | none =>
      certifyModelFileGlobal path eps actDerivBound (inputPath? := none) (inputDelta := inputDelta)
  | some ip =>
      if inputDelta < 0 then
        return .error "delta must be nonnegative"
      certifyModelFileLocal path eps actDerivBound ip inputDelta

end Nfp.Sound
