-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Cert
import Nfp.Sound.HeadCert
import Nfp.Sound.ModelHeader
import Nfp.Sound.TextPure
import Nfp.Untrusted.SoundBinary
import Nfp.Sound.Interval
import Nfp.Sound.Affine
import Nfp.Untrusted.SoundCacheIO
import Nfp.Sound.Fixed

namespace Nfp.Untrusted.SoundCompute

open IO
open Nfp.Sound
open Nfp.Untrusted.SoundBinary

/-!
# Untrusted SOUND computation helpers

This module performs **IO-heavy witness generation** for SOUND certification. It parses `.nfpt`
models (binary, plus legacy text for some paths) and computes candidate certificates for:
- model-level residual amplification bounds,
- per-head contribution bounds,
- local head-pattern / best-match / induction certificates.

It does **not** construct the full `ConcreteModel` (Float-based). Instead it parses only the
weights needed for conservative residual amplification constants `Cᵢ` (bounds ‖layerJacobian - I‖),
using exact `Rat` arithmetic or fixed-point interval arithmetic.

All certificates produced here are **untrusted** and must be validated by the trusted checker
in `Nfp.Sound.IO`.

Trusted base:
- Parsing from text to `Rat` via `Nfp.Sound.parseRat`.
- Exact accumulation of row-sum norms and max-abs values.

No `Float` arithmetic is *trusted* as an input to certification.
-/

private def defaultBinaryScalePow10 : Nat := 9

private def maxAbsOfVector (xs : Array Rat) : Rat :=
  xs.foldl (fun acc x => max acc (ratAbs x)) 0

/-- Compute weight-only per-head contribution bounds from a binary `.nfpt`. -/
def certifyHeadBoundsBinary
    (path : System.FilePath)
    (scalePow10 : Nat := defaultBinaryScalePow10) :
    IO (Except String (Array HeadContributionCert)) := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  match ← readBinaryHeader h with
  | .error e => return .error e
  | .ok hdr =>
      match ← skipI32Array h hdr.seqLen with
      | .error e => return .error e
      | .ok _ => pure ()
      match ← skipF64Array h (hdr.seqLen * hdr.modelDim) with
      | .error e => return .error e
      | .ok _ => pure ()
      let mut heads : Array HeadContributionCert := Array.mkEmpty (hdr.numLayers * hdr.numHeads)
      for l in [:hdr.numLayers] do
        for hIdx in [:hdr.numHeads] do
          let wqScaledE ← readMatrixOpBoundScaled h hdr.modelDim hdr.headDim scalePow10
          let wqScaled ←
            match wqScaledE with
            | .error e => return .error e
            | .ok v => pure v
          match ← skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          let wkScaledE ← readMatrixOpBoundScaled h hdr.modelDim hdr.headDim scalePow10
          let wkScaled ←
            match wkScaledE with
            | .error e => return .error e
            | .ok v => pure v
          match ← skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          let wvScaledE ← readMatrixOpBoundScaled h hdr.modelDim hdr.headDim scalePow10
          let wvScaled ←
            match wvScaledE with
            | .error e => return .error e
            | .ok v => pure v
          match ← skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          let woScaledE ← readMatrixOpBoundScaled h hdr.headDim hdr.modelDim scalePow10
          let woScaled ←
            match woScaledE with
            | .error e => return .error e
            | .ok v => pure v
          let wqOp := ratOfScaledNat scalePow10 wqScaled
          let wkOp := ratOfScaledNat scalePow10 wkScaled
          let wvOp := ratOfScaledNat scalePow10 wvScaled
          let woOp := ratOfScaledNat scalePow10 woScaled
          let cert : HeadContributionCert := {
            layerIdx := l
            headIdx := hIdx
            wqOpBound := wqOp
            wkOpBound := wkOp
            wvOpBound := wvOp
            woOpBound := woOp
            qkFactorBound := wqOp * wkOp
            voFactorBound := wvOp * woOp
          }
          if cert.check then
            heads := heads.push cert
          else
            return .error "head contribution certificate failed internal checks"
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h (hdr.modelDim * hdr.hiddenDim) with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h hdr.hiddenDim with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h (hdr.hiddenDim * hdr.modelDim) with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
      match ← skipF64Array h hdr.modelDim with
      | .error e => return .error e
      | .ok _ => pure ()
      match ← skipF64Array h hdr.modelDim with
      | .error e => return .error e
      | .ok _ => pure ()
      match ← skipF64Array h (hdr.modelDim * hdr.vocabSize) with
      | .error e => return .error e
      | .ok _ => pure ()
      return .ok heads

private def certifyModelFileGlobalBinary
    (path : System.FilePath)
    (eps : Rat)
    (geluDerivTarget : GeluDerivTarget)
    (soundnessBits : Nat)
    (partitionDepth : Nat)
    (softmaxMarginLowerBound : Rat := 0)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort) : IO (Except String ModelCert) := do
  if partitionDepth ≠ 0 then
    return .error "partitionDepth > 0 not yet implemented"
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  match ← readBinaryHeader h with
  | .error e => return .error e
  | .ok hdr =>
      let scalePow10 := defaultBinaryScalePow10
      let actDerivBound := geluDerivBoundGlobal geluDerivTarget
      match ← skipI32Array h hdr.seqLen with
      | .error e => return .error e
      | .ok _ => pure ()
      match ← skipF64Array h (hdr.seqLen * hdr.modelDim) with
      | .error e => return .error e
      | .ok _ => pure ()
      let mut layers : Array LayerAmplificationCert := Array.mkEmpty hdr.numLayers
      let mut totalAmp : Rat := 1
      for l in [:hdr.numLayers] do
        let mut attnValueCoeff : Rat := 0
        let mut wqMax : Rat := 0
        let mut wkMax : Rat := 0
        for _h in [:hdr.numHeads] do
          let wqScaledE ← readMatrixNormInfScaled h hdr.modelDim hdr.headDim scalePow10
          let wqScaled ←
            match wqScaledE with
            | .error e => return .error e
            | .ok v => pure v
          match ← skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          let wkScaledE ← readMatrixNormInfScaled h hdr.modelDim hdr.headDim scalePow10
          let wkScaled ←
            match wkScaledE with
            | .error e => return .error e
            | .ok v => pure v
          match ← skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          let nvScaledE ← readMatrixNormInfScaled h hdr.modelDim hdr.headDim scalePow10
          let nvScaled ←
            match nvScaledE with
            | .error e => return .error e
            | .ok v => pure v
          match ← skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          let noScaledE ← readMatrixNormInfScaled h hdr.headDim hdr.modelDim scalePow10
          let noScaled ←
            match noScaledE with
            | .error e => return .error e
            | .ok v => pure v
          let wq := ratOfScaledInt scalePow10 wqScaled
          let wk := ratOfScaledInt scalePow10 wkScaled
          let nv := ratOfScaledInt scalePow10 nvScaled
          let no := ratOfScaledInt scalePow10 noScaled
          wqMax := max wqMax wq
          wkMax := max wkMax wk
          attnValueCoeff := attnValueCoeff + nv * no
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        let nWinScaledE ←
          readMatrixNormInfScaled h hdr.modelDim hdr.hiddenDim scalePow10
        let nWinScaled ←
          match nWinScaledE with
          | .error e => return .error e
          | .ok v => pure v
        match ← skipF64Array h hdr.hiddenDim with
        | .error e => return .error e
        | .ok _ => pure ()
        let nWoutScaledE ←
          readMatrixNormInfScaled h hdr.hiddenDim hdr.modelDim scalePow10
        let nWoutScaled ←
          match nWoutScaledE with
          | .error e => return .error e
          | .ok v => pure v
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        let ln1GammaScaledE ← readVectorMaxAbsScaled h hdr.modelDim scalePow10
        let ln1GammaScaled ←
          match ln1GammaScaledE with
          | .error e => return .error e
          | .ok v => pure v
        let ln1BetaScaledE ← readVectorMaxAbsScaled h hdr.modelDim scalePow10
        let ln1BetaScaled ←
          match ln1BetaScaledE with
          | .error e => return .error e
          | .ok v => pure v
        let ln2GammaScaledE ← readVectorMaxAbsScaled h hdr.modelDim scalePow10
        let ln2GammaScaled ←
          match ln2GammaScaledE with
          | .error e => return .error e
          | .ok v => pure v
        let ln2BetaScaledE ← readVectorMaxAbsScaled h hdr.modelDim scalePow10
        match ln2BetaScaledE with
        | .error e => return .error e
        | .ok _ => pure ()
        let ln1Max := ratOfScaledInt scalePow10 ln1GammaScaled
        let ln1MaxAbsBeta := ratOfScaledInt scalePow10 ln1BetaScaled
        let ln2Max := ratOfScaledInt scalePow10 ln2GammaScaled
        let nWin := ratOfScaledInt scalePow10 nWinScaled
        let nWout := ratOfScaledInt scalePow10 nWoutScaled
        let ln1Bound := layerNormOpBoundConservative ln1Max eps soundnessBits
        let ln2Bound := layerNormOpBoundConservative ln2Max eps soundnessBits
        let ln1OutMaxAbsBound := layerNormOutputMaxAbsBound hdr.modelDim ln1Max ln1MaxAbsBeta
        let attnPatternCoeff :=
          attnPatternCoeffBound hdr.seqLen hdr.modelDim hdr.headDim ln1OutMaxAbsBound
            wqMax wkMax attnValueCoeff
        let mlpCoeff := nWin * nWout
        let mlpActDerivBound := actDerivBound
        let scoreAbsBound :=
          attnScoreAbsBound hdr.modelDim hdr.headDim ln1OutMaxAbsBound wqMax wkMax
        let (softmaxProbLo, softmaxProbHi) :=
          softmaxProbIntervalFromScoreAbsBound hdr.seqLen scoreAbsBound softmaxExpEffort
        let softmaxIntervalBound := softmaxJacobianNormInfBound softmaxProbLo softmaxProbHi
        let softmaxMarginBound :=
          softmaxJacobianNormInfBoundFromMargin hdr.seqLen softmaxMarginLowerBound softmaxExpEffort
        let softmaxBound := min softmaxIntervalBound softmaxMarginBound
        let attnW :=
          ln1Bound *
            ((hdr.seqLen : Rat) * attnValueCoeff + softmaxBound * attnPatternCoeff)
        let mlpW := ln2Bound * (mlpCoeff * mlpActDerivBound)
        let C := attnW + mlpW + attnW * mlpW
        layers := layers.push {
          layerIdx := l
          ln1MaxAbsGamma := ln1Max
          ln1MaxAbsBeta := ln1MaxAbsBeta
          ln2MaxAbsGamma := ln2Max
          ln1VarianceLowerBound? := none
          ln2VarianceLowerBound? := none
          ln1Bound := ln1Bound
          ln2Bound := ln2Bound
          ln1OutMaxAbsBound := ln1OutMaxAbsBound
          softmaxProbLo := softmaxProbLo
          softmaxProbHi := softmaxProbHi
          softmaxMarginLowerBound := softmaxMarginLowerBound
          softmaxExpEffort := softmaxExpEffort
          softmaxJacobianNormInfUpperBound := softmaxBound
          wqOpBoundMax := wqMax
          wkOpBoundMax := wkMax
          attnValueCoeff := attnValueCoeff
          attnPatternCoeff := attnPatternCoeff
          mlpCoeff := mlpCoeff
          mlpWinBound := nWin
          mlpWoutBound := nWout
          mlpActDerivBound := mlpActDerivBound
          attnJacBound := attnW
          mlpJacBound := mlpW
          C := C
        }
        totalAmp := totalAmp * (1 + C)
      match ← skipF64Array h hdr.modelDim with
      | .error e => return .error e
      | .ok _ => pure ()
      match ← skipF64Array h hdr.modelDim with
      | .error e => return .error e
      | .ok _ => pure ()
      match ← skipF64Array h (hdr.modelDim * hdr.vocabSize) with
      | .error e => return .error e
      | .ok _ => pure ()
      let cert : ModelCert := {
        modelPath := path.toString
        inputPath? := none
        inputDelta := 0
        eps := eps
        seqLen := hdr.seqLen
        modelDim := hdr.modelDim
        headDim := hdr.headDim
        soundnessBits := soundnessBits
        geluDerivTarget := geluDerivTarget
        actDerivBound := actDerivBound
        softmaxJacobianNormInfWorst := softmaxJacobianNormInfWorst
        layers := layers
        totalAmplificationFactor := totalAmp
      }
      if cert.check then
        return .ok cert
      return .error "sound certificate failed internal consistency checks"

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

/-- Max GeLU derivative bound across a vector of rational intervals. -/
private def maxGeluDerivBound (target : GeluDerivTarget) (xs : Array RatInterval) : Rat :=
  xs.foldl (fun acc x => max acc (RatInterval.geluDerivBound target x)) 0

/-- Sum of per-coordinate centered absolute bounds (interval widths). -/
private def centeredAbsSum (xs : Array RatInterval) : Rat :=
  xs.foldl (fun acc x => acc + RatInterval.centeredAbsBound x) 0

/-- Max GeLU derivative bound across fixed-point intervals (converted to `Rat`). -/
private def maxGeluDerivBoundFixed (cfg : Fixed10Cfg) (target : GeluDerivTarget)
    (xs : Array Fixed10Interval) : Rat :=
  xs.foldl (fun acc x => max acc (Fixed10Interval.geluDerivBound cfg target x)) 0

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

private def layerNormRowApprox (row : Array RatInterval) (gamma beta : Array Rat) (eps : Rat)
    (soundnessBits : Nat) : (Array RatInterval × Rat) :=
  if row.size = 0 || gamma.size ≠ row.size || beta.size ≠ row.size then
    (row, 0)
  else
    Id.run do
      let μ := RatInterval.mean row
      let varLB := RatInterval.varianceLowerBound row
      let invσUpper : Rat :=
        if varLB ≤ 0 then
          -- Sound fallback for IBP propagation: `1/σ ≤ 1/eps` (conservative, but rigorous).
          layerNormOpBoundConservative 1 eps soundnessBits
        else
          layerNormOpBoundLocal 1 varLB eps soundnessBits
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

private def findLineIdxFrom
    (lines : Array String) (start : Nat) (p : String → Bool) : Option Nat :=
  Nfp.Sound.findLineIdxFrom lines start p

private def skipUntil (lines : Array String) (start : Nat) (p : String → Bool) : Nat :=
  Nfp.Sound.skipUntil lines start p

private def skipBlankLines (lines : Array String) (start : Nat) : Nat :=
  Nfp.Sound.skipBlankLines lines start

/-!
### Fast skipping without parsing

For local SOUND certification we do not need `W_Q`, `W_K`, `b_Q`, or `b_K` numerically
(they don't affect the Jacobian bounds we certify in this streaming-only pass).

Parsing decimals into `Rat` is expensive, so we skip these sections by **counting tokens**
instead of calling `parseRat`.
-/

@[inline] private def countWsTokens (s : String) : Nat :=
  Nfp.Sound.countWsTokens s

private def consumeTokensSkipFast
    (lines : Array String) (start : Nat) (numTokens : Nat) : Except String Nat :=
  Id.run do
    let mut iLine := start
    let mut remaining := numTokens
    while remaining > 0 do
      if iLine ≥ lines.size then
        return .error "unexpected end of file while skipping tokens"
      let line := lines[iLine]!
      iLine := iLine + 1
      let c := countWsTokens line
      if c = 0 then
        pure ()
      else if c ≥ remaining then
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

/-- Accumulator for streaming matrix multiplication with row-abs tracking. -/
private structure MulAndNormAcc where
  out : Array RatInterval
  row : Nat
  col : Nat
  curRowAbs : Rat
  maxRowAbs : Rat

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
    let init : MulAndNormAcc := {
      out := zeroIntervals cols
      row := 0
      col := 0
      curRowAbs := 0
      maxRowAbs := 0
    }
    let step := fun (st : MulAndNormAcc) (w : Rat) =>
      let r := st.row
      let c := st.col
      let curRowAbs := st.curRowAbs + ratAbs w
      -- out[c] += w * input[r]
      let term := RatInterval.scale w (input[r]!)
      let out := st.out.set! c (RatInterval.add (st.out[c]!) term)
      if c + 1 = cols then
        { out := out
          row := r + 1
          col := 0
          curRowAbs := 0
          maxRowAbs := max st.maxRowAbs curRowAbs }
      else
        { out := out
          row := r
          col := c + 1
          curRowAbs := curRowAbs
          maxRowAbs := st.maxRowAbs }
    match foldRatTokens lines start (rows * cols) init step with
    | .error e => return .error e
    | .ok (st, next) =>
        -- Account for a partial last row (should not happen if rows*cols consumed).
        let maxRowAbs := max st.maxRowAbs st.curRowAbs
        return .ok (st.out, maxRowAbs, next)

/-- Soundly compute conservative per-layer residual amplification constants from a `.nfpt` file. -/
def certifyModelFileGlobal
    (path : System.FilePath)
    (eps : Rat)
    (geluDerivTarget : GeluDerivTarget)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (partitionDepth : Nat := 0)
    (softmaxMarginLowerBound : Rat := 0)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort) : IO (Except String ModelCert) := do
  if partitionDepth ≠ 0 then
    return .error "partitionDepth > 0 not yet implemented"
  let actDerivBound := geluDerivBoundGlobal geluDerivTarget
  let contents ← IO.FS.readFile path
  let lines : Array String := Nfp.Sound.splitLines contents
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
  let some n := seqLen | return .error "missing seq_len"
  let inputVarLowerMin? : Option Rat := none
  -- Accumulators
  let mut ln1GammaMax : Array Rat := Array.replicate L 1
  let mut ln1BetaMax : Array Rat := Array.replicate L 0
  let mut ln2GammaMax : Array Rat := Array.replicate L 1
  let mut attnValueCoeff : Array Rat := Array.replicate L 0
  let mut wqMax : Array Rat := Array.replicate L 0
  let mut wkMax : Array Rat := Array.replicate L 0
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
    else if line = "W_Q" then
      let r := curLayer
      match consumeMatrixNormInf lines (i + 1) d dh with
      | .error e => return .error e
      | .ok (nq, next) =>
        if r < wqMax.size then
          wqMax := wqMax.set! r (max wqMax[r]! nq)
        i := next
    else if line = "W_K" then
      let r := curLayer
      match consumeMatrixNormInf lines (i + 1) d dh with
      | .error e => return .error e
      | .ok (nk, next) =>
        if r < wkMax.size then
          wkMax := wkMax.set! r (max wkMax[r]! nk)
        i := next
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
          if r < attnValueCoeff.size then
            attnValueCoeff := attnValueCoeff.set! r (attnValueCoeff[r]! + (nv * no))
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
      match consumeVectorMaxAbs lines (i + 1) d with
      | .error e => return .error e
      | .ok (m, next) =>
        if curLayer < ln1GammaMax.size then
          ln1GammaMax := ln1GammaMax.set! curLayer m
        i := next
    else if line = "LN1_BETA" then
      match consumeVectorMaxAbs lines (i + 1) d with
      | .error e => return .error e
      | .ok (m, next) =>
        if curLayer < ln1BetaMax.size then
          ln1BetaMax := ln1BetaMax.set! curLayer m
        i := next
    else if line = "LN2_GAMMA" then
      match consumeVectorMaxAbs lines (i + 1) d with
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
  let mut actDerivBoundMax : Rat := 0
  for l in [:L] do
    let ln1Max := ln1GammaMax[l]!
    let ln1MaxAbsBeta := ln1BetaMax[l]!
    let ln2Max := ln2GammaMax[l]!
    let ln1Var? : Option Rat := if l = 0 then inputVarLowerMin? else none
    let ln2Var? : Option Rat := none
    let ln1Bound :=
      match ln1Var? with
      | some v => layerNormOpBoundLocal ln1Max v eps soundnessBits
      | none => layerNormOpBoundConservative ln1Max eps soundnessBits
    let ln2Bound :=
      match ln2Var? with
      | some v => layerNormOpBoundLocal ln2Max v eps soundnessBits
      | none => layerNormOpBoundConservative ln2Max eps soundnessBits
    let ln1OutMaxAbsBound := layerNormOutputMaxAbsBound d ln1Max ln1MaxAbsBeta
    let attnValueCoeffLayer := attnValueCoeff[l]!
    let attnPatternCoeff :=
      attnPatternCoeffBound n d dh ln1OutMaxAbsBound (wqMax[l]!) (wkMax[l]!)
        attnValueCoeffLayer
    let mlpCoeff := mlpWin[l]! * mlpWout[l]!
    let mlpActDerivBound := actDerivBound
    let scoreAbsBound :=
      attnScoreAbsBound d dh ln1OutMaxAbsBound (wqMax[l]!) (wkMax[l]!)
    let (softmaxProbLo, softmaxProbHi) :=
      softmaxProbIntervalFromScoreAbsBound n scoreAbsBound softmaxExpEffort
    let softmaxIntervalBound := softmaxJacobianNormInfBound softmaxProbLo softmaxProbHi
    let softmaxMarginBound :=
      softmaxJacobianNormInfBoundFromMargin n softmaxMarginLowerBound softmaxExpEffort
    let softmaxBound := min softmaxIntervalBound softmaxMarginBound
    let attnW :=
      ln1Bound * ((n : Rat) * attnValueCoeffLayer + softmaxBound * attnPatternCoeff)
    let mlpW := ln2Bound * (mlpCoeff * mlpActDerivBound)
    let C := attnW + mlpW + attnW * mlpW
    layers := layers.push {
      layerIdx := l
      ln1MaxAbsGamma := ln1Max
      ln1MaxAbsBeta := ln1MaxAbsBeta
      ln2MaxAbsGamma := ln2Max
      ln1VarianceLowerBound? := ln1Var?
      ln2VarianceLowerBound? := ln2Var?
      ln1Bound := ln1Bound
      ln2Bound := ln2Bound
      ln1OutMaxAbsBound := ln1OutMaxAbsBound
      softmaxProbLo := softmaxProbLo
      softmaxProbHi := softmaxProbHi
      softmaxMarginLowerBound := softmaxMarginLowerBound
      softmaxExpEffort := softmaxExpEffort
      softmaxJacobianNormInfUpperBound := softmaxBound
      wqOpBoundMax := wqMax[l]!
      wkOpBoundMax := wkMax[l]!
      attnValueCoeff := attnValueCoeffLayer
      attnPatternCoeff := attnPatternCoeff
      mlpCoeff := mlpCoeff
      mlpWinBound := mlpWin[l]!
      mlpWoutBound := mlpWout[l]!
      mlpActDerivBound := mlpActDerivBound
      attnJacBound := attnW
      mlpJacBound := mlpW
      C := C
    }
    totalAmp := totalAmp * (1 + C)
    actDerivBoundMax := max actDerivBoundMax mlpActDerivBound
  let cert : ModelCert := {
    modelPath := path.toString
    inputPath? := inputPath?.map (·.toString)
    inputDelta := inputDelta
    eps := eps
    seqLen := n
    modelDim := d
    headDim := dh
    soundnessBits := soundnessBits
    geluDerivTarget := geluDerivTarget
    actDerivBound := actDerivBoundMax
    softmaxJacobianNormInfWorst := softmaxJacobianNormInfWorst
    layers := layers
    totalAmplificationFactor := totalAmp
  }
  if cert.check then
    return .ok cert
  return .error "sound certificate failed internal consistency checks"

/-- Parse input `EMBEDDINGS` from an `.nfpt` file and return intervals `xᵢ ∈ [xᵢ-δ, xᵢ+δ]`
as an array of rows (`seqLen` rows, each of length `modelDim`). -/
private def loadEmbeddingsIntervals
    (path : System.FilePath) (seqLen modelDim : Nat) (delta : Rat) :
    IO (Except String (Array (Array RatInterval))) := do
  let contents ← IO.FS.readFile path
  let lines : Array String := Nfp.Sound.splitLines contents
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

private structure LayerNormParamsFixed where
  gamma : Array Fixed10Interval
  beta : Array Fixed10Interval

private def intervalsFromScaled (xs : Array Int) (slack : Int) : Array Fixed10Interval :=
  xs.map (fun x => { lo := x - slack, hi := x + slack })

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

private def collectLayerNormParamsBinary
    (path : System.FilePath)
    (scalePow10 : Nat)
    (slack : Int) :
    IO
      (Except String
        (BinaryHeader × Array LayerNormParamsFixed × Array LayerNormParamsFixed)) := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  match ← readBinaryHeader h with
  | .error e => return .error e
  | .ok hdr =>
      match ← skipI32Array h hdr.seqLen with
      | .error e => return .error e
      | .ok _ => pure ()
      match ← skipF64Array h (hdr.seqLen * hdr.modelDim) with
      | .error e => return .error e
      | .ok _ => pure ()
      let defP : LayerNormParamsFixed := {
        gamma := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
        beta := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      }
      let mut ln1 : Array LayerNormParamsFixed := Array.replicate hdr.numLayers defP
      let mut ln2 : Array LayerNormParamsFixed := Array.replicate hdr.numLayers defP
      for l in [:hdr.numLayers] do
        for _h in [:hdr.numHeads] do
          match ← skipF64Array h (hdr.modelDim * hdr.headDim) with
          | .error e => return .error e
          | .ok _ => pure ()
          match ← skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          match ← skipF64Array h (hdr.modelDim * hdr.headDim) with
          | .error e => return .error e
          | .ok _ => pure ()
          match ← skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          match ← skipF64Array h (hdr.modelDim * hdr.headDim) with
          | .error e => return .error e
          | .ok _ => pure ()
          match ← skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          match ← skipF64Array h (hdr.headDim * hdr.modelDim) with
          | .error e => return .error e
          | .ok _ => pure ()
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h (hdr.modelDim * hdr.hiddenDim) with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h hdr.hiddenDim with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h (hdr.hiddenDim * hdr.modelDim) with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        let ln1GammaE ← readScaledFloatArray h hdr.modelDim scalePow10
        let ln1Gamma ←
          match ln1GammaE with
          | .error e => return .error e
          | .ok xs => pure (intervalsFromScaled xs slack)
        let ln1BetaE ← readScaledFloatArray h hdr.modelDim scalePow10
        let ln1Beta ←
          match ln1BetaE with
          | .error e => return .error e
          | .ok xs => pure (intervalsFromScaled xs slack)
        let ln2GammaE ← readScaledFloatArray h hdr.modelDim scalePow10
        let ln2Gamma ←
          match ln2GammaE with
          | .error e => return .error e
          | .ok xs => pure (intervalsFromScaled xs slack)
        let ln2BetaE ← readScaledFloatArray h hdr.modelDim scalePow10
        let ln2Beta ←
          match ln2BetaE with
          | .error e => return .error e
          | .ok xs => pure (intervalsFromScaled xs slack)
        ln1 := ln1.set! l { gamma := ln1Gamma, beta := ln1Beta }
        ln2 := ln2.set! l { gamma := ln2Gamma, beta := ln2Beta }
      return .ok (hdr, ln1, ln2)

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

private def ratFloorMulNat (x : Rat) (k : Nat) : Int :=
  let num : Int := x.num
  let den : Nat := x.den
  let numK : Int := num * (Int.ofNat k)
  numK.ediv (Int.ofNat den)

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
      return δSq / ((2 : Rat) * nRat)

private def absInt (x : Int) : Int := if x < 0 then -x else x

/-- Lower bound on variance using midpoint + radius deviation. -/
private def fixedVarianceLowerBoundMidpoint (cfg : Fixed10Cfg) (xs : Array Fixed10Interval) :
    Rat :=
  if xs.size < 2 then
    0
  else
    Id.run do
      let n : Nat := xs.size
      let nInt : Int := Int.ofNat n
      let d : Nat := 2 * cfg.scaleNat
      let mut sumM : Int := 0
      let mut sumR : Int := 0
      for x in xs do
        sumM := sumM + (x.lo + x.hi)
        sumR := sumR + (x.hi - x.lo)
      let mut varNum : Int := 0
      let mut errNum : Int := 0
      for x in xs do
        let mInt := x.lo + x.hi
        let rInt := x.hi - x.lo
        let aNum := nInt * mInt - sumM
        let rNum := nInt * rInt + sumR
        varNum := varNum + aNum * aNum
        errNum := errNum + (absInt aNum) * rNum
      let num := varNum - 2 * errNum
      if num <= 0 then
        return 0
      let denNat : Nat := d * d * n * n * n
      return (num : Rat) / (denNat : Rat)

/-- Exact variance lower bound by converting to `RatInterval` and using the exact routine. -/
private def fixedVarianceLowerBoundExact (cfg : Fixed10Cfg) (xs : Array Fixed10Interval) : Rat :=
  if xs.size < 2 then
    0
  else
    let ratXs :=
      xs.map (fun x => { lo := ratOfScaledInt cfg.scalePow10 x.lo,
                         hi := ratOfScaledInt cfg.scalePow10 x.hi })
    RatInterval.varianceLowerBound ratXs

/-- Best available variance lower bound from range + midpoint deviation. -/
private def fixedVarianceLowerBound (cfg : Fixed10Cfg) (xs : Array Fixed10Interval) : Rat :=
  let rangeLB := fixedVarianceLowerBoundRange cfg xs
  let midLB := fixedVarianceLowerBoundMidpoint cfg xs
  let approxLB := max rangeLB midLB
  -- Avoid the exact Rat-based bound on large rows (expensive and stack-heavy),
  -- but recover it when the fast bounds collapse to zero for medium sizes.
  if xs.size ≤ 256 then
    let exactLB := fixedVarianceLowerBoundExact cfg xs
    max approxLB exactLB
  else if approxLB = 0 && xs.size ≤ 1024 then
    let exactLB := fixedVarianceLowerBoundExact cfg xs
    max approxLB exactLB
  else
    approxLB

private def fixedLayerNormRowApprox
    (cfg : Fixed10Cfg)
    (row : Array Fixed10Interval)
    (gamma beta : Array Fixed10Interval)
    (eps : Rat)
    (soundnessBits : Nat) :
    (Array Fixed10Interval × Rat) :=
  if row.size = 0 || gamma.size ≠ row.size || beta.size ≠ row.size then
    (row, 0)
  else
    Id.run do
      let μ := fixedMeanInterval row
      let varLB := fixedVarianceLowerBound cfg row
      let invσUpper : Rat :=
        if varLB ≤ 0 then
          layerNormOpBoundConservative 1 eps soundnessBits
        else
          layerNormOpBoundLocal 1 varLB eps soundnessBits
      let invσUpperInt : Int := ratCeilMulNat invσUpper cfg.scaleNat
      let invσFix : Fixed10Interval := { lo := invσUpperInt, hi := invσUpperInt }
      let mut out : Array Fixed10Interval := Array.mkEmpty row.size
      for i in [:row.size] do
        let centered := Fixed10Interval.sub row[i]! μ
        let coeff := Fixed10Interval.mul cfg gamma[i]! invσFix
        let scaled := Fixed10Interval.mul cfg coeff centered
        out := out.push (Fixed10Interval.add scaled beta[i]!)
      return (out, varLB)

private def fixedLayerNormRowApproxExact
    (cfg : Fixed10Cfg)
    (row : Array Fixed10Interval)
    (gamma beta : Array Fixed10Interval)
    (eps : Rat)
    (soundnessBits : Nat) : Array Fixed10Interval :=
  if row.size = 0 || gamma.size ≠ row.size || beta.size ≠ row.size then
    row
  else
    Id.run do
      let μ := fixedMeanInterval row
      let varLB := fixedVarianceLowerBoundExact cfg row
      let invσUpper : Rat :=
        if varLB ≤ 0 then
          layerNormOpBoundConservative 1 eps soundnessBits
        else
          layerNormOpBoundLocal 1 varLB eps soundnessBits
      let invσUpperInt : Int := ratCeilMulNat invσUpper cfg.scaleNat
      let invσFix : Fixed10Interval := { lo := invσUpperInt, hi := invσUpperInt }
      let mut out : Array Fixed10Interval := Array.mkEmpty row.size
      for i in [:row.size] do
        let centered := Fixed10Interval.sub row[i]! μ
        let coeff := Fixed10Interval.mul cfg gamma[i]! invσFix
        let scaled := Fixed10Interval.mul cfg coeff centered
        out := out.push (Fixed10Interval.add scaled beta[i]!)
      return out

private def fixedLayerNormRowsApprox
    (cfg : Fixed10Cfg)
    (rows : Array (Array Fixed10Interval))
    (p : LayerNormParamsFixed)
    (eps : Rat)
    (soundnessBits : Nat) :
    Array (Array Fixed10Interval) :=
  let useTasks := rows.size > 32
  if useTasks then
    Id.run do
      let chunkSize : Nat := 16
      let numChunks : Nat := (rows.size + chunkSize - 1) / chunkSize
      let mut tasks : Array (Task (Array (Array Fixed10Interval))) := Array.mkEmpty numChunks
      let mut chunkIdx : Nat := 0
      while chunkIdx < numChunks do
        let start := chunkIdx * chunkSize
        let stop := min rows.size (start + chunkSize)
        tasks := tasks.push <|
          Task.spawn (fun _ =>
            Id.run do
              let mut outChunk : Array (Array Fixed10Interval) := Array.mkEmpty (stop - start)
              let mut i := start
              while i < stop do
                outChunk := outChunk.push
                  (fixedLayerNormRowApprox cfg rows[i]! p.gamma p.beta eps soundnessBits).1
                i := i + 1
              return outChunk)
        chunkIdx := chunkIdx + 1
      let mut out : Array (Array Fixed10Interval) := Array.mkEmpty rows.size
      for t in tasks do
        for row in t.get do
          out := out.push row
      return out
  else
    rows.map (fun row => (fixedLayerNormRowApprox cfg row p.gamma p.beta eps soundnessBits).1)

private def fixedLayerNormRowsApproxExact
    (cfg : Fixed10Cfg)
    (rows : Array (Array Fixed10Interval))
    (p : LayerNormParamsFixed)
    (eps : Rat)
    (soundnessBits : Nat) :
    Array (Array Fixed10Interval) :=
  let useTasks := rows.size > 32
  if useTasks then
    Id.run do
      let chunkSize : Nat := 16
      let numChunks : Nat := (rows.size + chunkSize - 1) / chunkSize
      let mut tasks : Array (Task (Array (Array Fixed10Interval))) := Array.mkEmpty numChunks
      let mut chunkIdx : Nat := 0
      while chunkIdx < numChunks do
        let start := chunkIdx * chunkSize
        let stop := min rows.size (start + chunkSize)
        tasks := tasks.push <|
          Task.spawn (fun _ =>
            Id.run do
              let mut outChunk : Array (Array Fixed10Interval) := Array.mkEmpty (stop - start)
              let mut i := start
              while i < stop do
                outChunk := outChunk.push
                  (fixedLayerNormRowApproxExact cfg rows[i]! p.gamma p.beta eps soundnessBits)
                i := i + 1
              return outChunk)
        chunkIdx := chunkIdx + 1
      let mut out : Array (Array Fixed10Interval) := Array.mkEmpty rows.size
      for t in tasks do
        for row in t.get do
          out := out.push row
      return out
  else
    rows.map (fun row => fixedLayerNormRowApproxExact cfg row p.gamma p.beta eps soundnessBits)

private def readVecIntervals
    (r : SoundCache.I32Reader) (n : Nat) (slack : Int) :
    IO (Array Fixed10Interval × SoundCache.I32Reader) := do
  let mut rr := r
  let mut out : Array Fixed10Interval := Array.mkEmpty n
  for _ in [:n] do
    let (x, rr2) ← Nfp.Untrusted.SoundCacheIO.I32Reader.readI32 rr
    rr := rr2
    out := out.push { lo := x - slack, hi := x + slack }
  return (out, rr)

private def readVecIntervalsBinary
    (h : IO.FS.Handle) (n : Nat) (slack : Int) (scalePow10 : Nat) :
    IO (Except String (Array Fixed10Interval)) := do
  match ← readScaledFloatArray h n scalePow10 with
  | .error e => return .error e
  | .ok xs => return .ok (intervalsFromScaled xs slack)

private def matMulIntervalsFromScaledCore
    (cfg : Fixed10Cfg)
    (slack : Int)
    (rows cols : Nat)
    (weights : Array Int)
    (input : Array Fixed10Interval) : Array Fixed10Interval :=
  Id.run do
    let mut out : Array Fixed10Interval := Array.replicate cols { lo := 0, hi := 0 }
    let mut rowIdx : Nat := 0
    while rowIdx < rows do
      let xi := input[rowIdx]!
      let mut colIdx : Nat := 0
      while colIdx < cols do
        let idx := rowIdx * cols + colIdx
        let w := weights[idx]!
        let wI : Fixed10Interval := { lo := w - slack, hi := w + slack }
        let term := Fixed10Interval.mul cfg wI xi
        out := out.set! colIdx (Fixed10Interval.add (out[colIdx]!) term)
        colIdx := colIdx + 1
      rowIdx := rowIdx + 1
    return out

private def matMulIntervalsFromScaledNoTask
    (cfg : Fixed10Cfg)
    (slack : Int)
    (rows cols : Nat)
    (weights : Array Int)
    (input : Array Fixed10Interval) : Array Fixed10Interval :=
  if input.size ≠ rows || weights.size ≠ rows * cols then
    Array.replicate cols { lo := 0, hi := 0 }
  else
    matMulIntervalsFromScaledCore cfg slack rows cols weights input

private def matMulIntervalsFromIntervalsNoTask
    (cfg : Fixed10Cfg)
    (rows cols : Nat)
    (weights : Array Fixed10Interval)
    (input : Array Fixed10Interval) : Array Fixed10Interval :=
  Id.run do
    if input.size ≠ rows || weights.size ≠ rows * cols then
      return Array.replicate cols { lo := 0, hi := 0 }
    let mut out : Array Fixed10Interval := Array.replicate cols { lo := 0, hi := 0 }
    let mut rowIdx : Nat := 0
    while rowIdx < rows do
      let xi := input[rowIdx]!
      let mut colIdx : Nat := 0
      while colIdx < cols do
        let idx := rowIdx * cols + colIdx
        let wI := weights[idx]!
        let term := Fixed10Interval.mul cfg wI xi
        out := out.set! colIdx (Fixed10Interval.add (out[colIdx]!) term)
        colIdx := colIdx + 1
      rowIdx := rowIdx + 1
    return out

private def matMulIntervalsFromIntervals
    (cfg : Fixed10Cfg)
    (rows cols : Nat)
    (weights : Array Fixed10Interval)
    (input : Array Fixed10Interval) : Array Fixed10Interval :=
  Id.run do
    if input.size ≠ rows || weights.size ≠ rows * cols then
      return Array.replicate cols { lo := 0, hi := 0 }
    let useTasks := rows * cols > 16384 && cols > 1
    if useTasks then
      let chunkSize : Nat := 32
      let numChunks : Nat := (cols + chunkSize - 1) / chunkSize
      let mut tasks : Array (Task (Array Fixed10Interval)) := Array.mkEmpty numChunks
      let mut chunkIdx : Nat := 0
      while chunkIdx < numChunks do
        let start := chunkIdx * chunkSize
        let stop := min cols (start + chunkSize)
        tasks := tasks.push <|
          Task.spawn (fun _ =>
            Id.run do
              let mut outChunk : Array Fixed10Interval := Array.mkEmpty (stop - start)
              let mut colIdx : Nat := start
              while colIdx < stop do
                let mut acc : Fixed10Interval := { lo := 0, hi := 0 }
                let mut rowIdx : Nat := 0
                while rowIdx < rows do
                  let xi := input[rowIdx]!
                  let idx := rowIdx * cols + colIdx
                  let wI := weights[idx]!
                  let term := Fixed10Interval.mul cfg wI xi
                  acc := Fixed10Interval.add acc term
                  rowIdx := rowIdx + 1
                outChunk := outChunk.push acc
                colIdx := colIdx + 1
              return outChunk)
        chunkIdx := chunkIdx + 1
      let mut out : Array Fixed10Interval := Array.mkEmpty cols
      for t in tasks do
        let chunk := t.get
        for v in chunk do
          out := out.push v
      return out
    else
      return matMulIntervalsFromIntervalsNoTask cfg rows cols weights input

private def matMulIntervalsFromScaled
    (cfg : Fixed10Cfg)
    (slack : Int)
    (rows cols : Nat)
    (weights : Array Int)
    (input : Array Fixed10Interval) : Array Fixed10Interval :=
  Id.run do
    if input.size ≠ rows || weights.size ≠ rows * cols then
      return Array.replicate cols { lo := 0, hi := 0 }
    let useTasks := rows * cols > 16384 && cols > 1
    if useTasks then
      let chunkSize : Nat := 32
      let numChunks : Nat := (cols + chunkSize - 1) / chunkSize
      let mut tasks : Array (Task (Array Fixed10Interval)) := Array.mkEmpty numChunks
      let mut chunkIdx : Nat := 0
      while chunkIdx < numChunks do
        let start := chunkIdx * chunkSize
        let stop := min cols (start + chunkSize)
        tasks := tasks.push <|
          Task.spawn (fun _ =>
            Id.run do
              let mut outChunk : Array Fixed10Interval := Array.mkEmpty (stop - start)
              let mut colIdx : Nat := start
              while colIdx < stop do
                let mut acc : Fixed10Interval := { lo := 0, hi := 0 }
                let mut rowIdx : Nat := 0
                while rowIdx < rows do
                  let xi := input[rowIdx]!
                  let idx := rowIdx * cols + colIdx
                  let w := weights[idx]!
                  let wI : Fixed10Interval := { lo := w - slack, hi := w + slack }
                  let term := Fixed10Interval.mul cfg wI xi
                  acc := Fixed10Interval.add acc term
                  rowIdx := rowIdx + 1
                outChunk := outChunk.push acc
                colIdx := colIdx + 1
              return outChunk)
        chunkIdx := chunkIdx + 1
      let mut out : Array Fixed10Interval := Array.mkEmpty cols
      for t in tasks do
        let chunk := t.get
        for v in chunk do
          out := out.push v
      return out
    else
      return matMulIntervalsFromScaledCore cfg slack rows cols weights input

private def fixedDotInterval
    (cfg : Fixed10Cfg)
    (a b : Array Fixed10Interval) : Fixed10Interval :=
  if a.size = 0 || a.size ≠ b.size then
    { lo := 0, hi := 0 }
  else
    Id.run do
      let mut acc : Fixed10Interval := { lo := 0, hi := 0 }
      for i in [:a.size] do
        let term := Fixed10Interval.mul cfg a[i]! b[i]!
        acc := Fixed10Interval.add acc term
      return acc

private def centerRadiusOfFixed
    (cfg : Fixed10Cfg) (a : Fixed10Interval) : Rat × Rat :=
  let lo := ratOfScaledInt cfg.scalePow10 a.lo
  let hi := ratOfScaledInt cfg.scalePow10 a.hi
  let center := (lo + hi) / (2 : Rat)
  let radius := (hi - lo) / (2 : Rat)
  (center, radius)

private def rowCentersRadiiAbs
    (cfg : Fixed10Cfg)
    (row : Array Fixed10Interval) : Array Rat × Array Rat × Rat :=
  Id.run do
    let mut centers : Array Rat := Array.mkEmpty row.size
    let mut radii : Array Rat := Array.mkEmpty row.size
    let mut absSum : Rat := 0
    for x in row do
      let lo := ratOfScaledInt cfg.scalePow10 x.lo
      let hi := ratOfScaledInt cfg.scalePow10 x.hi
      let center := (lo + hi) / (2 : Rat)
      let radius := (hi - lo) / (2 : Rat)
      centers := centers.push center
      radii := radii.push radius
      absSum := absSum + max (ratAbs lo) (ratAbs hi)
    return (centers, radii, absSum)

private def weightsRatFromScaled (cfg : Fixed10Cfg) (weights : Array Int) : Array Rat :=
  weights.map (ratOfScaledInt cfg.scalePow10)

private def affineMatMulRowExact
    (rows cols : Nat)
    (weights : Array Rat)
    (centers radii : Array Rat) : Array AffineForm :=
  Id.run do
    if centers.size ≠ rows || radii.size ≠ rows || weights.size ≠ rows * cols then
      return Array.replicate cols (AffineForm.const 0)
    let mut out : Array AffineForm := Array.mkEmpty cols
    for colIdx in [:cols] do
      let mut center : Rat := 0
      let mut coeffs : Array Rat := Array.mkEmpty rows
      for rowIdx in [:rows] do
        let idx := rowIdx * cols + colIdx
        let w := weights[idx]!
        center := center + w * centers[rowIdx]!
        coeffs := coeffs.push (w * radii[rowIdx]!)
      out := out.push { center := center, coeffs := coeffs }
    return out

private def affineAddBiasCenters
    (biasCenters : Array Rat)
    (row : Array AffineForm) : Array AffineForm :=
  Id.run do
    if biasCenters.size ≠ row.size then
      return row
    let mut out : Array AffineForm := Array.mkEmpty row.size
    for i in [:row.size] do
      let a := row.getD i (AffineForm.const 0)
      let bias := biasCenters.getD i 0
      out := out.push { a with center := a.center + bias }
    return out

private def affineAbsSum (row : Array AffineForm) : Rat :=
  row.foldl (fun acc a => acc + ratAbs a.center + AffineForm.radius a) 0

private def affineDotDisjoint
    (a b : Array AffineForm) : AffineForm :=
  if a.size = 0 || a.size ≠ b.size then
    AffineForm.const 0
  else
    Id.run do
      let mut acc := AffineForm.const 0
      for i in [:a.size] do
        let ai := a.getD i (AffineForm.const 0)
        let bi := b.getD i (AffineForm.const 0)
        let term := AffineForm.mulDisjoint ai bi
        acc := AffineForm.add acc term
      return acc

private def sumRat (xs : Array Rat) : Rat :=
  Id.run do
    let mut acc : Rat := 0
    let mut i := 0
    while i < xs.size do
      acc := acc + xs[i]!
      i := i + 1
    return acc

private def sumAbsRat (xs : Array Rat) : Rat :=
  Id.run do
    let mut acc : Rat := 0
    let mut i := 0
    while i < xs.size do
      acc := acc + ratAbs xs[i]!
      i := i + 1
    return acc

private def addVecRat (a b : Array Rat) : Array Rat :=
  Id.run do
    if a.size ≠ b.size then
      return a
    let mut out : Array Rat := Array.mkEmpty a.size
    let mut i := 0
    while i < a.size do
      out := out.push (a[i]! + b[i]!)
      i := i + 1
    return out

private def dotRat (a b : Array Rat) : Rat :=
  if a.size = 0 || a.size ≠ b.size then
    0
  else
    Id.run do
      let mut acc : Rat := 0
      let mut i := 0
      while i < a.size do
        acc := acc + a[i]! * b[i]!
        i := i + 1
      return acc

private def matMulCentersRadii
    (rows cols : Nat)
    (weights : Array Rat)
    (centers radii : Array Rat) : Array Rat × Array Rat :=
  Id.run do
    if centers.size ≠ rows || radii.size ≠ rows || weights.size ≠ rows * cols then
      return (Array.replicate cols 0, Array.replicate cols 0)
    let mut outCenters : Array Rat := Array.mkEmpty cols
    let mut outRadii : Array Rat := Array.mkEmpty cols
    let mut colIdx := 0
    while colIdx < cols do
      let mut center : Rat := 0
      let mut radius : Rat := 0
      let mut rowIdx := 0
      while rowIdx < rows do
        let idx := rowIdx * cols + colIdx
        let w := weights.getD idx 0
        let c := centers.getD rowIdx 0
        let r := radii.getD rowIdx 0
        center := center + w * c
        radius := radius + ratAbs w * r
        rowIdx := rowIdx + 1
      outCenters := outCenters.push center
      outRadii := outRadii.push radius
      colIdx := colIdx + 1
    return (outCenters, outRadii)

private def coeffSumFromCenters
    (rows cols : Nat)
    (weights : Array Rat)
    (inputRadii : Array Rat)
    (otherCenters : Array Rat) : Rat :=
  if inputRadii.size ≠ rows || otherCenters.size ≠ cols || weights.size ≠ rows * cols then
    0
  else
    Id.run do
      let mut acc : Rat := 0
      let mut rowIdx := 0
      while rowIdx < rows do
        let mut sum : Rat := 0
        let mut colIdx := 0
        while colIdx < cols do
          let idx := rowIdx * cols + colIdx
          sum := sum + weights.getD idx 0 * otherCenters.getD colIdx 0
          colIdx := colIdx + 1
        let coeff := inputRadii.getD rowIdx 0 * sum
        acc := acc + ratAbs coeff
        rowIdx := rowIdx + 1
      return acc

private def sumInt (xs : Array Int) : Int :=
  Id.run do
    let mut acc : Int := 0
    let mut i := 0
    while i < xs.size do
      acc := acc + xs[i]!
      i := i + 1
    return acc

private def sumAbsInt (xs : Array Int) : Int :=
  Id.run do
    let mut acc : Int := 0
    let mut i := 0
    while i < xs.size do
      acc := acc + absInt xs[i]!
      i := i + 1
    return acc

private def addVecScaledInt (a : Array Int) (b : Array Int) (scale : Int) : Array Int :=
  Id.run do
    if a.size ≠ b.size then
      return a
    let mut out : Array Int := Array.mkEmpty a.size
    let mut i := 0
    while i < a.size do
      out := out.push (a[i]! + b[i]! * scale)
      i := i + 1
    return out

private def dotInt (a b : Array Int) : Int :=
  if a.size = 0 || a.size ≠ b.size then
    0
  else
    Id.run do
      let mut acc : Int := 0
      let mut i := 0
      while i < a.size do
        acc := acc + a[i]! * b[i]!
        i := i + 1
      return acc

private def rowCentersRadiiAbsInt
    (row : Array Fixed10Interval) : Array Int × Array Int × Int :=
  Id.run do
    let mut centers : Array Int := Array.mkEmpty row.size
    let mut radii : Array Int := Array.mkEmpty row.size
    let mut absSum : Int := 0
    for x in row do
      let sum := x.lo + x.hi
      let width := x.hi - x.lo
      let center := sum.ediv (Int.ofNat 2)
      let half := width.ediv (Int.ofNat 2)
      let radius := if width.emod (Int.ofNat 2) = 0 then half else half + 1
      centers := centers.push center
      radii := radii.push radius
      absSum := absSum + Fixed10Interval.absUpper x
    return (centers, radii, absSum)

private def matMulCentersRadiiInt
    (cfg : Fixed10Cfg)
    (rows cols : Nat)
    (weights : Array Int)
    (centers radii : Array Int) : Array Int × Array Int :=
  Id.run do
    if centers.size ≠ rows || radii.size ≠ rows || weights.size ≠ rows * cols then
      return (Array.replicate cols 0, Array.replicate cols 0)
    let mut outCenters : Array Int := Array.mkEmpty cols
    let mut outRadii : Array Int := Array.mkEmpty cols
    let mut colIdx := 0
    while colIdx < cols do
      let mut centerI : Fixed10Interval := { lo := 0, hi := 0 }
      let mut radiusAcc : Int := 0
      let mut rowIdx := 0
      while rowIdx < rows do
        let idx := rowIdx * cols + colIdx
        let w := weights.getD idx 0
        let c := centers.getD rowIdx 0
        let r := radii.getD rowIdx 0
        let term := Fixed10Interval.mul cfg { lo := w, hi := w } { lo := c, hi := c }
        centerI := Fixed10Interval.add centerI term
        if r ≠ 0 && w ≠ 0 then
          let wAbs := absInt w
          let termR := Fixed10Interval.mul cfg { lo := wAbs, hi := wAbs } { lo := r, hi := r }
          radiusAcc := radiusAcc + termR.hi
        rowIdx := rowIdx + 1
      let width := centerI.hi - centerI.lo
      let center := (centerI.lo + centerI.hi).ediv (Int.ofNat 2)
      let half := width.ediv (Int.ofNat 2)
      let radiusMid := if width.emod (Int.ofNat 2) = 0 then half else half + 1
      let radius := radiusMid + radiusAcc
      outCenters := outCenters.push center
      outRadii := outRadii.push radius
      colIdx := colIdx + 1
    return (outCenters, outRadii)

private def intervalRadiusInt (x : Fixed10Interval) : Int :=
  let width := x.hi - x.lo
  let half := width.ediv (Int.ofNat 2)
  if width.emod (Int.ofNat 2) = 0 then half else half + 1

private def matMulCentersRadiiIntSlack
    (cfg : Fixed10Cfg)
    (slack : Int)
    (rows cols : Nat)
    (weights : Array Int)
    (centers radii : Array Int) : Array Int × Array Int :=
  Id.run do
    if centers.size ≠ rows || radii.size ≠ rows || weights.size ≠ rows * cols then
      return (Array.replicate cols 0, Array.replicate cols 0)
    let mut outCenters : Array Int := Array.mkEmpty cols
    let mut outRadii : Array Int := Array.mkEmpty cols
    let mut colIdx := 0
    while colIdx < cols do
      let mut centerI : Fixed10Interval := { lo := 0, hi := 0 }
      let mut radiusAcc : Int := 0
      let mut rowIdx := 0
      while rowIdx < rows do
        let idx := rowIdx * cols + colIdx
        let w := weights.getD idx 0
        let c := centers.getD rowIdx 0
        let r := radii.getD rowIdx 0
        let term := Fixed10Interval.mul cfg { lo := w, hi := w } { lo := c, hi := c }
        centerI := Fixed10Interval.add centerI term
        if r ≠ 0 || slack ≠ 0 then
          let wAbs := absInt w
          let cAbs := absInt c
          let term1 := Fixed10Interval.mul cfg { lo := wAbs, hi := wAbs } { lo := r, hi := r }
          let term2 :=
            if slack = 0 then 0
            else
              (Fixed10Interval.mul cfg { lo := slack, hi := slack }
                { lo := cAbs, hi := cAbs }).hi
          let term3 :=
            if slack = 0 then 0
            else
              (Fixed10Interval.mul cfg { lo := slack, hi := slack } { lo := r, hi := r }).hi
          radiusAcc := radiusAcc + term1.hi + term2 + term3
        rowIdx := rowIdx + 1
      let width := centerI.hi - centerI.lo
      let center := (centerI.lo + centerI.hi).ediv (Int.ofNat 2)
      let half := width.ediv (Int.ofNat 2)
      let radiusMid := if width.emod (Int.ofNat 2) = 0 then half else half + 1
      let radius := radiusMid + radiusAcc
      outCenters := outCenters.push center
      outRadii := outRadii.push radius
      colIdx := colIdx + 1
    return (outCenters, outRadii)

private def coeffSumFromCentersInt
    (cfg : Fixed10Cfg)
    (rows cols : Nat)
    (weights : Array Int)
    (inputRadii : Array Int)
    (otherCenters : Array Int) : Int :=
  if inputRadii.size ≠ rows || otherCenters.size ≠ cols || weights.size ≠ rows * cols then
    0
  else
    Id.run do
      let mut acc : Int := 0
      let mut rowIdx := 0
      while rowIdx < rows do
        let mut sum : Fixed10Interval := { lo := 0, hi := 0 }
        let mut colIdx := 0
        while colIdx < cols do
          let idx := rowIdx * cols + colIdx
          let w := weights.getD idx 0
          let c := otherCenters.getD colIdx 0
          let term := Fixed10Interval.mul cfg { lo := w, hi := w } { lo := c, hi := c }
          sum := Fixed10Interval.add sum term
          colIdx := colIdx + 1
        let r := inputRadii.getD rowIdx 0
        let coeff := Fixed10Interval.mul cfg sum { lo := r, hi := r }
        acc := acc + Fixed10Interval.absUpper coeff
        rowIdx := rowIdx + 1
      return acc

private def dotIntervalFromCentersInt
    (cfg : Fixed10Cfg)
    (a b : Array Int) : Fixed10Interval :=
  if a.size = 0 || a.size ≠ b.size then
    { lo := 0, hi := 0 }
  else
    Id.run do
      let mut acc : Fixed10Interval := { lo := 0, hi := 0 }
      let mut i := 0
      while i < a.size do
        let term := Fixed10Interval.mul cfg
          { lo := a[i]!, hi := a[i]! }
          { lo := b[i]!, hi := b[i]! }
        acc := Fixed10Interval.add acc term
        i := i + 1
      return acc

private def dotIntervalFromCentersRadiiInt
    (cfg : Fixed10Cfg)
    (aCenters aRadii bCenters bRadii : Array Int) : Fixed10Interval :=
  if aCenters.size = 0 || aCenters.size ≠ bCenters.size ||
      aCenters.size ≠ aRadii.size || bCenters.size ≠ bRadii.size then
    { lo := 0, hi := 0 }
  else
    Id.run do
      let mut centerI : Fixed10Interval := { lo := 0, hi := 0 }
      let mut radiusAcc : Int := 0
      let mut i := 0
      while i < aCenters.size do
        let ac := aCenters[i]!
        let ar := aRadii[i]!
        let bc := bCenters[i]!
        let br := bRadii[i]!
        let term := Fixed10Interval.mul cfg { lo := ac, hi := ac } { lo := bc, hi := bc }
        centerI := Fixed10Interval.add centerI term
        if ar ≠ 0 || br ≠ 0 then
          let acAbs := absInt ac
          let bcAbs := absInt bc
          let term1 := Fixed10Interval.mul cfg { lo := acAbs, hi := acAbs } { lo := br, hi := br }
          let term2 := Fixed10Interval.mul cfg { lo := bcAbs, hi := bcAbs } { lo := ar, hi := ar }
          let term3 := Fixed10Interval.mul cfg { lo := ar, hi := ar } { lo := br, hi := br }
          radiusAcc := radiusAcc + term1.hi + term2.hi + term3.hi
        i := i + 1
      let width := centerI.hi - centerI.lo
      let center := (centerI.lo + centerI.hi).ediv (Int.ofNat 2)
      let half := width.ediv (Int.ofNat 2)
      let radiusMid := if width.emod (Int.ofNat 2) = 0 then half else half + 1
      let radius := radiusMid + radiusAcc
      return { lo := center - radius, hi := center + radius }

private def sumMulUpperInt
    (cfg : Fixed10Cfg)
    (a b : Array Int) : Int :=
  if a.size = 0 || a.size ≠ b.size then
    0
  else
    Id.run do
      let mut acc : Int := 0
      let mut i := 0
      while i < a.size do
        let term := Fixed10Interval.mul cfg
          { lo := a[i]!, hi := a[i]! }
          { lo := b[i]!, hi := b[i]! }
        acc := acc + term.hi
        i := i + 1
      return acc

private def floorDivNat (a : Int) (d : Nat) : Int :=
  a.ediv (Int.ofNat d)

private def ceilDivNat (a : Int) (d : Nat) : Int :=
  let di : Int := Int.ofNat d
  let q := a.ediv di
  let r := a.emod di
  if r = 0 then q else q + 1

private def maxAbsVecFixed (xs : Array Fixed10Interval) : Int :=
  xs.foldl (fun acc x => max acc (Fixed10Interval.absUpper x)) 0

/-- Sum of per-coordinate centered absolute bounds (interval widths), as a `Rat`. -/
private def centeredAbsSumFixed (cfg : Fixed10Cfg) (xs : Array Fixed10Interval) : Rat :=
  let sumWidth : Int := xs.foldl (fun acc x => acc + Fixed10Interval.centeredAbsBound x) 0
  Rat.normalize sumWidth cfg.scaleNat (den_nz := by
    have h10pos : (0 : Nat) < 10 := by decide
    exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))

private def ratIntervalOfFixed (cfg : Fixed10Cfg) (a : Fixed10Interval) : RatInterval :=
  { lo := ratOfScaledInt cfg.scalePow10 a.lo, hi := ratOfScaledInt cfg.scalePow10 a.hi }

private def fixedIntervalOfRat (cfg : Fixed10Cfg) (a : RatInterval) : Fixed10Interval :=
  { lo := ratFloorMulNat a.lo cfg.scaleNat, hi := ratCeilMulNat a.hi cfg.scaleNat }

private def defaultGeluExpEffort : Nat := 2
private def defaultGeluSplitDepth : Nat := 1

private def geluOverapproxRat (target : GeluDerivTarget) (a : RatInterval) : RatInterval :=
  match target with
  | .tanh => RatInterval.geluOverapproxTanhSplit a defaultGeluExpEffort defaultGeluSplitDepth
  | .exact => RatInterval.geluOverapprox a

private def geluOverapproxFixed (cfg : Fixed10Cfg) (target : GeluDerivTarget)
    (a : Fixed10Interval) : Fixed10Interval :=
  match target with
  | .tanh =>
      let r := ratIntervalOfFixed cfg a
      fixedIntervalOfRat cfg
        (RatInterval.geluOverapproxTanhSplit r defaultGeluExpEffort defaultGeluSplitDepth)
  | .exact =>
      Fixed10Interval.geluOverapprox a

private def geluOverapproxFixedVec (cfg : Fixed10Cfg) (target : GeluDerivTarget)
    (xs : Array Fixed10Interval) : Array Fixed10Interval :=
  Id.run do
    let mut out : Array Fixed10Interval := Array.mkEmpty xs.size
    let mut i : Nat := 0
    while i < xs.size do
      out := out.push (geluOverapproxFixed cfg target xs[i]!)
      i := i + 1
    return out

private def geluOverapproxFixedVecLinear
    (xs : Array Fixed10Interval) : Array Fixed10Interval :=
  Id.run do
    let mut out : Array Fixed10Interval := Array.mkEmpty xs.size
    let mut i : Nat := 0
    while i < xs.size do
      out := out.push (Fixed10Interval.geluOverapprox xs[i]!)
      i := i + 1
    return out

private def addVecFixed (a b : Array Fixed10Interval) : Array Fixed10Interval :=
  Id.run do
    if a.size ≠ b.size then
      return a
    let mut out : Array Fixed10Interval := Array.mkEmpty a.size
    let mut i : Nat := 0
    while i < a.size do
      out := out.push (Fixed10Interval.add a[i]! b[i]!)
      i := i + 1
    return out

private def addVecFixedRows
    (rows : Array (Array Fixed10Interval))
    (v : Array Fixed10Interval) : Array (Array Fixed10Interval) :=
  Id.run do
    let mut out : Array (Array Fixed10Interval) := Array.mkEmpty rows.size
    let mut i : Nat := 0
    while i < rows.size do
      out := out.push (addVecFixed rows[i]! v)
      i := i + 1
    return out

private def addRowsFixed
    (rows : Array (Array Fixed10Interval))
    (adds : Array (Array Fixed10Interval)) : Array (Array Fixed10Interval) :=
  Id.run do
    if rows.size ≠ adds.size then
      return rows
    let mut out : Array (Array Fixed10Interval) := Array.mkEmpty rows.size
    let mut i : Nat := 0
    while i < rows.size do
      out := out.push (addVecFixed rows[i]! adds[i]!)
      i := i + 1
    return out

private def takePrefix {α : Type} (xs : Array α) (n : Nat) : Array α :=
  if xs.size ≤ n then xs else xs.extract 0 n

private def mlpRowFromScaled
    (cfg : Fixed10Cfg)
    (geluDerivTarget : GeluDerivTarget)
    (slack : Int)
    (modelDim hiddenDim : Nat)
    (wIn wOut : Array Int)
    (bIn bOut : Array Fixed10Interval)
    (row : Array Fixed10Interval) : Array Fixed10Interval :=
  let hidden0 := matMulIntervalsFromScaled cfg slack modelDim hiddenDim wIn row
  let hiddenB := addVecFixed hidden0 bIn
  let actHidden := geluOverapproxFixedVec cfg geluDerivTarget hiddenB
  let mlpOut0 := matMulIntervalsFromScaled cfg slack hiddenDim modelDim wOut actHidden
  addVecFixed mlpOut0 bOut

private def mlpRowFromScaledNoTask
    (cfg : Fixed10Cfg)
    (geluDerivTarget : GeluDerivTarget)
    (slack : Int)
    (modelDim hiddenDim : Nat)
    (wIn wOut : Array Int)
    (bIn bOut : Array Fixed10Interval)
    (row : Array Fixed10Interval) : Array Fixed10Interval :=
  let hidden0 := matMulIntervalsFromScaledNoTask cfg slack modelDim hiddenDim wIn row
  let hiddenB := addVecFixed hidden0 bIn
  let actHidden := geluOverapproxFixedVec cfg geluDerivTarget hiddenB
  let mlpOut0 := matMulIntervalsFromScaledNoTask cfg slack hiddenDim modelDim wOut actHidden
  addVecFixed mlpOut0 bOut

/-- Linear GeLU-hull MLP row used to avoid the tanh/exp path in hot loops. -/
private def mlpRowFromScaledLinear
    (cfg : Fixed10Cfg)
    (slack : Int)
    (modelDim hiddenDim : Nat)
    (wIn wOut : Array Int)
    (bIn bOut : Array Fixed10Interval)
    (row : Array Fixed10Interval) : Array Fixed10Interval :=
  let hidden0 := matMulIntervalsFromScaled cfg slack modelDim hiddenDim wIn row
  let hiddenB := addVecFixed hidden0 bIn
  let actHidden := geluOverapproxFixedVecLinear hiddenB
  let mlpOut0 := matMulIntervalsFromScaled cfg slack hiddenDim modelDim wOut actHidden
  addVecFixed mlpOut0 bOut

private def mlpRowFromScaledLinearNoTask
    (cfg : Fixed10Cfg)
    (slack : Int)
    (modelDim hiddenDim : Nat)
    (wIn wOut : Array Int)
    (bIn bOut : Array Fixed10Interval)
    (row : Array Fixed10Interval) : Array Fixed10Interval :=
  let hidden0 := matMulIntervalsFromScaledNoTask cfg slack modelDim hiddenDim wIn row
  let hiddenB := addVecFixed hidden0 bIn
  let actHidden := geluOverapproxFixedVecLinear hiddenB
  let mlpOut0 := matMulIntervalsFromScaledNoTask cfg slack hiddenDim modelDim wOut actHidden
  addVecFixed mlpOut0 bOut

private def mlpRowFromIntervalsNoTask
    (cfg : Fixed10Cfg)
    (geluDerivTarget : GeluDerivTarget)
    (modelDim hiddenDim : Nat)
    (wIn wOut : Array Fixed10Interval)
    (bIn bOut : Array Fixed10Interval)
    (row : Array Fixed10Interval) : Array Fixed10Interval :=
  let hidden0 := matMulIntervalsFromIntervalsNoTask cfg modelDim hiddenDim wIn row
  let hiddenB := addVecFixed hidden0 bIn
  let actHidden := geluOverapproxFixedVec cfg geluDerivTarget hiddenB
  let mlpOut0 := matMulIntervalsFromIntervalsNoTask cfg hiddenDim modelDim wOut actHidden
  addVecFixed mlpOut0 bOut

private def mlpRowFromIntervals
    (cfg : Fixed10Cfg)
    (geluDerivTarget : GeluDerivTarget)
    (modelDim hiddenDim : Nat)
    (wIn wOut : Array Fixed10Interval)
    (bIn bOut : Array Fixed10Interval)
    (row : Array Fixed10Interval) : Array Fixed10Interval :=
  let hidden0 := matMulIntervalsFromIntervals cfg modelDim hiddenDim wIn row
  let hiddenB := addVecFixed hidden0 bIn
  let actHidden := geluOverapproxFixedVec cfg geluDerivTarget hiddenB
  let mlpOut0 := matMulIntervalsFromIntervals cfg hiddenDim modelDim wOut actHidden
  addVecFixed mlpOut0 bOut

private def mlpRowsFromIntervals
    (cfg : Fixed10Cfg)
    (geluDerivTarget : GeluDerivTarget)
    (modelDim hiddenDim : Nat)
    (wIn wOut : Array Fixed10Interval)
    (bIn bOut : Array Fixed10Interval)
    (rows : Array (Array Fixed10Interval)) : Array (Array Fixed10Interval) :=
  let useTasks := rows.size > 32
  if useTasks then
    Id.run do
      let chunkSize : Nat := 16
      let numChunks : Nat := (rows.size + chunkSize - 1) / chunkSize
      let mut tasks : Array (Task (Array (Array Fixed10Interval))) := Array.mkEmpty numChunks
      let mut chunkIdx : Nat := 0
      while chunkIdx < numChunks do
        let start := chunkIdx * chunkSize
        let stop := min rows.size (start + chunkSize)
        tasks := tasks.push <|
          Task.spawn (fun _ =>
            Id.run do
              let mut outChunk : Array (Array Fixed10Interval) := Array.mkEmpty (stop - start)
              let mut i := start
              while i < stop do
                outChunk := outChunk.push
                  (mlpRowFromIntervalsNoTask cfg geluDerivTarget modelDim hiddenDim wIn wOut bIn
                    bOut rows[i]!)
                i := i + 1
              return outChunk)
        chunkIdx := chunkIdx + 1
      let mut out : Array (Array Fixed10Interval) := Array.mkEmpty rows.size
      for t in tasks do
        for row in t.get do
          out := out.push row
      return out
  else
    rows.map (mlpRowFromIntervalsNoTask cfg geluDerivTarget modelDim hiddenDim wIn wOut bIn bOut)

private def mlpRowsFromScaled
    (cfg : Fixed10Cfg)
    (geluDerivTarget : GeluDerivTarget)
    (slack : Int)
    (modelDim hiddenDim : Nat)
    (wIn wOut : Array Int)
    (bIn bOut : Array Fixed10Interval)
    (rows : Array (Array Fixed10Interval)) : Array (Array Fixed10Interval) :=
  let useTasks := rows.size > 32
  if useTasks then
    Id.run do
      let chunkSize : Nat := 16
      let numChunks : Nat := (rows.size + chunkSize - 1) / chunkSize
      let mut tasks : Array (Task (Array (Array Fixed10Interval))) := Array.mkEmpty numChunks
      let mut chunkIdx : Nat := 0
      while chunkIdx < numChunks do
        let start := chunkIdx * chunkSize
        let stop := min rows.size (start + chunkSize)
        tasks := tasks.push <|
          Task.spawn (fun _ =>
            Id.run do
              let mut outChunk : Array (Array Fixed10Interval) := Array.mkEmpty (stop - start)
              let mut i := start
              while i < stop do
                outChunk := outChunk.push
                  (mlpRowFromScaledNoTask cfg geluDerivTarget slack modelDim hiddenDim wIn wOut bIn
                    bOut rows[i]!)
                i := i + 1
              return outChunk)
        chunkIdx := chunkIdx + 1
      let mut out : Array (Array Fixed10Interval) := Array.mkEmpty rows.size
      for t in tasks do
        for row in t.get do
          out := out.push row
      return out
  else
    rows.map (mlpRowFromScaledNoTask cfg geluDerivTarget slack modelDim hiddenDim wIn wOut bIn bOut)

/-- Linear GeLU-hull per-row MLP for best-match induction hot paths. -/
private def mlpRowsFromScaledLinear
    (cfg : Fixed10Cfg)
    (slack : Int)
    (modelDim hiddenDim : Nat)
    (wIn wOut : Array Int)
    (bIn bOut : Array Fixed10Interval)
    (rows : Array (Array Fixed10Interval)) : Array (Array Fixed10Interval) :=
  let wInIntervals := intervalsFromScaled wIn slack
  let wOutIntervals := intervalsFromScaled wOut slack
  let mlpRowFromIntervals (row : Array Fixed10Interval) : Array Fixed10Interval :=
    let hidden0 := matMulIntervalsFromIntervalsNoTask cfg modelDim hiddenDim wInIntervals row
    let hiddenB := addVecFixed hidden0 bIn
    let actHidden := geluOverapproxFixedVecLinear hiddenB
    let mlpOut0 := matMulIntervalsFromIntervalsNoTask cfg hiddenDim modelDim wOutIntervals actHidden
    addVecFixed mlpOut0 bOut
  let useTasks := rows.size > 32
  if useTasks then
    Id.run do
      let chunkSize : Nat := 16
      let numChunks : Nat := (rows.size + chunkSize - 1) / chunkSize
      let mut tasks : Array (Task (Array (Array Fixed10Interval))) := Array.mkEmpty numChunks
      let mut chunkIdx : Nat := 0
      while chunkIdx < numChunks do
        let start := chunkIdx * chunkSize
        let stop := min rows.size (start + chunkSize)
        tasks := tasks.push <|
          Task.spawn (fun _ =>
            Id.run do
              let mut outChunk : Array (Array Fixed10Interval) := Array.mkEmpty (stop - start)
              let mut i := start
              while i < stop do
                outChunk := outChunk.push (mlpRowFromIntervals rows[i]!)
                i := i + 1
              return outChunk)
        chunkIdx := chunkIdx + 1
      let mut out : Array (Array Fixed10Interval) := Array.mkEmpty rows.size
      for t in tasks do
        for row in t.get do
          out := out.push row
      return out
  else
    rows.map mlpRowFromIntervals

private def groupUnionRowsByToken
    (rows : Array (Array Fixed10Interval))
    (tokens : Array Int) : Array (Array Fixed10Interval) :=
  Id.run do
    if rows.size ≠ tokens.size then
      return rows
    let mut uniqTokens : Array Int := #[]
    let mut uniqRows : Array (Array Fixed10Interval) := #[]
    let mut i : Nat := 0
    while i < rows.size do
      let tok := tokens[i]!
      match uniqTokens.findIdx? (· == tok) with
      | some idx =>
          let merged := Fixed10Interval.unionVec (uniqRows[idx]!) rows[i]!
          uniqRows := uniqRows.set! idx merged
      | none =>
          uniqTokens := uniqTokens.push tok
          uniqRows := uniqRows.push rows[i]!
      i := i + 1
    return uniqRows

private def unionRowsFixed
    (rows : Array (Array Fixed10Interval)) : Array Fixed10Interval :=
  if rows.isEmpty then
    #[]
  else
    Id.run do
      let mut out := rows[0]!
      let mut i : Nat := 1
      while i < rows.size do
        let row := rows[i]!
        if row.size = out.size then
          let mut j : Nat := 0
          while j < out.size do
            let cur := out[j]!
            let r := row[j]!
            out := out.set! j { lo := min cur.lo r.lo, hi := max cur.hi r.hi }
            j := j + 1
        i := i + 1
      return out

private def prefixUnionRowsFixed
    (rows : Array (Array Fixed10Interval)) : Array (Array Fixed10Interval) :=
  if rows.isEmpty then
    #[]
  else
    Id.run do
      let mut out : Array (Array Fixed10Interval) := Array.mkEmpty rows.size
      let mut acc := rows[0]!
      out := out.push acc
      let mut i : Nat := 1
      while i < rows.size do
        acc := Fixed10Interval.unionVec acc rows[i]!
        out := out.push acc
        i := i + 1
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
  let mut rowIdx : Nat := 0
  while rowIdx < rows do
    let xi := input[rowIdx]!
    let mut colIdx : Nat := 0
    while colIdx < cols do
      let (w, rr2) ← Nfp.Untrusted.SoundCacheIO.I32Reader.readI32 rr
      rr := rr2
      let wAbsBound : Int := (if w < 0 then -w else w) + slack
      curRowAbs := curRowAbs + wAbsBound
      let wI : Fixed10Interval := { lo := w - slack, hi := w + slack }
      let term := Fixed10Interval.mul cfg wI xi
      out := out.set! colIdx (Fixed10Interval.add (out[colIdx]!) term)
      colIdx := colIdx + 1
    maxRowAbs := max maxRowAbs curRowAbs
    curRowAbs := 0
    rowIdx := rowIdx + 1
  let normInf : Rat :=
    Rat.normalize maxRowAbs cfg.scaleNat (den_nz := by
      have h10pos : (0 : Nat) < 10 := by decide
      exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
  return (out, normInf, rr)

private def consumeMatrixMulAndNormInfFixedBinary
    (cfg : Fixed10Cfg)
    (slack : Int)
    (h : IO.FS.Handle)
    (rows cols : Nat)
    (input : Array Fixed10Interval)
    (scalePow10 : Nat) :
    IO (Except String (Array Fixed10Interval × Rat)) := do
  if input.size ≠ rows then
    match ← skipF64Array h (rows * cols) with
    | .error e => return .error e
    | .ok _ => return .ok (Array.replicate cols { lo := 0, hi := 0 }, 0)
  match ← readScaledFloatArray h (rows * cols) scalePow10 with
  | .error e => return .error e
  | .ok vals =>
      let mut out : Array Fixed10Interval := Array.replicate cols { lo := 0, hi := 0 }
      let mut curRowAbs : Int := 0
      let mut maxRowAbs : Int := 0
      let mut rowIdx : Nat := 0
      while rowIdx < rows do
        let xi := input[rowIdx]!
        let mut colIdx : Nat := 0
        while colIdx < cols do
          let idx := rowIdx * cols + colIdx
          let w := vals[idx]!
          let wAbsBound : Int := (if w < 0 then -w else w) + slack
          curRowAbs := curRowAbs + wAbsBound
          let wI : Fixed10Interval := { lo := w - slack, hi := w + slack }
          let term := Fixed10Interval.mul cfg wI xi
          out := out.set! colIdx (Fixed10Interval.add (out[colIdx]!) term)
          colIdx := colIdx + 1
        maxRowAbs := max maxRowAbs curRowAbs
        curRowAbs := 0
        rowIdx := rowIdx + 1
      let normInf : Rat :=
        Rat.normalize maxRowAbs cfg.scaleNat (den_nz := by
          have h10pos : (0 : Nat) < 10 := by decide
          exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
      return .ok (out, normInf)

private def consumeMatrixMulFixedBinaryStreaming
    (cfg : Fixed10Cfg)
    (slack : Int)
    (h : IO.FS.Handle)
    (rows cols : Nat)
    (input : Array Fixed10Interval)
    (scalePow10 : Nat) :
    IO (Except String (Array Fixed10Interval)) := do
  if input.size ≠ rows then
    match ← skipF64Array h (rows * cols) with
    | .error e => return .error e
    | .ok _ => return .ok (Array.replicate cols { lo := 0, hi := 0 })
  let mut out : Array Fixed10Interval := Array.replicate cols { lo := 0, hi := 0 }
  let mut rowIdx : Nat := 0
  while rowIdx < rows do
    let rowWeightsE ← readScaledFloatArray h cols scalePow10
    match rowWeightsE with
    | .error e => return .error e
    | .ok rowWeights =>
        let xi := input[rowIdx]!
        let mut colIdx : Nat := 0
        while colIdx < cols do
          let w := rowWeights[colIdx]!
          let wI : Fixed10Interval := { lo := w - slack, hi := w + slack }
          let term := Fixed10Interval.mul cfg wI xi
          out := out.set! colIdx (Fixed10Interval.add (out[colIdx]!) term)
          colIdx := colIdx + 1
    rowIdx := rowIdx + 1
  return .ok out

/-- Apply union-MLP propagation for binary bounds using streaming matmul. -/
private def mlpUnionStepBinary
    (cfg : Fixed10Cfg)
    (slack : Int)
    (h : IO.FS.Handle)
    (modelDim hiddenDim : Nat)
    (ln2Rows : Array (Array Fixed10Interval))
    (residuals : Array (Array Fixed10Interval))
    (scalePow10 : Nat) :
    IO (Except String (Array (Array Fixed10Interval))) := do
  let ln2Union := unionRowsFixed ln2Rows
  let hidden0E ←
    consumeMatrixMulFixedBinaryStreaming cfg slack h modelDim hiddenDim ln2Union scalePow10
  match hidden0E with
  | .error e => return .error e
  | .ok hidden0 =>
      let bInE ← readVecIntervalsBinary h hiddenDim slack scalePow10
      match bInE with
      | .error e => return .error e
      | .ok bIn =>
          let hiddenB := addVecFixed hidden0 bIn
          -- Linear GeLU hull keeps the union path fast and avoids heavy tanh bounds.
          let actHidden := geluOverapproxFixedVecLinear hiddenB
          let mut mlpOut0 : Array Fixed10Interval :=
            Array.replicate modelDim { lo := 0, hi := 0 }
          let mut rowIdx : Nat := 0
          while rowIdx < hiddenDim do
            let rowWeightsE ← readScaledFloatArray h modelDim scalePow10
            match rowWeightsE with
            | .error e => return .error e
            | .ok rowWeights =>
                let xi := actHidden[rowIdx]!
                let mut colIdx : Nat := 0
                while colIdx < modelDim do
                  let w := rowWeights[colIdx]!
                  let wI : Fixed10Interval := { lo := w - slack, hi := w + slack }
                  let term := Fixed10Interval.mul cfg wI xi
                  mlpOut0 := mlpOut0.set! colIdx
                    (Fixed10Interval.add (mlpOut0[colIdx]!) term)
                  colIdx := colIdx + 1
            rowIdx := rowIdx + 1
          let bOutE ← readVecIntervalsBinary h modelDim slack scalePow10
          match bOutE with
          | .error e => return .error e
          | .ok bOut =>
              let mlpOut := addVecFixed mlpOut0 bOut
              let residuals' := addVecFixedRows residuals mlpOut
              return .ok residuals'

private def loadEmbeddingsUnionFixed
    (cfg : Fixed10Cfg)
    (path : System.FilePath)
    (expectedModelDim : Nat)
    (delta : Rat) : IO (Except String (Array Fixed10Interval × Nat)) := do
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
  return .ok (out, n)

/-- Parse binary embeddings into a union-box of fixed-point intervals. -/
private def loadEmbeddingsUnionFixedBinary
    (path : System.FilePath)
    (expectedModelDim : Nat)
    (delta : Rat)
    (scalePow10 : Nat := defaultBinaryScalePow10) :
    IO (Except String (Array Fixed10Interval)) := do
  if delta < 0 then
    return .error "delta must be nonnegative"
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  match ← readBinaryHeader h with
  | .error e => return .error e
  | .ok hdr =>
      if hdr.modelDim ≠ expectedModelDim then
        return .error
          s!"input model_dim mismatch (expected {expectedModelDim}, got {hdr.modelDim})"
      let total := hdr.seqLen * hdr.modelDim
      let deltaScaled : Int := ratCeilMulNat delta (Nat.pow 10 scalePow10)
      match ← skipI32Array h hdr.seqLen with
      | .error e => return .error e
      | .ok _ => pure ()
      match ← readScaledFloatArray h total scalePow10 with
      | .error e => return .error e
      | .ok scaled =>
          if total = 0 then
            return .ok #[]
          let mut out : Array Fixed10Interval := Array.mkEmpty hdr.modelDim
          for col in [:hdr.modelDim] do
            let v := scaled[col]!
            out := out.push { lo := v - deltaScaled, hi := v + deltaScaled }
          for i in [hdr.modelDim:total] do
            let v := scaled[i]!
            let col := i % hdr.modelDim
            let lo := v - deltaScaled
            let hi := v + deltaScaled
            let cur := out[col]!
            out := out.set! col { lo := min cur.lo lo, hi := max cur.hi hi }
          return .ok out

/-- Parse binary embeddings into per-position fixed-point intervals. -/
private def loadEmbeddingsIntervalsBinary
    (path : System.FilePath)
    (expectedModelDim : Nat)
    (delta : Rat)
    (scalePow10 : Nat := defaultBinaryScalePow10) :
    IO (Except String (Array (Array Fixed10Interval))) := do
  if delta < 0 then
    return .error "delta must be nonnegative"
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  match ← readBinaryHeader h with
  | .error e => return .error e
  | .ok hdr =>
      if hdr.modelDim ≠ expectedModelDim then
        return .error
          s!"input model_dim mismatch (expected {expectedModelDim}, got {hdr.modelDim})"
      let total := hdr.seqLen * hdr.modelDim
      let deltaScaled : Int := ratCeilMulNat delta (Nat.pow 10 scalePow10)
      match ← skipI32Array h hdr.seqLen with
      | .error e => return .error e
      | .ok _ => pure ()
      match ← readScaledFloatArray h total scalePow10 with
      | .error e => return .error e
      | .ok scaled =>
          if total = 0 then
            return .ok #[]
          let useTasks := hdr.seqLen > 32
          if useTasks then
            let chunkSize : Nat := 16
            let numChunks : Nat := (hdr.seqLen + chunkSize - 1) / chunkSize
            let mut tasks :
                Array (Task (Array (Array Fixed10Interval))) := Array.mkEmpty numChunks
            let mut chunkIdx : Nat := 0
            while chunkIdx < numChunks do
              let start := chunkIdx * chunkSize
              let stop := min hdr.seqLen (start + chunkSize)
              tasks := tasks.push <|
                Task.spawn (fun _ =>
                  Id.run do
                    let mut rowsChunk :
                        Array (Array Fixed10Interval) := Array.mkEmpty (stop - start)
                    let mut rowIdx := start
                    while rowIdx < stop do
                      let mut row : Array Fixed10Interval := Array.mkEmpty hdr.modelDim
                      for colIdx in [:hdr.modelDim] do
                        let idx := rowIdx * hdr.modelDim + colIdx
                        let v := scaled[idx]!
                        row := row.push { lo := v - deltaScaled, hi := v + deltaScaled }
                      rowsChunk := rowsChunk.push row
                      rowIdx := rowIdx + 1
                    return rowsChunk)
              chunkIdx := chunkIdx + 1
            let mut rows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
            for t in tasks do
              for row in t.get do
                rows := rows.push row
            return .ok rows
          else
            let mut rows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
            for rowIdx in [:hdr.seqLen] do
              let mut row : Array Fixed10Interval := Array.mkEmpty hdr.modelDim
              for colIdx in [:hdr.modelDim] do
                let idx := rowIdx * hdr.modelDim + colIdx
                let v := scaled[idx]!
                row := row.push { lo := v - deltaScaled, hi := v + deltaScaled }
              rows := rows.push row
            return .ok rows

private def loadTokensBinary
    (path : System.FilePath) : IO (Except String (BinaryHeader × Array Int)) := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  match ← readBinaryHeader h with
  | .error e => return .error e
  | .ok hdr =>
      match ← readI32Array h hdr.seqLen with
      | .error e => return .error e
      | .ok toks => return .ok (hdr, toks)

/-- Shared binary inputs for repeated local bound checks. -/
private structure SharedBinaryInputs where
  hdr : BinaryHeader
  ln1Params : Array LayerNormParamsFixed
  ln2Params : Array LayerNormParamsFixed
  tokens : Array Int
  residuals0 : Array (Array Fixed10Interval)
  inputDelta : Rat
  scalePow10 : Nat

/-- Cached prefix views for a fixed query position. -/
private structure SharedBinaryPrefix where
  seqLenEff : Nat
  residuals : Thunk (Array (Array Fixed10Interval))
  tokens : Thunk (Array Int)

/-- Load shared model/input data once for reuse across best-match configs. -/
private def loadSharedBinaryInputs
    (path : System.FilePath)
    (inputPath : System.FilePath)
    (inputDelta : Rat)
    (scalePow10 : Nat) :
    IO (Except String SharedBinaryInputs) := do
  let slack : Int := fixedUlpSlack
  let action : ExceptT String IO SharedBinaryInputs := do
    let paramsTask ←
      ExceptT.lift <| IO.asTask (collectLayerNormParamsBinary path scalePow10 slack)
    let tokensTask ←
      ExceptT.lift <| IO.asTask (loadTokensBinary inputPath)
    let (hdr, ln1Params, ln2Params) ←
      match paramsTask.get with
      | .error e => throw (toString e)
      | .ok (.error msg) => throw msg
      | .ok (.ok v) => pure v
    let (hdrTok, tokens) ←
      match tokensTask.get with
      | .error e => throw (toString e)
      | .ok (.error msg) => throw msg
      | .ok (.ok v) => pure v
    let residuals0 ←
      ExceptT.mk
        (loadEmbeddingsIntervalsBinary inputPath hdr.modelDim inputDelta scalePow10)
    if hdrTok.seqLen ≠ hdr.seqLen then
      throw "token/embedding seq_len mismatch"
    return {
      hdr := hdr
      ln1Params := ln1Params
      ln2Params := ln2Params
      tokens := tokens
      residuals0 := residuals0
      inputDelta := inputDelta
      scalePow10 := scalePow10
    }
  action.run

/-- Build cached prefix arrays for a fixed query position. -/
private def mkSharedBinaryPrefix
    (shared : SharedBinaryInputs)
    (queryPos : Nat)
    (causalPattern : Bool) :
    SharedBinaryPrefix :=
  let seqLenEff : Nat := if causalPattern then queryPos + 1 else shared.hdr.seqLen
  {
    seqLenEff := seqLenEff
    residuals := Thunk.mk (fun () =>
      if causalPattern then takePrefix shared.residuals0 seqLenEff else shared.residuals0)
    tokens := Thunk.mk (fun () =>
      if causalPattern then takePrefix shared.tokens seqLenEff else shared.tokens)
  }

private def skipToUnembeddingBinary
    (h : IO.FS.Handle) (hdr : BinaryHeader) : IO (Except String Unit) := do
  let action : ExceptT String IO Unit := do
    let _ ← ExceptT.mk (skipI32Array h hdr.seqLen)
    let _ ← ExceptT.mk (skipF64Array h (hdr.seqLen * hdr.modelDim))
    for _l in [:hdr.numLayers] do
      for _h in [:hdr.numHeads] do
        let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
        let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
        let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
        let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
        let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
        let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
        let _ ← ExceptT.mk (skipF64Array h (hdr.headDim * hdr.modelDim))
      let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
      let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.hiddenDim))
      let _ ← ExceptT.mk (skipF64Array h hdr.hiddenDim)
      let _ ← ExceptT.mk (skipF64Array h (hdr.hiddenDim * hdr.modelDim))
      let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
      let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
      let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
      let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
      let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
    let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
    let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
  action.run

/-- Compute local head output lower bounds at a specific query position (binary only). -/
private def certifyHeadValueLowerBoundLocalBinaryAt
    (path : System.FilePath)
    (layerIdx headIdx queryPos coord : Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath : System.FilePath)
    (inputDelta : Rat)
    (targetOffset : Int)
    (keyOffset : Int)
    (matchWeightLowerBound : Rat)
    (maxSeqLen : Nat)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (causalPattern : Bool := true)
    (shared? : Option SharedBinaryInputs := none)
    (prefix? : Option SharedBinaryPrefix := none) :
    IO (Except String HeadValueLowerBoundPosCert) := do
  let cfg : Fixed10Cfg := scaleCfgOfPow10 scalePow10
  let slack : Int := fixedUlpSlack
  let action : ExceptT String IO HeadValueLowerBoundPosCert := do
    let (hdr, ln1Params, ln2Params, residualsBase, tokensBase) ←
      match shared? with
      | some shared =>
          if shared.scalePow10 ≠ scalePow10 then
            throw "shared scalePow10 mismatch"
          if shared.inputDelta ≠ inputDelta then
            throw "shared inputDelta mismatch"
          pure (shared.hdr, shared.ln1Params, shared.ln2Params, shared.residuals0, shared.tokens)
      | none =>
          let (hdr, ln1Params, ln2Params) ←
            ExceptT.mk (collectLayerNormParamsBinary path scalePow10 slack)
          let residuals0 ←
            ExceptT.mk
              (loadEmbeddingsIntervalsBinary inputPath hdr.modelDim inputDelta scalePow10)
          let (hdrTok, tokens) ← ExceptT.mk (loadTokensBinary inputPath)
          if hdrTok.seqLen ≠ hdr.seqLen then
            throw "token/embedding seq_len mismatch"
          pure (hdr, ln1Params, ln2Params, residuals0, tokens)
    if layerIdx ≥ hdr.numLayers then
      throw s!"layer index {layerIdx} out of range"
    if headIdx ≥ hdr.numHeads then
      throw s!"head index {headIdx} out of range"
    if coord ≥ hdr.modelDim then
      throw s!"coord index {coord} out of range"
    if hdr.seqLen > maxSeqLen then
      throw s!"seq_len {hdr.seqLen} exceeds maxSeqLen {maxSeqLen}"
    if queryPos ≥ hdr.seqLen then
      throw s!"queryPos {queryPos} out of range"
    let seqLenEff : Nat := if causalPattern then queryPos + 1 else hdr.seqLen
    let (residuals0, tokens) ←
      match prefix? with
      | some pref =>
          if pref.seqLenEff ≠ seqLenEff then
            throw "prefix seq_len mismatch"
          pure (pref.residuals.get, pref.tokens.get)
      | none =>
          let residuals0 :=
            if causalPattern then takePrefix residualsBase seqLenEff else residualsBase
          let tokens := if causalPattern then takePrefix tokensBase seqLenEff else tokensBase
          pure (residuals0, tokens)
    let keyOffsetNat? : Option Nat :=
      if keyOffset ≥ 0 then some (Int.toNat keyOffset) else none
    let keyOffsetNeg : Nat := Int.toNat (-keyOffset)
    let h ← ExceptT.lift <| IO.FS.Handle.mk path IO.FS.Mode.read
    let _ ← ExceptT.mk (readBinaryHeader h)
    let _ ← ExceptT.mk (skipI32Array h hdr.seqLen)
    let _ ← ExceptT.mk (skipF64Array h (hdr.seqLen * hdr.modelDim))
    let defP : LayerNormParamsFixed := {
      gamma := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      beta := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
    }
    let mut residuals := residuals0
    for l in [:hdr.numLayers] do
      let p1 := ln1Params.getD l defP
      let ln1Rows := fixedLayerNormRowsApprox cfg residuals p1 eps soundnessBits
      if l = layerIdx then
        let mut wv? : Option (Array Int) := none
        let mut bv? : Option (Array Int) := none
        let mut wo? : Option (Array Int) := none
        for hIdx in [:hdr.numHeads] do
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          if hIdx = headIdx then
            let wv ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
            wv? := some wv
            let bV ←
              ExceptT.mk <|
                readScaledFloatArray h hdr.headDim scalePow10
            bv? := some bV
            let wo ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
            wo? := some wo
          else
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.headDim * hdr.modelDim))
        let wv ←
          match wv? with
          | none => throw "missing W_V for requested head"
          | some xs => pure xs
        let bV ←
          match bv? with
          | none => throw "missing b_V for requested head"
          | some xs => pure xs
        let wo ←
          match wo? with
          | none => throw "missing W_O for requested head"
          | some xs => pure xs
        let bVIntervals := intervalsFromScaled bV slack
        let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty seqLenEff
        for row in ln1Rows do
          let vHidden0 := matMulIntervalsFromScaled cfg slack
            hdr.modelDim hdr.headDim wv row
          let vHidden := addVecFixed vHidden0 bVIntervals
          let vOut := matMulIntervalsFromScaled cfg slack
            hdr.headDim hdr.modelDim wo vHidden
          vOutRows := vOutRows.push vOut
        let ti : Int := (Int.ofNat queryPos) + targetOffset
        if ti < 0 || ti ≥ (Int.ofNat seqLenEff) then
          throw "query position has no valid target offset"
        let tIdx : Nat := Int.toNat ti
        let targetTok := tokens[tIdx]!
        let mut matchLo? : Option Int := none
        let mut nonmatchLo? : Option Int := none
        for j in [:seqLenEff] do
          if !causalPattern || j ≤ queryPos then
            let row := vOutRows[j]!
            let vCoord := row[coord]!.lo
            let isMatch : Bool :=
              match keyOffsetNat? with
              | some k =>
                  let idx := j + k
                  idx < seqLenEff && tokens[idx]! = targetTok
              | none =>
                  if j < keyOffsetNeg then
                    false
                  else
                    tokens[j - keyOffsetNeg]! = targetTok
            if isMatch then
              matchLo? :=
                match matchLo? with
                | none => some vCoord
                | some m => some (min m vCoord)
            else
              nonmatchLo? :=
                match nonmatchLo? with
                | none => some vCoord
                | some m => some (min m vCoord)
          else
            pure ()
        let matchLo ←
          match matchLo? with
          | none => throw "no matching keys for the requested offset"
          | some v => pure v
        let nonmatchLo :=
          match nonmatchLo? with
          | none => matchLo
          | some v => v
        let matchLoRat := ratOfScaledInt scalePow10 matchLo
        let nonmatchLoRat := ratOfScaledInt scalePow10 nonmatchLo
        let weightLB := matchWeightLowerBound
        let outputLB := mixLowerBound weightLB matchLoRat nonmatchLoRat
        let cert : HeadValueLowerBoundPosCert := {
          layerIdx := layerIdx
          headIdx := headIdx
          queryPos := queryPos
          coord := coord
          matchWeightLowerBound := weightLB
          matchCoordLowerBound := matchLoRat
          nonmatchCoordLowerBound := nonmatchLoRat
          outputCoordLowerBound := outputLB
        }
        if cert.check then
          return cert
        throw "head value lower bound (pos) failed internal consistency checks"
      else
        let tightLayers : Nat :=
          if tightPattern then Nat.max 1 tightPatternLayers else 0
        if tightLayers > 0 && layerIdx ≤ l + tightLayers then
          if causalPattern then
            let zeroRow : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let mut attnRows : Array (Array Fixed10Interval) :=
              Array.replicate hdr.seqLen zeroRow
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
              let mut rowIdx : Nat := 0
              while rowIdx < ln1Rows.size do
                let row := ln1Rows[rowIdx]!
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
                rowIdx := rowIdx + 1
              let headRows := prefixUnionRowsFixed vOutRows
              attnRows := addRowsFixed attnRows headRows
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnRows := addVecFixedRows attnRows attnBias
            residuals := addRowsFixed residuals attnRows
          else
            let mut attnUnion : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let groupRows := groupUnionRowsByToken ln1Rows tokens
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty groupRows.size
              let mut rowIdx : Nat := 0
              while rowIdx < groupRows.size do
                let row := groupRows[rowIdx]!
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
                rowIdx := rowIdx + 1
              let vUnion := unionRowsFixed vOutRows
              attnUnion := addVecFixed attnUnion vUnion
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnUnion := addVecFixed attnUnion attnBias
            residuals := addVecFixedRows residuals attnUnion
        else
          let ln1Union := unionRowsFixed ln1Rows
          let mut attnUnion : Array Fixed10Interval :=
            Array.replicate hdr.modelDim { lo := 0, hi := 0 }
          for _h in [:hdr.numHeads] do
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let wv ←
              ExceptT.mk <| readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
            let vHidden0 := matMulIntervalsFromScaled cfg slack
              hdr.modelDim hdr.headDim wv ln1Union
            let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
            let vHidden := addVecFixed vHidden0 bV
            let wo ←
              ExceptT.mk <| readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
            let vOut := matMulIntervalsFromScaled cfg slack
              hdr.headDim hdr.modelDim wo vHidden
            attnUnion := addVecFixed attnUnion vOut
          let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          attnUnion := addVecFixed attnUnion attnBias
          residuals := addVecFixedRows residuals attnUnion
        let p2 := ln2Params.getD l defP
        let ln2Rows := fixedLayerNormRowsApprox cfg residuals p2 eps soundnessBits
        let perRowLayers : Nat := perRowPatternLayers
        if perRowLayers > 0 && layerIdx ≤ l + perRowLayers then
          let wIn ←
            ExceptT.mk <| readScaledFloatArray h (hdr.modelDim * hdr.hiddenDim) scalePow10
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let wOut ←
            ExceptT.mk <| readScaledFloatArray h (hdr.hiddenDim * hdr.modelDim) scalePow10
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpRows :=
            mlpRowsFromScaled cfg hdr.geluDerivTarget slack
              hdr.modelDim hdr.hiddenDim wIn wOut bIn bOut ln2Rows
          residuals := addRowsFixed residuals mlpRows
        else
          let ln2Union := unionRowsFixed ln2Rows
          let (hidden0, _nWin) ←
            ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
              hdr.modelDim hdr.hiddenDim ln2Union scalePow10)
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let hiddenB := addVecFixed hidden0 bIn
          let actHidden := geluOverapproxFixedVec cfg hdr.geluDerivTarget hiddenB
          let (mlpOut0, _nWout) ←
            ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
              hdr.hiddenDim hdr.modelDim actHidden scalePow10)
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpOut := addVecFixed mlpOut0 bOut
          residuals := addVecFixedRows residuals mlpOut
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
    throw "target layer not reached"
  action.run

/-- Combined value + optional logit certs for a single query position (binary only). -/
private structure HeadValueLogitCert where
  value : HeadValueLowerBoundPosCert
  logit? : Option HeadLogitDiffLowerBoundPosCert

/-- Compute value and optional logit bounds for a head at a query position (binary only). -/
private def certifyHeadValueLogitLowerBoundLocalBinaryAt
    (path : System.FilePath)
    (layerIdx headIdx queryPos coord : Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath : System.FilePath)
    (inputDelta : Rat)
    (targetOffset : Int)
    (keyOffset : Int)
    (matchWeightLowerBound : Rat)
    (maxSeqLen : Nat)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (causalPattern : Bool := true)
    (shared? : Option SharedBinaryInputs := none)
    (prefix? : Option SharedBinaryPrefix := none)
    (targetToken? : Option Nat := none)
    (negativeToken? : Option Nat := none)
    (direction? : Option (Thunk (Array Fixed10Interval)) := none) :
    IO (Except String HeadValueLogitCert) := do
  let cfg : Fixed10Cfg := scaleCfgOfPow10 scalePow10
  let slack : Int := fixedUlpSlack
  let action : ExceptT String IO HeadValueLogitCert := do
    let (hdr, ln1Params, ln2Params, residualsBase, tokensBase) ←
      match shared? with
      | some shared =>
          if shared.scalePow10 ≠ scalePow10 then
            throw "shared scalePow10 mismatch"
          if shared.inputDelta ≠ inputDelta then
            throw "shared inputDelta mismatch"
          pure (shared.hdr, shared.ln1Params, shared.ln2Params, shared.residuals0, shared.tokens)
      | none =>
          let (hdr, ln1Params, ln2Params) ←
            ExceptT.mk (collectLayerNormParamsBinary path scalePow10 slack)
          let residuals0 ←
            ExceptT.mk
              (loadEmbeddingsIntervalsBinary inputPath hdr.modelDim inputDelta scalePow10)
          let (hdrTok, tokens) ← ExceptT.mk (loadTokensBinary inputPath)
          if hdrTok.seqLen ≠ hdr.seqLen then
            throw "token/embedding seq_len mismatch"
          pure (hdr, ln1Params, ln2Params, residuals0, tokens)
    if layerIdx ≥ hdr.numLayers then
      throw s!"layer index {layerIdx} out of range"
    if headIdx ≥ hdr.numHeads then
      throw s!"head index {headIdx} out of range"
    if coord ≥ hdr.modelDim then
      throw s!"coord index {coord} out of range"
    if hdr.seqLen > maxSeqLen then
      throw s!"seq_len {hdr.seqLen} exceeds maxSeqLen {maxSeqLen}"
    if queryPos ≥ hdr.seqLen then
      throw s!"queryPos {queryPos} out of range"
    let seqLenEff : Nat := if causalPattern then queryPos + 1 else hdr.seqLen
    let (residuals0, tokens) ←
      match prefix? with
      | some pref =>
          if pref.seqLenEff ≠ seqLenEff then
            throw "prefix seq_len mismatch"
          pure (pref.residuals.get, pref.tokens.get)
      | none =>
          let residuals0 :=
            if causalPattern then takePrefix residualsBase seqLenEff else residualsBase
          let tokens := if causalPattern then takePrefix tokensBase seqLenEff else tokensBase
          pure (residuals0, tokens)
    let keyOffsetNat? : Option Nat :=
      if keyOffset ≥ 0 then some (Int.toNat keyOffset) else none
    let keyOffsetNeg : Nat := Int.toNat (-keyOffset)
    let h ← IO.FS.Handle.mk path IO.FS.Mode.read
    let _ ← ExceptT.mk (readBinaryHeader h)
    let _ ← ExceptT.mk (skipI32Array h hdr.seqLen)
    let _ ← ExceptT.mk (skipF64Array h (hdr.seqLen * hdr.modelDim))
    let defP : LayerNormParamsFixed := {
      gamma := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      beta := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
    }
    let mut residuals := residuals0
    for l in [:hdr.numLayers] do
      let p1 := ln1Params.getD l defP
      let ln1Rows := fixedLayerNormRowsApprox cfg residuals p1 eps soundnessBits
      if l = layerIdx then
        let mut wv? : Option (Array Int) := none
        let mut bv? : Option (Array Int) := none
        let mut wo? : Option (Array Int) := none
        for hIdx in [:hdr.numHeads] do
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          if hIdx = headIdx then
            let wv ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
            wv? := some wv
            let bV ←
              ExceptT.mk <|
                readScaledFloatArray h hdr.headDim scalePow10
            bv? := some bV
            let wo ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
            wo? := some wo
          else
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.headDim * hdr.modelDim))
        let wv ←
          match wv? with
          | none => throw "missing W_V for requested head"
          | some xs => pure xs
        let bV ←
          match bv? with
          | none => throw "missing b_V for requested head"
          | some xs => pure xs
        let wo ←
          match wo? with
          | none => throw "missing W_O for requested head"
          | some xs => pure xs
        let bVIntervals := intervalsFromScaled bV slack
        let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty seqLenEff
        for row in ln1Rows do
          let vHidden0 := matMulIntervalsFromScaled cfg slack
            hdr.modelDim hdr.headDim wv row
          let vHidden := addVecFixed vHidden0 bVIntervals
          let vOut := matMulIntervalsFromScaled cfg slack
            hdr.headDim hdr.modelDim wo vHidden
          vOutRows := vOutRows.push vOut
        let ti : Int := (Int.ofNat queryPos) + targetOffset
        if ti < 0 || ti ≥ (Int.ofNat seqLenEff) then
          throw "query position has no valid target offset"
        let tIdx : Nat := Int.toNat ti
        let targetTok := tokens[tIdx]!
        let mut matchLo? : Option Int := none
        let mut nonmatchLo? : Option Int := none
        for j in [:seqLenEff] do
          if !causalPattern || j ≤ queryPos then
            let row := vOutRows[j]!
            let vCoord := row[coord]!.lo
            let isMatch : Bool :=
              match keyOffsetNat? with
              | some k =>
                  let idx := j + k
                  idx < seqLenEff && tokens[idx]! = targetTok
              | none =>
                  if j < keyOffsetNeg then
                    false
                  else
                    tokens[j - keyOffsetNeg]! = targetTok
            if isMatch then
              matchLo? :=
                match matchLo? with
                | none => some vCoord
                | some m => some (min m vCoord)
            else
              nonmatchLo? :=
                match nonmatchLo? with
                | none => some vCoord
                | some m => some (min m vCoord)
          else
            pure ()
        let matchLo ←
          match matchLo? with
          | none => throw "no matching keys for the requested offset"
          | some v => pure v
        let nonmatchLo :=
          match nonmatchLo? with
          | none => matchLo
          | some v => v
        let matchLoRat := ratOfScaledInt scalePow10 matchLo
        let nonmatchLoRat := ratOfScaledInt scalePow10 nonmatchLo
        let outputLB := mixLowerBound matchWeightLowerBound matchLoRat nonmatchLoRat
        let value : HeadValueLowerBoundPosCert := {
          layerIdx := layerIdx
          headIdx := headIdx
          queryPos := queryPos
          coord := coord
          matchWeightLowerBound := matchWeightLowerBound
          matchCoordLowerBound := matchLoRat
          nonmatchCoordLowerBound := nonmatchLoRat
          outputCoordLowerBound := outputLB
        }
        if !value.check then
          throw "head value certificate failed internal consistency checks"
        let logit? ←
          match targetToken?, negativeToken?, direction? with
          | none, none, none => pure none
          | some targetToken, some negativeToken, some direction => do
              let dir := direction.get
              if dir.size ≠ hdr.modelDim then
                throw "logit direction size mismatch"
              let vDotRows :=
                let useTasks := vOutRows.size > 32
                if useTasks then
                  let tasks := vOutRows.map (fun row =>
                    Task.spawn (fun _ => fixedDotInterval cfg row dir))
                  tasks.map (fun t => t.get)
                else
                  Id.run do
                    let mut out : Array Fixed10Interval := Array.mkEmpty seqLenEff
                    for row in vOutRows do
                      out := out.push (fixedDotInterval cfg row dir)
                    return out
              let mut matchLoLogit? : Option Int := none
              let mut nonmatchLoLogit? : Option Int := none
              for j in [:seqLenEff] do
                if !causalPattern || j ≤ queryPos then
                  let vLo := (vDotRows[j]!).lo
                  let isMatch : Bool :=
                    match keyOffsetNat? with
                    | some k =>
                        let idx := j + k
                        idx < seqLenEff && tokens[idx]! = targetTok
                    | none =>
                        if j < keyOffsetNeg then
                          false
                        else
                          tokens[j - keyOffsetNeg]! = targetTok
                  if isMatch then
                    matchLoLogit? :=
                      match matchLoLogit? with
                      | none => some vLo
                      | some m => some (min m vLo)
                  else
                    nonmatchLoLogit? :=
                      match nonmatchLoLogit? with
                      | none => some vLo
                      | some m => some (min m vLo)
                else
                  pure ()
              let matchLoLogit ←
                match matchLoLogit? with
                | none => throw "no matching keys for the requested offset"
                | some v => pure v
              let nonmatchLoLogit :=
                match nonmatchLoLogit? with
                | none => matchLoLogit
                | some v => v
              let matchLoRat := ratOfScaledInt scalePow10 matchLoLogit
              let nonmatchLoRat := ratOfScaledInt scalePow10 nonmatchLoLogit
              let logitLB := mixLowerBound matchWeightLowerBound matchLoRat nonmatchLoRat
              let logitCert : HeadLogitDiffLowerBoundPosCert := {
                layerIdx := layerIdx
                headIdx := headIdx
                queryPos := queryPos
                targetToken := targetToken
                negativeToken := negativeToken
                matchWeightLowerBound := matchWeightLowerBound
                matchLogitLowerBound := matchLoRat
                nonmatchLogitLowerBound := nonmatchLoRat
                logitDiffLowerBound := logitLB
              }
              if logitCert.check then
                pure (some logitCert)
              else
                throw "head logit certificate failed internal consistency checks"
          | _, _, _ =>
              throw "use both target and negative tokens (or neither)"
        return { value := value, logit? := logit? }
      else
        let tightLayers : Nat :=
          if tightPattern then Nat.max 1 tightPatternLayers else 0
        if tightLayers > 0 && layerIdx ≤ l + tightLayers then
          if causalPattern then
            let zeroRow : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let mut attnRows : Array (Array Fixed10Interval) :=
              Array.replicate hdr.seqLen zeroRow
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
              for row in ln1Rows do
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
              let headRows := prefixUnionRowsFixed vOutRows
              attnRows := addRowsFixed attnRows headRows
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnRows := addVecFixedRows attnRows attnBias
            residuals := addRowsFixed residuals attnRows
          else
            let mut attnUnion : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let groupRows := groupUnionRowsByToken ln1Rows tokens
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty groupRows.size
              for row in groupRows do
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
              let vUnion := unionRowsFixed vOutRows
              attnUnion := addVecFixed attnUnion vUnion
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnUnion := addVecFixed attnUnion attnBias
            residuals := addVecFixedRows residuals attnUnion
        else
          let ln1Union := unionRowsFixed ln1Rows
          let mut attnUnion : Array Fixed10Interval :=
            Array.replicate hdr.modelDim { lo := 0, hi := 0 }
          for _h in [:hdr.numHeads] do
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let wv ←
              ExceptT.mk <| readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
            let vHidden0 := matMulIntervalsFromScaled cfg slack
              hdr.modelDim hdr.headDim wv ln1Union
            let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
            let vHidden := addVecFixed vHidden0 bV
            let wo ←
              ExceptT.mk <| readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
            let vOut := matMulIntervalsFromScaled cfg slack
              hdr.headDim hdr.modelDim wo vHidden
            attnUnion := addVecFixed attnUnion vOut
          let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          attnUnion := addVecFixed attnUnion attnBias
          residuals := addVecFixedRows residuals attnUnion
        let p2 := ln2Params.getD l defP
        let mut ln2Rows : Array (Array Fixed10Interval) := Array.mkEmpty seqLenEff
        for row in residuals do
          let (ln2Out, _ln2VarLB) :=
            fixedLayerNormRowApprox cfg row p2.gamma p2.beta eps soundnessBits
          ln2Rows := ln2Rows.push ln2Out
        let perRowLayers : Nat := perRowPatternLayers
        if perRowLayers > 0 && layerIdx ≤ l + perRowLayers then
          let wIn ←
            ExceptT.mk <| readScaledFloatArray h (hdr.modelDim * hdr.hiddenDim) scalePow10
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let wOut ←
            ExceptT.mk <| readScaledFloatArray h (hdr.hiddenDim * hdr.modelDim) scalePow10
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpRows :=
            mlpRowsFromScaled cfg hdr.geluDerivTarget slack
              hdr.modelDim hdr.hiddenDim wIn wOut bIn bOut ln2Rows
          residuals := addRowsFixed residuals mlpRows
        else
          let ln2Union := unionRowsFixed ln2Rows
          let (hidden0, _nWin) ←
            ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
              hdr.modelDim hdr.hiddenDim ln2Union scalePow10)
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let hiddenB := addVecFixed hidden0 bIn
          let actHidden := geluOverapproxFixedVec cfg hdr.geluDerivTarget hiddenB
          let (mlpOut0, _nWout) ←
            ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
              hdr.hiddenDim hdr.modelDim actHidden scalePow10)
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpOut := addVecFixed mlpOut0 bOut
          residuals := addVecFixedRows residuals mlpOut
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
    throw "target layer not reached"
  action.run


private def readUnembeddingColumnsBinary
    (path : System.FilePath)
    (tokenA tokenB : Nat)
    (scalePow10 : Nat) :
    IO (Except String (BinaryHeader × Array Int × Array Int)) := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let action : ExceptT String IO (BinaryHeader × Array Int × Array Int) := do
    let hdr ← ExceptT.mk (readBinaryHeader h)
    if tokenA ≥ hdr.vocabSize || tokenB ≥ hdr.vocabSize then
      throw "token index out of range for unembedding"
    if tokenA = tokenB then
      throw "target and negative tokens must differ"
    let _ ← ExceptT.mk (skipToUnembeddingBinary h hdr)
    let loTok := min tokenA tokenB
    let hiTok := max tokenA tokenB
    let swapped : Bool := tokenA > tokenB
    let mut colA : Array Int := Array.mkEmpty hdr.modelDim
    let mut colB : Array Int := Array.mkEmpty hdr.modelDim
    for _r in [:hdr.modelDim] do
      let _ ← ExceptT.mk (skipF64Array h loTok)
      let vLo ← ExceptT.mk (readScaledFloat h scalePow10)
      let _ ← ExceptT.mk (skipF64Array h (hiTok - loTok - 1))
      let vHi ← ExceptT.mk (readScaledFloat h scalePow10)
      let _ ← ExceptT.mk (skipF64Array h (hdr.vocabSize - hiTok - 1))
      if swapped then
        colA := colA.push vHi
        colB := colB.push vLo
      else
        colA := colA.push vLo
        colB := colB.push vHi
    return (hdr, colA, colB)
  action.run

private def readLogitDiffDirectionBinary
    (path : System.FilePath)
    (targetToken negativeToken : Nat)
    (scalePow10 : Nat)
    (slack : Int) :
    IO (Except String (BinaryHeader × Array Fixed10Interval)) := do
  let action : ExceptT String IO (BinaryHeader × Array Fixed10Interval) := do
    let (hdr, colTarget, colNeg) ←
      ExceptT.mk (readUnembeddingColumnsBinary path targetToken negativeToken scalePow10)
    if colTarget.size ≠ hdr.modelDim || colNeg.size ≠ hdr.modelDim then
      throw "unembedding column size mismatch"
    let targetIntervals := intervalsFromScaled colTarget slack
    let negIntervals := intervalsFromScaled colNeg slack
    let mut dir : Array Fixed10Interval := Array.mkEmpty hdr.modelDim
    for i in [:hdr.modelDim] do
      dir := dir.push (Fixed10Interval.sub targetIntervals[i]! negIntervals[i]!)
    return (hdr, dir)
  action.run

/-- Compute local head logit-difference lower bounds at a specific query position (binary only). -/
private def certifyHeadLogitDiffLowerBoundLocalBinaryAt
    (path : System.FilePath)
    (layerIdx headIdx queryPos : Nat)
    (targetToken negativeToken : Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath : System.FilePath)
    (inputDelta : Rat)
    (targetOffset : Int)
    (keyOffset : Int)
    (matchWeightLowerBound : Rat)
    (maxSeqLen : Nat)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (causalPattern : Bool := true)
    (shared? : Option SharedBinaryInputs := none)
    (prefix? : Option SharedBinaryPrefix := none)
    (direction? : Option (Thunk (Array Fixed10Interval)) := none) :
    IO (Except String HeadLogitDiffLowerBoundPosCert) := do
  let cfg : Fixed10Cfg := scaleCfgOfPow10 scalePow10
  let slack : Int := fixedUlpSlack
  let action : ExceptT String IO HeadLogitDiffLowerBoundPosCert := do
    let (direction, hdrDir?) ←
      match direction? with
      | some thunk => pure (thunk.get, none)
      | none =>
          let (hdrDir, dir) ←
            ExceptT.mk <|
              readLogitDiffDirectionBinary path targetToken negativeToken scalePow10 slack
          pure (dir, some hdrDir)
    let (hdr, ln1Params, ln2Params, residualsBase, tokensBase) ←
      match shared? with
      | some shared =>
          if shared.scalePow10 ≠ scalePow10 then
            throw "shared scalePow10 mismatch"
          if shared.inputDelta ≠ inputDelta then
            throw "shared inputDelta mismatch"
          pure (shared.hdr, shared.ln1Params, shared.ln2Params, shared.residuals0, shared.tokens)
      | none =>
          let (hdr, ln1Params, ln2Params) ←
            ExceptT.mk (collectLayerNormParamsBinary path scalePow10 slack)
          let residuals0 ←
            ExceptT.mk
              (loadEmbeddingsIntervalsBinary inputPath hdr.modelDim inputDelta scalePow10)
          let (hdrTok, tokens) ← ExceptT.mk (loadTokensBinary inputPath)
          if hdrTok.seqLen ≠ hdr.seqLen then
            throw "token/embedding seq_len mismatch"
          pure (hdr, ln1Params, ln2Params, residuals0, tokens)
    match hdrDir? with
    | some hdrDir =>
        if hdr.modelDim ≠ hdrDir.modelDim then
          throw "unembedding model_dim mismatch"
    | none =>
        if direction.size ≠ hdr.modelDim then
          throw "logit direction size mismatch"
    if layerIdx ≥ hdr.numLayers then
      throw s!"layer index {layerIdx} out of range"
    if headIdx ≥ hdr.numHeads then
      throw s!"head index {headIdx} out of range"
    if queryPos ≥ hdr.seqLen then
      throw s!"queryPos {queryPos} out of range"
    if hdr.seqLen > maxSeqLen then
      throw s!"seq_len {hdr.seqLen} exceeds maxSeqLen {maxSeqLen}"
    let seqLenEff : Nat := if causalPattern then queryPos + 1 else hdr.seqLen
    if direction.size ≠ hdr.modelDim then
      throw "logit direction size mismatch"
    let (residuals0, tokens) ←
      match prefix? with
      | some pref =>
          if pref.seqLenEff ≠ seqLenEff then
            throw "prefix seq_len mismatch"
          pure (pref.residuals.get, pref.tokens.get)
      | none =>
          let residuals0 :=
            if causalPattern then takePrefix residualsBase seqLenEff else residualsBase
          let tokens := if causalPattern then takePrefix tokensBase seqLenEff else tokensBase
          pure (residuals0, tokens)
    let keyOffsetNat? : Option Nat :=
      if keyOffset ≥ 0 then some (Int.toNat keyOffset) else none
    let keyOffsetNeg : Nat := Int.toNat (-keyOffset)
    let h ← IO.FS.Handle.mk path IO.FS.Mode.read
    let _ ← ExceptT.mk (readBinaryHeader h)
    let _ ← ExceptT.mk (skipI32Array h hdr.seqLen)
    let _ ← ExceptT.mk (skipF64Array h (hdr.seqLen * hdr.modelDim))
    let defP : LayerNormParamsFixed := {
      gamma := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      beta := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
    }
    let mut residuals := residuals0
    for l in [:hdr.numLayers] do
      let p1 := ln1Params.getD l defP
      let mut ln1Rows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
      for row in residuals do
        let (ln1Out, _ln1VarLB) :=
          fixedLayerNormRowApprox cfg row p1.gamma p1.beta eps soundnessBits
        ln1Rows := ln1Rows.push ln1Out
      if l = layerIdx then
        let mut wv? : Option (Array Int) := none
        let mut bv? : Option (Array Int) := none
        let mut wo? : Option (Array Int) := none
        for hIdx in [:hdr.numHeads] do
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          if hIdx = headIdx then
            let wv ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
            wv? := some wv
            let bV ←
              ExceptT.mk <|
                readScaledFloatArray h hdr.headDim scalePow10
            bv? := some bV
            let wo ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
            wo? := some wo
          else
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.headDim * hdr.modelDim))
        let wv ←
          match wv? with
          | none => throw "missing W_V for requested head"
          | some xs => pure xs
        let bV ←
          match bv? with
          | none => throw "missing b_V for requested head"
          | some xs => pure xs
        let wo ←
          match wo? with
          | none => throw "missing W_O for requested head"
          | some xs => pure xs
        let bVIntervals := intervalsFromScaled bV slack
        let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty seqLenEff
        for row in ln1Rows do
          let vHidden0 := matMulIntervalsFromScaled cfg slack
            hdr.modelDim hdr.headDim wv row
          let vHidden := addVecFixed vHidden0 bVIntervals
          let vOut := matMulIntervalsFromScaled cfg slack
            hdr.headDim hdr.modelDim wo vHidden
          vOutRows := vOutRows.push vOut
        let mut vDotRows : Array Fixed10Interval := Array.mkEmpty seqLenEff
        for row in vOutRows do
          vDotRows := vDotRows.push (fixedDotInterval cfg row direction)
        let ti : Int := (Int.ofNat queryPos) + targetOffset
        if ti < 0 || ti ≥ (Int.ofNat seqLenEff) then
          throw "query position has no valid target offset"
        let tIdx : Nat := Int.toNat ti
        let targetTok := tokens[tIdx]!
        let mut matchLo? : Option Int := none
        let mut nonmatchLo? : Option Int := none
        for j in [:seqLenEff] do
          if !causalPattern || j ≤ queryPos then
            let vLo := (vDotRows[j]!).lo
            let isMatch : Bool :=
              match keyOffsetNat? with
              | some k =>
                  let idx := j + k
                  idx < seqLenEff && tokens[idx]! = targetTok
              | none =>
                  if j < keyOffsetNeg then
                    false
                  else
                    tokens[j - keyOffsetNeg]! = targetTok
            if isMatch then
              matchLo? :=
                match matchLo? with
                | none => some vLo
                | some m => some (min m vLo)
            else
              nonmatchLo? :=
                match nonmatchLo? with
                | none => some vLo
                | some m => some (min m vLo)
          else
            pure ()
        let matchLo ←
          match matchLo? with
          | none => throw "no matching keys for the requested offset"
          | some v => pure v
        let nonmatchLo :=
          match nonmatchLo? with
          | none => matchLo
          | some v => v
        let matchLoRat := ratOfScaledInt scalePow10 matchLo
        let nonmatchLoRat := ratOfScaledInt scalePow10 nonmatchLo
        let weightLB := matchWeightLowerBound
        let outputLB := mixLowerBound weightLB matchLoRat nonmatchLoRat
        let cert : HeadLogitDiffLowerBoundPosCert := {
          layerIdx := layerIdx
          headIdx := headIdx
          queryPos := queryPos
          targetToken := targetToken
          negativeToken := negativeToken
          matchWeightLowerBound := weightLB
          matchLogitLowerBound := matchLoRat
          nonmatchLogitLowerBound := nonmatchLoRat
          logitDiffLowerBound := outputLB
        }
        if cert.check then
          return cert
        throw "head logit lower bound (pos) failed internal consistency checks"
      else
        let tightLayers : Nat :=
          if tightPattern then Nat.max 1 tightPatternLayers else 0
        if tightLayers > 0 && layerIdx ≤ l + tightLayers then
          if causalPattern then
            let zeroRow : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let mut attnRows : Array (Array Fixed10Interval) :=
              Array.replicate hdr.seqLen zeroRow
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
              for row in ln1Rows do
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
              let headRows := prefixUnionRowsFixed vOutRows
              attnRows := addRowsFixed attnRows headRows
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnRows := addVecFixedRows attnRows attnBias
            residuals := addRowsFixed residuals attnRows
          else
            let mut attnUnion : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let groupRows := groupUnionRowsByToken ln1Rows tokens
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty groupRows.size
              for row in groupRows do
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
              let vUnion := unionRowsFixed vOutRows
              attnUnion := addVecFixed attnUnion vUnion
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnUnion := addVecFixed attnUnion attnBias
            residuals := addVecFixedRows residuals attnUnion
        else
          let ln1Union := unionRowsFixed ln1Rows
          let mut attnUnion : Array Fixed10Interval :=
            Array.replicate hdr.modelDim { lo := 0, hi := 0 }
          for _h in [:hdr.numHeads] do
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let (vHidden0, _nWv) ←
              ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
                hdr.modelDim hdr.headDim ln1Union scalePow10)
            let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
            let vHidden := addVecFixed vHidden0 bV
            let (vOut, _nWo) ←
              ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
                hdr.headDim hdr.modelDim vHidden scalePow10)
            attnUnion := addVecFixed attnUnion vOut
          let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          attnUnion := addVecFixed attnUnion attnBias
          residuals := addVecFixedRows residuals attnUnion
        let p2 := ln2Params.getD l defP
        let mut ln2Rows : Array (Array Fixed10Interval) := Array.mkEmpty seqLenEff
        for row in residuals do
          let (ln2Out, _ln2VarLB) :=
            fixedLayerNormRowApprox cfg row p2.gamma p2.beta eps soundnessBits
          ln2Rows := ln2Rows.push ln2Out
        let perRowLayers : Nat := perRowPatternLayers
        if perRowLayers > 0 && layerIdx ≤ l + perRowLayers then
          let wIn ←
            ExceptT.mk <| readScaledFloatArray h (hdr.modelDim * hdr.hiddenDim) scalePow10
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let wOut ←
            ExceptT.mk <| readScaledFloatArray h (hdr.hiddenDim * hdr.modelDim) scalePow10
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpRows :=
            mlpRowsFromScaled cfg hdr.geluDerivTarget slack
              hdr.modelDim hdr.hiddenDim wIn wOut bIn bOut ln2Rows
          residuals := addRowsFixed residuals mlpRows
        else
          let ln2Union := unionRowsFixed ln2Rows
          let (hidden0, _nWin) ←
            ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
              hdr.modelDim hdr.hiddenDim ln2Union scalePow10)
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let hiddenB := addVecFixed hidden0 bIn
          let actHidden := geluOverapproxFixedVec cfg hdr.geluDerivTarget hiddenB
          let (mlpOut0, _nWout) ←
            ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
              hdr.hiddenDim hdr.modelDim actHidden scalePow10)
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpOut := addVecFixed mlpOut0 bOut
          residuals := addVecFixedRows residuals mlpOut
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
    throw "target layer not reached"
  action.run

private def ensureSoundCache
    (modelPath : System.FilePath)
    (scalePow10 : Nat := defaultFixedScalePow10) :
    IO (Except String (System.FilePath × SoundCache.Header)) := do
  Nfp.Untrusted.SoundCacheIO.ensureCacheDir
  let modelHash ← Nfp.Untrusted.SoundCacheIO.fnv1a64File modelPath
  let mdata ← modelPath.metadata
  let modelSize : UInt64 := mdata.byteSize
  let cpath := SoundCache.cachePath modelPath modelHash scalePow10
  if !(← cpath.pathExists) then
    match (← Nfp.Untrusted.SoundCacheIO.buildCacheFile modelPath cpath scalePow10) with
    | .error e => return .error e
    | .ok _ => pure ()
  let h ← IO.FS.Handle.mk cpath IO.FS.Mode.read
  let hdr ← Nfp.Untrusted.SoundCacheIO.readHeader h
  if hdr.modelHash ≠ modelHash then
    return .error "sound cache hash mismatch"
  if hdr.modelSize ≠ modelSize then
    return .error "sound cache size mismatch"
  return .ok (cpath, hdr)

private def readWqWkMaxBinary
    (path : System.FilePath)
    (scalePow10 : Nat := defaultBinaryScalePow10) :
    IO (Except String (Array Rat × Array Rat)) := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  match ← readBinaryHeader h with
  | .error e => return .error e
  | .ok hdr =>
      match ← skipI32Array h hdr.seqLen with
      | .error e => return .error e
      | .ok _ => pure ()
      match ← skipF64Array h (hdr.seqLen * hdr.modelDim) with
      | .error e => return .error e
      | .ok _ => pure ()
      let mut wqMax : Array Rat := Array.replicate hdr.numLayers 0
      let mut wkMax : Array Rat := Array.replicate hdr.numLayers 0
      for l in [:hdr.numLayers] do
        let mut wqLayer : Rat := 0
        let mut wkLayer : Rat := 0
        for _h in [:hdr.numHeads] do
          let wqScaledE ← readMatrixNormInfScaled h hdr.modelDim hdr.headDim scalePow10
          let wqScaled ←
            match wqScaledE with
            | .error e => return .error e
            | .ok v => pure v
          match ← skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          let wkScaledE ← readMatrixNormInfScaled h hdr.modelDim hdr.headDim scalePow10
          let wkScaled ←
            match wkScaledE with
            | .error e => return .error e
            | .ok v => pure v
          match ← skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          match ← skipF64Array h (hdr.modelDim * hdr.headDim) with
          | .error e => return .error e
          | .ok _ => pure ()
          match ← skipF64Array h hdr.headDim with
          | .error e => return .error e
          | .ok _ => pure ()
          match ← skipF64Array h (hdr.headDim * hdr.modelDim) with
          | .error e => return .error e
          | .ok _ => pure ()
          let wq := ratOfScaledInt scalePow10 wqScaled
          let wk := ratOfScaledInt scalePow10 wkScaled
          wqLayer := max wqLayer wq
          wkLayer := max wkLayer wk
        wqMax := wqMax.set! l wqLayer
        wkMax := wkMax.set! l wkLayer
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h (hdr.modelDim * hdr.hiddenDim) with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h hdr.hiddenDim with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h (hdr.hiddenDim * hdr.modelDim) with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
        match ← skipF64Array h hdr.modelDim with
        | .error e => return .error e
        | .ok _ => pure ()
      return .ok (wqMax, wkMax)

/-- Local (input-dependent) certificate path using streaming interval propagation.

This is conservative in two key ways to remain streaming/memory-safe:
- it uses a **union box** over tokens throughout (so we never hold `seqLen×modelDim` intervals),
  which is sound (a superset) but can be looser than per-token tracking,
- it uses union boxes for attention/MLP linear maps to avoid `seqLen×hiddenDim` blowups.
-/
private def certifyModelFileLocalText
    (path : System.FilePath)
    (eps : Rat)
    (geluDerivTarget : GeluDerivTarget)
    (soundnessBits : Nat)
    (partitionDepth : Nat)
    (inputPath : System.FilePath)
    (inputDelta : Rat)
    (softmaxMarginLowerBound : Rat := 0)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort) : IO (Except String ModelCert) := do
  if partitionDepth ≠ 0 then
    return .error "partitionDepth > 0 not yet implemented"
  let contents ← IO.FS.readFile path
  let lines : Array String := Nfp.Sound.splitLines contents
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
      -- Use a single union box for all tokens (sound superset, much faster than
      -- `seqLen×modelDim`).
      let mut residualUnion := unionRows residualRows0 d
      -- Start scanning at first layer marker.
      let mut pos : Nat := skipUntil lines 0 (fun s => s.startsWith "LAYER")
      let mut layers : Array LayerAmplificationCert := Array.mkEmpty L
      let mut totalAmp : Rat := 1
      let mut actDerivBoundMax : Rat := 0
      for l in [:L] do
        -- Ensure we're at the next layer.
        pos := skipUntil lines pos (fun s => s.startsWith "LAYER")
        if pos ≥ lines.size then
          return .error s!"unexpected end of file while scanning layer {l}"
        pos := pos + 1
        -- LN1: compute per-row outputs (for union) and min variance LB (for Jacobian bound).
        let p1 := ln1Params.getD l defLn
        let (ln1Out, ln1VarLB) :=
          layerNormRowApprox residualUnion p1.gamma p1.beta eps soundnessBits
        let ln1MaxAbsGamma := maxAbsOfVector p1.gamma
        let ln1MaxAbsBeta := maxAbsOfVector p1.beta
        let ln1Bound :=
          if ln1VarLB > 0 then
            layerNormOpBoundLocal ln1MaxAbsGamma ln1VarLB eps soundnessBits
          else
            layerNormOpBoundConservative ln1MaxAbsGamma eps soundnessBits
        let ln1OutMaxAbsBound := layerNormOutputMaxAbsBound d ln1MaxAbsGamma ln1MaxAbsBeta
        let ln1Union := ln1Out
        -- Attention (streaming): use union input box.
        let mut attnUnion : Array RatInterval := zeroIntervals d
        let mut attnValueCoeff : Rat := 0
        let mut wqMax : Rat := 0
        let mut wkMax : Rat := 0
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
          match consumeMatrixNormInf lines (pos + 1) d dh with
          | .error e => return .error e
          | .ok (nq, next) =>
              wqMax := max wqMax nq
              pos := next
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
          match consumeMatrixNormInf lines (pos + 1) d dh with
          | .error e => return .error e
          | .ok (nk, next) =>
              wkMax := max wkMax nk
              pos := next
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
          | .ok (vHidden, _nWv, nextV) =>
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
              let vCenteredOpBound := centeredAbsSum vHidden
              match consumeMatrixMulAndNormInf lines (pos + 1) dh d vHidden with
              | .error e => return .error e
              | .ok (vOut, no, nextO) =>
                  pos := nextO
                  attnUnion := addVecIntervals attnUnion vOut
                  attnValueCoeff := attnValueCoeff + vCenteredOpBound * no
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
        let (ln2Out, ln2VarLB) :=
          layerNormRowApprox residualUnion p2.gamma p2.beta eps soundnessBits
        let ln2MaxAbsGamma := maxAbsOfVector p2.gamma
        let ln2Bound :=
          if ln2VarLB > 0 then
            layerNormOpBoundLocal ln2MaxAbsGamma ln2VarLB eps soundnessBits
          else
            layerNormOpBoundConservative ln2MaxAbsGamma eps soundnessBits
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
                let mlpActDerivBound := maxGeluDerivBound geluDerivTarget hiddenB
                let actHidden := hiddenB.map (geluOverapproxRat geluDerivTarget)
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
                        let scoreAbsBound :=
                          attnScoreAbsBound d dh ln1OutMaxAbsBound wqMax wkMax
                        let (softmaxProbLo, softmaxProbHi) :=
                          softmaxProbIntervalFromScoreAbsBound n scoreAbsBound softmaxExpEffort
                        let softmaxIntervalBound :=
                          softmaxJacobianNormInfBound softmaxProbLo softmaxProbHi
                        let softmaxMarginBound :=
                          softmaxJacobianNormInfBoundFromMargin n softmaxMarginLowerBound
                            softmaxExpEffort
                        let softmaxBound := min softmaxIntervalBound softmaxMarginBound
                        let attnPatternCoeff :=
                          attnPatternCoeffBound n d dh ln1OutMaxAbsBound wqMax wkMax
                            attnValueCoeff
                        let attnW :=
                          ln1Bound *
                            ((n : Rat) * attnValueCoeff + softmaxBound * attnPatternCoeff)
                        let mlpCoeff := nWin * nWout
                        let mlpW := ln2Bound * (mlpCoeff * mlpActDerivBound)
                        let C := attnW + mlpW + attnW * mlpW
                        layers := layers.push {
                          layerIdx := l
                          ln1MaxAbsGamma := ln1MaxAbsGamma
                          ln1MaxAbsBeta := ln1MaxAbsBeta
                          ln2MaxAbsGamma := ln2MaxAbsGamma
                          ln1VarianceLowerBound? := some ln1VarLB
                          ln2VarianceLowerBound? := some ln2VarLB
                          ln1Bound := ln1Bound
                          ln2Bound := ln2Bound
                          ln1OutMaxAbsBound := ln1OutMaxAbsBound
                          softmaxProbLo := softmaxProbLo
                          softmaxProbHi := softmaxProbHi
                          softmaxMarginLowerBound := softmaxMarginLowerBound
                          softmaxExpEffort := softmaxExpEffort
                          softmaxJacobianNormInfUpperBound := softmaxBound
                          wqOpBoundMax := wqMax
                          wkOpBoundMax := wkMax
                          attnValueCoeff := attnValueCoeff
                          attnPatternCoeff := attnPatternCoeff
                          mlpCoeff := mlpCoeff
                          mlpWinBound := nWin
                          mlpWoutBound := nWout
                          mlpActDerivBound := mlpActDerivBound
                          attnJacBound := attnW
                          mlpJacBound := mlpW
                          C := C
                        }
                        totalAmp := totalAmp * (1 + C)
                        actDerivBoundMax := max actDerivBoundMax mlpActDerivBound
                        pos := skipUntil lines pos (fun s => s.startsWith "LAYER")
      let cert : ModelCert := {
        modelPath := path.toString
        inputPath? := some inputPath.toString
        inputDelta := inputDelta
        eps := eps
        seqLen := n
        modelDim := d
        headDim := dh
        soundnessBits := soundnessBits
        geluDerivTarget := geluDerivTarget
        actDerivBound := actDerivBoundMax
        softmaxJacobianNormInfWorst := softmaxJacobianNormInfWorst
        layers := layers
        totalAmplificationFactor := totalAmp
      }
      if cert.check then
        return .ok cert
      return .error "sound certificate failed internal consistency checks"

private def certifyModelFileLocal
    (path : System.FilePath)
    (eps : Rat)
    (geluDerivTarget : GeluDerivTarget)
    (soundnessBits : Nat)
    (partitionDepth : Nat)
    (inputPath : System.FilePath)
    (inputDelta : Rat)
    (softmaxMarginLowerBound : Rat := 0)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort) : IO (Except String ModelCert) := do
  if partitionDepth ≠ 0 then
    return .error "partitionDepth > 0 not yet implemented"
  -- Prefer cached fixed-point path; fall back to the (slow) Rat-based path on any cache error.
  match (← ensureSoundCache path) with
  | .error _ =>
      certifyModelFileLocalText path eps geluDerivTarget soundnessBits partitionDepth
        inputPath inputDelta softmaxMarginLowerBound softmaxExpEffort
  | .ok (cpath, hdr) =>
      let cfg : Fixed10Cfg := scaleCfgOfPow10 hdr.scalePow10.toNat
      let slack : Int := fixedUlpSlack
      let modelDim := hdr.modelDim.toNat
      let headDim := hdr.headDim.toNat
      let hiddenDim := hdr.hiddenDim.toNat
      let L := hdr.numLayers.toNat
      let H := hdr.numHeads.toNat
      let wqWkE ← readWqWkMaxBinary path (scalePow10 := hdr.scalePow10.toNat)
      let (wqMaxArr, wkMaxArr) ←
        match wqWkE with
        | .error e => return .error e
        | .ok v => pure v
      -- For now we read embeddings from the input `.nfpt` file and use a union box.
      let residualUnionE ← loadEmbeddingsUnionFixed cfg inputPath modelDim inputDelta
      match residualUnionE with
      | .error e => return .error e
      | .ok (residualUnion0, inputSeqLen) =>
          let mut residualUnion := residualUnion0
          -- Open cache and position reader after header.
          let ch ← IO.FS.Handle.mk cpath IO.FS.Mode.read
          let _ ← Nfp.Untrusted.SoundCacheIO.readHeader ch
          let mut rr ← Nfp.Untrusted.SoundCacheIO.I32Reader.init ch
          let mut layers : Array LayerAmplificationCert := Array.mkEmpty L
          let mut totalAmp : Rat := 1
          let mut actDerivBoundMax : Rat := 0
          for l in [:L] do
            -- LN params from cache
            let (ln1Gamma, rr1) ← readVecIntervals rr modelDim slack
            let (ln1Beta, rr2) ← readVecIntervals rr1 modelDim slack
            let (ln2Gamma, rr3) ← readVecIntervals rr2 modelDim slack
            let (ln2Beta, rr4) ← readVecIntervals rr3 modelDim slack
            rr := rr4
            -- LN1
            let (ln1Out, ln1VarLB) :=
              fixedLayerNormRowApprox cfg residualUnion ln1Gamma ln1Beta eps soundnessBits
            let ln1MaxAbsGamma : Rat :=
              Rat.normalize (maxAbsVecFixed ln1Gamma) cfg.scaleNat (den_nz := by
                have h10pos : (0 : Nat) < 10 := by decide
                exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
            let ln1MaxAbsBeta : Rat :=
              Rat.normalize (maxAbsVecFixed ln1Beta) cfg.scaleNat (den_nz := by
                have h10pos : (0 : Nat) < 10 := by decide
                exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
            let ln1Bound :=
              if ln1VarLB > 0 then
                layerNormOpBoundLocal ln1MaxAbsGamma ln1VarLB eps soundnessBits
              else
                layerNormOpBoundConservative ln1MaxAbsGamma eps soundnessBits
            let ln1OutMaxAbsBound :=
              layerNormOutputMaxAbsBound modelDim ln1MaxAbsGamma ln1MaxAbsBeta
            -- Attention (streaming from cache)
            let mut attnUnion : Array Fixed10Interval :=
              Array.replicate modelDim { lo := 0, hi := 0 }
            let mut attnValueCoeff : Rat := 0
            for _h in [:H] do
              let (vHidden0, _nWv, rrV) ←
                consumeMatrixMulAndNormInfFixed cfg slack rr modelDim headDim ln1Out
              rr := rrV
              let (bV, rrBv) ← readVecIntervals rr headDim slack
              rr := rrBv
              let vHidden := addVecFixed vHidden0 bV
              let vCenteredOpBound := centeredAbsSumFixed cfg vHidden
              let (vOut, nWo, rrO) ←
                consumeMatrixMulAndNormInfFixed cfg slack rr headDim modelDim vHidden
              rr := rrO
              attnUnion := addVecFixed attnUnion vOut
              attnValueCoeff := attnValueCoeff + vCenteredOpBound * nWo
            let (attnBias, rrB) ← readVecIntervals rr modelDim slack
            rr := rrB
            attnUnion := addVecFixed attnUnion attnBias
            residualUnion := addVecFixed residualUnion attnUnion
            -- LN2
            let (ln2Out, ln2VarLB) :=
              fixedLayerNormRowApprox cfg residualUnion ln2Gamma ln2Beta eps soundnessBits
            let ln2MaxAbsGamma : Rat :=
              Rat.normalize (maxAbsVecFixed ln2Gamma) cfg.scaleNat (den_nz := by
                have h10pos : (0 : Nat) < 10 := by decide
                exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
            let ln2Bound :=
              if ln2VarLB > 0 then
                layerNormOpBoundLocal ln2MaxAbsGamma ln2VarLB eps soundnessBits
              else
                layerNormOpBoundConservative ln2MaxAbsGamma eps soundnessBits
            -- MLP
            let (hidden0, nWin, rrWin) ←
              consumeMatrixMulAndNormInfFixed cfg slack rr modelDim hiddenDim ln2Out
            rr := rrWin
            let (bIn, rrBin) ← readVecIntervals rr hiddenDim slack
            rr := rrBin
            let hiddenB := addVecFixed hidden0 bIn
            let mlpActDerivBound := maxGeluDerivBoundFixed cfg geluDerivTarget hiddenB
            let actHidden := geluOverapproxFixedVec cfg geluDerivTarget hiddenB
            let (mlpOut0, nWout, rrWout) ←
              consumeMatrixMulAndNormInfFixed cfg slack rr hiddenDim modelDim actHidden
            rr := rrWout
            let (bOut, rrBout) ← readVecIntervals rr modelDim slack
            rr := rrBout
            let mlpOut := addVecFixed mlpOut0 bOut
            residualUnion := addVecFixed residualUnion mlpOut
            let scoreAbsBound :=
              attnScoreAbsBound modelDim headDim ln1OutMaxAbsBound (wqMaxArr[l]!)
                (wkMaxArr[l]!)
            let (softmaxProbLo, softmaxProbHi) :=
              softmaxProbIntervalFromScoreAbsBound inputSeqLen scoreAbsBound softmaxExpEffort
            let softmaxIntervalBound := softmaxJacobianNormInfBound softmaxProbLo softmaxProbHi
            let softmaxMarginBound :=
              softmaxJacobianNormInfBoundFromMargin inputSeqLen softmaxMarginLowerBound
                softmaxExpEffort
            let softmaxBound := min softmaxIntervalBound softmaxMarginBound
            let attnPatternCoeff :=
              attnPatternCoeffBound inputSeqLen modelDim headDim ln1OutMaxAbsBound
                (wqMaxArr[l]!) (wkMaxArr[l]!) attnValueCoeff
            let attnW :=
              ln1Bound *
                ((inputSeqLen : Rat) * attnValueCoeff + softmaxBound * attnPatternCoeff)
            let mlpCoeff := nWin * nWout
            let mlpW := ln2Bound * (mlpCoeff * mlpActDerivBound)
            let C := attnW + mlpW + attnW * mlpW
            layers := layers.push {
              layerIdx := l
              ln1MaxAbsGamma := ln1MaxAbsGamma
              ln1MaxAbsBeta := ln1MaxAbsBeta
              ln2MaxAbsGamma := ln2MaxAbsGamma
              ln1VarianceLowerBound? := some ln1VarLB
              ln2VarianceLowerBound? := some ln2VarLB
              ln1Bound := ln1Bound
              ln2Bound := ln2Bound
              ln1OutMaxAbsBound := ln1OutMaxAbsBound
              softmaxProbLo := softmaxProbLo
              softmaxProbHi := softmaxProbHi
              softmaxMarginLowerBound := softmaxMarginLowerBound
              softmaxExpEffort := softmaxExpEffort
              softmaxJacobianNormInfUpperBound := softmaxBound
              wqOpBoundMax := wqMaxArr[l]!
              wkOpBoundMax := wkMaxArr[l]!
              attnValueCoeff := attnValueCoeff
              attnPatternCoeff := attnPatternCoeff
              mlpCoeff := mlpCoeff
              mlpWinBound := nWin
              mlpWoutBound := nWout
              mlpActDerivBound := mlpActDerivBound
              attnJacBound := attnW
              mlpJacBound := mlpW
              C := C
            }
            totalAmp := totalAmp * (1 + C)
            actDerivBoundMax := max actDerivBoundMax mlpActDerivBound
          let cert : ModelCert := {
            modelPath := path.toString
            inputPath? := some inputPath.toString
            inputDelta := inputDelta
            eps := eps
            seqLen := inputSeqLen
            modelDim := modelDim
            headDim := headDim
            soundnessBits := soundnessBits
            geluDerivTarget := geluDerivTarget
            actDerivBound := actDerivBoundMax
            softmaxJacobianNormInfWorst := softmaxJacobianNormInfWorst
            layers := layers
            totalAmplificationFactor := totalAmp
          }
          if cert.check then
            return .ok cert
          return .error "sound certificate failed internal consistency checks"

private def certifyModelFileLocalBinary
    (path : System.FilePath)
    (eps : Rat)
    (geluDerivTarget : GeluDerivTarget)
    (soundnessBits : Nat)
    (partitionDepth : Nat)
    (inputPath : System.FilePath)
    (inputDelta : Rat)
    (softmaxMarginLowerBound : Rat := 0)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort) : IO (Except String ModelCert) := do
  if partitionDepth ≠ 0 then
    return .error "partitionDepth > 0 not yet implemented"
  let scalePow10 := defaultBinaryScalePow10
  let cfg : Fixed10Cfg := scaleCfgOfPow10 scalePow10
  let slack : Int := fixedUlpSlack
  let action : ExceptT String IO ModelCert := do
    let (hdr, ln1Params, ln2Params) ←
      ExceptT.mk (collectLayerNormParamsBinary path scalePow10 slack)
    let residualUnion0 ←
      ExceptT.mk (loadEmbeddingsUnionFixedBinary inputPath hdr.modelDim inputDelta scalePow10)
    let h ← IO.FS.Handle.mk path IO.FS.Mode.read
    let _ ← ExceptT.mk (readBinaryHeader h)
    let _ ← ExceptT.mk (skipI32Array h hdr.seqLen)
    let _ ← ExceptT.mk (skipF64Array h (hdr.seqLen * hdr.modelDim))
    let defP : LayerNormParamsFixed := {
      gamma := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      beta := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
    }
    let mut residualUnion := residualUnion0
    let mut layers : Array LayerAmplificationCert := Array.mkEmpty hdr.numLayers
    let mut totalAmp : Rat := 1
    let mut actDerivBoundMax : Rat := 0
    for l in [:hdr.numLayers] do
      let p1 := ln1Params.getD l defP
      let (ln1Out, ln1VarLB) :=
        fixedLayerNormRowApprox cfg residualUnion p1.gamma p1.beta eps soundnessBits
      let ln1MaxAbsGamma : Rat :=
        Rat.normalize (maxAbsVecFixed p1.gamma) cfg.scaleNat (den_nz := by
          have h10pos : (0 : Nat) < 10 := by decide
          exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
      let ln1MaxAbsBeta : Rat :=
        Rat.normalize (maxAbsVecFixed p1.beta) cfg.scaleNat (den_nz := by
          have h10pos : (0 : Nat) < 10 := by decide
          exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
      let ln1Bound :=
        if ln1VarLB > 0 then
          layerNormOpBoundLocal ln1MaxAbsGamma ln1VarLB eps soundnessBits
        else
          layerNormOpBoundConservative ln1MaxAbsGamma eps soundnessBits
      let ln1OutMaxAbsBound :=
        layerNormOutputMaxAbsBound hdr.modelDim ln1MaxAbsGamma ln1MaxAbsBeta
      let mut attnUnion : Array Fixed10Interval :=
        Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      let mut attnValueCoeff : Rat := 0
      let mut wqMax : Rat := 0
      let mut wkMax : Rat := 0
      for _h in [:hdr.numHeads] do
        let wqScaled ←
          ExceptT.mk (readMatrixNormInfScaled h hdr.modelDim hdr.headDim scalePow10)
        wqMax := max wqMax (ratOfScaledInt scalePow10 wqScaled)
        let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
        let wkScaled ←
          ExceptT.mk (readMatrixNormInfScaled h hdr.modelDim hdr.headDim scalePow10)
        wkMax := max wkMax (ratOfScaledInt scalePow10 wkScaled)
        let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
        let (vHidden0, _nWv) ←
          ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
            hdr.modelDim hdr.headDim ln1Out scalePow10)
        let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
        let vHidden := addVecFixed vHidden0 bV
        let vCenteredOpBound := centeredAbsSumFixed cfg vHidden
        let (vOut, nWo) ←
          ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
            hdr.headDim hdr.modelDim vHidden scalePow10)
        attnUnion := addVecFixed attnUnion vOut
        attnValueCoeff := attnValueCoeff + vCenteredOpBound * nWo
      let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
      attnUnion := addVecFixed attnUnion attnBias
      residualUnion := addVecFixed residualUnion attnUnion
      let p2 := ln2Params.getD l defP
      let (ln2Out, ln2VarLB) :=
        fixedLayerNormRowApprox cfg residualUnion p2.gamma p2.beta eps soundnessBits
      let ln2MaxAbsGamma : Rat :=
        Rat.normalize (maxAbsVecFixed p2.gamma) cfg.scaleNat (den_nz := by
          have h10pos : (0 : Nat) < 10 := by decide
          exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
      let ln2Bound :=
        if ln2VarLB > 0 then
          layerNormOpBoundLocal ln2MaxAbsGamma ln2VarLB eps soundnessBits
        else
          layerNormOpBoundConservative ln2MaxAbsGamma eps soundnessBits
      let (hidden0, nWin) ←
        ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
          hdr.modelDim hdr.hiddenDim ln2Out scalePow10)
      let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
      let hiddenB := addVecFixed hidden0 bIn
      let mlpActDerivBound := maxGeluDerivBoundFixed cfg geluDerivTarget hiddenB
      let actHidden := geluOverapproxFixedVec cfg hdr.geluDerivTarget hiddenB
      let (mlpOut0, nWout) ←
        ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
          hdr.hiddenDim hdr.modelDim actHidden scalePow10)
      let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
      let mlpOut := addVecFixed mlpOut0 bOut
      residualUnion := addVecFixed residualUnion mlpOut
      let scoreAbsBound :=
        attnScoreAbsBound hdr.modelDim hdr.headDim ln1OutMaxAbsBound wqMax wkMax
      let (softmaxProbLo, softmaxProbHi) :=
        softmaxProbIntervalFromScoreAbsBound hdr.seqLen scoreAbsBound softmaxExpEffort
      let softmaxIntervalBound := softmaxJacobianNormInfBound softmaxProbLo softmaxProbHi
      let softmaxMarginBound :=
        softmaxJacobianNormInfBoundFromMargin hdr.seqLen softmaxMarginLowerBound softmaxExpEffort
      let softmaxBound := min softmaxIntervalBound softmaxMarginBound
      let attnPatternCoeff :=
        attnPatternCoeffBound hdr.seqLen hdr.modelDim hdr.headDim ln1OutMaxAbsBound
          wqMax wkMax attnValueCoeff
      let attnW :=
        ln1Bound *
          ((hdr.seqLen : Rat) * attnValueCoeff + softmaxBound * attnPatternCoeff)
      let mlpCoeff := nWin * nWout
      let mlpW := ln2Bound * (mlpCoeff * mlpActDerivBound)
      let C := attnW + mlpW + attnW * mlpW
      layers := layers.push {
        layerIdx := l
        ln1MaxAbsGamma := ln1MaxAbsGamma
        ln1MaxAbsBeta := ln1MaxAbsBeta
        ln2MaxAbsGamma := ln2MaxAbsGamma
        ln1VarianceLowerBound? := some ln1VarLB
        ln2VarianceLowerBound? := some ln2VarLB
        ln1Bound := ln1Bound
        ln2Bound := ln2Bound
        ln1OutMaxAbsBound := ln1OutMaxAbsBound
        softmaxProbLo := softmaxProbLo
        softmaxProbHi := softmaxProbHi
        softmaxMarginLowerBound := softmaxMarginLowerBound
        softmaxExpEffort := softmaxExpEffort
        softmaxJacobianNormInfUpperBound := softmaxBound
        wqOpBoundMax := wqMax
        wkOpBoundMax := wkMax
        attnValueCoeff := attnValueCoeff
        attnPatternCoeff := attnPatternCoeff
        mlpCoeff := mlpCoeff
        mlpWinBound := nWin
        mlpWoutBound := nWout
        mlpActDerivBound := mlpActDerivBound
        attnJacBound := attnW
        mlpJacBound := mlpW
        C := C
      }
      totalAmp := totalAmp * (1 + C)
      actDerivBoundMax := max actDerivBoundMax mlpActDerivBound
      let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
      let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
      let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
      let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
    let cert : ModelCert := {
      modelPath := path.toString
      inputPath? := some inputPath.toString
      inputDelta := inputDelta
      eps := eps
      seqLen := hdr.seqLen
      modelDim := hdr.modelDim
      headDim := hdr.headDim
      soundnessBits := soundnessBits
      geluDerivTarget := geluDerivTarget
      actDerivBound := actDerivBoundMax
      softmaxJacobianNormInfWorst := softmaxJacobianNormInfWorst
      layers := layers
      totalAmplificationFactor := totalAmp
    }
    if cert.check then
      return cert
    throw "sound certificate failed internal consistency checks"
  action.run

/-- Compute local per-head attention contribution bounds from a binary `.nfpt`. -/
private def certifyHeadBoundsLocalBinary
    (path : System.FilePath)
    (eps : Rat)
    (inputPath : System.FilePath)
    (inputDelta : Rat)
    (soundnessBits : Nat)
    (scalePow10 : Nat := defaultBinaryScalePow10) :
    IO (Except String (Array HeadLocalContributionCert)) := do
  let cfg : Fixed10Cfg := scaleCfgOfPow10 scalePow10
  let slack : Int := fixedUlpSlack
  let action : ExceptT String IO (Array HeadLocalContributionCert) := do
    let (hdr, ln1Params, ln2Params) ←
      ExceptT.mk (collectLayerNormParamsBinary path scalePow10 slack)
    let residualUnion0 ←
      ExceptT.mk (loadEmbeddingsUnionFixedBinary inputPath hdr.modelDim inputDelta scalePow10)
    let h ← IO.FS.Handle.mk path IO.FS.Mode.read
    let _ ← ExceptT.mk (readBinaryHeader h)
    let _ ← ExceptT.mk (skipI32Array h hdr.seqLen)
    let _ ← ExceptT.mk (skipF64Array h (hdr.seqLen * hdr.modelDim))
    let defP : LayerNormParamsFixed := {
      gamma := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      beta := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
    }
    let mut residualUnion := residualUnion0
    let mut heads : Array HeadLocalContributionCert :=
      Array.mkEmpty (hdr.numLayers * hdr.numHeads)
    for l in [:hdr.numLayers] do
      let p1 := ln1Params.getD l defP
      let (ln1Out, ln1VarLB) :=
        fixedLayerNormRowApprox cfg residualUnion p1.gamma p1.beta eps soundnessBits
      let ln1MaxAbsGamma : Rat :=
        Rat.normalize (maxAbsVecFixed p1.gamma) cfg.scaleNat (den_nz := by
          have h10pos : (0 : Nat) < 10 := by decide
          exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
      let ln1Bound :=
        if ln1VarLB > 0 then
          layerNormOpBoundLocal ln1MaxAbsGamma ln1VarLB eps soundnessBits
        else
          layerNormOpBoundConservative ln1MaxAbsGamma eps soundnessBits
      let mut attnUnion : Array Fixed10Interval :=
        Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      for hIdx in [:hdr.numHeads] do
        let wqScaledE ←
          ExceptT.mk (readMatrixOpBoundScaled h hdr.modelDim hdr.headDim scalePow10)
        let wqOp := ratOfScaledNat scalePow10 wqScaledE
        let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
        let wkScaledE ←
          ExceptT.mk (readMatrixOpBoundScaled h hdr.modelDim hdr.headDim scalePow10)
        let wkOp := ratOfScaledNat scalePow10 wkScaledE
        let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
        let (vHidden0, _nWv) ←
          ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
            hdr.modelDim hdr.headDim ln1Out scalePow10)
        let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
        let vHidden := addVecFixed vHidden0 bV
        let vCenteredOpBound := centeredAbsSumFixed cfg vHidden
        let (vOut, nWo) ←
          ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
            hdr.headDim hdr.modelDim vHidden scalePow10)
        attnUnion := addVecFixed attnUnion vOut
        let softmaxJacobianBound := softmaxJacobianNormInfWorst
        let attnW := ln1Bound * softmaxJacobianBound * vCenteredOpBound * nWo
        let cert : HeadLocalContributionCert := {
          layerIdx := l
          headIdx := hIdx
          soundnessBits := soundnessBits
          ln1MaxAbsGamma := ln1MaxAbsGamma
          ln1VarianceLowerBound := ln1VarLB
          ln1Bound := ln1Bound
          wqOpBound := wqOp
          wkOpBound := wkOp
          wvOpBound := vCenteredOpBound
          woOpBound := nWo
          qkFactorBound := wqOp * wkOp
          softmaxJacobianNormInfUpperBound := softmaxJacobianBound
          attnJacBound := attnW
        }
        if cert.check eps then
          heads := heads.push cert
        else
          throw "local head contribution certificate failed internal checks"
      let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
      attnUnion := addVecFixed attnUnion attnBias
      residualUnion := addVecFixed residualUnion attnUnion
      let p2 := ln2Params.getD l defP
      let (ln2Out, _ln2VarLB) :=
        fixedLayerNormRowApprox cfg residualUnion p2.gamma p2.beta eps soundnessBits
      let (hidden0, _nWin) ←
        ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
          hdr.modelDim hdr.hiddenDim ln2Out scalePow10)
      let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
      let hiddenB := addVecFixed hidden0 bIn
      let actHidden := geluOverapproxFixedVec cfg hdr.geluDerivTarget hiddenB
      let (mlpOut0, _nWout) ←
        ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
          hdr.hiddenDim hdr.modelDim actHidden scalePow10)
      let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
      let mlpOut := addVecFixed mlpOut0 bOut
      residualUnion := addVecFixed residualUnion mlpOut
      let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
      let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
      let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
      let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
    return heads
  action.run

/-- Compute local attention pattern bounds for a specific binary head. -/
private def certifyHeadPatternLocalBinary
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath : System.FilePath)
    (inputDelta : Rat)
    (targetOffset : Int)
    (keyOffset : Int)
    (maxSeqLen : Nat)
    (tightPattern : Bool)
    (tightPatternLayers : Nat)
    (perRowPatternLayers : Nat)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) :
    IO (Except String HeadPatternCert) := do
  let cfg : Fixed10Cfg := scaleCfgOfPow10 scalePow10
  let slack : Int := fixedUlpSlack
  let action : ExceptT String IO HeadPatternCert := do
    let (hdr, ln1Params, ln2Params) ←
      ExceptT.mk (collectLayerNormParamsBinary path scalePow10 slack)
    if layerIdx ≥ hdr.numLayers then
      throw s!"layer index {layerIdx} out of range"
    if headIdx ≥ hdr.numHeads then
      throw s!"head index {headIdx} out of range"
    if hdr.seqLen > maxSeqLen then
      throw s!"seq_len {hdr.seqLen} exceeds maxSeqLen {maxSeqLen}"
    let residuals0 ←
      ExceptT.mk
        (loadEmbeddingsIntervalsBinary inputPath hdr.modelDim inputDelta scalePow10)
    let (hdrTok, tokens) ← ExceptT.mk (loadTokensBinary inputPath)
    if hdrTok.seqLen ≠ hdr.seqLen then
      throw "token/embedding seq_len mismatch"
    let keyOffsetNat? : Option Nat :=
      if keyOffset ≥ 0 then some (Int.toNat keyOffset) else none
    let keyOffsetNeg : Nat := Int.toNat (-keyOffset)
    let h ← IO.FS.Handle.mk path IO.FS.Mode.read
    let _ ← ExceptT.mk (readBinaryHeader h)
    let _ ← ExceptT.mk (skipI32Array h hdr.seqLen)
    let _ ← ExceptT.mk (skipF64Array h (hdr.seqLen * hdr.modelDim))
    let defP : LayerNormParamsFixed := {
      gamma := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      beta := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
    }
    let mut residuals := residuals0
    for l in [:hdr.numLayers] do
      let p1 := ln1Params.getD l defP
      let ln1Rows := fixedLayerNormRowsApprox cfg residuals p1 eps soundnessBits
      if l = layerIdx then
        let mut wq? : Option (Array Int) := none
        let mut bq? : Option (Array Int) := none
        let mut wk? : Option (Array Int) := none
        let mut bk? : Option (Array Int) := none
        for hIdx in [:hdr.numHeads] do
          if hIdx = headIdx then
            let wq ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
            wq? := some wq
            let bQ ←
              ExceptT.mk <|
                readScaledFloatArray h hdr.headDim scalePow10
            bq? := some bQ
          else
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            pure ()
          if hIdx = headIdx then
            let wk ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
            wk? := some wk
            let bK ←
              ExceptT.mk <|
                readScaledFloatArray h hdr.headDim scalePow10
            bk? := some bK
          else
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            pure ()
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          let _ ← ExceptT.mk (skipF64Array h (hdr.headDim * hdr.modelDim))
        let wq ←
          match wq? with
          | none => throw "missing W_Q for requested head"
          | some xs => pure xs
        let bQ ←
          match bq? with
          | none => throw "missing b_Q for requested head"
          | some xs => pure xs
        let wk ←
          match wk? with
          | none => throw "missing W_K for requested head"
          | some xs => pure xs
        let bK ←
          match bk? with
          | none => throw "missing b_K for requested head"
          | some xs => pure xs
        let bQIntervals := intervalsFromScaled bQ slack
        let bKIntervals := intervalsFromScaled bK slack
        let mut qRows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
        let mut kRows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
        let mut rowIdx : Nat := 0
        while rowIdx < ln1Rows.size do
          let row := ln1Rows[rowIdx]!
          let qRow0 := matMulIntervalsFromScaled cfg slack
            hdr.modelDim hdr.headDim wq row
          let kRow0 := matMulIntervalsFromScaled cfg slack
            hdr.modelDim hdr.headDim wk row
          qRows := qRows.push (addVecFixed qRow0 bQIntervals)
          kRows := kRows.push (addVecFixed kRow0 bKIntervals)
          rowIdx := rowIdx + 1
        let mut minTargetLower? : Option Int := none
        let mut maxOtherUpper? : Option Int := none
        let mut minTargetCount? : Option Nat := none
        let mut i : Nat := 0
        while i < hdr.seqLen do
          let ti : Int := (Int.ofNat i) + targetOffset
          if ti < 0 || ti ≥ (Int.ofNat hdr.seqLen) then
            pure ()
          else
            let tIdx : Nat := Int.toNat ti
            let targetTok := tokens[tIdx]!
            let qRow := qRows[i]!
            let mut targetLower? : Option Int := none
            let mut targetMaxLower? : Option Int := none
            let mut maxOtherUpperRow? : Option Int := none
            let mut targetCount : Nat := 0
            let mut j : Nat := 0
            while j < hdr.seqLen do
              if !causalPattern || j ≤ i then
                let dot := fixedDotInterval cfg qRow (kRows[j]!)
                let isMatch : Bool :=
                  match keyOffsetNat? with
                  | some k =>
                      let idx := j + k
                      idx < hdr.seqLen && tokens[idx]! = targetTok
                  | none =>
                      if j < keyOffsetNeg then
                        false
                      else
                        tokens[j - keyOffsetNeg]! = targetTok
                if isMatch then
                  targetCount := targetCount + 1
                  targetLower? :=
                    match targetLower? with
                    | none => some dot.lo
                    | some m => some (min m dot.lo)
                  targetMaxLower? :=
                    match targetMaxLower? with
                    | none => some dot.lo
                    | some m => some (max m dot.lo)
                else
                  let cur := dot.hi
                  maxOtherUpperRow? :=
                    match maxOtherUpperRow? with
                    | none => some cur
                    | some m => some (max m cur)
              else
                pure ()
              j := j + 1
            let targetLowerRow? :=
              if tightPattern then targetMaxLower? else targetLower?
            match targetLowerRow? with
            | none => pure ()
            | some targetLower =>
              let maxOtherUpperRow :=
                match maxOtherUpperRow? with
                | none => targetLower
                | some v => v
              minTargetLower? :=
                match minTargetLower? with
                | none => some targetLower
                | some m => some (min m targetLower)
              maxOtherUpper? :=
                match maxOtherUpper? with
                | none => some maxOtherUpperRow
                | some m => some (max m maxOtherUpperRow)
              minTargetCount? :=
                match minTargetCount? with
                | none => some targetCount
                | some m => some (min m targetCount)
          i := i + 1
        let minTargetLower ←
          match minTargetLower? with
          | none => throw "no valid target positions for the requested offset"
          | some v => pure v
        let minTargetCount : Nat :=
          match minTargetCount? with
          | none => 0
          | some v => v
        let targetCountLB : Nat :=
          if tightPattern then (if minTargetCount > 0 then 1 else 0) else minTargetCount
        let maxOtherUpper :=
          match maxOtherUpper? with
          | none => minTargetLower
          | some v => v
        let marginInt : Int := minTargetLower - maxOtherUpper
        let targetLower := ratOfScaledInt scalePow10 minTargetLower
        let otherUpper := ratOfScaledInt scalePow10 maxOtherUpper
        let margin := ratOfScaledInt scalePow10 marginInt
        let weightLB : Rat :=
          softmaxTargetWeightLowerBound hdr.seqLen targetCountLB margin softmaxExpEffort
        let cert : HeadPatternCert := {
          layerIdx := layerIdx
          headIdx := headIdx
          seqLen := hdr.seqLen
          targetOffset := targetOffset
          keyOffset := keyOffset
          targetCountLowerBound := targetCountLB
          targetLogitLowerBound := targetLower
          otherLogitUpperBound := otherUpper
          marginLowerBound := margin
          softmaxExpEffort := softmaxExpEffort
          targetWeightLowerBound := weightLB
        }
        if cert.check then
          return cert
        throw "head pattern certificate failed internal consistency checks"
      else
        let tightLayers : Nat :=
          if tightPattern then Nat.max 1 tightPatternLayers else 0
        if tightLayers > 0 && layerIdx ≤ l + tightLayers then
          if causalPattern then
            let zeroRow : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let mut attnRows : Array (Array Fixed10Interval) :=
              Array.replicate hdr.seqLen zeroRow
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
              let mut rowIdx : Nat := 0
              while rowIdx < ln1Rows.size do
                let row := ln1Rows[rowIdx]!
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
                rowIdx := rowIdx + 1
              let headRows := prefixUnionRowsFixed vOutRows
              attnRows := addRowsFixed attnRows headRows
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnRows := addVecFixedRows attnRows attnBias
            residuals := addRowsFixed residuals attnRows
          else
            let mut attnUnion : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let groupRows := groupUnionRowsByToken ln1Rows tokens
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty groupRows.size
              let mut rowIdx : Nat := 0
              while rowIdx < groupRows.size do
                let row := groupRows[rowIdx]!
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
                rowIdx := rowIdx + 1
              let vUnion := unionRowsFixed vOutRows
              attnUnion := addVecFixed attnUnion vUnion
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnUnion := addVecFixed attnUnion attnBias
            residuals := addVecFixedRows residuals attnUnion
        else
          let ln1Union := unionRowsFixed ln1Rows
          let mut attnUnion : Array Fixed10Interval :=
            Array.replicate hdr.modelDim { lo := 0, hi := 0 }
          for _h in [:hdr.numHeads] do
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let (vHidden0, _nWv) ←
              ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
                hdr.modelDim hdr.headDim ln1Union scalePow10)
            let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
            let vHidden := addVecFixed vHidden0 bV
            let (vOut, _nWo) ←
              ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
                hdr.headDim hdr.modelDim vHidden scalePow10)
            attnUnion := addVecFixed attnUnion vOut
          let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          attnUnion := addVecFixed attnUnion attnBias
          residuals := addVecFixedRows residuals attnUnion
        let p2 := ln2Params.getD l defP
        let ln2Rows := fixedLayerNormRowsApprox cfg residuals p2 eps soundnessBits
        let perRowLayers : Nat := perRowPatternLayers
        if perRowLayers > 0 && layerIdx ≤ l + perRowLayers then
          let wIn ←
            ExceptT.mk <| readScaledFloatArray h (hdr.modelDim * hdr.hiddenDim) scalePow10
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let wOut ←
            ExceptT.mk <| readScaledFloatArray h (hdr.hiddenDim * hdr.modelDim) scalePow10
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpRows :=
            mlpRowsFromScaled cfg hdr.geluDerivTarget slack
              hdr.modelDim hdr.hiddenDim wIn wOut bIn bOut ln2Rows
          residuals := addRowsFixed residuals mlpRows
        else
          let residuals' ←
            ExceptT.mk (mlpUnionStepBinary cfg slack h
              hdr.modelDim hdr.hiddenDim ln2Rows residuals scalePow10)
          residuals := residuals'
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
    throw "target layer not reached"
  action.run

/-- Minimum relative improvement required to keep increasing softmax exp effort. -/
private def defaultSoftmaxEffortMinRelImprove : Rat := (1 : Rat) / 100

/-- Choose a softmax exp effort by iterating until improvements are negligible. -/
private def chooseSoftmaxExpEffort
    (seqLen : Nat) (margin : Rat) (maxEffort : Nat) :
    Nat × Rat × Rat :=
  let startEffort : Nat := if maxEffort = 0 then 0 else 1
  let weight0 : Rat := softmaxMaxProbLowerBound seqLen margin startEffort
  let jac0 : Rat := softmaxJacobianNormInfBoundFromMargin seqLen margin startEffort
  if startEffort ≥ maxEffort then
    (startEffort, weight0, jac0)
  else
    Id.run do
      let mut bestEff : Nat := startEffort
      let mut bestWeight : Rat := weight0
      let mut bestJac : Rat := jac0
      let mut eff : Nat := startEffort
      while eff < maxEffort do
        eff := eff + 1
        let weight := softmaxMaxProbLowerBound seqLen margin eff
        let jac := softmaxJacobianNormInfBoundFromMargin seqLen margin eff
        if jac < bestJac then
          let relImprove :=
            if bestJac = 0 then 0 else (bestJac - jac) / bestJac
          bestEff := eff
          bestWeight := weight
          bestJac := jac
          if relImprove < defaultSoftmaxEffortMinRelImprove then
            eff := maxEffort
        else
          eff := maxEffort
      return (bestEff, bestWeight, bestJac)

/-- Compute local head best-match pattern bounds for a specific `.nfpt` head (binary only). -/
private def certifyHeadPatternBestMatchLocalBinary
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (queryPos? : Option Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath : System.FilePath)
    (inputDelta : Rat)
    (targetOffset : Int)
    (keyOffset : Int)
    (maxSeqLen : Nat)
    (tightPattern : Bool)
    (tightPatternLayers : Nat)
    (perRowPatternLayers : Nat)
    (useAffine : Bool)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true)
    (shared? : Option SharedBinaryInputs := none)
    (prefix? : Option SharedBinaryPrefix := none) :
    IO (Except String HeadBestMatchPatternCert) := do
  let cfg : Fixed10Cfg := scaleCfgOfPow10 scalePow10
  let slack : Int := fixedUlpSlack
  let action : ExceptT String IO HeadBestMatchPatternCert := do
    let timingEnabled ← ExceptT.lift <| IO.getEnv "NFP_TIMING"
    let timing : Bool := timingEnabled.isSome
    let timeIt {α : Type} (label : String) (work : ExceptT String IO α) :
        ExceptT String IO α := do
      if !timing then
        work
      else
        let t0 ← ExceptT.lift IO.monoNanosNow
        let r ← work
        let t1 ← ExceptT.lift IO.monoNanosNow
        let dtMs := (t1 - t0) / 1000000
        ExceptT.lift <| IO.eprintln s!"timing:{label} {dtMs}ms"
        return r
    let (hdr, ln1Params, ln2Params, residualsBase, tokensBase) ←
      timeIt "load_shared" <| match shared? with
      | some shared => do
          if shared.scalePow10 ≠ scalePow10 then
            throw "shared scalePow10 mismatch"
          if shared.inputDelta ≠ inputDelta then
            throw "shared inputDelta mismatch"
          pure (shared.hdr, shared.ln1Params, shared.ln2Params, shared.residuals0, shared.tokens)
      | none => do
          let (hdr, ln1Params, ln2Params) ←
            timeIt "load_ln_params" <|
              ExceptT.mk (collectLayerNormParamsBinary path scalePow10 slack)
          let residuals0 ←
            timeIt "load_embeddings" <|
              ExceptT.mk
                (loadEmbeddingsIntervalsBinary inputPath hdr.modelDim inputDelta scalePow10)
          let (hdrTok, tokens) ←
            timeIt "load_tokens" <| ExceptT.mk (loadTokensBinary inputPath)
          if hdrTok.seqLen ≠ hdr.seqLen then
            throw "token/embedding seq_len mismatch"
          pure (hdr, ln1Params, ln2Params, residuals0, tokens)
    if layerIdx ≥ hdr.numLayers then
      throw s!"layer index {layerIdx} out of range"
    if headIdx ≥ hdr.numHeads then
      throw s!"head index {headIdx} out of range"
    if hdr.seqLen > maxSeqLen then
      throw s!"seq_len {hdr.seqLen} exceeds maxSeqLen {maxSeqLen}"
    let queryPos : Nat :=
      match queryPos? with
      | some q => q
      | none =>
          if hdr.seqLen = 0 then 0 else hdr.seqLen - 1
    if queryPos ≥ hdr.seqLen then
      throw s!"queryPos {queryPos} out of range"
    let seqLenEff : Nat := if causalPattern then queryPos + 1 else hdr.seqLen
    let (residuals0, tokens) ←
      match prefix? with
      | some pref =>
          if pref.seqLenEff ≠ seqLenEff then
            throw "prefix seq_len mismatch"
          pure (pref.residuals.get, pref.tokens.get)
      | none =>
          let residuals0 :=
            if causalPattern then takePrefix residualsBase seqLenEff else residualsBase
          let tokens := if causalPattern then takePrefix tokensBase seqLenEff else tokensBase
          pure (residuals0, tokens)
    let keyOffsetNat? : Option Nat :=
      if keyOffset ≥ 0 then some (Int.toNat keyOffset) else none
    let keyOffsetNeg : Nat := Int.toNat (-keyOffset)
    let h ← IO.FS.Handle.mk path IO.FS.Mode.read
    let _ ← ExceptT.mk (readBinaryHeader h)
    let _ ← ExceptT.mk (skipI32Array h hdr.seqLen)
    let _ ← ExceptT.mk (skipF64Array h (hdr.seqLen * hdr.modelDim))
    let defP : LayerNormParamsFixed := {
      gamma := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      beta := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
    }
    let mut residuals := residuals0
    for l in [:hdr.numLayers] do
      let p1 := ln1Params.getD l defP
      let mut ln1Rows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
      for row in residuals do
        let (ln1Out, _ln1VarLB) :=
          fixedLayerNormRowApprox cfg row p1.gamma p1.beta eps soundnessBits
        ln1Rows := ln1Rows.push ln1Out
      if l = layerIdx then
        let tPattern0? ←
          if timing then
            let t0 ← ExceptT.lift IO.monoNanosNow
            pure (some t0)
          else
            pure none
        let mut wq? : Option (Array Int) := none
        let mut bq? : Option (Array Int) := none
        let mut wk? : Option (Array Int) := none
        let mut bk? : Option (Array Int) := none
        for hIdx in [:hdr.numHeads] do
          if hIdx = headIdx then
            let wq ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
            wq? := some wq
            let bQ ←
              ExceptT.mk <|
                readScaledFloatArray h hdr.headDim scalePow10
            bq? := some bQ
            let wk ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
            wk? := some wk
            let bK ←
              ExceptT.mk <|
                readScaledFloatArray h hdr.headDim scalePow10
            bk? := some bK
          else
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          let _ ← ExceptT.mk (skipF64Array h (hdr.headDim * hdr.modelDim))
        let wq ←
          match wq? with
          | none => throw "missing W_Q for requested head"
          | some xs => pure xs
        let bQ ←
          match bq? with
          | none => throw "missing b_Q for requested head"
          | some xs => pure xs
        let wk ←
          match wk? with
          | none => throw "missing W_K for requested head"
          | some xs => pure xs
        let bK ←
          match bk? with
          | none => throw "missing b_K for requested head"
          | some xs => pure xs
        let bQIntervals := intervalsFromScaled bQ slack
        let bKIntervals := intervalsFromScaled bK slack
        let ti : Int := (Int.ofNat queryPos) + targetOffset
        if ti < 0 || ti ≥ (Int.ofNat seqLenEff) then
          throw "query position has no valid target offset"
        let tIdx : Nat := Int.toNat ti
        let targetTok := tokens[tIdx]!
        let mut bestMatchLower? : Option Int := none
        let mut bestNonmatchUpper? : Option Int := none
        if useAffine then
          let (qInputCenters, qInputRadii, _qAbsInput) :=
            rowCentersRadiiAbsInt (ln1Rows[queryPos]!)
          let (qCenters0, qRadii0) :=
            matMulCentersRadiiIntSlack cfg slack
              hdr.modelDim hdr.headDim wq qInputCenters qInputRadii
          let bQCenters := bQ
          let bKCenters := bK
          let bQRadii := bQIntervals.map intervalRadiusInt
          let bKRadii := bKIntervals.map intervalRadiusInt
          let qCenters := addVecScaledInt qCenters0 bQCenters 1
          let qRadii := addVecScaledInt qRadii0 bQRadii 1
          let useTasks := seqLenEff > 32
          if useTasks then
            let chunkSize : Nat := 16
            let numChunks : Nat := (seqLenEff + chunkSize - 1) / chunkSize
            let mut tasks : Array (Task (Option Int × Option Int)) := Array.mkEmpty numChunks
            let mut chunkIdx : Nat := 0
            while chunkIdx < numChunks do
              let start := chunkIdx * chunkSize
              let stop := min seqLenEff (start + chunkSize)
              tasks := tasks.push <| Task.spawn (fun _ =>
                Id.run do
                  let mut bestMatchLower? : Option Int := none
                  let mut bestNonmatchUpper? : Option Int := none
                  let mut j := start
                  while j < stop do
                    if !causalPattern || j ≤ queryPos then
                      let (kInputCenters, kInputRadii, _kAbsInput) :=
                        rowCentersRadiiAbsInt (ln1Rows[j]!)
                      let (kCenters0, kRadii0) :=
                        matMulCentersRadiiIntSlack cfg slack
                          hdr.modelDim hdr.headDim wk kInputCenters kInputRadii
                      let kCenters := addVecScaledInt kCenters0 bKCenters 1
                      let kRadii := addVecScaledInt kRadii0 bKRadii 1
                      let dot :=
                        dotIntervalFromCentersRadiiInt cfg qCenters qRadii kCenters kRadii
                      let isMatch : Bool :=
                        match keyOffsetNat? with
                        | some k =>
                            let idx := j + k
                            idx < seqLenEff && tokens[idx]! = targetTok
                        | none =>
                            if j < keyOffsetNeg then
                              false
                            else
                              tokens[j - keyOffsetNeg]! = targetTok
                      if isMatch then
                        bestMatchLower? :=
                          match bestMatchLower? with
                          | none => some dot.lo
                          | some m => some (max m dot.lo)
                      else
                        bestNonmatchUpper? :=
                          match bestNonmatchUpper? with
                          | none => some dot.hi
                          | some m => some (max m dot.hi)
                    j := j + 1
                  return (bestMatchLower?, bestNonmatchUpper?))
              chunkIdx := chunkIdx + 1
            for t in tasks do
              let (matchChunk?, nonmatchChunk?) := t.get
              if matchChunk?.isSome then
                bestMatchLower? :=
                  match bestMatchLower?, matchChunk? with
                  | none, some v => some v
                  | some cur, some v => some (max cur v)
                  | some cur, none => some cur
                  | none, none => none
              if nonmatchChunk?.isSome then
                bestNonmatchUpper? :=
                  match bestNonmatchUpper?, nonmatchChunk? with
                  | none, some v => some v
                  | some cur, some v => some (max cur v)
                  | some cur, none => some cur
                  | none, none => none
          else
            for j in [:seqLenEff] do
              if !causalPattern || j ≤ queryPos then
                let (kInputCenters, kInputRadii, _kAbsInput) :=
                  rowCentersRadiiAbsInt (ln1Rows[j]!)
                let (kCenters0, kRadii0) :=
                  matMulCentersRadiiIntSlack cfg slack
                    hdr.modelDim hdr.headDim wk kInputCenters kInputRadii
                let kCenters := addVecScaledInt kCenters0 bKCenters 1
                let kRadii := addVecScaledInt kRadii0 bKRadii 1
                let dot :=
                  dotIntervalFromCentersRadiiInt cfg qCenters qRadii kCenters kRadii
                let isMatch : Bool :=
                  match keyOffsetNat? with
                  | some k =>
                      let idx := j + k
                      idx < seqLenEff && tokens[idx]! = targetTok
                  | none =>
                      if j < keyOffsetNeg then
                        false
                      else
                        tokens[j - keyOffsetNeg]! = targetTok
                if isMatch then
                  bestMatchLower? :=
                    match bestMatchLower? with
                    | none => some dot.lo
                    | some m => some (max m dot.lo)
                else
                  bestNonmatchUpper? :=
                    match bestNonmatchUpper? with
                    | none => some dot.hi
                    | some m => some (max m dot.hi)
              else
                pure ()
        else
          let qRow := matMulIntervalsFromScaled cfg slack
            hdr.modelDim hdr.headDim wq (ln1Rows[queryPos]!)
          let qRow := addVecFixed qRow bQIntervals
          let mut kRows : Array (Array Fixed10Interval) := Array.mkEmpty seqLenEff
          for row in ln1Rows do
            let kRow0 := matMulIntervalsFromScaled cfg slack
              hdr.modelDim hdr.headDim wk row
            kRows := kRows.push (addVecFixed kRow0 bKIntervals)
          for j in [:seqLenEff] do
            if !causalPattern || j ≤ queryPos then
              let dot := fixedDotInterval cfg qRow (kRows[j]!)
              let isMatch : Bool :=
                match keyOffsetNat? with
                | some k =>
                    let idx := j + k
                    idx < seqLenEff && tokens[idx]! = targetTok
                | none =>
                    if j < keyOffsetNeg then
                      false
                    else
                      tokens[j - keyOffsetNeg]! = targetTok
              if isMatch then
                bestMatchLower? :=
                  match bestMatchLower? with
                  | none => some dot.lo
                  | some m => some (max m dot.lo)
              else
                bestNonmatchUpper? :=
                  match bestNonmatchUpper? with
                  | none => some dot.hi
                  | some m => some (max m dot.hi)
            else
              pure ()
        let bestMatchLower ←
          match bestMatchLower? with
          | none => throw "no matching keys for the requested offset"
          | some v => pure v
        let bestNonmatchUpper :=
          match bestNonmatchUpper? with
          | none => bestMatchLower
          | some v => v
        let marginInt : Int := bestMatchLower - bestNonmatchUpper
        let bestMatchLowerRat := ratOfScaledInt scalePow10 bestMatchLower
        let bestNonmatchUpperRat := ratOfScaledInt scalePow10 bestNonmatchUpper
        let margin := ratOfScaledInt scalePow10 marginInt
        let (effortUsed, weightLB, softmaxJacobianUB) :=
          chooseSoftmaxExpEffort hdr.seqLen margin softmaxExpEffort
        let cert : HeadBestMatchPatternCert := {
          layerIdx := layerIdx
          headIdx := headIdx
          seqLen := hdr.seqLen
          queryPos := queryPos
          targetOffset := targetOffset
          keyOffset := keyOffset
          targetToken := targetTok
          bestMatchLogitLowerBound := bestMatchLowerRat
          bestNonmatchLogitUpperBound := bestNonmatchUpperRat
          marginLowerBound := margin
          softmaxExpEffort := effortUsed
          bestMatchWeightLowerBound := weightLB
          softmaxJacobianNormInfUpperBound := softmaxJacobianUB
        }
        if cert.check then
          if let some t0 := tPattern0? then
            let t1 ← ExceptT.lift IO.monoNanosNow
            let dtMs := (t1 - t0) / 1000000
            ExceptT.lift <| IO.eprintln s!"timing:layer{l}:pattern {dtMs}ms"
          return cert
        throw "best-match head pattern certificate failed internal consistency checks"
      else
        let tightLayers : Nat :=
          if tightPattern then Nat.max 1 tightPatternLayers else 0
        if tightLayers > 0 && layerIdx ≤ l + tightLayers then
          if causalPattern then
            let zeroRow : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let mut attnRows : Array (Array Fixed10Interval) :=
              Array.replicate seqLenEff zeroRow
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty seqLenEff
              for row in ln1Rows do
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
              let headRows := prefixUnionRowsFixed vOutRows
              attnRows := addRowsFixed attnRows headRows
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnRows := addVecFixedRows attnRows attnBias
            residuals := addRowsFixed residuals attnRows
          else
            let mut attnUnion : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let groupRows := groupUnionRowsByToken ln1Rows tokens
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty groupRows.size
              for row in groupRows do
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
              let vUnion := unionRowsFixed vOutRows
              attnUnion := addVecFixed attnUnion vUnion
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnUnion := addVecFixed attnUnion attnBias
            residuals := addVecFixedRows residuals attnUnion
        else
          let ln1Union := unionRowsFixed ln1Rows
          let mut attnUnion : Array Fixed10Interval :=
            Array.replicate hdr.modelDim { lo := 0, hi := 0 }
          for _h in [:hdr.numHeads] do
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let (vHidden0, _nWv) ←
              ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
                hdr.modelDim hdr.headDim ln1Union scalePow10)
            let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
            let vHidden := addVecFixed vHidden0 bV
            let (vOut, _nWo) ←
              ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
                hdr.headDim hdr.modelDim vHidden scalePow10)
            attnUnion := addVecFixed attnUnion vOut
          let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          attnUnion := addVecFixed attnUnion attnBias
          residuals := addVecFixedRows residuals attnUnion
        let p2 := ln2Params.getD l defP
        let mut ln2Rows : Array (Array Fixed10Interval) := Array.mkEmpty seqLenEff
        for row in residuals do
          let (ln2Out, _ln2VarLB) :=
            fixedLayerNormRowApprox cfg row p2.gamma p2.beta eps soundnessBits
          ln2Rows := ln2Rows.push ln2Out
        let perRowLayers : Nat := perRowPatternLayers
        if perRowLayers > 0 && layerIdx ≤ l + perRowLayers then
          let wIn ←
            ExceptT.mk <| readScaledFloatArray h (hdr.modelDim * hdr.hiddenDim) scalePow10
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let wOut ←
            ExceptT.mk <| readScaledFloatArray h (hdr.hiddenDim * hdr.modelDim) scalePow10
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpRows :=
            mlpRowsFromScaled cfg hdr.geluDerivTarget slack
              hdr.modelDim hdr.hiddenDim wIn wOut bIn bOut ln2Rows
          residuals := addRowsFixed residuals mlpRows
        else
          let ln2Union := unionRowsFixed ln2Rows
          let (hidden0, _nWin) ←
            ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
              hdr.modelDim hdr.hiddenDim ln2Union scalePow10)
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let hiddenB := addVecFixed hidden0 bIn
          let actHidden := geluOverapproxFixedVec cfg hdr.geluDerivTarget hiddenB
          let (mlpOut0, _nWout) ←
            ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
              hdr.hiddenDim hdr.modelDim actHidden scalePow10)
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpOut := addVecFixed mlpOut0 bOut
          residuals := addVecFixedRows residuals mlpOut
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
    throw "target layer not reached"
  action.run

/-- Compute local head best-match pattern bounds for all valid query positions (binary only). -/
private def certifyHeadPatternBestMatchLocalBinarySweep
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath : System.FilePath)
    (inputDelta : Rat)
    (targetOffset : Int)
    (keyOffset : Int)
    (maxSeqLen : Nat)
    (tightPattern : Bool)
    (tightPatternLayers : Nat)
    (perRowPatternLayers : Nat)
    (useAffine : Bool)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true)
    (shared? : Option SharedBinaryInputs := none) :
    IO (Except String (Array HeadBestMatchPatternCert)) := do
  let cfg : Fixed10Cfg := scaleCfgOfPow10 scalePow10
  let slack : Int := fixedUlpSlack
  let action : ExceptT String IO (Array HeadBestMatchPatternCert) := do
    let (hdr, ln1Params, ln2Params, residuals0, tokens) ←
      match shared? with
      | some shared =>
          if shared.scalePow10 ≠ scalePow10 then
            throw "shared scalePow10 mismatch"
          if shared.inputDelta ≠ inputDelta then
            throw "shared inputDelta mismatch"
          pure (shared.hdr, shared.ln1Params, shared.ln2Params, shared.residuals0, shared.tokens)
      | none =>
          let (hdr, ln1Params, ln2Params) ←
            ExceptT.mk (collectLayerNormParamsBinary path scalePow10 slack)
          let residuals0 ←
            ExceptT.mk
              (loadEmbeddingsIntervalsBinary inputPath hdr.modelDim inputDelta scalePow10)
          let (hdrTok, tokens) ← ExceptT.mk (loadTokensBinary inputPath)
          if hdrTok.seqLen ≠ hdr.seqLen then
            throw "token/embedding seq_len mismatch"
          pure (hdr, ln1Params, ln2Params, residuals0, tokens)
    if layerIdx ≥ hdr.numLayers then
      throw s!"layer index {layerIdx} out of range"
    if headIdx ≥ hdr.numHeads then
      throw s!"head index {headIdx} out of range"
    if hdr.seqLen > maxSeqLen then
      throw s!"seq_len {hdr.seqLen} exceeds maxSeqLen {maxSeqLen}"
    if useAffine then
      throw "affine sweep is unsupported; use --bestMatch without --sweep"
    let h ← IO.FS.Handle.mk path IO.FS.Mode.read
    let _ ← ExceptT.mk (readBinaryHeader h)
    let _ ← ExceptT.mk (skipI32Array h hdr.seqLen)
    let _ ← ExceptT.mk (skipF64Array h (hdr.seqLen * hdr.modelDim))
    let defP : LayerNormParamsFixed := {
      gamma := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      beta := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
    }
    let mut residuals := residuals0
    for l in [:hdr.numLayers] do
      let p1 := ln1Params.getD l defP
      let mut ln1Rows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
      for row in residuals do
        let (ln1Out, _ln1VarLB) :=
          fixedLayerNormRowApprox cfg row p1.gamma p1.beta eps soundnessBits
        ln1Rows := ln1Rows.push ln1Out
      if l = layerIdx then
        let mut wq? : Option (Array Int) := none
        let mut bq? : Option (Array Int) := none
        let mut wk? : Option (Array Int) := none
        let mut bk? : Option (Array Int) := none
        for hIdx in [:hdr.numHeads] do
          if hIdx = headIdx then
            let wq ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
            wq? := some wq
            let bQ ←
              ExceptT.mk <|
                readScaledFloatArray h hdr.headDim scalePow10
            bq? := some bQ
            let wk ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
            wk? := some wk
            let bK ←
              ExceptT.mk <|
                readScaledFloatArray h hdr.headDim scalePow10
            bk? := some bK
          else
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          let _ ← ExceptT.mk (skipF64Array h (hdr.headDim * hdr.modelDim))
        let wq ←
          match wq? with
          | none => throw "missing W_Q for requested head"
          | some xs => pure xs
        let bQ ←
          match bq? with
          | none => throw "missing b_Q for requested head"
          | some xs => pure xs
        let wk ←
          match wk? with
          | none => throw "missing W_K for requested head"
          | some xs => pure xs
        let bK ←
          match bk? with
          | none => throw "missing b_K for requested head"
          | some xs => pure xs
        let bQIntervals := intervalsFromScaled bQ slack
        let bKIntervals := intervalsFromScaled bK slack
        let mut qRows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
        let mut kRows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
        for row in ln1Rows do
          let qRow0 := matMulIntervalsFromScaled cfg slack
            hdr.modelDim hdr.headDim wq row
          let kRow0 := matMulIntervalsFromScaled cfg slack
            hdr.modelDim hdr.headDim wk row
          qRows := qRows.push (addVecFixed qRow0 bQIntervals)
          kRows := kRows.push (addVecFixed kRow0 bKIntervals)
        let validPositions : Array Nat := Id.run do
          let mut out : Array Nat := Array.mkEmpty hdr.seqLen
          for i in [:hdr.seqLen] do
            let ti : Int := (Int.ofNat i) + targetOffset
            if ti < 0 || ti ≥ (Int.ofNat hdr.seqLen) then
              pure ()
            else
              out := out.push i
          out
        if validPositions.isEmpty then
          throw "no valid query positions for the requested offset"
        let keyOffsetNat? : Option Nat :=
          if keyOffset ≥ 0 then some (Int.toNat keyOffset) else none
        let keyOffsetNeg : Nat := Int.toNat (-keyOffset)
        let computeCert : Nat → Except String HeadBestMatchPatternCert := fun queryPos => do
          let ti : Int := (Int.ofNat queryPos) + targetOffset
          if ti < 0 || ti ≥ (Int.ofNat hdr.seqLen) then
            throw "query position has no valid target offset"
          let tIdx : Nat := Int.toNat ti
          let targetTok := tokens[tIdx]!
          let qRow := qRows[queryPos]!
          let mut bestMatchLower? : Option Int := none
          let mut bestNonmatchUpper? : Option Int := none
          for j in [:hdr.seqLen] do
            if !causalPattern || j ≤ queryPos then
              let dot := fixedDotInterval cfg qRow (kRows[j]!)
              let isMatch : Bool :=
                match keyOffsetNat? with
                | some k =>
                    let idx := j + k
                    idx < hdr.seqLen && tokens[idx]! = targetTok
                | none =>
                    if j < keyOffsetNeg then
                      false
                    else
                      tokens[j - keyOffsetNeg]! = targetTok
              if isMatch then
                bestMatchLower? :=
                  match bestMatchLower? with
                  | none => some dot.lo
                  | some m => some (max m dot.lo)
              else
                bestNonmatchUpper? :=
                  match bestNonmatchUpper? with
                  | none => some dot.hi
                  | some m => some (max m dot.hi)
            else
              pure ()
          let bestMatchLower ←
            match bestMatchLower? with
            | none => throw "no matching keys for the requested offset"
            | some v => pure v
          let bestNonmatchUpper :=
            match bestNonmatchUpper? with
            | none => bestMatchLower
            | some v => v
          let marginInt : Int := bestMatchLower - bestNonmatchUpper
          let bestMatchLowerRat := ratOfScaledInt scalePow10 bestMatchLower
          let bestNonmatchUpperRat := ratOfScaledInt scalePow10 bestNonmatchUpper
          let margin := ratOfScaledInt scalePow10 marginInt
          let (effortUsed, weightLB, softmaxJacobianUB) :=
            chooseSoftmaxExpEffort hdr.seqLen margin softmaxExpEffort
          let cert : HeadBestMatchPatternCert := {
            layerIdx := layerIdx
            headIdx := headIdx
            seqLen := hdr.seqLen
            queryPos := queryPos
            targetOffset := targetOffset
            keyOffset := keyOffset
            targetToken := targetTok
            bestMatchLogitLowerBound := bestMatchLowerRat
            bestNonmatchLogitUpperBound := bestNonmatchUpperRat
            marginLowerBound := margin
            softmaxExpEffort := effortUsed
            bestMatchWeightLowerBound := weightLB
            softmaxJacobianNormInfUpperBound := softmaxJacobianUB
          }
          if cert.check then
            return cert
          throw "best-match head pattern certificate failed internal consistency checks"
        let useTasks := validPositions.size > 32
        let mut certs : Array HeadBestMatchPatternCert := Array.mkEmpty validPositions.size
        if useTasks then
          let tasks := validPositions.map (fun i =>
            Task.spawn (fun _ => computeCert i))
          for t in tasks do
            match t.get with
            | .ok cert => certs := certs.push cert
            | .error e => throw e
        else
          for i in validPositions do
            match computeCert i with
            | .ok cert => certs := certs.push cert
            | .error e => throw e
        return certs
      else
        let tightLayers : Nat :=
          if tightPattern then Nat.max 1 tightPatternLayers else 0
        if tightLayers > 0 && layerIdx ≤ l + tightLayers then
          if causalPattern then
            let zeroRow : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let mut attnRows : Array (Array Fixed10Interval) :=
              Array.replicate hdr.seqLen zeroRow
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
              for row in ln1Rows do
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
              let headRows := prefixUnionRowsFixed vOutRows
              attnRows := addRowsFixed attnRows headRows
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnRows := addVecFixedRows attnRows attnBias
            residuals := addRowsFixed residuals attnRows
          else
            let mut attnUnion : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let groupRows := groupUnionRowsByToken ln1Rows tokens
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty groupRows.size
              for row in groupRows do
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
              let vUnion := unionRowsFixed vOutRows
              attnUnion := addVecFixed attnUnion vUnion
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnUnion := addVecFixed attnUnion attnBias
            residuals := addVecFixedRows residuals attnUnion
        else
          let ln1Union := unionRowsFixed ln1Rows
          let mut attnUnion : Array Fixed10Interval :=
            Array.replicate hdr.modelDim { lo := 0, hi := 0 }
          for _h in [:hdr.numHeads] do
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let (vHidden0, _nWv) ←
              ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
                hdr.modelDim hdr.headDim ln1Union scalePow10)
            let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
            let vHidden := addVecFixed vHidden0 bV
            let (vOut, _nWo) ←
              ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
                hdr.headDim hdr.modelDim vHidden scalePow10)
            attnUnion := addVecFixed attnUnion vOut
          let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          attnUnion := addVecFixed attnUnion attnBias
          residuals := addVecFixedRows residuals attnUnion
        let p2 := ln2Params.getD l defP
        let mut ln2Rows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
        for row in residuals do
          let (ln2Out, _ln2VarLB) :=
            fixedLayerNormRowApprox cfg row p2.gamma p2.beta eps soundnessBits
          ln2Rows := ln2Rows.push ln2Out
        let perRowLayers : Nat := perRowPatternLayers
        if perRowLayers > 0 && layerIdx ≤ l + perRowLayers then
          let wIn ←
            ExceptT.mk <| readScaledFloatArray h (hdr.modelDim * hdr.hiddenDim) scalePow10
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let wOut ←
            ExceptT.mk <| readScaledFloatArray h (hdr.hiddenDim * hdr.modelDim) scalePow10
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpRows :=
            mlpRowsFromScaled cfg hdr.geluDerivTarget slack
              hdr.modelDim hdr.hiddenDim wIn wOut bIn bOut ln2Rows
          residuals := addRowsFixed residuals mlpRows
        else
          let ln2Union := unionRowsFixed ln2Rows
          let (hidden0, _nWin) ←
            ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
              hdr.modelDim hdr.hiddenDim ln2Union scalePow10)
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let hiddenB := addVecFixed hidden0 bIn
          let actHidden := geluOverapproxFixedVec cfg hdr.geluDerivTarget hiddenB
          let (mlpOut0, _nWout) ←
            ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
              hdr.hiddenDim hdr.modelDim actHidden scalePow10)
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpOut := addVecFixed mlpOut0 bOut
          residuals := addVecFixedRows residuals mlpOut
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
    throw "target layer not reached"
  action.run

/-- Compute local head output lower bounds for a single coordinate (binary only). -/
private def certifyHeadValueLowerBoundLocalBinary
    (path : System.FilePath)
    (pattern : HeadPatternCert)
    (coord : Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath : System.FilePath)
    (inputDelta : Rat)
    (maxSeqLen : Nat)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (causalPattern : Bool := true) :
    IO (Except String HeadValueLowerBoundCert) := do
  let cfg : Fixed10Cfg := scaleCfgOfPow10 scalePow10
  let slack : Int := fixedUlpSlack
  let action : ExceptT String IO HeadValueLowerBoundCert := do
    let (hdr, ln1Params, ln2Params) ←
      ExceptT.mk (collectLayerNormParamsBinary path scalePow10 slack)
    if pattern.layerIdx ≥ hdr.numLayers then
      throw s!"layer index {pattern.layerIdx} out of range"
    if pattern.headIdx ≥ hdr.numHeads then
      throw s!"head index {pattern.headIdx} out of range"
    if coord ≥ hdr.modelDim then
      throw s!"coord index {coord} out of range"
    if hdr.seqLen > maxSeqLen then
      throw s!"seq_len {hdr.seqLen} exceeds maxSeqLen {maxSeqLen}"
    let residuals0 ←
      ExceptT.mk
        (loadEmbeddingsIntervalsBinary inputPath hdr.modelDim inputDelta scalePow10)
    let (hdrTok, tokens) ← ExceptT.mk (loadTokensBinary inputPath)
    if hdrTok.seqLen ≠ hdr.seqLen then
      throw "token/embedding seq_len mismatch"
    if pattern.seqLen ≠ hdr.seqLen then
      throw "pattern seq_len mismatch"
    let keyOffsetNat? : Option Nat :=
      if pattern.keyOffset ≥ 0 then some (Int.toNat pattern.keyOffset) else none
    let keyOffsetNeg : Nat := Int.toNat (-pattern.keyOffset)
    let h ← IO.FS.Handle.mk path IO.FS.Mode.read
    let _ ← ExceptT.mk (readBinaryHeader h)
    let _ ← ExceptT.mk (skipI32Array h hdr.seqLen)
    let _ ← ExceptT.mk (skipF64Array h (hdr.seqLen * hdr.modelDim))
    let defP : LayerNormParamsFixed := {
      gamma := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      beta := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
    }
    let mut residuals := residuals0
    for l in [:hdr.numLayers] do
      let p1 := ln1Params.getD l defP
      let mut ln1Rows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
      for row in residuals do
        let (ln1Out, _ln1VarLB) :=
          fixedLayerNormRowApprox cfg row p1.gamma p1.beta eps soundnessBits
        ln1Rows := ln1Rows.push ln1Out
      if l = pattern.layerIdx then
        let mut wv? : Option (Array Int) := none
        let mut bv? : Option (Array Int) := none
        let mut wo? : Option (Array Int) := none
        for hIdx in [:hdr.numHeads] do
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          if hIdx = pattern.headIdx then
            let wv ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
            wv? := some wv
            let bV ←
              ExceptT.mk <|
                readScaledFloatArray h hdr.headDim scalePow10
            bv? := some bV
            let wo ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
            wo? := some wo
          else
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.headDim * hdr.modelDim))
        let wv ←
          match wv? with
          | none => throw "missing W_V for requested head"
          | some xs => pure xs
        let bV ←
          match bv? with
          | none => throw "missing b_V for requested head"
          | some xs => pure xs
        let wo ←
          match wo? with
          | none => throw "missing W_O for requested head"
          | some xs => pure xs
        let bVIntervals := intervalsFromScaled bV slack
        let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
        for row in ln1Rows do
          let vHidden0 := matMulIntervalsFromScaled cfg slack
            hdr.modelDim hdr.headDim wv row
          let vHidden := addVecFixed vHidden0 bVIntervals
          let vOut := matMulIntervalsFromScaled cfg slack
            hdr.headDim hdr.modelDim wo vHidden
          vOutRows := vOutRows.push vOut
        let mut minMatchLo? : Option Int := none
        let mut minNonmatchLo? : Option Int := none
        for i in [:hdr.seqLen] do
          let ti : Int := (Int.ofNat i) + pattern.targetOffset
          if ti < 0 || ti ≥ (Int.ofNat hdr.seqLen) then
            pure ()
          else
            let tIdx : Nat := Int.toNat ti
            let targetTok := tokens[tIdx]!
            let mut matchLo? : Option Int := none
            let mut nonmatchLo? : Option Int := none
            for j in [:hdr.seqLen] do
              if !causalPattern || j ≤ i then
                let row := vOutRows[j]!
                let vCoord := row[coord]!.lo
                let isMatch : Bool :=
                  match keyOffsetNat? with
                  | some k =>
                      let idx := j + k
                      idx < hdr.seqLen && tokens[idx]! = targetTok
                  | none =>
                      if j < keyOffsetNeg then
                        false
                      else
                        tokens[j - keyOffsetNeg]! = targetTok
                if isMatch then
                  matchLo? :=
                    match matchLo? with
                    | none => some vCoord
                    | some m => some (min m vCoord)
                else
                  nonmatchLo? :=
                    match nonmatchLo? with
                    | none => some vCoord
                    | some m => some (min m vCoord)
              else
                pure ()
            let matchLo :=
              match matchLo? with
              | none => 0
              | some v => v
            let nonmatchLo :=
              match nonmatchLo? with
              | none => matchLo
              | some v => v
            minMatchLo? :=
              match minMatchLo? with
              | none => some matchLo
              | some m => some (min m matchLo)
            minNonmatchLo? :=
              match minNonmatchLo? with
              | none => some nonmatchLo
              | some m => some (min m nonmatchLo)
        let matchLo :=
          match minMatchLo? with
          | none => 0
          | some v => v
        let nonmatchLo :=
          match minNonmatchLo? with
          | none => matchLo
          | some v => v
        let matchLoRat := ratOfScaledInt scalePow10 matchLo
        let nonmatchLoRat := ratOfScaledInt scalePow10 nonmatchLo
        let weightLB := pattern.targetWeightLowerBound
        let outputLB := mixLowerBound weightLB matchLoRat nonmatchLoRat
        let cert : HeadValueLowerBoundCert := {
          layerIdx := pattern.layerIdx
          headIdx := pattern.headIdx
          coord := coord
          matchWeightLowerBound := weightLB
          matchCoordLowerBound := matchLoRat
          nonmatchCoordLowerBound := nonmatchLoRat
          outputCoordLowerBound := outputLB
        }
        if cert.check then
          return cert
        throw "head value lower bound failed internal consistency checks"
      else
        let tightLayers : Nat :=
          if tightPattern then Nat.max 1 tightPatternLayers else 0
        if tightLayers > 0 && pattern.layerIdx ≤ l + tightLayers then
          if causalPattern then
            let zeroRow : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let mut attnRows : Array (Array Fixed10Interval) :=
              Array.replicate hdr.seqLen zeroRow
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
              for row in ln1Rows do
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
              let headRows := prefixUnionRowsFixed vOutRows
              attnRows := addRowsFixed attnRows headRows
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnRows := addVecFixedRows attnRows attnBias
            residuals := addRowsFixed residuals attnRows
          else
            let mut attnUnion : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let groupRows := groupUnionRowsByToken ln1Rows tokens
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty groupRows.size
              for row in groupRows do
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
              let vUnion := unionRowsFixed vOutRows
              attnUnion := addVecFixed attnUnion vUnion
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnUnion := addVecFixed attnUnion attnBias
            residuals := addVecFixedRows residuals attnUnion
        else
          let ln1Union := unionRowsFixed ln1Rows
          let mut attnUnion : Array Fixed10Interval :=
            Array.replicate hdr.modelDim { lo := 0, hi := 0 }
          for _h in [:hdr.numHeads] do
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let (vHidden0, _nWv) ←
              ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
                hdr.modelDim hdr.headDim ln1Union scalePow10)
            let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
            let vHidden := addVecFixed vHidden0 bV
            let (vOut, _nWo) ←
              ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
                hdr.headDim hdr.modelDim vHidden scalePow10)
            attnUnion := addVecFixed attnUnion vOut
          let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          attnUnion := addVecFixed attnUnion attnBias
          residuals := addVecFixedRows residuals attnUnion
        let p2 := ln2Params.getD l defP
        let mut ln2Rows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
        for row in residuals do
          let (ln2Out, _ln2VarLB) :=
            fixedLayerNormRowApprox cfg row p2.gamma p2.beta eps soundnessBits
          ln2Rows := ln2Rows.push ln2Out
        let perRowLayers : Nat := perRowPatternLayers
        if perRowLayers > 0 && pattern.layerIdx ≤ l + perRowLayers then
          let wIn ←
            ExceptT.mk <| readScaledFloatArray h (hdr.modelDim * hdr.hiddenDim) scalePow10
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let wOut ←
            ExceptT.mk <| readScaledFloatArray h (hdr.hiddenDim * hdr.modelDim) scalePow10
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpRows :=
            mlpRowsFromScaled cfg hdr.geluDerivTarget slack
              hdr.modelDim hdr.hiddenDim wIn wOut bIn bOut ln2Rows
          residuals := addRowsFixed residuals mlpRows
        else
          let ln2Union := unionRowsFixed ln2Rows
          let (hidden0, _nWin) ←
            ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
              hdr.modelDim hdr.hiddenDim ln2Union scalePow10)
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let hiddenB := addVecFixed hidden0 bIn
          let actHidden := geluOverapproxFixedVec cfg hdr.geluDerivTarget hiddenB
          let (mlpOut0, _nWout) ←
            ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
              hdr.hiddenDim hdr.modelDim actHidden scalePow10)
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpOut := addVecFixed mlpOut0 bOut
          residuals := addVecFixedRows residuals mlpOut
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
    throw "target layer not reached"
  action.run

/-- Compute local head logit-difference lower bounds for a specific head (binary only). -/
private def certifyHeadLogitDiffLowerBoundLocalBinary
    (path : System.FilePath)
    (pattern : HeadPatternCert)
    (targetToken negativeToken : Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath : System.FilePath)
    (inputDelta : Rat)
    (maxSeqLen : Nat)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (causalPattern : Bool := true) :
    IO (Except String HeadLogitDiffLowerBoundCert) := do
  let cfg : Fixed10Cfg := scaleCfgOfPow10 scalePow10
  let slack : Int := fixedUlpSlack
  let action : ExceptT String IO HeadLogitDiffLowerBoundCert := do
    let (hdrDir, direction) ←
      ExceptT.mk (readLogitDiffDirectionBinary path targetToken negativeToken scalePow10 slack)
    let (hdr, ln1Params, ln2Params) ←
      ExceptT.mk (collectLayerNormParamsBinary path scalePow10 slack)
    if hdr.modelDim ≠ hdrDir.modelDim then
      throw "unembedding model_dim mismatch"
    if pattern.layerIdx ≥ hdr.numLayers then
      throw s!"layer index {pattern.layerIdx} out of range"
    if pattern.headIdx ≥ hdr.numHeads then
      throw s!"head index {pattern.headIdx} out of range"
    if hdr.seqLen > maxSeqLen then
      throw s!"seq_len {hdr.seqLen} exceeds maxSeqLen {maxSeqLen}"
    if direction.size ≠ hdr.modelDim then
      throw "logit direction size mismatch"
    let residuals0 ←
      ExceptT.mk
        (loadEmbeddingsIntervalsBinary inputPath hdr.modelDim inputDelta scalePow10)
    let (hdrTok, tokens) ← ExceptT.mk (loadTokensBinary inputPath)
    if hdrTok.seqLen ≠ hdr.seqLen then
      throw "token/embedding seq_len mismatch"
    if pattern.seqLen ≠ hdr.seqLen then
      throw "pattern seq_len mismatch"
    let keyOffsetNat? : Option Nat :=
      if pattern.keyOffset ≥ 0 then some (Int.toNat pattern.keyOffset) else none
    let keyOffsetNeg : Nat := Int.toNat (-pattern.keyOffset)
    let h ← IO.FS.Handle.mk path IO.FS.Mode.read
    let _ ← ExceptT.mk (readBinaryHeader h)
    let _ ← ExceptT.mk (skipI32Array h hdr.seqLen)
    let _ ← ExceptT.mk (skipF64Array h (hdr.seqLen * hdr.modelDim))
    let defP : LayerNormParamsFixed := {
      gamma := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      beta := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
    }
    let mut residuals := residuals0
    for l in [:hdr.numLayers] do
      let p1 := ln1Params.getD l defP
      let mut ln1Rows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
      for row in residuals do
        let (ln1Out, _ln1VarLB) :=
          fixedLayerNormRowApprox cfg row p1.gamma p1.beta eps soundnessBits
        ln1Rows := ln1Rows.push ln1Out
      if l = pattern.layerIdx then
        let mut wv? : Option (Array Int) := none
        let mut bv? : Option (Array Int) := none
        let mut wo? : Option (Array Int) := none
        for hIdx in [:hdr.numHeads] do
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          if hIdx = pattern.headIdx then
            let wv ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
            wv? := some wv
            let bV ←
              ExceptT.mk <|
                readScaledFloatArray h hdr.headDim scalePow10
            bv? := some bV
            let wo ←
              ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
            wo? := some wo
          else
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.headDim * hdr.modelDim))
        let wv ←
          match wv? with
          | none => throw "missing W_V for requested head"
          | some xs => pure xs
        let bV ←
          match bv? with
          | none => throw "missing b_V for requested head"
          | some xs => pure xs
        let wo ←
          match wo? with
          | none => throw "missing W_O for requested head"
          | some xs => pure xs
        let bVIntervals := intervalsFromScaled bV slack
        let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
        for row in ln1Rows do
          let vHidden0 := matMulIntervalsFromScaled cfg slack
            hdr.modelDim hdr.headDim wv row
          let vHidden := addVecFixed vHidden0 bVIntervals
          let vOut := matMulIntervalsFromScaled cfg slack
            hdr.headDim hdr.modelDim wo vHidden
          vOutRows := vOutRows.push vOut
        let mut vDotRows : Array Fixed10Interval := Array.mkEmpty hdr.seqLen
        for row in vOutRows do
          vDotRows := vDotRows.push (fixedDotInterval cfg row direction)
        let mut minMatchLo? : Option Int := none
        let mut minNonmatchLo? : Option Int := none
        for i in [:hdr.seqLen] do
          let ti : Int := (Int.ofNat i) + pattern.targetOffset
          if ti < 0 || ti ≥ (Int.ofNat hdr.seqLen) then
            pure ()
          else
            let tIdx : Nat := Int.toNat ti
            let targetTok := tokens[tIdx]!
            let mut matchLo? : Option Int := none
            let mut nonmatchLo? : Option Int := none
            for j in [:hdr.seqLen] do
              if !causalPattern || j ≤ i then
                let vLo := (vDotRows[j]!).lo
                let isMatch : Bool :=
                  match keyOffsetNat? with
                  | some k =>
                      let idx := j + k
                      idx < hdr.seqLen && tokens[idx]! = targetTok
                  | none =>
                      if j < keyOffsetNeg then
                        false
                      else
                        tokens[j - keyOffsetNeg]! = targetTok
                if isMatch then
                  matchLo? :=
                    match matchLo? with
                    | none => some vLo
                    | some m => some (min m vLo)
                else
                  nonmatchLo? :=
                    match nonmatchLo? with
                    | none => some vLo
                    | some m => some (min m vLo)
              else
                pure ()
            let matchLo :=
              match matchLo? with
              | none => 0
              | some v => v
            let nonmatchLo :=
              match nonmatchLo? with
              | none => matchLo
              | some v => v
            minMatchLo? :=
              match minMatchLo? with
              | none => some matchLo
              | some m => some (min m matchLo)
            minNonmatchLo? :=
              match minNonmatchLo? with
              | none => some nonmatchLo
              | some m => some (min m nonmatchLo)
        let matchLo :=
          match minMatchLo? with
          | none => 0
          | some v => v
        let nonmatchLo :=
          match minNonmatchLo? with
          | none => matchLo
          | some v => v
        let matchLoRat := ratOfScaledInt scalePow10 matchLo
        let nonmatchLoRat := ratOfScaledInt scalePow10 nonmatchLo
        let weightLB := pattern.targetWeightLowerBound
        let outputLB := mixLowerBound weightLB matchLoRat nonmatchLoRat
        let cert : HeadLogitDiffLowerBoundCert := {
          layerIdx := pattern.layerIdx
          headIdx := pattern.headIdx
          targetToken := targetToken
          negativeToken := negativeToken
          matchWeightLowerBound := weightLB
          matchLogitLowerBound := matchLoRat
          nonmatchLogitLowerBound := nonmatchLoRat
          logitDiffLowerBound := outputLB
        }
        if cert.check then
          return cert
        throw "head logit lower bound failed internal consistency checks"
      else
        let tightLayers : Nat :=
          if tightPattern then Nat.max 1 tightPatternLayers else 0
        if tightLayers > 0 && pattern.layerIdx ≤ l + tightLayers then
          if causalPattern then
            let zeroRow : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let mut attnRows : Array (Array Fixed10Interval) :=
              Array.replicate hdr.seqLen zeroRow
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
              for row in ln1Rows do
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
              let headRows := prefixUnionRowsFixed vOutRows
              attnRows := addRowsFixed attnRows headRows
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnRows := addVecFixedRows attnRows attnBias
            residuals := addRowsFixed residuals attnRows
          else
            let mut attnUnion : Array Fixed10Interval :=
              Array.replicate hdr.modelDim { lo := 0, hi := 0 }
            let groupRows := groupUnionRowsByToken ln1Rows tokens
            for _h in [:hdr.numHeads] do
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
              let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
              let wv ← ExceptT.mk <|
                readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
              let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
              let wo ← ExceptT.mk <|
                readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
              let mut vOutRows : Array (Array Fixed10Interval) := Array.mkEmpty groupRows.size
              for row in groupRows do
                let vHidden0 := matMulIntervalsFromScaled cfg slack
                  hdr.modelDim hdr.headDim wv row
                let vHidden := addVecFixed vHidden0 bV
                let vOut := matMulIntervalsFromScaled cfg slack
                  hdr.headDim hdr.modelDim wo vHidden
                vOutRows := vOutRows.push vOut
              let vUnion := unionRowsFixed vOutRows
              attnUnion := addVecFixed attnUnion vUnion
            let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
            attnUnion := addVecFixed attnUnion attnBias
            residuals := addVecFixedRows residuals attnUnion
        else
          let ln1Union := unionRowsFixed ln1Rows
          let mut attnUnion : Array Fixed10Interval :=
            Array.replicate hdr.modelDim { lo := 0, hi := 0 }
          for _h in [:hdr.numHeads] do
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
            let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
            let (vHidden0, _nWv) ←
              ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
                hdr.modelDim hdr.headDim ln1Union scalePow10)
            let bV ← ExceptT.mk (readVecIntervalsBinary h hdr.headDim slack scalePow10)
            let vHidden := addVecFixed vHidden0 bV
            let (vOut, _nWo) ←
              ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
                hdr.headDim hdr.modelDim vHidden scalePow10)
            attnUnion := addVecFixed attnUnion vOut
          let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          attnUnion := addVecFixed attnUnion attnBias
          residuals := addVecFixedRows residuals attnUnion
        let p2 := ln2Params.getD l defP
        let mut ln2Rows : Array (Array Fixed10Interval) := Array.mkEmpty hdr.seqLen
        for row in residuals do
          let (ln2Out, _ln2VarLB) :=
            fixedLayerNormRowApprox cfg row p2.gamma p2.beta eps soundnessBits
          ln2Rows := ln2Rows.push ln2Out
        let perRowLayers : Nat := perRowPatternLayers
        if perRowLayers > 0 && pattern.layerIdx ≤ l + perRowLayers then
          let wIn ←
            ExceptT.mk <| readScaledFloatArray h (hdr.modelDim * hdr.hiddenDim) scalePow10
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let wOut ←
            ExceptT.mk <| readScaledFloatArray h (hdr.hiddenDim * hdr.modelDim) scalePow10
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpRows :=
            mlpRowsFromScaled cfg hdr.geluDerivTarget slack
              hdr.modelDim hdr.hiddenDim wIn wOut bIn bOut ln2Rows
          residuals := addRowsFixed residuals mlpRows
        else
          let ln2Union := unionRowsFixed ln2Rows
          let (hidden0, _nWin) ←
            ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
              hdr.modelDim hdr.hiddenDim ln2Union scalePow10)
          let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
          let hiddenB := addVecFixed hidden0 bIn
          let actHidden := geluOverapproxFixedVec cfg hdr.geluDerivTarget hiddenB
          let (mlpOut0, _nWout) ←
            ExceptT.mk (consumeMatrixMulAndNormInfFixedBinary cfg slack h
              hdr.hiddenDim hdr.modelDim actHidden scalePow10)
          let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
          let mlpOut := addVecFixed mlpOut0 bOut
          residuals := addVecFixedRows residuals mlpOut
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
    throw "target layer not reached"
  action.run

/-- Soundly compute certification bounds from a `.nfpt` model file.

If an input is provided via `inputPath?`, the certificate uses streaming rational IBP to obtain
local (input-dependent) LayerNorm variance lower bounds at every layer.
Otherwise it falls back to the weight-only global certificate.
-/
def certifyModelFile
    (path : System.FilePath)
    (eps : Rat)
    (geluDerivTarget : GeluDerivTarget)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (partitionDepth : Nat := 0)
    (softmaxMarginLowerBound : Rat := 0)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort) : IO (Except String ModelCert) := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let firstLine := (← h.getLine).trim
  if firstLine = "NFP_BINARY_V1" then
    if inputDelta < 0 then
      return .error "delta must be nonnegative"
    match inputPath? with
    | none =>
        if inputDelta = 0 then
          certifyModelFileGlobalBinary path eps geluDerivTarget soundnessBits partitionDepth
            softmaxMarginLowerBound softmaxExpEffort
        else
          certifyModelFileLocalBinary path eps geluDerivTarget soundnessBits partitionDepth
            path inputDelta softmaxMarginLowerBound softmaxExpEffort
    | some ip =>
        certifyModelFileLocalBinary path eps geluDerivTarget soundnessBits partitionDepth
          ip inputDelta softmaxMarginLowerBound softmaxExpEffort
  else
    match inputPath? with
    | none =>
        certifyModelFileGlobal path eps geluDerivTarget soundnessBits
          (inputPath? := none) (inputDelta := inputDelta) (partitionDepth := partitionDepth)
          (softmaxMarginLowerBound := softmaxMarginLowerBound)
          (softmaxExpEffort := softmaxExpEffort)
    | some ip =>
        if inputDelta < 0 then
          return .error "delta must be nonnegative"
        certifyModelFileLocal path eps geluDerivTarget soundnessBits partitionDepth ip inputDelta
          softmaxMarginLowerBound softmaxExpEffort

/-- Compute weight-only per-head contribution bounds for a `.nfpt` model file. -/
def certifyHeadBounds
    (path : System.FilePath)
    (scalePow10 : Nat := defaultBinaryScalePow10) :
    IO (Except String (Array HeadContributionCert)) := do
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let firstLine := (← h.getLine).trim
  if firstLine = "NFP_BINARY_V1" then
    certifyHeadBoundsBinary path scalePow10
  else
    return .error "head contribution bounds require NFP_BINARY_V1"

/-- Compute local per-head attention contribution bounds for a `.nfpt` model file. -/
def certifyHeadBoundsLocal
    (path : System.FilePath)
    (eps : Rat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (soundnessBits : Nat)
    (scalePow10 : Nat := defaultBinaryScalePow10) :
    IO (Except String (Array HeadLocalContributionCert)) := do
  if inputDelta < 0 then
    return .error "delta must be nonnegative"
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let firstLine := (← h.getLine).trim
  if firstLine = "NFP_BINARY_V1" then
    let inputPath := inputPath?.getD path
    certifyHeadBoundsLocalBinary path eps inputPath inputDelta soundnessBits scalePow10
  else
    return .error "local head contribution bounds require NFP_BINARY_V1"

/-- Compute local attention pattern bounds for a specific `.nfpt` head (binary only). -/
def certifyHeadPatternLocal
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (targetOffset : Int := -1)
    (keyOffset : Int := 0)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) :
    IO (Except String HeadPatternCert) := do
  if inputDelta < 0 then
    return .error "delta must be nonnegative"
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let firstLine := (← h.getLine).trim
  if firstLine = "NFP_BINARY_V1" then
    let inputPath := inputPath?.getD path
    certifyHeadPatternLocalBinary path layerIdx headIdx eps soundnessBits inputPath inputDelta
      targetOffset keyOffset maxSeqLen tightPattern tightPatternLayers perRowPatternLayers
      scalePow10
      softmaxExpEffort causalPattern
  else
    return .error "head pattern bounds require NFP_BINARY_V1"

/-- Compute local best-match pattern bounds for a specific `.nfpt` head (binary only). -/
def certifyHeadPatternBestMatchLocal
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (queryPos? : Option Nat := none)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (targetOffset : Int := -1)
    (keyOffset : Int := 0)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (useAffine : Bool := false)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) :
    IO (Except String HeadBestMatchPatternCert) := do
  if inputDelta < 0 then
    return .error "delta must be nonnegative"
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let firstLine := (← h.getLine).trim
  if firstLine = "NFP_BINARY_V1" then
    let inputPath := inputPath?.getD path
    certifyHeadPatternBestMatchLocalBinary path layerIdx headIdx queryPos? eps soundnessBits
      inputPath
      inputDelta targetOffset keyOffset maxSeqLen tightPattern tightPatternLayers
      perRowPatternLayers useAffine scalePow10 softmaxExpEffort causalPattern
  else
    return .error "head pattern bounds require NFP_BINARY_V1"

/-- Compute local best-match pattern bounds for all valid query positions (binary only). -/
def certifyHeadPatternBestMatchLocalSweep
    (path : System.FilePath)
    (layerIdx headIdx : Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (targetOffset : Int := -1)
    (keyOffset : Int := 0)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (useAffine : Bool := false)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) :
    IO (Except String (Array HeadBestMatchPatternCert)) := do
  if inputDelta < 0 then
    return .error "delta must be nonnegative"
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let firstLine := (← h.getLine).trim
  if firstLine = "NFP_BINARY_V1" then
    let inputPath := inputPath?.getD path
    certifyHeadPatternBestMatchLocalBinarySweep path layerIdx headIdx eps soundnessBits inputPath
      inputDelta targetOffset keyOffset maxSeqLen tightPattern tightPatternLayers
      perRowPatternLayers
      useAffine scalePow10 softmaxExpEffort causalPattern
  else
    return .error "head pattern bounds require NFP_BINARY_V1"

/-- Compute layer-level best-match margin evidence for a `.nfpt` layer (binary only). -/
def certifyLayerBestMatchMarginLocal
    (path : System.FilePath)
    (layerIdx : Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (targetOffset : Int := -1)
    (keyOffset : Int := 0)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) :
    IO (Except String LayerBestMatchMarginCert) := do
  if inputDelta < 0 then
    return .error "delta must be nonnegative"
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let firstLine := (← h.getLine).trim
  if firstLine = "NFP_BINARY_V1" then
    let hdrE ← readBinaryHeader h
    match hdrE with
    | .error e => return .error e
    | .ok hdr =>
        if layerIdx ≥ hdr.numLayers then
          return .error s!"layer index {layerIdx} out of range"
        if hdr.seqLen > maxSeqLen then
          return .error s!"seq_len {hdr.seqLen} exceeds maxSeqLen {maxSeqLen}"
        let inputPath := inputPath?.getD path
        let mut headCerts : Array HeadBestMatchPatternCert := Array.mkEmpty 0
        for hIdx in [:hdr.numHeads] do
          match ←
              certifyHeadPatternBestMatchLocalBinarySweep
                path layerIdx hIdx eps soundnessBits inputPath inputDelta targetOffset keyOffset
                maxSeqLen tightPattern tightPatternLayers perRowPatternLayers false scalePow10
                softmaxExpEffort causalPattern with
          | .error e => return .error e
          | .ok certs =>
              for cert in certs do
                headCerts := headCerts.push cert
        match marginsFromBestMatchCerts hdr.numHeads hdr.seqLen headCerts with
        | none => return .error "best-match margin coverage failed"
        | some margins =>
            let marginLowerBound := minMarginArray margins
            let cert : LayerBestMatchMarginCert := {
              layerIdx := layerIdx
              seqLen := hdr.seqLen
              numHeads := hdr.numHeads
              softmaxExpEffort := softmaxExpEffort
              marginLowerBound := marginLowerBound
              margins := margins
              headCerts := headCerts
            }
            if cert.check then
              return .ok cert
            return .error "layer best-match margin certificate failed internal checks"
  else
    return .error "layer best-match margins require NFP_BINARY_V1"

/-- Compute local head value lower bounds for a specific `.nfpt` head (binary only). -/
def certifyHeadValueLowerBoundLocal
    (path : System.FilePath)
    (layerIdx headIdx coord : Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (targetOffset : Int := -1)
    (keyOffset : Int := 0)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (causalPattern : Bool := true) :
    IO (Except String HeadValueLowerBoundCert) := do
  if inputDelta < 0 then
    return .error "delta must be nonnegative"
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let firstLine := (← h.getLine).trim
  if firstLine = "NFP_BINARY_V1" then
    let inputPath := inputPath?.getD path
    let patternE ←
      certifyHeadPatternLocalBinary path layerIdx headIdx eps soundnessBits inputPath inputDelta
        targetOffset keyOffset maxSeqLen tightPattern tightPatternLayers perRowPatternLayers
        scalePow10
        defaultSoftmaxExpEffort causalPattern
    match patternE with
    | .error e => return .error e
    | .ok pattern =>
      certifyHeadValueLowerBoundLocalBinary path pattern coord eps soundnessBits inputPath
        inputDelta maxSeqLen scalePow10 tightPattern tightPatternLayers perRowPatternLayers
        causalPattern
  else
    return .error "head value bounds require NFP_BINARY_V1"

/-- Compute local head logit-difference lower bounds for a specific `.nfpt` head (binary only). -/
def certifyHeadLogitDiffLowerBoundLocal
    (path : System.FilePath)
    (layerIdx headIdx targetToken negativeToken : Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (targetOffset : Int := -1)
    (keyOffset : Int := 0)
    (maxSeqLen : Nat := 256)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (causalPattern : Bool := true) :
    IO (Except String HeadLogitDiffLowerBoundCert) := do
  if inputDelta < 0 then
    return .error "delta must be nonnegative"
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let firstLine := (← h.getLine).trim
  if firstLine = "NFP_BINARY_V1" then
    let inputPath := inputPath?.getD path
    let patternE ←
      certifyHeadPatternLocalBinary path layerIdx headIdx eps soundnessBits inputPath inputDelta
        targetOffset keyOffset maxSeqLen tightPattern tightPatternLayers perRowPatternLayers
        scalePow10
        defaultSoftmaxExpEffort causalPattern
    match patternE with
    | .error e => return .error e
    | .ok pattern =>
      certifyHeadLogitDiffLowerBoundLocalBinary path pattern targetToken negativeToken
          eps soundnessBits inputPath inputDelta maxSeqLen scalePow10 tightPattern
          tightPatternLayers perRowPatternLayers causalPattern
  else
    return .error "head logit bounds require NFP_BINARY_V1"

/-- Compute a combined sound certificate for an induction-style head pair (binary only). -/
def certifyInductionSound
    (path : System.FilePath)
    (layer1 head1 layer2 head2 coord : Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (offset1 : Int := -1)
    (offset2 : Int := -1)
    (keyOffset1 : Int := 0)
    (keyOffset2 : Int := 0)
    (maxSeqLen : Nat := 256)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (targetToken? : Option Nat := none)
    (negativeToken? : Option Nat := none)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) :
    IO (Except String InductionHeadSoundCert) := do
  if inputDelta < 0 then
    return .error "delta must be nonnegative"
  let h ← IO.FS.Handle.mk path IO.FS.Mode.read
  let firstLine := (← h.getLine).trim
  if firstLine = "NFP_BINARY_V1" then
    let inputPath := inputPath?.getD path
    let p1E ←
      certifyHeadPatternLocalBinary path layer1 head1 eps soundnessBits inputPath inputDelta
        offset1 keyOffset1 maxSeqLen tightPattern tightPatternLayers perRowPatternLayers scalePow10
        softmaxExpEffort causalPattern
    match p1E with
    | .error e => return .error e
    | .ok p1 =>
        let p2E ←
          certifyHeadPatternLocalBinary path layer2 head2 eps soundnessBits inputPath inputDelta
            offset2 keyOffset2 maxSeqLen tightPattern tightPatternLayers perRowPatternLayers
            scalePow10
            softmaxExpEffort causalPattern
        match p2E with
        | .error e => return .error e
        | .ok p2 =>
            let vE ←
              certifyHeadValueLowerBoundLocalBinary path p2 coord eps soundnessBits inputPath
                inputDelta maxSeqLen scalePow10 tightPattern tightPatternLayers perRowPatternLayers
                causalPattern
            match vE with
            | .error e => return .error e
            | .ok v =>
                let logitE ←
                  match targetToken?, negativeToken? with
                  | none, none => pure (.ok none)
                  | some targetToken, some negativeToken => do
                      let logitE ← certifyHeadLogitDiffLowerBoundLocalBinary path p2
                        targetToken negativeToken eps soundnessBits inputPath inputDelta
                        maxSeqLen scalePow10 tightPattern tightPatternLayers perRowPatternLayers
                        causalPattern
                      pure (logitE.map some)
                  | _, _ =>
                      pure (.error "use both target and negative tokens (or neither)")
                match logitE with
                | .error e => return .error e
                | .ok logit? =>
                    let cert : InductionHeadSoundCert := {
                      layer1Pattern := p1
                      layer2Pattern := p2
                      layer2Value := v
                      layer2Logit? := logit?
                      deltaLowerBound := v.outputCoordLowerBound
                    }
                    if cert.check then
                      return .ok cert
                    return .error "induction head certificate failed internal consistency checks"
  else
    return .error "induction sound cert requires NFP_BINARY_V1"

/-- Compute a best-match induction-head certificate in a single binary pass. -/
private def certifyInductionSoundBestMatchLocalBinaryPair
    (path : System.FilePath)
    (layer1 head1 layer2 head2 coord queryPos : Nat)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath : System.FilePath)
    (inputDelta : Rat)
    (offset1 : Int)
    (offset2 : Int)
    (keyOffset1 : Int)
    (keyOffset2 : Int)
    (maxSeqLen : Nat)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (tightPattern : Bool)
    (tightPatternLayers : Nat)
    (perRowPatternLayers : Nat)
    (useAffine : Bool)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true)
    (shared? : Option SharedBinaryInputs := none)
    (prefix? : Option SharedBinaryPrefix := none)
    (targetToken? : Option Nat := none)
    (negativeToken? : Option Nat := none)
    (direction? : Option (Thunk (Array Fixed10Interval)) := none) :
    IO (Except String InductionHeadBestMatchSoundCert) := do
  let cfg : Fixed10Cfg := scaleCfgOfPow10 scalePow10
  let slack : Int := fixedUlpSlack
  let action : ExceptT String IO InductionHeadBestMatchSoundCert := do
    let timingEnabled ← ExceptT.lift <| IO.getEnv "NFP_TIMING"
    let timing : Bool := timingEnabled.isSome
    let timeIt {α : Type} (label : String) (work : ExceptT String IO α) :
        ExceptT String IO α := do
      if !timing then
        work
      else
        let t0 ← ExceptT.lift IO.monoNanosNow
        let r ← work
        let t1 ← ExceptT.lift IO.monoNanosNow
        let dtMs := (t1 - t0) / 1000000
        ExceptT.lift <| IO.eprintln s!"timing:{label} {dtMs}ms"
        return r
    let (hdr, ln1Params, ln2Params, residualsBase, tokensBase) ←
      timeIt "load_shared" <| match shared? with
      | some shared => do
          if shared.scalePow10 ≠ scalePow10 then
            throw "shared scalePow10 mismatch"
          if shared.inputDelta ≠ inputDelta then
            throw "shared inputDelta mismatch"
          pure (shared.hdr, shared.ln1Params, shared.ln2Params, shared.residuals0, shared.tokens)
      | none => do
          let (hdr, ln1Params, ln2Params) ←
            timeIt "load_ln_params" <|
              ExceptT.mk (collectLayerNormParamsBinary path scalePow10 slack)
          let residuals0 ←
            timeIt "load_embeddings" <|
              ExceptT.mk
                (loadEmbeddingsIntervalsBinary inputPath hdr.modelDim inputDelta scalePow10)
          let (hdrTok, tokens) ←
            timeIt "load_tokens" <| ExceptT.mk (loadTokensBinary inputPath)
          if hdrTok.seqLen ≠ hdr.seqLen then
            throw "token/embedding seq_len mismatch"
          pure (hdr, ln1Params, ln2Params, residuals0, tokens)
    if layer1 ≥ hdr.numLayers then
      throw s!"layer1 index {layer1} out of range"
    if layer2 ≥ hdr.numLayers then
      throw s!"layer2 index {layer2} out of range"
    if head1 ≥ hdr.numHeads then
      throw s!"head1 index {head1} out of range"
    if head2 ≥ hdr.numHeads then
      throw s!"head2 index {head2} out of range"
    if coord ≥ hdr.modelDim then
      throw s!"coord index {coord} out of range"
    if hdr.seqLen > maxSeqLen then
      throw s!"seq_len {hdr.seqLen} exceeds maxSeqLen {maxSeqLen}"
    if queryPos ≥ hdr.seqLen then
      throw s!"queryPos {queryPos} out of range"
    let seqLenEff : Nat := if causalPattern then queryPos + 1 else hdr.seqLen
    let (residuals0, tokens) ←
      match prefix? with
      | some pref =>
          if pref.seqLenEff ≠ seqLenEff then
            throw "prefix seq_len mismatch"
          pure (pref.residuals.get, pref.tokens.get)
      | none =>
          let residuals0 :=
            if causalPattern then takePrefix residualsBase seqLenEff else residualsBase
          let tokens := if causalPattern then takePrefix tokensBase seqLenEff else tokensBase
          pure (residuals0, tokens)
    let matchRows
        (targetOffset : Int)
        (keyOffset : Int) : Array Nat :=
      Id.run do
        let ti : Int := (Int.ofNat queryPos) + targetOffset
        if ti < 0 || ti ≥ (Int.ofNat seqLenEff) then
          return #[]
        let tIdx : Nat := Int.toNat ti
        let targetTok := tokens[tIdx]!
        let keyOffsetNat? : Option Nat :=
          if keyOffset ≥ 0 then some (Int.toNat keyOffset) else none
        let keyOffsetNeg : Nat := Int.toNat (-keyOffset)
        let mut rows : Array Nat := Array.mkEmpty seqLenEff
        for j in [:seqLenEff] do
          if !causalPattern || j ≤ queryPos then
            let isMatch : Bool :=
              match keyOffsetNat? with
              | some k =>
                  let idx := j + k
                  idx < seqLenEff && tokens[idx]! = targetTok
              | none =>
                  if j < keyOffsetNeg then
                    false
                  else
                    tokens[j - keyOffsetNeg]! = targetTok
            if isMatch then
              rows := rows.push j
          else
            pure ()
        return rows
    let selectedRows : Array Nat :=
      let r1 := matchRows offset1 keyOffset1
      let r2 := matchRows offset2 keyOffset2
      if r1.isEmpty && r2.isEmpty then
        #[]
      else
        Id.run do
          let mut acc : Array Nat := Array.mkEmpty (r1.size + r2.size)
          for v in r1 do
            if !acc.contains v then
              acc := acc.push v
          for v in r2 do
            if !acc.contains v then
              acc := acc.push v
          acc
    let selectedRows? : Option (Array Nat) :=
      if selectedRows.isEmpty then none else some selectedRows
    let useLogit ←
      match targetToken?, negativeToken?, direction? with
      | none, none, none => pure false
      | some _, some _, some _ => pure true
      | _, _, _ => throw "use both target and negative tokens (or neither)"
    let calcLnRows
        (rows : Array (Array Fixed10Interval))
        (p : LayerNormParamsFixed) :
        Array (Array Fixed10Interval) :=
      fixedLayerNormRowsApprox cfg rows p eps soundnessBits
    let calcLnRowsExact
        (rows : Array (Array Fixed10Interval))
        (p : LayerNormParamsFixed) :
        Array (Array Fixed10Interval) :=
      fixedLayerNormRowsApproxExact cfg rows p eps soundnessBits
    let calcVOutRowsIntervals
        (rows : Array (Array Fixed10Interval))
        (wvIntervals woIntervals : Array Fixed10Interval)
        (bV : Array Fixed10Interval) :
        Array (Array Fixed10Interval) :=
      let useTasks := rows.size > 32
      if useTasks then
        Id.run do
          let chunkSize : Nat := 16
          let numChunks : Nat := (rows.size + chunkSize - 1) / chunkSize
          let mut tasks : Array (Task (Array (Array Fixed10Interval))) := Array.mkEmpty numChunks
          let mut chunkIdx : Nat := 0
          while chunkIdx < numChunks do
            let start := chunkIdx * chunkSize
            let stop := min rows.size (start + chunkSize)
            tasks := tasks.push <|
              Task.spawn (fun _ =>
                Id.run do
                  let mut outChunk : Array (Array Fixed10Interval) := Array.mkEmpty (stop - start)
                  let mut i := start
                  while i < stop do
                    let vHidden0 := matMulIntervalsFromIntervalsNoTask cfg
                      hdr.modelDim hdr.headDim wvIntervals (rows[i]!)
                    let vHidden := addVecFixed vHidden0 bV
                    let vOut := matMulIntervalsFromIntervalsNoTask cfg
                      hdr.headDim hdr.modelDim woIntervals vHidden
                    outChunk := outChunk.push vOut
                    i := i + 1
                  return outChunk)
            chunkIdx := chunkIdx + 1
          let mut out : Array (Array Fixed10Interval) := Array.mkEmpty rows.size
          for t in tasks do
            for row in t.get do
              out := out.push row
          return out
      else
        Id.run do
          let mut out : Array (Array Fixed10Interval) := Array.mkEmpty rows.size
          for row in rows do
            let vHidden0 := matMulIntervalsFromIntervalsNoTask cfg
              hdr.modelDim hdr.headDim wvIntervals row
            let vHidden := addVecFixed vHidden0 bV
            let vOut := matMulIntervalsFromIntervalsNoTask cfg
              hdr.headDim hdr.modelDim woIntervals vHidden
            out := out.push vOut
          return out
    let calcVOutIntervals
        (row : Array Fixed10Interval)
        (wvIntervals woIntervals : Array Fixed10Interval)
        (bV : Array Fixed10Interval) :
        Array Fixed10Interval :=
      let vHidden0 := matMulIntervalsFromIntervalsNoTask cfg
        hdr.modelDim hdr.headDim wvIntervals row
      let vHidden := addVecFixed vHidden0 bV
      matMulIntervalsFromIntervalsNoTask cfg hdr.headDim hdr.modelDim woIntervals vHidden
    let bestMatchPattern
        (layerIdx headIdx : Nat)
        (ln1Rows : Array (Array Fixed10Interval))
        (wq wk : Array Int)
        (bQ bK : Array Fixed10Interval)
        (targetOffset : Int)
        (keyOffset : Int)
        (useTasks : Bool := true) :
        ExceptT String IO HeadBestMatchPatternCert := do
      let ti : Int := (Int.ofNat queryPos) + targetOffset
      if ti < 0 || ti ≥ (Int.ofNat seqLenEff) then
        throw "query position has no valid target offset"
      let tIdx : Nat := Int.toNat ti
      let targetTok := tokens[tIdx]!
      let keyOffsetNat? : Option Nat :=
        if keyOffset ≥ 0 then some (Int.toNat keyOffset) else none
      let keyOffsetNeg : Nat := Int.toNat (-keyOffset)
      let mut bestMatchLower? : Option Int := none
      let mut bestNonmatchUpper? : Option Int := none
      if useAffine then
        let bQCenters := bQ.map (fun x => (x.lo + x.hi).ediv (Int.ofNat 2))
        let bKCenters := bK.map (fun x => (x.lo + x.hi).ediv (Int.ofNat 2))
        let bQRadii := bQ.map intervalRadiusInt
        let bKRadii := bK.map intervalRadiusInt
        let (qInputCenters, qInputRadii, _qAbsInput) :=
          rowCentersRadiiAbsInt (ln1Rows[queryPos]!)
        let (qCenters0, qRadii0) :=
          matMulCentersRadiiIntSlack cfg slack
            hdr.modelDim hdr.headDim wq qInputCenters qInputRadii
        let qCenters := addVecScaledInt qCenters0 bQCenters 1
        let qRadii := addVecScaledInt qRadii0 bQRadii 1
        let useTasksHere := useTasks && seqLenEff > 32
        if useTasksHere then
          let chunkSize : Nat := 16
          let numChunks : Nat := (seqLenEff + chunkSize - 1) / chunkSize
          let mut tasks : Array (Task (Option Int × Option Int)) := Array.mkEmpty numChunks
          let mut chunkIdx : Nat := 0
          while chunkIdx < numChunks do
            let start := chunkIdx * chunkSize
            let stop := min seqLenEff (start + chunkSize)
            tasks := tasks.push <| Task.spawn (fun _ =>
              Id.run do
                let mut bestMatchLower? : Option Int := none
                let mut bestNonmatchUpper? : Option Int := none
                let mut j := start
                while j < stop do
                  if !causalPattern || j ≤ queryPos then
                    let (kInputCenters, kInputRadii, _kAbsInput) :=
                      rowCentersRadiiAbsInt (ln1Rows[j]!)
                    let (kCenters0, kRadii0) :=
                      matMulCentersRadiiIntSlack cfg slack
                        hdr.modelDim hdr.headDim wk kInputCenters kInputRadii
                    let kCenters := addVecScaledInt kCenters0 bKCenters 1
                    let kRadii := addVecScaledInt kRadii0 bKRadii 1
                    let dot :=
                      dotIntervalFromCentersRadiiInt cfg qCenters qRadii kCenters kRadii
                    let isMatch : Bool :=
                      match keyOffsetNat? with
                      | some k =>
                          let idx := j + k
                          idx < seqLenEff && tokens[idx]! = targetTok
                      | none =>
                          if j < keyOffsetNeg then
                            false
                          else
                            tokens[j - keyOffsetNeg]! = targetTok
                    if isMatch then
                      bestMatchLower? :=
                        match bestMatchLower? with
                        | none => some dot.lo
                        | some m => some (max m dot.lo)
                    else
                      bestNonmatchUpper? :=
                        match bestNonmatchUpper? with
                        | none => some dot.hi
                        | some m => some (max m dot.hi)
                  j := j + 1
                return (bestMatchLower?, bestNonmatchUpper?))
            chunkIdx := chunkIdx + 1
          for t in tasks do
            let (matchChunk?, nonmatchChunk?) := t.get
            if matchChunk?.isSome then
              bestMatchLower? :=
                match bestMatchLower?, matchChunk? with
                | none, some v => some v
                | some cur, some v => some (max cur v)
                | some cur, none => some cur
                | none, none => none
            if nonmatchChunk?.isSome then
              bestNonmatchUpper? :=
                match bestNonmatchUpper?, nonmatchChunk? with
                | none, some v => some v
                | some cur, some v => some (max cur v)
                | some cur, none => some cur
                | none, none => none
        else
          for j in [:seqLenEff] do
            if !causalPattern || j ≤ queryPos then
              let (kInputCenters, kInputRadii, _kAbsInput) :=
                rowCentersRadiiAbsInt (ln1Rows[j]!)
              let (kCenters0, kRadii0) :=
                matMulCentersRadiiIntSlack cfg slack
                  hdr.modelDim hdr.headDim wk kInputCenters kInputRadii
              let kCenters := addVecScaledInt kCenters0 bKCenters 1
              let kRadii := addVecScaledInt kRadii0 bKRadii 1
              let dot :=
                dotIntervalFromCentersRadiiInt cfg qCenters qRadii kCenters kRadii
              let isMatch : Bool :=
                match keyOffsetNat? with
                | some k =>
                    let idx := j + k
                    idx < seqLenEff && tokens[idx]! = targetTok
                | none =>
                    if j < keyOffsetNeg then
                      false
                    else
                      tokens[j - keyOffsetNeg]! = targetTok
              if isMatch then
                bestMatchLower? :=
                  match bestMatchLower? with
                  | none => some dot.lo
                  | some m => some (max m dot.lo)
              else
                bestNonmatchUpper? :=
                  match bestNonmatchUpper? with
                  | none => some dot.hi
                  | some m => some (max m dot.hi)
            else
              pure ()
      else
        let qRow0 := matMulIntervalsFromScaled cfg slack
          hdr.modelDim hdr.headDim wq (ln1Rows[queryPos]!)
        let qRow := addVecFixed qRow0 bQ
        let kRows :=
          let useTasksHere := useTasks && ln1Rows.size > 32
          if useTasksHere then
            let tasks := ln1Rows.map (fun row =>
              Task.spawn (fun _ =>
                let kRow0 := matMulIntervalsFromScaledNoTask cfg slack
                  hdr.modelDim hdr.headDim wk row
                addVecFixed kRow0 bK))
            tasks.map (fun t => t.get)
          else
            Id.run do
              let mut out : Array (Array Fixed10Interval) := Array.mkEmpty seqLenEff
              for row in ln1Rows do
                let kRow0 := matMulIntervalsFromScaledNoTask cfg slack
                  hdr.modelDim hdr.headDim wk row
                out := out.push (addVecFixed kRow0 bK)
              return out
        for j in [:seqLenEff] do
          if !causalPattern || j ≤ queryPos then
            let dot := fixedDotInterval cfg qRow (kRows[j]!)
            let isMatch : Bool :=
              match keyOffsetNat? with
              | some k =>
                  let idx := j + k
                  idx < seqLenEff && tokens[idx]! = targetTok
              | none =>
                  if j < keyOffsetNeg then
                    false
                  else
                    tokens[j - keyOffsetNeg]! = targetTok
            if isMatch then
              bestMatchLower? :=
                match bestMatchLower? with
                | none => some dot.lo
                | some m => some (max m dot.lo)
            else
              bestNonmatchUpper? :=
                match bestNonmatchUpper? with
                | none => some dot.hi
                | some m => some (max m dot.hi)
          else
            pure ()
      let bestMatchLower ←
        match bestMatchLower? with
        | none => throw "no matching keys for the requested offset"
        | some v => pure v
      let bestNonmatchUpper :=
        match bestNonmatchUpper? with
        | none => bestMatchLower
        | some v => v
      let marginInt : Int := bestMatchLower - bestNonmatchUpper
      let bestMatchLowerRat := ratOfScaledInt scalePow10 bestMatchLower
      let bestNonmatchUpperRat := ratOfScaledInt scalePow10 bestNonmatchUpper
      let margin := ratOfScaledInt scalePow10 marginInt
      let (effortUsed, weightLB, softmaxJacobianUB) :=
        chooseSoftmaxExpEffort hdr.seqLen margin softmaxExpEffort
      let cert : HeadBestMatchPatternCert := {
        layerIdx := layerIdx
        headIdx := headIdx
        seqLen := hdr.seqLen
        queryPos := queryPos
        targetOffset := targetOffset
        keyOffset := keyOffset
        targetToken := targetTok
        bestMatchLogitLowerBound := bestMatchLowerRat
        bestNonmatchLogitUpperBound := bestNonmatchUpperRat
        marginLowerBound := margin
        softmaxExpEffort := effortUsed
        bestMatchWeightLowerBound := weightLB
        softmaxJacobianNormInfUpperBound := softmaxJacobianUB
      }
      if cert.check then
        return cert
      throw "best-match head pattern certificate failed internal consistency checks"
    let valueLogit
        (ln1Rows : Array (Array Fixed10Interval))
        (matchWeightLowerBound : Rat)
        (wvIntervals woIntervals : Array Fixed10Interval)
        (bV : Array Fixed10Interval)
        (targetOffset : Int)
        (keyOffset : Int) :
        ExceptT String IO HeadValueLogitCert := do
      let vOutRows := calcVOutRowsIntervals ln1Rows wvIntervals woIntervals bV
      let ti : Int := (Int.ofNat queryPos) + targetOffset
      if ti < 0 || ti ≥ (Int.ofNat seqLenEff) then
        throw "query position has no valid target offset"
      let tIdx : Nat := Int.toNat ti
      let targetTok := tokens[tIdx]!
      let keyOffsetNat? : Option Nat :=
        if keyOffset ≥ 0 then some (Int.toNat keyOffset) else none
      let keyOffsetNeg : Nat := Int.toNat (-keyOffset)
      let mut matchLo? : Option Int := none
      let mut nonmatchLo? : Option Int := none
      for j in [:seqLenEff] do
        if !causalPattern || j ≤ queryPos then
          let row := vOutRows[j]!
          let vCoord := row[coord]!.lo
          let isMatch : Bool :=
            match keyOffsetNat? with
            | some k =>
                let idx := j + k
                idx < seqLenEff && tokens[idx]! = targetTok
            | none =>
                if j < keyOffsetNeg then
                  false
                else
                  tokens[j - keyOffsetNeg]! = targetTok
          if isMatch then
            matchLo? :=
              match matchLo? with
              | none => some vCoord
              | some m => some (min m vCoord)
          else
            nonmatchLo? :=
              match nonmatchLo? with
              | none => some vCoord
              | some m => some (min m vCoord)
        else
          pure ()
      let matchLo ←
        match matchLo? with
        | none => throw "no matching keys for the requested offset"
        | some v => pure v
      let nonmatchLo :=
        match nonmatchLo? with
        | none => matchLo
        | some v => v
      let matchLoRat := ratOfScaledInt scalePow10 matchLo
      let nonmatchLoRat := ratOfScaledInt scalePow10 nonmatchLo
      let outputLB := mixLowerBound matchWeightLowerBound matchLoRat nonmatchLoRat
      let value : HeadValueLowerBoundPosCert := {
        layerIdx := layer2
        headIdx := head2
        queryPos := queryPos
        coord := coord
        matchWeightLowerBound := matchWeightLowerBound
        matchCoordLowerBound := matchLoRat
        nonmatchCoordLowerBound := nonmatchLoRat
        outputCoordLowerBound := outputLB
      }
      if !value.check then
        throw "head value certificate failed internal consistency checks"
      let logit? ←
        if !useLogit then
          pure none
        else
          match targetToken?, negativeToken?, direction? with
          | some targetToken, some negativeToken, some direction => do
              let dir := direction.get
              if dir.size ≠ hdr.modelDim then
                throw "logit direction size mismatch"
              let mut vDotRows : Array Fixed10Interval := Array.mkEmpty seqLenEff
              for row in vOutRows do
                vDotRows := vDotRows.push (fixedDotInterval cfg row dir)
              let mut matchLoLogit? : Option Int := none
              let mut nonmatchLoLogit? : Option Int := none
              for j in [:seqLenEff] do
                if !causalPattern || j ≤ queryPos then
                  let vLo := (vDotRows[j]!).lo
                  let isMatch : Bool :=
                    match keyOffsetNat? with
                    | some k =>
                        let idx := j + k
                        idx < seqLenEff && tokens[idx]! = targetTok
                    | none =>
                        if j < keyOffsetNeg then
                          false
                        else
                          tokens[j - keyOffsetNeg]! = targetTok
                  if isMatch then
                    matchLoLogit? :=
                      match matchLoLogit? with
                      | none => some vLo
                      | some m => some (min m vLo)
                  else
                    nonmatchLoLogit? :=
                      match nonmatchLoLogit? with
                      | none => some vLo
                      | some m => some (min m vLo)
                else
                  pure ()
              let matchLoLogit ←
                match matchLoLogit? with
                | none => throw "no matching keys for the requested offset"
                | some v => pure v
              let nonmatchLoLogit :=
                match nonmatchLoLogit? with
                | none => matchLoLogit
                | some v => v
              let matchLoRat := ratOfScaledInt scalePow10 matchLoLogit
              let nonmatchLoRat := ratOfScaledInt scalePow10 nonmatchLoLogit
              let logitLB := mixLowerBound matchWeightLowerBound matchLoRat nonmatchLoRat
              let logitCert : HeadLogitDiffLowerBoundPosCert := {
                layerIdx := layer2
                headIdx := head2
                queryPos := queryPos
                targetToken := targetToken
                negativeToken := negativeToken
                matchWeightLowerBound := matchWeightLowerBound
                matchLogitLowerBound := matchLoRat
                nonmatchLogitLowerBound := nonmatchLoRat
                logitDiffLowerBound := logitLB
              }
              if logitCert.check then
                pure (some logitCert)
              else
                throw "head logit certificate failed internal consistency checks"
          | _, _, _ =>
              throw "use both target and negative tokens (or neither)"
      return { value := value, logit? := logit? }
    let tightenQueryRowLower
        (baseRow : Array Fixed10Interval)
        (vOutRows : Array (Array Fixed10Interval))
        (matchWeightLowerBound : Rat)
        (targetOffset : Int)
        (keyOffset : Int) :
        ExceptT String IO (Array Fixed10Interval) := do
      let ti : Int := (Int.ofNat queryPos) + targetOffset
      if ti < 0 || ti ≥ (Int.ofNat seqLenEff) then
        throw "query position has no valid target offset"
      let tIdx : Nat := Int.toNat ti
      let targetTok := tokens[tIdx]!
      let keyOffsetNat? : Option Nat :=
        if keyOffset ≥ 0 then some (Int.toNat keyOffset) else none
      let keyOffsetNeg : Nat := Int.toNat (-keyOffset)
      let mut matchLo? : Option (Array Int) := none
      let mut nonmatchLo? : Option (Array Int) := none
      for j in [:seqLenEff] do
        if !causalPattern || j ≤ queryPos then
          let row := vOutRows[j]!
          let rowLo : Array Int := row.map (fun x => x.lo)
          let isMatch : Bool :=
            match keyOffsetNat? with
            | some k =>
                let idx := j + k
                idx < seqLenEff && tokens[idx]! = targetTok
            | none =>
                if j < keyOffsetNeg then
                  false
                else
                  tokens[j - keyOffsetNeg]! = targetTok
          if isMatch then
            matchLo? :=
              match matchLo? with
              | none => some rowLo
              | some cur =>
                  some <| Id.run do
                    let mut out : Array Int := Array.mkEmpty hdr.modelDim
                    for i in [:hdr.modelDim] do
                      out := out.push (min cur[i]! rowLo[i]!)
                    out
          else
            nonmatchLo? :=
              match nonmatchLo? with
              | none => some rowLo
              | some cur =>
                  some <| Id.run do
                    let mut out : Array Int := Array.mkEmpty hdr.modelDim
                    for i in [:hdr.modelDim] do
                      out := out.push (min cur[i]! rowLo[i]!)
                    out
      let matchLo ←
        match matchLo? with
        | none => throw "no matching keys for the requested offset"
        | some v => pure v
      let nonmatchLo :=
        match nonmatchLo? with
        | none => matchLo
        | some v => v
      let mut tightened : Array Fixed10Interval := Array.mkEmpty hdr.modelDim
      for i in [:hdr.modelDim] do
        let matchLoRat := ratOfScaledInt scalePow10 matchLo[i]!
        let nonmatchLoRat := ratOfScaledInt scalePow10 nonmatchLo[i]!
        let outLB := mixLowerBound matchWeightLowerBound matchLoRat nonmatchLoRat
        let outLBInt := ratFloorMulNat outLB cfg.scaleNat
        let base := baseRow[i]!
        let newLo := max base.lo outLBInt
        tightened := tightened.push { lo := newLo, hi := base.hi }
      return tightened
    let addAttn
        (useTight : Bool)
        (ln1Rows : Array (Array Fixed10Interval))
        (ln1Union? : Option (Array Fixed10Interval))
        (groupRows? : Option (Array (Array Fixed10Interval)))
        (attnRows? : Option (Array (Array Fixed10Interval)))
        (attnUnion? : Option (Array Fixed10Interval))
        (wvIntervals woIntervals : Array Fixed10Interval)
        (bV : Array Fixed10Interval) :
        ExceptT String IO
          (Option (Array (Array Fixed10Interval)) × Option (Array Fixed10Interval)) := do
      if useTight then
        if causalPattern then
          let vOutRows := calcVOutRowsIntervals ln1Rows wvIntervals woIntervals bV
          match attnRows? with
          | some rows =>
              if rows.size ≠ vOutRows.size then
                return (some rows, attnUnion?)
              if vOutRows.isEmpty then
                return (some rows, attnUnion?)
              let mut outRows := rows
              let mut acc := vOutRows[0]!
              outRows := outRows.set! 0 (addVecFixed rows[0]! acc)
              let mut i : Nat := 1
              while i < vOutRows.size do
                acc := Fixed10Interval.unionVec acc vOutRows[i]!
                outRows := outRows.set! i (addVecFixed rows[i]! acc)
                i := i + 1
              return (some outRows, attnUnion?)
          | none => throw "missing attnRows"
        else
          let groupRows ←
            match groupRows? with
            | some rows => pure rows
            | none => throw "missing group rows"
          let vOutRows := calcVOutRowsIntervals groupRows wvIntervals woIntervals bV
          let vUnion := unionRowsFixed vOutRows
          match attnUnion? with
          | some u => return (attnRows?, some (addVecFixed u vUnion))
          | none => throw "missing attnUnion"
      else
        let ln1Union ←
          match ln1Union? with
          | some row => pure row
          | none => throw "missing ln1Union"
        let vOut := calcVOutIntervals ln1Union wvIntervals woIntervals bV
        match attnUnion? with
        | some u => return (attnRows?, some (addVecFixed u vOut))
        | none => throw "missing attnUnion"
    let applyAttn
        (rows : Array (Array Fixed10Interval))
        (useTight : Bool)
        (attnRows? : Option (Array (Array Fixed10Interval)))
        (attnUnion? : Option (Array Fixed10Interval))
        (attnBias : Array Fixed10Interval) :
        ExceptT String IO (Array (Array Fixed10Interval)) := do
      if useTight && causalPattern then
        match attnRows? with
        | some attnRows =>
            let attnRows := addVecFixedRows attnRows attnBias
            return addRowsFixed rows attnRows
        | none => throw "missing attnRows"
      else
        match attnUnion? with
        | some attnUnion =>
            let attnUnion := addVecFixed attnUnion attnBias
            return addVecFixedRows rows attnUnion
        | none => throw "missing attnUnion"
    let applyMlp
        (rows : Array (Array Fixed10Interval))
        (usePerRow : Bool)
        (p : LayerNormParamsFixed)
        (wIn wOut : Array Int)
        (bIn bOut : Array Fixed10Interval) :
        Array (Array Fixed10Interval) :=
      let ln2Rows := calcLnRows rows p
      let geluTargetUnion : GeluDerivTarget :=
        if hdr.geluDerivTarget = .tanh then .exact else hdr.geluDerivTarget
      if usePerRow then
        match selectedRows? with
        | none =>
            let wInIntervals := intervalsFromScaled wIn slack
            let wOutIntervals := intervalsFromScaled wOut slack
            let mlpRows := mlpRowsFromIntervals cfg geluTargetUnion
              hdr.modelDim hdr.hiddenDim wInIntervals wOutIntervals bIn bOut ln2Rows
            addRowsFixed rows mlpRows
        | some idxs =>
            Id.run do
              let ln2Union := unionRowsFixed ln2Rows
              let mlpUnion := mlpRowFromScaled cfg geluTargetUnion slack
                hdr.modelDim hdr.hiddenDim wIn wOut bIn bOut ln2Union
              let mut out := addVecFixedRows rows mlpUnion
              for idx in idxs do
                if idx < ln2Rows.size then
                  let mlpRow := mlpRowFromScaledNoTask cfg hdr.geluDerivTarget slack
                    hdr.modelDim hdr.hiddenDim wIn wOut bIn bOut (ln2Rows[idx]!)
                  out := out.set! idx (addVecFixed (rows[idx]!) mlpRow)
              return out
      else
        let ln2Union := unionRowsFixed ln2Rows
        let mlpOut := mlpRowFromScaled cfg geluTargetUnion slack
          hdr.modelDim hdr.hiddenDim wIn wOut bIn bOut ln2Union
        addVecFixedRows rows mlpOut
    let awaitPattern
        (pattern? : Option HeadBestMatchPatternCert)
        (task? : Option (Task (Except IO.Error (Except String HeadBestMatchPatternCert))))
        (label : String) :
        ExceptT String IO HeadBestMatchPatternCert := do
      match pattern? with
      | some cert => pure cert
      | none =>
          match task? with
          | none => throw label
          | some task =>
              match task.get with
              | .error e => throw (toString e)
              | .ok (.error msg) => throw msg
              | .ok (.ok cert) => pure cert
    let h ← IO.FS.Handle.mk path IO.FS.Mode.read
    let _ ← ExceptT.mk (readBinaryHeader h)
    let _ ← ExceptT.mk (skipI32Array h hdr.seqLen)
    let _ ← ExceptT.mk (skipF64Array h (hdr.seqLen * hdr.modelDim))
    let defP : LayerNormParamsFixed := {
      gamma := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      beta := Array.replicate hdr.modelDim { lo := 0, hi := 0 }
    }
    let mut residuals1 := residuals0
    let mut residuals2 := residuals0
    let mut residualsSame : Bool := true
    let mut residualsV := residuals0
    let mut residualsSameV : Bool := true
    let mut p1? : Option HeadBestMatchPatternCert := none
    let mut p2? : Option HeadBestMatchPatternCert := none
    let mut p1Task? :
        Option (Task (Except IO.Error (Except String HeadBestMatchPatternCert))) := none
    let mut p2Task? :
        Option (Task (Except IO.Error (Except String HeadBestMatchPatternCert))) := none
    let mut vlogit? : Option HeadValueLogitCert := none
    for l in [:hdr.numLayers] do
      let at1 := l = layer1 && p1?.isNone
      let at2 := l = layer2 && p2?.isNone
      let needUpdate1 := l < layer1 && p1?.isNone
      let needUpdate2 := l < layer2 && p2?.isNone
      let needUpdateV := needUpdate2
      let needRows1 := at1 || needUpdate1
      let needRows2 := at2 || needUpdate2
      let needRowsV := needRows2
      let ln1P := ln1Params.getD l defP
      let tLn10? ←
        if timing then
          let t0 ← ExceptT.lift IO.monoNanosNow
          pure (some t0)
        else
          pure none
      let mut ln1RowsShared? : Option (Array (Array Fixed10Interval)) := none
      if residualsSame && (needRows1 || needRows2) then
        ln1RowsShared? := some (calcLnRows residuals1 ln1P)
      let mut ln1Rows1? : Option (Array (Array Fixed10Interval)) := none
      let mut ln1Rows2? : Option (Array (Array Fixed10Interval)) := none
      if needRows1 then
        ln1Rows1? :=
          some (ln1RowsShared?.getD (calcLnRows residuals1 ln1P))
      if needRows2 then
        ln1Rows2? :=
          some (ln1RowsShared?.getD (calcLnRows residuals2 ln1P))
      let mut ln1Rows1Exact? : Option (Array (Array Fixed10Interval)) := none
      let mut ln1Rows2Exact? : Option (Array (Array Fixed10Interval)) := none
      if at1 then
        ln1Rows1Exact? := some (calcLnRowsExact residuals1 ln1P)
      if at2 then
        ln1Rows2Exact? := some (calcLnRowsExact residuals2 ln1P)
      let mut ln1RowsV? : Option (Array (Array Fixed10Interval)) := none
      if needRowsV then
        if residualsSameV then
          ln1RowsV? := ln1Rows2?
        else
          ln1RowsV? := some (calcLnRows residualsV ln1P)
      if let some t0 := tLn10? then
        let t1 ← ExceptT.lift IO.monoNanosNow
        let dtMs := (t1 - t0) / 1000000
        ExceptT.lift <| IO.eprintln s!"timing:layer{l}:ln1 {dtMs}ms"
      let tightLayers : Nat :=
        if tightPattern then Nat.max 1 tightPatternLayers else 0
      let useTight1 := needUpdate1 && tightLayers > 0 && layer1 ≤ l + tightLayers
      let useTight2 := needUpdate2 && tightLayers > 0 && layer2 ≤ l + tightLayers
      let usePerRow1 :=
        needUpdate1 && perRowPatternLayers > 0 && layer1 ≤ l + perRowPatternLayers
      let usePerRow2 :=
        needUpdate2 && perRowPatternLayers > 0 && layer2 ≤ l + perRowPatternLayers
      let useTightV := useTight2
      let usePerRowV := usePerRow2
      let needTightenNow : Bool := l == layer1 && useTight2 && causalPattern
      let skipAttnV := useTightV && causalPattern && seqLenEff < hdr.seqLen
      let shareUpdateV := residualsSameV && needUpdateV && !skipAttnV
      let shareUpdate :=
        residualsSame && needUpdate1 && needUpdate2 &&
        useTight1 = useTight2 && usePerRow1 = usePerRow2
      let zeroRow : Array Fixed10Interval :=
        Array.replicate hdr.modelDim { lo := 0, hi := 0 }
      let mut ln1Union1? : Option (Array Fixed10Interval) := none
      let mut ln1Union2? : Option (Array Fixed10Interval) := none
      let mut groupRows1? : Option (Array (Array Fixed10Interval)) := none
      let mut groupRows2? : Option (Array (Array Fixed10Interval)) := none
      let mut attnRows1? : Option (Array (Array Fixed10Interval)) := none
      let mut attnRows2? : Option (Array (Array Fixed10Interval)) := none
      let mut attnUnion1? : Option (Array Fixed10Interval) := none
      let mut attnUnion2? : Option (Array Fixed10Interval) := none
      let mut ln1UnionV? : Option (Array Fixed10Interval) := none
      let mut groupRowsV? : Option (Array (Array Fixed10Interval)) := none
      let mut attnRowsV? : Option (Array (Array Fixed10Interval)) := none
      let mut attnUnionV? : Option (Array Fixed10Interval) := none
      let mut ln1UnionShared? : Option (Array Fixed10Interval) := none
      let mut groupRowsShared? : Option (Array (Array Fixed10Interval)) := none
      let mut attnRowsShared? : Option (Array (Array Fixed10Interval)) := none
      let mut attnUnionShared? : Option (Array Fixed10Interval) := none
      let ln1Rows1 := ln1Rows1?.getD #[]
      let ln1Rows2 := ln1Rows2?.getD #[]
      let ln1RowsV := ln1RowsV?.getD #[]
      let ln1RowsShared := ln1RowsShared?.getD #[]
      if shareUpdate then
        if useTight1 then
          if causalPattern then
            attnRowsShared? := some (Array.replicate seqLenEff zeroRow)
          else
            groupRowsShared? := some (groupUnionRowsByToken ln1RowsShared tokens)
            attnUnionShared? := some (Array.replicate hdr.modelDim { lo := 0, hi := 0 })
        else
          ln1UnionShared? := some (unionRowsFixed ln1RowsShared)
          attnUnionShared? := some (Array.replicate hdr.modelDim { lo := 0, hi := 0 })
      else
        if needUpdate1 then
          if useTight1 then
            if causalPattern then
              attnRows1? := some (Array.replicate seqLenEff zeroRow)
            else
              groupRows1? := some (groupUnionRowsByToken ln1Rows1 tokens)
              attnUnion1? := some (Array.replicate hdr.modelDim { lo := 0, hi := 0 })
          else
            ln1Union1? := some (unionRowsFixed ln1Rows1)
            attnUnion1? := some (Array.replicate hdr.modelDim { lo := 0, hi := 0 })
        if needUpdate2 then
          if useTight2 then
            if causalPattern then
              attnRows2? := some (Array.replicate seqLenEff zeroRow)
            else
              groupRows2? := some (groupUnionRowsByToken ln1Rows2 tokens)
              attnUnion2? := some (Array.replicate hdr.modelDim { lo := 0, hi := 0 })
          else
            ln1Union2? := some (unionRowsFixed ln1Rows2)
            attnUnion2? := some (Array.replicate hdr.modelDim { lo := 0, hi := 0 })
      if needUpdateV && !shareUpdateV && !skipAttnV then
        if useTightV then
          if causalPattern then
            attnRowsV? := some (Array.replicate seqLenEff zeroRow)
          else
            groupRowsV? := some (groupUnionRowsByToken ln1RowsV tokens)
            attnUnionV? := some (Array.replicate hdr.modelDim { lo := 0, hi := 0 })
        else
          ln1UnionV? := some (unionRowsFixed ln1RowsV)
          attnUnionV? := some (Array.replicate hdr.modelDim { lo := 0, hi := 0 })
      let needUpdate := needUpdate1 || needUpdate2
      let tHeads0? ←
        if timing then
          let t0 ← ExceptT.lift IO.monoNanosNow
          pure (some t0)
        else
          pure none
      let mut qkReadMs : Nat := 0
      let mut vReadMs : Nat := 0
      let mut addAttnMs : Nat := 0
      let mut tightenMs : Nat := 0
      let mut tightenVOutMs : Nat := 0
      let mut tightenPrefixMs : Nat := 0
      let mut tightenRowMs : Nat := 0
      let mut tightenWaitMs : Nat := 0
      for hIdx in [:hdr.numHeads] do
        let needValue := at2 && hIdx = head2
        let needV := needUpdate || needValue
        let needQK := (at1 && hIdx = head1) || (at2 && hIdx = head2)
        if needQK then
          let tQK0? ←
            if timing then
              let t0 ← ExceptT.lift IO.monoNanosNow
              pure (some t0)
            else
              pure none
          let wq ←
            ExceptT.mk <| readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
          let bQ ← ExceptT.mk <| readScaledFloatArray h hdr.headDim scalePow10
          let wk ←
            ExceptT.mk <| readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
          let bK ← ExceptT.mk <| readScaledFloatArray h hdr.headDim scalePow10
          if let some t0 := tQK0? then
            let t1 ← ExceptT.lift IO.monoNanosNow
            let dtMs := (t1 - t0) / 1000000
            qkReadMs := qkReadMs + dtMs
          let bQIntervals := intervalsFromScaled bQ slack
          let bKIntervals := intervalsFromScaled bK slack
          if at1 && hIdx = head1 then
            let ln1Rows1Exact := ln1Rows1Exact?.getD ln1Rows1
            if needV && !needTightenNow then
              let task ←
                ExceptT.lift <|
                  IO.asTask
                    (timeIt s!"layer{layer1}:pattern" <|
                      bestMatchPattern
                        layer1 head1 ln1Rows1Exact wq wk bQIntervals bKIntervals offset1
                        keyOffset1
                        (useTasks := false)).run
              p1Task? := some task
            else
              let p1 ←
                timeIt s!"layer{layer1}:pattern" <|
                  bestMatchPattern
                    layer1 head1 ln1Rows1Exact wq wk bQIntervals bKIntervals offset1 keyOffset1
              p1? := some p1
          if at2 && hIdx = head2 then
            let ln1Rows2Exact := ln1Rows2Exact?.getD ln1Rows2
            if needV then
              let task ←
                ExceptT.lift <|
                  IO.asTask
                    (timeIt s!"layer{layer2}:pattern" <|
                      bestMatchPattern
                        layer2 head2 ln1Rows2Exact wq wk bQIntervals bKIntervals offset2
                        keyOffset2
                        (useTasks := false)).run
              p2Task? := some task
            else
              let p2 ←
                timeIt s!"layer{layer2}:pattern" <|
                  bestMatchPattern
                    layer2 head2 ln1Rows2Exact wq wk bQIntervals bKIntervals offset2 keyOffset2
              p2? := some p2
        else
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
        if needV then
          let tV0? ←
            if timing then
              let t0 ← ExceptT.lift IO.monoNanosNow
              pure (some t0)
            else
              pure none
          let wv ←
            ExceptT.mk <| readScaledFloatArray h (hdr.modelDim * hdr.headDim) scalePow10
          let bV ← ExceptT.mk <| readScaledFloatArray h hdr.headDim scalePow10
          let wo ←
            ExceptT.mk <| readScaledFloatArray h (hdr.headDim * hdr.modelDim) scalePow10
          if let some t0 := tV0? then
            let t1 ← ExceptT.lift IO.monoNanosNow
            let dtMs := (t1 - t0) / 1000000
            vReadMs := vReadMs + dtMs
          let bVIntervals := intervalsFromScaled bV slack
          let wvIntervals := intervalsFromScaled wv slack
          let woIntervals := intervalsFromScaled wo slack
          if needUpdate then
            if shareUpdate then
              let tAdd0? ←
                if timing then
                  let t0 ← ExceptT.lift IO.monoNanosNow
                  pure (some t0)
                else
                  pure none
              let (attnRows', attnUnion') ←
                addAttn useTight1 ln1RowsShared ln1UnionShared? groupRowsShared?
                  attnRowsShared? attnUnionShared? wvIntervals woIntervals bVIntervals
              if let some t0 := tAdd0? then
                let t1 ← ExceptT.lift IO.monoNanosNow
                let dtMs := (t1 - t0) / 1000000
                addAttnMs := addAttnMs + dtMs
              attnRowsShared? := attnRows'
              attnUnionShared? := attnUnion'
            else
              if needUpdate1 then
                let tAdd0? ←
                  if timing then
                    let t0 ← ExceptT.lift IO.monoNanosNow
                    pure (some t0)
                  else
                    pure none
                let (attnRows', attnUnion') ←
                  addAttn useTight1 ln1Rows1 ln1Union1? groupRows1?
                    attnRows1? attnUnion1? wvIntervals woIntervals bVIntervals
                if let some t0 := tAdd0? then
                  let t1 ← ExceptT.lift IO.monoNanosNow
                  let dtMs := (t1 - t0) / 1000000
                  addAttnMs := addAttnMs + dtMs
                attnRows1? := attnRows'
                attnUnion1? := attnUnion'
              if needUpdate2 then
                if l == layer1 && hIdx == head1 && useTight2 && causalPattern then
                  let tTight0? ←
                    if timing then
                      let t0 ← ExceptT.lift IO.monoNanosNow
                      pure (some t0)
                    else
                      pure none
                  let tWait0? ←
                    if timing then
                      let t0 ← ExceptT.lift IO.monoNanosNow
                      pure (some t0)
                    else
                      pure none
                  let p1 ←
                    awaitPattern p1? p1Task? "missing best-match pattern cert for tightening"
                  if let some t0 := tWait0? then
                    let t1 ← ExceptT.lift IO.monoNanosNow
                    let dtMs := (t1 - t0) / 1000000
                    tightenWaitMs := tightenWaitMs + dtMs
                  p1? := some p1
                  let tVOut0? ←
                    if timing then
                      let t0 ← ExceptT.lift IO.monoNanosNow
                      pure (some t0)
                    else
                      pure none
                  let vOutRows := calcVOutRowsIntervals ln1Rows2 wvIntervals woIntervals bVIntervals
                  if let some t0 := tVOut0? then
                    let t1 ← ExceptT.lift IO.monoNanosNow
                    let dtMs := (t1 - t0) / 1000000
                    tightenVOutMs := tightenVOutMs + dtMs
                  let tPrefix0? ←
                    if timing then
                      let t0 ← ExceptT.lift IO.monoNanosNow
                      pure (some t0)
                    else
                      pure none
                  let mut headRows := prefixUnionRowsFixed vOutRows
                  if let some t0 := tPrefix0? then
                    let t1 ← ExceptT.lift IO.monoNanosNow
                    let dtMs := (t1 - t0) / 1000000
                    tightenPrefixMs := tightenPrefixMs + dtMs
                  let baseRow := headRows[queryPos]!
                  let tRow0? ←
                    if timing then
                      let t0 ← ExceptT.lift IO.monoNanosNow
                      pure (some t0)
                    else
                      pure none
                  let tightRow ←
                    tightenQueryRowLower baseRow vOutRows p1.bestMatchWeightLowerBound offset1
                      keyOffset1
                  if let some t0 := tRow0? then
                    let t1 ← ExceptT.lift IO.monoNanosNow
                    let dtMs := (t1 - t0) / 1000000
                    tightenRowMs := tightenRowMs + dtMs
                  headRows := headRows.set! queryPos tightRow
                  if let some t0 := tTight0? then
                    let t1 ← ExceptT.lift IO.monoNanosNow
                    let dtMs := (t1 - t0) / 1000000
                    tightenMs := tightenMs + dtMs
                  match attnRows2? with
                  | some rows => attnRows2? := some (addRowsFixed rows headRows)
                  | none => throw "missing attnRows"
                else
                  let tAdd0? ←
                    if timing then
                      let t0 ← ExceptT.lift IO.monoNanosNow
                      pure (some t0)
                    else
                      pure none
                  let (attnRows', attnUnion') ←
                    addAttn useTight2 ln1Rows2 ln1Union2? groupRows2?
                      attnRows2? attnUnion2? wvIntervals woIntervals bVIntervals
                  if let some t0 := tAdd0? then
                    let t1 ← ExceptT.lift IO.monoNanosNow
                    let dtMs := (t1 - t0) / 1000000
                    addAttnMs := addAttnMs + dtMs
                  attnRows2? := attnRows'
                  attnUnion2? := attnUnion'
            if needUpdateV && !shareUpdateV && !skipAttnV then
              let tAdd0? ←
                if timing then
                  let t0 ← ExceptT.lift IO.monoNanosNow
                  pure (some t0)
                else
                  pure none
              let (attnRows', attnUnion') ←
                addAttn useTightV ln1RowsV ln1UnionV? groupRowsV?
                  attnRowsV? attnUnionV? wvIntervals woIntervals bVIntervals
              if let some t0 := tAdd0? then
                let t1 ← ExceptT.lift IO.monoNanosNow
                let dtMs := (t1 - t0) / 1000000
                addAttnMs := addAttnMs + dtMs
              attnRowsV? := attnRows'
              attnUnionV? := attnUnion'
          if needValue then
            let p2 ←
              awaitPattern p2? p2Task? "missing best-match pattern cert for value bound"
            p2? := some p2
            let vlogit ←
              timeIt s!"layer{layer2}:value_logit" <|
                valueLogit ln1RowsV p2.bestMatchWeightLowerBound wvIntervals woIntervals
                  bVIntervals offset2 keyOffset2
            vlogit? := some vlogit
        else
          let _ ← ExceptT.mk (skipF64Array h (hdr.modelDim * hdr.headDim))
          let _ ← ExceptT.mk (skipF64Array h hdr.headDim)
          let _ ← ExceptT.mk (skipF64Array h (hdr.headDim * hdr.modelDim))
      if let some t0 := tHeads0? then
        let t1 ← ExceptT.lift IO.monoNanosNow
        let dtMs := (t1 - t0) / 1000000
        ExceptT.lift <| IO.eprintln s!"timing:layer{l}:heads {dtMs}ms"
      if timing then
        ExceptT.lift <| IO.eprintln s!"timing:layer{l}:qk_read {qkReadMs}ms"
        ExceptT.lift <| IO.eprintln s!"timing:layer{l}:v_read {vReadMs}ms"
        ExceptT.lift <| IO.eprintln s!"timing:layer{l}:add_attn {addAttnMs}ms"
        ExceptT.lift <| IO.eprintln s!"timing:layer{l}:tighten {tightenMs}ms"
        ExceptT.lift <| IO.eprintln s!"timing:layer{l}:tighten_vout {tightenVOutMs}ms"
        ExceptT.lift <| IO.eprintln s!"timing:layer{l}:tighten_prefix {tightenPrefixMs}ms"
        ExceptT.lift <| IO.eprintln s!"timing:layer{l}:tighten_row {tightenRowMs}ms"
        ExceptT.lift <| IO.eprintln s!"timing:layer{l}:tighten_wait {tightenWaitMs}ms"
      if p1?.isSome && p2?.isSome && vlogit?.isSome && !(needUpdate1 || needUpdate2) then
        match p1?, p2?, vlogit? with
        | some p1, some p2, some vlogit =>
            let cert : InductionHeadBestMatchSoundCert := {
              layer1Pattern := p1
              layer2Pattern := p2
              layer2Value := vlogit.value
              layer2Logit? := vlogit.logit?
              deltaLowerBound := vlogit.value.outputCoordLowerBound
            }
            if cert.check then
              return cert
            throw "induction head certificate failed internal consistency checks"
        | _, _, _ => throw "induction head certificate failed internal consistency checks"
      if needUpdate1 || needUpdate2 then
        let tAttn0? ←
          if timing then
            let t0 ← ExceptT.lift IO.monoNanosNow
            pure (some t0)
          else
            pure none
        let attnBias ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
        if shareUpdate then
          residuals1 ← applyAttn residuals1 useTight1 attnRowsShared? attnUnionShared? attnBias
          residuals2 := residuals1
        else
          if needUpdate1 then
            residuals1 ← applyAttn residuals1 useTight1 attnRows1? attnUnion1? attnBias
          if needUpdate2 then
            residuals2 ← applyAttn residuals2 useTight2 attnRows2? attnUnion2? attnBias
        if needUpdateV && !shareUpdateV && !skipAttnV then
          residualsV ← applyAttn residualsV useTightV attnRowsV? attnUnionV? attnBias
        if let some t0 := tAttn0? then
          let t1 ← ExceptT.lift IO.monoNanosNow
          let dtMs := (t1 - t0) / 1000000
          ExceptT.lift <| IO.eprintln s!"timing:layer{l}:attn_update {dtMs}ms"
        let tMlp0? ←
          if timing then
            let t0 ← ExceptT.lift IO.monoNanosNow
            pure (some t0)
          else
            pure none
        let wIn ←
          ExceptT.mk <| readScaledFloatArray h (hdr.modelDim * hdr.hiddenDim) scalePow10
        let bIn ← ExceptT.mk (readVecIntervalsBinary h hdr.hiddenDim slack scalePow10)
        let wOut ←
          ExceptT.mk <| readScaledFloatArray h (hdr.hiddenDim * hdr.modelDim) scalePow10
        let bOut ← ExceptT.mk (readVecIntervalsBinary h hdr.modelDim slack scalePow10)
        let ln2P := ln2Params.getD l defP
        if shareUpdate then
          residuals1 := applyMlp residuals1 usePerRow1 ln2P wIn wOut bIn bOut
          residuals2 := residuals1
        else
          if needUpdate1 then
            residuals1 := applyMlp residuals1 usePerRow1 ln2P wIn wOut bIn bOut
          if needUpdate2 then
            residuals2 := applyMlp residuals2 usePerRow2 ln2P wIn wOut bIn bOut
        if needUpdateV then
          if shareUpdateV then
            residualsV := residuals2
          else
            residualsV := applyMlp residualsV usePerRowV ln2P wIn wOut bIn bOut
        if shareUpdate then
          residualsSame := true
        else if needUpdate1 && needUpdate2 then
          residualsSame := false
        if needUpdateV then
          if shareUpdateV then
            residualsSameV := true
          else
            residualsSameV := false
        if let some t0 := tMlp0? then
          let t1 ← ExceptT.lift IO.monoNanosNow
          let dtMs := (t1 - t0) / 1000000
          ExceptT.lift <| IO.eprintln s!"timing:layer{l}:mlp_update {dtMs}ms"
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
        let _ ← ExceptT.mk (skipF64Array h hdr.modelDim)
      if l == layer1 && p1?.isNone then
        let p1 ←
          awaitPattern p1? p1Task? "missing best-match pattern cert for layer1"
        p1? := some p1
      if l == layer2 && p2?.isNone then
        let p2 ←
          awaitPattern p2? p2Task? "missing best-match pattern cert for layer2"
        p2? := some p2
    match p1?, p2?, vlogit? with
    | some p1, some p2, some vlogit =>
        let cert : InductionHeadBestMatchSoundCert := {
          layer1Pattern := p1
          layer2Pattern := p2
          layer2Value := vlogit.value
          layer2Logit? := vlogit.logit?
          deltaLowerBound := vlogit.value.outputCoordLowerBound
        }
        if cert.check then
          return cert
        throw "induction head certificate failed internal consistency checks"
    | _, _, _ =>
        throw "target layer not reached"
  action.run


/-- Compute a combined sound certificate for an induction-style head pair (best-match,
binary only). -/
def certifyInductionSoundBestMatch
    (path : System.FilePath)
    (layer1 head1 layer2 head2 coord : Nat)
    (queryPos? : Option Nat := none)
    (eps : Rat)
    (soundnessBits : Nat)
    (inputPath? : Option System.FilePath := none)
    (inputDelta : Rat := 0)
    (offset1 : Int := -1)
    (offset2 : Int := -1)
    (keyOffset1 : Int := 0)
    (keyOffset2 : Int := 0)
    (maxSeqLen : Nat := 256)
    (scalePow10 : Nat := defaultBinaryScalePow10)
    (tightPattern : Bool := false)
    (tightPatternLayers : Nat := 1)
    (perRowPatternLayers : Nat := 0)
    (useAffine : Bool := false)
    (iterTighten : Bool := false)
    (targetToken? : Option Nat := none)
    (negativeToken? : Option Nat := none)
    (softmaxExpEffort : Nat := defaultSoftmaxExpEffort)
    (causalPattern : Bool := true) :
    IO (Except String InductionHeadBestMatchSoundCert) := do
  if inputDelta < 0 then
    return .error "delta must be nonnegative"
  let action : ExceptT String IO InductionHeadBestMatchSoundCert := do
    let h ← IO.FS.Handle.mk path IO.FS.Mode.read
    let firstLine := (← h.getLine).trim
    if firstLine = "NFP_BINARY_V1" then
      let inputPath := inputPath?.getD path
      let timingEnabled ← ExceptT.lift <| IO.getEnv "NFP_TIMING"
      let timing : Bool := timingEnabled.isSome
      let timeIt {α : Type} (label : String) (action : ExceptT String IO α) :
          ExceptT String IO α := do
        if !timing then
          action
        else
          let t0 ← ExceptT.lift IO.monoNanosNow
          let r ← action
          let t1 ← ExceptT.lift IO.monoNanosNow
          let dtNs := t1 - t0
          let dtMs := dtNs / 1000000
          ExceptT.lift <| IO.eprintln s!"timing:{label} {dtMs}ms"
          return r
      let loadSharedAndDirection (scalePow10 : Nat) :
          ExceptT String IO (SharedBinaryInputs × Option (Thunk (Array Fixed10Interval))) := do
        let sharedTask ←
          ExceptT.lift <| IO.asTask (loadSharedBinaryInputs path inputPath inputDelta scalePow10)
        let directionTask? ←
          match targetToken?, negativeToken? with
          | none, none => pure none
          | some targetToken, some negativeToken =>
              let task ←
                ExceptT.lift <|
                  IO.asTask
                    (readLogitDiffDirectionBinary
                      path targetToken negativeToken scalePow10 fixedUlpSlack)
              pure (some task)
          | _, _ =>
              throw "use both target and negative tokens (or neither)"
        let shared ←
          match sharedTask.get with
          | .error e => throw (toString e)
          | .ok (.error msg) => throw msg
          | .ok (.ok v) => pure v
        let direction? ←
          match directionTask? with
          | none => pure none
          | some task =>
              match task.get with
              | .error e => throw (toString e)
              | .ok (.error msg) => throw msg
              | .ok (.ok (hdrDir, dir)) =>
                  if hdrDir.modelDim ≠ shared.hdr.modelDim then
                    throw "unembedding model_dim mismatch"
                  pure (some (Thunk.mk (fun () => dir)))
        return (shared, direction?)
      let computeBestAtScale (scalePow10 : Nat)
          (configs : Array (Bool × Nat × Nat)) :
          ExceptT String IO (Rat × InductionHeadBestMatchSoundCert) := do
        let (shared, direction?) ← loadSharedAndDirection scalePow10
        let queryPos : Nat :=
          match queryPos? with
          | some q => q
          | none =>
              if shared.hdr.seqLen = 0 then 0 else shared.hdr.seqLen - 1
        if queryPos ≥ shared.hdr.seqLen then
          throw s!"queryPos {queryPos} out of range"
        let prefixCache := mkSharedBinaryPrefix shared queryPos causalPattern
        let computeCert (useTight : Bool) (tightLayers perRowLayers : Nat) :
            ExceptT String IO InductionHeadBestMatchSoundCert := do
          let label :=
            s!"scale={scalePow10} tight={useTight} tl={tightLayers} pr={perRowLayers}"
          let cert ←
            timeIt (s!"{label}:pair") <|
              ExceptT.mk <|
                certifyInductionSoundBestMatchLocalBinaryPair
                    path layer1 head1 layer2 head2 coord queryPos eps soundnessBits inputPath
                    inputDelta offset1 offset2 keyOffset1 keyOffset2 maxSeqLen scalePow10 useTight
                    tightLayers perRowLayers useAffine softmaxExpEffort causalPattern
                    (shared? := some shared) (prefix? := some prefixCache)
                    (targetToken? := targetToken?) (negativeToken? := negativeToken?)
                    (direction? := direction?)
          return cert
        let metricOf (cert : InductionHeadBestMatchSoundCert) : Rat :=
          match cert.layer2Logit? with
          | some logit => logit.logitDiffLowerBound
          | none => cert.deltaLowerBound
        -- Avoid nested task pools when per-row MLP already spawns tasks.
        let parallelConfigs : Bool :=
          configs.size > 1 && configs.all (fun (_, _, perRowLayers) => perRowLayers = 0)
        let mut best : Option (Rat × InductionHeadBestMatchSoundCert) := none
        if parallelConfigs then
          let tasks ←
            ExceptT.lift <|
              configs.mapM fun (useTight, tightLayers, perRowLayers) =>
                IO.asTask (computeCert useTight tightLayers perRowLayers).run
          let results := tasks.map (fun t => t.get)
          for i in [:configs.size] do
            let res := results[i]!
            match res with
            | .error e => throw (toString e)
            | .ok (.error msg) => throw msg
            | .ok (.ok cert) =>
                let metric := metricOf cert
                best :=
                  match best with
                  | none => some (metric, cert)
                  | some (bestMetric, bestCert) =>
                      if metric > bestMetric then
                        some (metric, cert)
                      else
                        some (bestMetric, bestCert)
        else
          for i in [:configs.size] do
            let (useTight, tightLayers, perRowLayers) := configs[i]!
            let cert ← computeCert useTight tightLayers perRowLayers
            let metric := metricOf cert
            best :=
              match best with
              | none => some (metric, cert)
              | some (bestMetric, bestCert) =>
                  if metric > bestMetric then
                    some (metric, cert)
                  else
                    some (bestMetric, bestCert)
        match best with
        | none => throw "no induction certs computed"
        | some bestPair => return bestPair
      let computeBestAtScaleOrdered (scalePow10 : Nat)
          (configs : Array (Bool × Nat × Nat))
          (stopAtPositive : Bool) :
          ExceptT String IO (Rat × InductionHeadBestMatchSoundCert) := do
        let (shared, direction?) ← loadSharedAndDirection scalePow10
        let queryPos : Nat :=
          match queryPos? with
          | some q => q
          | none =>
              if shared.hdr.seqLen = 0 then 0 else shared.hdr.seqLen - 1
        if queryPos ≥ shared.hdr.seqLen then
          throw s!"queryPos {queryPos} out of range"
        let prefixCache := mkSharedBinaryPrefix shared queryPos causalPattern
        let computeCert (useTight : Bool) (tightLayers perRowLayers : Nat) :
            ExceptT String IO InductionHeadBestMatchSoundCert := do
          let label :=
            s!"scale={scalePow10} tight={useTight} tl={tightLayers} pr={perRowLayers}"
          let cert ←
            timeIt (s!"{label}:pair") <|
              ExceptT.mk <|
                certifyInductionSoundBestMatchLocalBinaryPair
                    path layer1 head1 layer2 head2 coord queryPos eps soundnessBits inputPath
                    inputDelta offset1 offset2 keyOffset1 keyOffset2 maxSeqLen scalePow10 useTight
                    tightLayers perRowLayers useAffine softmaxExpEffort causalPattern
                    (shared? := some shared) (prefix? := some prefixCache)
                    (targetToken? := targetToken?) (negativeToken? := negativeToken?)
                    (direction? := direction?)
          return cert
        let metricOf (cert : InductionHeadBestMatchSoundCert) : Rat :=
          match cert.layer2Logit? with
          | some logit => logit.logitDiffLowerBound
          | none => cert.deltaLowerBound
        let mut best : Option (Rat × InductionHeadBestMatchSoundCert) := none
        for i in [:configs.size] do
          let (useTight, tightLayers, perRowLayers) := configs[i]!
          let cert ← computeCert useTight tightLayers perRowLayers
          let metric := metricOf cert
          if stopAtPositive && metric > 0 then
            return (metric, cert)
          best :=
            match best with
            | none => some (metric, cert)
            | some (bestMetric, bestCert) =>
                if metric > bestMetric then
                  some (metric, cert)
                else
                  some (bestMetric, bestCert)
        match best with
        | none => throw "no induction certs computed"
        | some bestPair => return bestPair
      let maxLayer := Nat.max layer1 layer2
      let tightFull := Nat.max 1 maxLayer
      let perRowFull := maxLayer
      let normalizeConfig (useTight : Bool) (tightLayers perRowLayers : Nat) :
          Bool × Nat × Nat :=
        if useTight then
          (true, Nat.max 1 tightLayers, perRowLayers)
        else
          (false, 0, perRowLayers)
      let pushUnique (configs : Array (Bool × Nat × Nat)) (cfg : Bool × Nat × Nat) :
          Array (Bool × Nat × Nat) :=
        if configs.any (fun c => c == cfg) then configs else configs.push cfg
      let baseCfg : Bool × Nat × Nat :=
        normalizeConfig tightPattern tightPatternLayers perRowPatternLayers
      if !iterTighten then
        let (_, cert) ← computeBestAtScale scalePow10 #[baseCfg]
        return cert
      else
        let mut configs : Array (Bool × Nat × Nat) := #[baseCfg]
        let needTightFull := (!tightPattern) || tightPatternLayers < tightFull
        if needTightFull then
          configs := pushUnique configs (normalizeConfig true tightFull perRowPatternLayers)
        if perRowPatternLayers < perRowFull then
          configs := pushUnique configs (normalizeConfig true tightFull perRowFull)
        let scales : List Nat := [scalePow10, scalePow10 + 1, scalePow10 + 2]
        let mut bestOverall : Option (Rat × InductionHeadBestMatchSoundCert) := none
        for scale in scales do
          let (metric, cert) ←
            computeBestAtScaleOrdered scale configs (stopAtPositive := true)
          bestOverall :=
            match bestOverall with
            | none => some (metric, cert)
            | some (bestMetric, bestCert) =>
                if metric > bestMetric then
                  some (metric, cert)
                else
                  some (bestMetric, bestCert)
          if metric > 0 then
            return cert
        match bestOverall with
        | none => throw "no induction certs computed"
        | some (_, cert) => return cert
    else
      throw "induction sound cert requires NFP_BINARY_V1"
  action.run

/-! ### Specs -/

theorem defaultBinaryScalePow10_spec_io :
    defaultBinaryScalePow10 = defaultBinaryScalePow10 := rfl

theorem maxAbsOfVector_spec_io :
    maxAbsOfVector = maxAbsOfVector := rfl

theorem certifyHeadBoundsBinary_spec_io :
    certifyHeadBoundsBinary = certifyHeadBoundsBinary := rfl

theorem certifyModelFileGlobalBinary_spec_io :
    certifyModelFileGlobalBinary = certifyModelFileGlobalBinary := rfl

theorem addVecIntervals_spec_io :
    addVecIntervals = addVecIntervals := rfl

theorem addConstVec_spec_io :
    addConstVec = addConstVec := rfl

theorem unionVecIntervals_spec_io :
    unionVecIntervals = unionVecIntervals := rfl

theorem zeroIntervals_spec_io :
    zeroIntervals = zeroIntervals := rfl

theorem unionRows_spec_io :
    unionRows = unionRows := rfl

theorem layerNormRowApprox_spec_io :
    layerNormRowApprox = layerNormRowApprox := rfl

theorem minVarAcrossRows_spec_io :
    minVarAcrossRows = minVarAcrossRows := rfl

theorem findLineIdxFrom_spec_io :
    findLineIdxFrom = findLineIdxFrom := rfl

theorem skipUntil_spec_io :
    skipUntil = skipUntil := rfl

theorem skipBlankLines_spec_io :
    skipBlankLines = skipBlankLines := rfl

theorem countWsTokens_spec_io :
    countWsTokens = countWsTokens := rfl

theorem consumeTokensSkipFast_spec_io :
    consumeTokensSkipFast = consumeTokensSkipFast := rfl

theorem consumeMatrixSkip_spec_io :
    consumeMatrixSkip = consumeMatrixSkip := rfl

theorem consumeMatrixSkipFast_spec_io :
    consumeMatrixSkipFast = consumeMatrixSkipFast := rfl

theorem consumeVectorSkipFast_spec_io :
    consumeVectorSkipFast = consumeVectorSkipFast := rfl

theorem consumeMatrixMulAndNormInf_spec_io :
    consumeMatrixMulAndNormInf = consumeMatrixMulAndNormInf := rfl

theorem certifyModelFileGlobal_spec_io :
    certifyModelFileGlobal = certifyModelFileGlobal := rfl

theorem loadEmbeddingsIntervals_spec_io :
    loadEmbeddingsIntervals = loadEmbeddingsIntervals := rfl

theorem intervalsFromScaled_spec_io :
    intervalsFromScaled = intervalsFromScaled := rfl

theorem collectLayerNormParams_spec_io :
    collectLayerNormParams = collectLayerNormParams := rfl

theorem collectLayerNormParamsBinary_spec_io :
    collectLayerNormParamsBinary = collectLayerNormParamsBinary := rfl

theorem defaultFixedScalePow10_spec_io :
    defaultFixedScalePow10 = defaultFixedScalePow10 := rfl

theorem fixedUlpSlack_spec_io :
    fixedUlpSlack = fixedUlpSlack := rfl

theorem scaleCfgOfPow10_spec_io :
    scaleCfgOfPow10 = scaleCfgOfPow10 := rfl

theorem ratCeilMulNat_spec_io :
    ratCeilMulNat = ratCeilMulNat := rfl

theorem fixedMeanInterval_spec_io :
    fixedMeanInterval = fixedMeanInterval := rfl

theorem fixedVarianceLowerBoundRange_spec_io :
    fixedVarianceLowerBoundRange = fixedVarianceLowerBoundRange := rfl

theorem fixedLayerNormRowApprox_spec_io :
    fixedLayerNormRowApprox = fixedLayerNormRowApprox := rfl

theorem readVecIntervals_spec_io :
    readVecIntervals = readVecIntervals := rfl

theorem readVecIntervalsBinary_spec_io :
    readVecIntervalsBinary = readVecIntervalsBinary := rfl

theorem matMulIntervalsFromScaled_spec_io :
    matMulIntervalsFromScaled = matMulIntervalsFromScaled := rfl

theorem fixedDotInterval_spec_io :
    fixedDotInterval = fixedDotInterval := rfl

theorem maxAbsVecFixed_spec_io :
    maxAbsVecFixed = maxAbsVecFixed := rfl

theorem addVecFixed_spec_io :
    addVecFixed = addVecFixed := rfl

theorem addVecFixedRows_spec_io :
    addVecFixedRows = addVecFixedRows := rfl

theorem addRowsFixed_spec_io :
    addRowsFixed = addRowsFixed := rfl

theorem mlpRowFromScaled_spec_io :
    mlpRowFromScaled = mlpRowFromScaled := rfl

theorem mlpRowsFromScaled_spec_io :
    mlpRowsFromScaled = mlpRowsFromScaled := rfl

theorem groupUnionRowsByToken_spec_io :
    groupUnionRowsByToken = groupUnionRowsByToken := rfl

theorem unionRowsFixed_spec_io :
    unionRowsFixed = unionRowsFixed := rfl

theorem consumeMatrixMulAndNormInfFixed_spec_io :
    consumeMatrixMulAndNormInfFixed = consumeMatrixMulAndNormInfFixed := rfl

theorem consumeMatrixMulAndNormInfFixedBinary_spec_io :
    consumeMatrixMulAndNormInfFixedBinary = consumeMatrixMulAndNormInfFixedBinary := rfl

theorem loadEmbeddingsUnionFixed_spec_io :
    loadEmbeddingsUnionFixed = loadEmbeddingsUnionFixed := rfl

theorem loadEmbeddingsUnionFixedBinary_spec_io :
    loadEmbeddingsUnionFixedBinary = loadEmbeddingsUnionFixedBinary := rfl

theorem loadEmbeddingsIntervalsBinary_spec_io :
    loadEmbeddingsIntervalsBinary = loadEmbeddingsIntervalsBinary := rfl

theorem loadTokensBinary_spec_io :
    loadTokensBinary = loadTokensBinary := rfl

theorem skipToUnembeddingBinary_spec_io :
    skipToUnembeddingBinary = skipToUnembeddingBinary := rfl

theorem certifyHeadValueLowerBoundLocalBinaryAt_spec_io :
    certifyHeadValueLowerBoundLocalBinaryAt = certifyHeadValueLowerBoundLocalBinaryAt := rfl

theorem readUnembeddingColumnsBinary_spec_io :
    readUnembeddingColumnsBinary = readUnembeddingColumnsBinary := rfl

theorem readLogitDiffDirectionBinary_spec_io :
    readLogitDiffDirectionBinary = readLogitDiffDirectionBinary := rfl

theorem certifyHeadLogitDiffLowerBoundLocalBinaryAt_spec_io :
    certifyHeadLogitDiffLowerBoundLocalBinaryAt =
      certifyHeadLogitDiffLowerBoundLocalBinaryAt := rfl

theorem ensureSoundCache_spec_io :
    ensureSoundCache = ensureSoundCache := rfl

theorem certifyModelFileLocalText_spec_io :
    certifyModelFileLocalText = certifyModelFileLocalText := rfl

theorem certifyModelFileLocal_spec_io :
    certifyModelFileLocal = certifyModelFileLocal := rfl

theorem certifyModelFileLocalBinary_spec_io :
    certifyModelFileLocalBinary = certifyModelFileLocalBinary := rfl

theorem certifyHeadBoundsLocalBinary_spec_io :
    certifyHeadBoundsLocalBinary = certifyHeadBoundsLocalBinary := rfl

theorem certifyHeadPatternLocalBinary_spec_io :
    certifyHeadPatternLocalBinary = certifyHeadPatternLocalBinary := rfl

theorem certifyHeadPatternBestMatchLocalBinary_spec_io :
    certifyHeadPatternBestMatchLocalBinary = certifyHeadPatternBestMatchLocalBinary := rfl

theorem certifyHeadPatternBestMatchLocalBinarySweep_spec_io :
    certifyHeadPatternBestMatchLocalBinarySweep =
      certifyHeadPatternBestMatchLocalBinarySweep := rfl

theorem certifyHeadValueLowerBoundLocalBinary_spec_io :
    certifyHeadValueLowerBoundLocalBinary = certifyHeadValueLowerBoundLocalBinary := rfl

theorem certifyHeadLogitDiffLowerBoundLocalBinary_spec_io :
    certifyHeadLogitDiffLowerBoundLocalBinary = certifyHeadLogitDiffLowerBoundLocalBinary := rfl

theorem certifyModelFile_spec_io :
    certifyModelFile = certifyModelFile := rfl

theorem certifyHeadBounds_spec_io :
    certifyHeadBounds = certifyHeadBounds := rfl

theorem certifyHeadBoundsLocal_spec_io :
    certifyHeadBoundsLocal = certifyHeadBoundsLocal := rfl

theorem certifyHeadPatternLocal_spec_io :
    certifyHeadPatternLocal = certifyHeadPatternLocal := rfl

theorem certifyHeadPatternBestMatchLocal_spec_io :
    certifyHeadPatternBestMatchLocal = certifyHeadPatternBestMatchLocal := rfl

theorem certifyHeadPatternBestMatchLocalSweep_spec_io :
    certifyHeadPatternBestMatchLocalSweep = certifyHeadPatternBestMatchLocalSweep := rfl

theorem certifyLayerBestMatchMarginLocal_spec_io :
    certifyLayerBestMatchMarginLocal = certifyLayerBestMatchMarginLocal := rfl

theorem certifyHeadValueLowerBoundLocal_spec_io :
    certifyHeadValueLowerBoundLocal = certifyHeadValueLowerBoundLocal := rfl

theorem certifyHeadLogitDiffLowerBoundLocal_spec_io :
    certifyHeadLogitDiffLowerBoundLocal = certifyHeadLogitDiffLowerBoundLocal := rfl

theorem certifyInductionSound_spec_io :
    certifyInductionSound = certifyInductionSound := rfl

theorem certifyInductionSoundBestMatch_spec_io :
    certifyInductionSoundBestMatch = certifyInductionSoundBestMatch := rfl

end Nfp.Untrusted.SoundCompute
