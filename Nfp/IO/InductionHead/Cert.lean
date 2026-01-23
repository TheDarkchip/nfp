-- SPDX-License-Identifier: AGPL-3.0-or-later
module
public import Mathlib.Data.Finset.Insert
public import Nfp.Circuit.Cert.InductionHead
public import Nfp.Circuit.Cert.LogitDiff
public import Nfp.IO.InductionHead.Tokens
public import Nfp.IO.Parse.Basic
public import Nfp.IO.Stripe
public import Nfp.IO.Util
public import Nfp.Bounds.LayerNorm
public import Nfp.Model.InductionPrompt
public import Nfp.Sound.Induction.ScoreSlice
/-!
Untrusted parsing and checking for explicit induction-head certificates.
All sequence indices in the certificate payload are 0-based (file-format convention) and
are converted to `Fin` indices internally. Supported certificate kinds:
- `kind onehot-approx` (proxy bounds only)
-- `kind induction-aligned` (requires `period` and checks prefix-matching metrics)
-/
public section
namespace Nfp
namespace IO
open Nfp.Circuit
open Nfp.IO.Parse
namespace InductionHeadCert
/-- State for parsing induction-head certificates. -/
structure ParseState (seq : Nat) where
  /-- Optional certificate kind tag. -/
  kind : Option String
  /-- Optional induction prompt period (for `kind induction-aligned`). -/
  period : Option Nat
  /-- Optional epsilon bound. -/
  eps : Option Rat
  /-- Optional margin bound. -/
  margin : Option Rat
  /-- Active query set. -/
  active : Finset (Fin seq)
  /-- Whether any active entries were parsed. -/
  activeSeen : Bool
  /-- Optional predecessor pointer per query. -/
  prev : Array (Option (Fin seq))
  /-- Optional score matrix entries. -/
  scores : Array (Array (Option Rat))
  /-- Optional weight matrix entries. -/
  weights : Array (Array (Option Rat))
  /-- Optional copy-logit matrix entries. -/
  copyLogits : Array (Array (Option Rat))
  /-- Whether any copy-logit entries were parsed. -/
  copyLogitsSeen : Bool
  /-- Optional per-query epsilon bounds. -/
  epsAt : Array (Option Rat)
  /-- Optional per-key weight bounds. -/
  weightBoundAt : Array (Array (Option Rat))
  /-- Optional lower bound for values. -/
  lo : Option Rat
  /-- Optional upper bound for values. -/
  hi : Option Rat
  /-- Optional per-key lower bounds. -/
  valsLo : Array (Option Rat)
  /-- Optional per-key upper bounds. -/
  valsHi : Array (Option Rat)
  /-- Optional per-key exact values. -/
  vals : Array (Option Rat)
  /-- Optional direction target index. -/
  directionTarget : Option Nat
  /-- Optional direction negative index. -/
  directionNegative : Option Nat
  /-- Optional TL exclude_bos flag. -/
  tlExcludeBos : Option Bool
  /-- Optional TL exclude_current_token flag. -/
  tlExcludeCurrent : Option Bool
  /-- Optional TL mul score lower bound. -/
  tlScoreLB : Option Rat
  /-- Optional model hidden dimension (for score anchoring). -/
  modelDModel : Option Nat
  /-- Optional model head dimension (for score anchoring). -/
  modelHeadDim : Option Nat
  /-- Optional model layer index (for score anchoring). -/
  modelLayer : Option Nat
  /-- Optional model head index (for score anchoring). -/
  modelHead : Option Nat
  /-- Optional model score scale factor. -/
  modelScoreScale : Option Rat
  /-- Optional model score mask value. -/
  modelScoreMask : Option Rat
  /-- Optional model mask-causal flag. -/
  modelMaskCausal : Option Bool
  /-- Optional model residual inputs (seq × d_model). -/
  modelResid : Array (Array (Option Rat))
  /-- Optional model embedding inputs (seq × d_model). -/
  modelEmbed : Array (Array (Option Rat))
  /-- Optional LayerNorm epsilon. -/
  modelLnEps : Option Rat
  /-- Optional LayerNorm bound slack. -/
  modelLnSlack : Option Rat
  /-- Optional LayerNorm sqrt bound scale. -/
  modelLnScale : Option Nat
  /-- Optional LayerNorm fast-path toggle. -/
  modelLnFast : Option Bool
  /-- Optional model-decimals hint for fixed-denominator checks. -/
  modelDecimals : Option Nat
  /-- Optional LayerNorm gamma. -/
  modelLnGamma : Array (Option Rat)
  /-- Optional LayerNorm beta. -/
  modelLnBeta : Array (Option Rat)
  /-- Optional model Wq slice (d_model × head_dim). -/
  modelWq : Array (Array (Option Rat))
  /-- Optional model Wk slice (d_model × head_dim). -/
  modelWk : Array (Array (Option Rat))
  /-- Optional model bq slice (head_dim). -/
  modelBq : Array (Option Rat)
  /-- Optional model bk slice (head_dim). -/
  modelBk : Array (Option Rat)
/-- Minimal model slice for recomputing attention scores. -/
structure ModelSlice (seq : Nat) where
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
  /-- Residual inputs (post-layernorm). -/
  resid : Fin seq → Fin dModel → Rat
  /-- Q projection slice. -/
  wq : Fin dModel → Fin headDim → Rat
  /-- K projection slice. -/
  wk : Fin dModel → Fin headDim → Rat
  /-- Q bias slice. -/
  bq : Fin headDim → Rat
  /-- K bias slice. -/
  bk : Fin headDim → Rat
/-- LayerNorm inputs for anchoring post-LN residuals. -/
structure ModelLnSlice (seq : Nat) where
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
  lnGamma : Fin dModel → Rat
  /-- LayerNorm beta. -/
  lnBeta : Fin dModel → Rat
  /-- Pre-LN embeddings/residual stream. -/
  embed : Fin seq → Fin dModel → Rat
/-- Initialize a parse state. -/
def initState (seq : Nat) : ParseState seq :=
  let row : Array (Option Rat) := Array.replicate seq none
  { kind := none
    period := none
    eps := none
    margin := none
    active := ∅
    activeSeen := false
    prev := Array.replicate seq none
    scores := Array.replicate seq row
    weights := Array.replicate seq row
    epsAt := Array.replicate seq none
    weightBoundAt := Array.replicate seq row
    copyLogits := Array.replicate seq row
    copyLogitsSeen := false
    lo := none
    hi := none
    valsLo := Array.replicate seq none
    valsHi := Array.replicate seq none
    vals := Array.replicate seq none
    directionTarget := none
    directionNegative := none
    tlExcludeBos := none
    tlExcludeCurrent := none
    tlScoreLB := none
    modelDModel := none
    modelHeadDim := none
    modelLayer := none
    modelHead := none
    modelScoreScale := none
    modelScoreMask := none
    modelMaskCausal := none
    modelResid := #[]
    modelEmbed := #[]
    modelLnEps := none
    modelLnSlack := none
    modelLnScale := none
    modelLnFast := none
    modelDecimals := none
    modelLnGamma := #[]
    modelLnBeta := #[]
    modelWq := #[]
    modelWk := #[]
    modelBq := #[]
    modelBk := #[] }
private def toIndex0 {seq : Nat} (label : String) (idx : Nat) : Except String (Fin seq) := do
  if h : idx < seq then
    return ⟨idx, h⟩
  else
    throw s!"{label} index out of range: {idx}"
private def toIndex0Sized (label : String) (size idx : Nat) : Except String (Fin size) := do
  if h : idx < size then
    return ⟨idx, h⟩
  else
    throw s!"{label} index out of range: {idx}"
private def setActive {seq : Nat} (st : ParseState seq) (q : Nat) :
    Except String (ParseState seq) := do
  let qFin ← toIndex0 (seq := seq) "q" q
  if qFin ∈ st.active then
    throw s!"duplicate active entry for q={q}"
  else
    return { st with active := insert qFin st.active, activeSeen := true }
private def setPrev {seq : Nat} (st : ParseState seq) (q k : Nat) :
    Except String (ParseState seq) := do
  let qFin ← toIndex0 (seq := seq) "q" q
  let kFin ← toIndex0 (seq := seq) "k" k
  match st.prev[qFin.1]! with
  | some _ =>
      throw s!"duplicate prev entry for q={q}"
  | none =>
      let prev' := st.prev.set! qFin.1 (some kFin)
      return { st with prev := prev' }
private def setVecEntry {seq : Nat} (arr : Array (Option Rat)) (idx : Nat) (v : Rat) :
    Except String (Array (Option Rat)) := do
  let kFin ← toIndex0 (seq := seq) "k" idx
  match arr[kFin.1]! with
  | some _ =>
      throw s!"duplicate entry for k={idx}"
  | none =>
      return arr.set! kFin.1 (some v)
private def setMatrixEntry {seq : Nat} (mat : Array (Array (Option Rat)))
    (q k : Nat) (v : Rat) : Except String (Array (Array (Option Rat))) := do
  let qFin ← toIndex0 (seq := seq) "q" q
  let kFin ← toIndex0 (seq := seq) "k" k
  let row := mat[qFin.1]!
  match row[kFin.1]! with
  | some _ =>
      throw s!"duplicate matrix entry at ({q}, {k})"
  | none =>
      let row' := row.set! kFin.1 (some v)
      return mat.set! qFin.1 row'
private def allocModelResid (seq dModel : Nat) : Array (Array (Option Rat)) :=
  Array.replicate seq (Array.replicate dModel none)
private def allocModelEmbed (seq dModel : Nat) : Array (Array (Option Rat)) :=
  Array.replicate seq (Array.replicate dModel none)
private def allocModelLnVec (dModel : Nat) : Array (Option Rat) :=
  Array.replicate dModel none
private def allocModelW (dModel headDim : Nat) : Array (Array (Option Rat)) :=
  Array.replicate dModel (Array.replicate headDim none)
private def allocModelB (headDim : Nat) : Array (Option Rat) :=
  Array.replicate headDim none
private def ensureModelResid {seq : Nat} (st : ParseState seq) :
    Except String (ParseState seq × Nat) := do
  let dModel ←
    match st.modelDModel with
    | some v => pure v
    | none => throw "model-d-model required before model-resid"
  if dModel = 0 then
    throw "model-d-model must be positive"
  let st :=
    if st.modelResid.isEmpty then
      { st with modelResid := allocModelResid seq dModel }
    else
      st
  return (st, dModel)
private def ensureModelEmbed {seq : Nat} (st : ParseState seq) :
    Except String (ParseState seq × Nat) := do
  let dModel ←
    match st.modelDModel with
    | some v => pure v
    | none => throw "model-d-model required before model-embed"
  if dModel = 0 then
    throw "model-d-model must be positive"
  let st :=
    if st.modelEmbed.isEmpty then
      { st with modelEmbed := allocModelEmbed seq dModel }
    else
      st
  let st :=
    if st.modelLnGamma.isEmpty then
      { st with modelLnGamma := allocModelLnVec dModel, modelLnBeta := allocModelLnVec dModel }
    else
      st
  return (st, dModel)
private def ensureModelW {seq : Nat} (st : ParseState seq) :
    Except String (ParseState seq × Nat × Nat) := do
  let dModel ←
    match st.modelDModel with
    | some v => pure v
    | none => throw "model-d-model required before model-wq/wk/bq/bk"
  let headDim ←
    match st.modelHeadDim with
    | some v => pure v
    | none => throw "model-head-dim required before model-wq/wk/bq/bk"
  if dModel = 0 then
    throw "model-d-model must be positive"
  if headDim = 0 then
    throw "model-head-dim must be positive"
  let st :=
    if st.modelWq.isEmpty then
      { st with modelWq := allocModelW dModel headDim, modelWk := allocModelW dModel headDim }
    else
      st
  let st :=
    if st.modelBq.isEmpty then
      { st with modelBq := allocModelB headDim, modelBk := allocModelB headDim }
  else
      st
  return (st, dModel, headDim)
private def setModelResidEntry {seq : Nat} (st : ParseState seq) (q i : Nat) (v : Rat) :
    Except String (ParseState seq) := do
  let (st, dModel) ← ensureModelResid st
  let qFin ← toIndex0 (seq := seq) "q" q
  let iFin ← toIndex0Sized "i" dModel i
  let row := st.modelResid[qFin.1]!
  match row[iFin.1]! with
  | some _ =>
      throw s!"duplicate model-resid entry at ({q}, {i})"
  | none =>
      let row' := row.set! iFin.1 (some v)
      return { st with modelResid := st.modelResid.set! qFin.1 row' }
private def setModelEmbedEntry {seq : Nat} (st : ParseState seq) (q i : Nat) (v : Rat) :
    Except String (ParseState seq) := do
  let (st, dModel) ← ensureModelEmbed st
  let qFin ← toIndex0 (seq := seq) "q" q
  let iFin ← toIndex0Sized "i" dModel i
  let row := st.modelEmbed[qFin.1]!
  match row[iFin.1]! with
  | some _ =>
      throw s!"duplicate model-embed entry at ({q}, {i})"
  | none =>
      let row' := row.set! iFin.1 (some v)
      return { st with modelEmbed := st.modelEmbed.set! qFin.1 row' }
private def setModelLnEntry {seq : Nat} (st : ParseState seq)
    (i : Nat) (v : Rat) (which : String) : Except String (ParseState seq) := do
  let (st, dModel) ← ensureModelEmbed st
  let iFin ← toIndex0Sized "i" dModel i
  let arr := if which = "gamma" then st.modelLnGamma else st.modelLnBeta
  match arr[iFin.1]! with
  | some _ =>
      throw s!"duplicate model-ln-{which} entry for i={i}"
  | none =>
      let arr' := arr.set! iFin.1 (some v)
      if which = "gamma" then
        return { st with modelLnGamma := arr' }
      else
        return { st with modelLnBeta := arr' }
private def setModelWEntry {seq : Nat} (st : ParseState seq)
    (i j : Nat) (v : Rat) (which : String) :
    Except String (ParseState seq) := do
  let (st, dModel, headDim) ← ensureModelW st
  let iFin ← toIndex0Sized "i" dModel i
  let jFin ← toIndex0Sized "j" headDim j
  let mat := if which = "wq" then st.modelWq else st.modelWk
  let row := mat[iFin.1]!
  match row[jFin.1]! with
  | some _ =>
      throw s!"duplicate model-{which} entry at ({i}, {j})"
  | none =>
      let row' := row.set! jFin.1 (some v)
      let mat' := mat.set! iFin.1 row'
      if which = "wq" then
        return { st with modelWq := mat' }
      else
        return { st with modelWk := mat' }
private def setModelBEntry {seq : Nat} (st : ParseState seq)
    (j : Nat) (v : Rat) (which : String) :
    Except String (ParseState seq) := do
  let (st, _dModel, headDim) ← ensureModelW st
  let jFin ← toIndex0Sized "j" headDim j
  let arr := if which = "bq" then st.modelBq else st.modelBk
  match arr[jFin.1]! with
  | some _ =>
      throw s!"duplicate model-{which} entry for j={j}"
  | none =>
      let arr' := arr.set! jFin.1 (some v)
      if which = "bq" then
        return { st with modelBq := arr' }
      else
        return { st with modelBk := arr' }
/-- Parse a boolean literal. -/
private def parseBool (s : String) : Except String Bool := do
  match s.toLower with
  | "true" => pure true
  | "false" => pure false
  | "1" => pure true
  | "0" => pure false
  | _ => throw s!"expected Bool, got '{s}'"
/-- Parse a tokenized line into the parse state. -/
def parseLine {seq : Nat} (st : ParseState seq) (tokens : List String) :
    Except String (ParseState seq) := do
  match tokens with
  | ["kind", k] =>
      if st.kind.isSome then
        throw "duplicate kind entry"
      else
        return { st with kind := some k }
  | ["period", val] =>
      if st.period.isSome then
        throw "duplicate period entry"
      else
        return { st with period := some (← parseNat val) }
  | ["eps", val] =>
      if st.eps.isSome then
        throw "duplicate eps entry"
      else
        return { st with eps := some (← parseRat val) }
  | ["margin", val] =>
      if st.margin.isSome then
        throw "duplicate margin entry"
      else
        return { st with margin := some (← parseRat val) }
  | ["active", q] =>
      setActive st (← parseNat q)
  | ["prev", q, k] =>
      setPrev st (← parseNat q) (← parseNat k)
  | ["score", q, k, val] =>
      let mat ← setMatrixEntry (seq := seq) st.scores (← parseNat q) (← parseNat k)
        (← parseRat val)
      return { st with scores := mat }
  | ["weight", q, k, val] =>
      let mat ← setMatrixEntry (seq := seq) st.weights (← parseNat q) (← parseNat k)
        (← parseRat val)
      return { st with weights := mat }
  | ["copy-logit", q, k, val] =>
      let mat ← setMatrixEntry (seq := seq) st.copyLogits (← parseNat q) (← parseNat k)
        (← parseRat val)
      return { st with copyLogits := mat, copyLogitsSeen := true }
  | ["eps-at", q, val] =>
      let arr ← setVecEntry (seq := seq) st.epsAt (← parseNat q) (← parseRat val)
      return { st with epsAt := arr }
  | ["weight-bound", q, k, val] =>
      let mat ← setMatrixEntry (seq := seq) st.weightBoundAt (← parseNat q) (← parseNat k)
        (← parseRat val)
      return { st with weightBoundAt := mat }
  | ["lo", val] =>
      if st.lo.isSome then
        throw "duplicate lo entry"
      else
        return { st with lo := some (← parseRat val) }
  | ["hi", val] =>
      if st.hi.isSome then
        throw "duplicate hi entry"
      else
        return { st with hi := some (← parseRat val) }
  | ["val", k, val] =>
      let arr ← setVecEntry (seq := seq) st.vals (← parseNat k) (← parseRat val)
      return { st with vals := arr }
  | ["val-lo", k, val] =>
      let arr ← setVecEntry (seq := seq) st.valsLo (← parseNat k) (← parseRat val)
      return { st with valsLo := arr }
  | ["val-hi", k, val] =>
      let arr ← setVecEntry (seq := seq) st.valsHi (← parseNat k) (← parseRat val)
      return { st with valsHi := arr }
  | ["direction-target", tok] =>
      if st.directionTarget.isSome then
        throw "duplicate direction-target entry"
      else
        return { st with directionTarget := some (← parseNat tok) }
  | ["direction-negative", tok] =>
      if st.directionNegative.isSome then
        throw "duplicate direction-negative entry"
      else
        return { st with directionNegative := some (← parseNat tok) }
  | ["tl-exclude-bos", val] =>
      if st.tlExcludeBos.isSome then
        throw "duplicate tl-exclude-bos entry"
      else
        return { st with tlExcludeBos := some (← parseBool val) }
  | ["tl-exclude-current-token", val] =>
      if st.tlExcludeCurrent.isSome then
        throw "duplicate tl-exclude-current-token entry"
      else
        return { st with tlExcludeCurrent := some (← parseBool val) }
  | ["tl-score-lb", val] =>
      if st.tlScoreLB.isSome then
        throw "duplicate tl-score-lb entry"
      else
        return { st with tlScoreLB := some (← parseRat val) }
  | ["model-d-model", val] =>
      if st.modelDModel.isSome then
        throw "duplicate model-d-model entry"
      else
        let dModel := (← parseNat val)
        let mut st' := { st with modelDModel := some dModel }
        if !st'.modelResid.isEmpty then
          return st'
        if dModel > 0 then
          st' := { st' with modelResid := allocModelResid seq dModel }
        if let some headDim := st'.modelHeadDim then
          if headDim > 0 then
            if st'.modelWq.isEmpty then
              st' := { st' with modelWq := allocModelW dModel headDim,
                                modelWk := allocModelW dModel headDim }
            if st'.modelBq.isEmpty then
              st' := { st' with modelBq := allocModelB headDim,
                                modelBk := allocModelB headDim }
        return st'
  | ["model-head-dim", val] =>
      if st.modelHeadDim.isSome then
        throw "duplicate model-head-dim entry"
      else
        let headDim := (← parseNat val)
        let mut st' := { st with modelHeadDim := some headDim }
        if let some dModel := st'.modelDModel then
          if dModel > 0 && headDim > 0 then
            if st'.modelWq.isEmpty then
              st' := { st' with modelWq := allocModelW dModel headDim,
                                modelWk := allocModelW dModel headDim }
            if st'.modelBq.isEmpty then
              st' := { st' with modelBq := allocModelB headDim,
                                modelBk := allocModelB headDim }
        return st'
  | ["model-layer", val] =>
      if st.modelLayer.isSome then
        throw "duplicate model-layer entry"
      else
        return { st with modelLayer := some (← parseNat val) }
  | ["model-head", val] =>
      if st.modelHead.isSome then
        throw "duplicate model-head entry"
      else
        return { st with modelHead := some (← parseNat val) }
  | ["model-score-scale", val] =>
      if st.modelScoreScale.isSome then
        throw "duplicate model-score-scale entry"
      else
        return { st with modelScoreScale := some (← parseRat val) }
  | ["model-score-mask", val] =>
      if st.modelScoreMask.isSome then
        throw "duplicate model-score-mask entry"
      else
        return { st with modelScoreMask := some (← parseRat val) }
  | ["model-mask-causal", val] =>
      if st.modelMaskCausal.isSome then
        throw "duplicate model-mask-causal entry"
      else
        return { st with modelMaskCausal := some (← parseBool val) }
  | ["model-ln-eps", val] =>
      if st.modelLnEps.isSome then
        throw "duplicate model-ln-eps entry"
      else
        return { st with modelLnEps := some (← parseRat val) }
  | ["model-ln-slack", val] =>
      if st.modelLnSlack.isSome then
        throw "duplicate model-ln-slack entry"
      else
        return { st with modelLnSlack := some (← parseRat val) }
  | ["model-ln-scale", val] =>
      if st.modelLnScale.isSome then
        throw "duplicate model-ln-scale entry"
      else
        return { st with modelLnScale := some (← parseNat val) }
  | ["model-ln-fast", val] =>
      if st.modelLnFast.isSome then
        throw "duplicate model-ln-fast entry"
      else
        return { st with modelLnFast := some (← parseBool val) }
  | ["model-decimals", val] =>
      if st.modelDecimals.isSome then
        throw "duplicate model-decimals entry"
      else
        return { st with modelDecimals := some (← parseNat val) }
  | ["model-ln-gamma", i, val] =>
      setModelLnEntry st (← parseNat i) (← parseRat val) "gamma"
  | ["model-ln-beta", i, val] =>
      setModelLnEntry st (← parseNat i) (← parseRat val) "beta"
  | ["model-embed", q, i, val] =>
      setModelEmbedEntry st (← parseNat q) (← parseNat i) (← parseRat val)
  | ["model-resid", q, i, val] =>
      setModelResidEntry st (← parseNat q) (← parseNat i) (← parseRat val)
  | ["model-wq", i, j, val] =>
      setModelWEntry st (← parseNat i) (← parseNat j) (← parseRat val) "wq"
  | ["model-wk", i, j, val] =>
      setModelWEntry st (← parseNat i) (← parseNat j) (← parseRat val) "wk"
  | ["model-bq", j, val] =>
      setModelBEntry st (← parseNat j) (← parseRat val) "bq"
  | ["model-bk", j, val] =>
      setModelBEntry st (← parseNat j) (← parseRat val) "bk"
  | _ =>
      throw s!"unrecognized line: '{String.intercalate " " tokens}'"
/-- Extract the `seq` header from tokenized lines. -/
def parseSeq (tokens : List (List String)) : Except String Nat := do
  let mut seq? : Option Nat := none
  for t in tokens do
    match t with
    | ["seq", n] =>
        if seq?.isSome then
          throw "duplicate seq entry"
        else
          seq? := some (← parseNat n)
    | _ => pure ()
  match seq? with
  | some v => pure v
  | none => throw "missing seq entry"
/-- Parsed induction-head certificate payload plus kind metadata. -/
structure InductionHeadCertPayload (seq : Nat) where
  /-- Certificate kind tag. -/
  kind : String
  /-- Optional prompt period (only for `kind induction-aligned`). -/
  period? : Option Nat
  /-- Optional model slice for score anchoring. -/
  modelSlice? : Option (InductionHeadCert.ModelSlice seq)
  /-- Optional LayerNorm inputs for anchoring residuals. -/
  modelLnSlice? : Option (InductionHeadCert.ModelLnSlice seq)
  /-- Optional TL exclude_bos flag. -/
  tlExcludeBos? : Option Bool
  /-- Optional TL exclude_current_token flag. -/
  tlExcludeCurrent? : Option Bool
  /-- Optional TL mul score lower bound. -/
  tlScoreLB? : Option Rat
  /-- Optional copy-logit matrix (may be present for `kind induction-aligned`). -/
  copyLogits? : Option (Fin seq → Fin seq → Rat)
  /-- Verified certificate payload. -/
  cert : Circuit.InductionHeadCert seq
private def finalizeStateCore {seq : Nat} (hpos : 0 < seq) (st : ParseState seq) :
    Except String (Circuit.InductionHeadCert seq) := do
  let eps ←
    match st.eps with
    | some v => pure v
    | none => throw "missing eps entry"
  let margin ←
    match st.margin with
    | some v => pure v
    | none => throw "missing margin entry"
  let lo ←
    match st.lo with
    | some v => pure v
    | none => throw "missing lo entry"
  let hi ←
    match st.hi with
    | some v => pure v
    | none => throw "missing hi entry"
  if !st.prev.all Option.isSome then
    throw "missing prev entries"
  if !st.scores.all (fun row => row.all Option.isSome) then
    throw "missing score entries"
  if !st.weights.all (fun row => row.all Option.isSome) then
    throw "missing weight entries"
  if !st.epsAt.all Option.isSome then
    throw "missing eps-at entries"
  if !st.weightBoundAt.all (fun row => row.all Option.isSome) then
    throw "missing weight-bound entries"
  if !st.valsLo.all Option.isSome then
    throw "missing val-lo entries"
  if !st.valsHi.all Option.isSome then
    throw "missing val-hi entries"
  if !st.vals.all Option.isSome then
    throw "missing val entries"
  let defaultPrev : Fin seq := ⟨0, hpos⟩
  let prevFun : Fin seq → Fin seq := fun q =>
    (st.prev[q.1]!).getD defaultPrev
  let scoresFun : Fin seq → Fin seq → Rat := fun q k =>
    let row := st.scores[q.1]!
    (row[k.1]!).getD 0
  let weightsFun : Fin seq → Fin seq → Rat := fun q k =>
    let row := st.weights[q.1]!
    (row[k.1]!).getD 0
  let epsAtFun : Fin seq → Rat := fun q =>
    (st.epsAt[q.1]!).getD 0
  let weightBoundAtFun : Fin seq → Fin seq → Rat := fun q k =>
    let row := st.weightBoundAt[q.1]!
    (row[k.1]!).getD 0
  let valsLoFun : Fin seq → Rat := fun k =>
    (st.valsLo[k.1]!).getD 0
  let valsHiFun : Fin seq → Rat := fun k =>
    (st.valsHi[k.1]!).getD 0
  let valsFun : Fin seq → Rat := fun k =>
    (st.vals[k.1]!).getD 0
  let direction ←
    match st.directionTarget, st.directionNegative with
    | none, none => pure none
    | some target, some negative =>
        pure (some { target := target, negative := negative })
    | _, _ =>
        throw "direction metadata requires both direction-target and direction-negative"
  let values : Circuit.ValueIntervalCert seq :=
    { lo := lo
      hi := hi
      valsLo := valsLoFun
      valsHi := valsHiFun
      vals := valsFun
      direction := direction }
  let active :=
    if st.activeSeen then
      st.active
    else
      (Finset.univ : Finset (Fin seq)).erase defaultPrev
  pure
    { eps := eps
      epsAt := epsAtFun
      weightBoundAt := weightBoundAtFun
      margin := margin
      active := active
      prev := prevFun
      scores := scoresFun
      weights := weightsFun
      values := values }
private def finalizeCopyLogits? {seq : Nat} (st : ParseState seq) :
    Except String (Option (Fin seq → Fin seq → Rat)) := do
  if !st.copyLogitsSeen then
    return none
  if !st.copyLogits.all (fun row => row.all Option.isSome) then
    throw "missing copy-logit entries"
  let copyLogitsFun : Fin seq → Fin seq → Rat := fun q k =>
    let row := st.copyLogits[q.1]!
    (row[k.1]!).getD 0
  return some copyLogitsFun
private def finalizeModelSlice? {seq : Nat} (st : ParseState seq) :
    Except String (Option (ModelSlice seq)) := do
  let any :=
    st.modelDModel.isSome || st.modelHeadDim.isSome || st.modelLayer.isSome ||
      st.modelHead.isSome || st.modelScoreScale.isSome || st.modelScoreMask.isSome ||
      st.modelMaskCausal.isSome ||
      !st.modelResid.isEmpty || !st.modelWq.isEmpty || !st.modelWk.isEmpty ||
      !st.modelBq.isEmpty || !st.modelBk.isEmpty
  if !any then
    return none
  let dModel ←
    match st.modelDModel with
    | some v => pure v
    | none => throw "missing model-d-model entry"
  let headDim ←
    match st.modelHeadDim with
    | some v => pure v
    | none => throw "missing model-head-dim entry"
  let layer ←
    match st.modelLayer with
    | some v => pure v
    | none => throw "missing model-layer entry"
  let head ←
    match st.modelHead with
    | some v => pure v
    | none => throw "missing model-head entry"
  let scoreScale ←
    match st.modelScoreScale with
    | some v => pure v
    | none => throw "missing model-score-scale entry"
  let scoreMask ←
    match st.modelScoreMask with
    | some v => pure v
    | none => throw "missing model-score-mask entry"
  let maskCausal := st.modelMaskCausal.getD true
  if dModel = 0 then
    throw "model-d-model must be positive"
  if headDim = 0 then
    throw "model-head-dim must be positive"
  if st.modelResid.size ≠ seq then
    throw "model-resid has unexpected row count"
  if !st.modelResid.all (fun row => row.size = dModel) then
    throw "model-resid has unexpected column count"
  if !st.modelResid.all (fun row => row.all Option.isSome) then
    throw "missing model-resid entries"
  if st.modelWq.size ≠ dModel || st.modelWk.size ≠ dModel then
    throw "model-wq/wk has unexpected row count"
  if !st.modelWq.all (fun row => row.size = headDim) then
    throw "model-wq has unexpected column count"
  if !st.modelWk.all (fun row => row.size = headDim) then
    throw "model-wk has unexpected column count"
  if !st.modelWq.all (fun row => row.all Option.isSome) then
    throw "missing model-wq entries"
  if !st.modelWk.all (fun row => row.all Option.isSome) then
    throw "missing model-wk entries"
  if st.modelBq.size ≠ headDim || st.modelBk.size ≠ headDim then
    throw "model-bq/bk has unexpected length"
  if !st.modelBq.all Option.isSome then
    throw "missing model-bq entries"
  if !st.modelBk.all Option.isSome then
    throw "missing model-bk entries"
  let residFun : Fin seq → Fin dModel → Rat := fun q i =>
    let row := st.modelResid[q.1]!
    (row[i.1]!).getD 0
  let wqFun : Fin dModel → Fin headDim → Rat := fun i j =>
    let row := st.modelWq[i.1]!
    (row[j.1]!).getD 0
  let wkFun : Fin dModel → Fin headDim → Rat := fun i j =>
    let row := st.modelWk[i.1]!
    (row[j.1]!).getD 0
  let bqFun : Fin headDim → Rat := fun j =>
    (st.modelBq[j.1]!).getD 0
  let bkFun : Fin headDim → Rat := fun j =>
    (st.modelBk[j.1]!).getD 0
  return some
    { dModel := dModel
      headDim := headDim
      layer := layer
      head := head
      scoreScale := scoreScale
      scoreMask := scoreMask
      maskCausal := maskCausal
      resid := residFun
      wq := wqFun
      wk := wkFun
      bq := bqFun
      bk := bkFun }
private def finalizeModelLnSlice? {seq : Nat} (st : ParseState seq) :
    Except String (Option (ModelLnSlice seq)) := do
  let any :=
    st.modelLnEps.isSome || st.modelLnSlack.isSome || st.modelLnScale.isSome ||
      st.modelLnFast.isSome || st.modelDecimals.isSome ||
      !st.modelEmbed.isEmpty ||
      !st.modelLnGamma.isEmpty || !st.modelLnBeta.isEmpty
  if !any then
    return none
  let dModel ←
    match st.modelDModel with
    | some v => pure v
    | none => throw "missing model-d-model entry"
  let lnEps ←
    match st.modelLnEps with
    | some v => pure v
    | none => throw "missing model-ln-eps entry"
  let lnSlack := st.modelLnSlack.getD 0
  if lnSlack < 0 then
    throw "model-ln-slack must be nonnegative"
  let lnScale? := st.modelLnScale
  if let some scale := lnScale? then
    if scale = 0 then
      throw "model-ln-scale must be positive"
  let lnFast? := st.modelLnFast
  let lnDecimals? := st.modelDecimals
  if dModel = 0 then
    throw "model-d-model must be positive"
  if st.modelEmbed.size ≠ seq then
    throw "model-embed has unexpected row count"
  if !st.modelEmbed.all (fun row => row.size = dModel) then
    throw "model-embed has unexpected column count"
  if !st.modelEmbed.all (fun row => row.all Option.isSome) then
    throw "missing model-embed entries"
  if st.modelLnGamma.size ≠ dModel || st.modelLnBeta.size ≠ dModel then
    throw "model-ln-gamma/beta has unexpected length"
  if !st.modelLnGamma.all Option.isSome then
    throw "missing model-ln-gamma entries"
  if !st.modelLnBeta.all Option.isSome then
    throw "missing model-ln-beta entries"
  let embedFun : Fin seq → Fin dModel → Rat := fun q i =>
    let row := st.modelEmbed[q.1]!
    (row[i.1]!).getD 0
  let gammaFun : Fin dModel → Rat := fun i =>
    (st.modelLnGamma[i.1]!).getD 0
  let betaFun : Fin dModel → Rat := fun i =>
    (st.modelLnBeta[i.1]!).getD 0
  return some
    { dModel := dModel
      lnEps := lnEps
      lnSlack := lnSlack
      lnScale? := lnScale?
      lnFast? := lnFast?
      lnDecimals? := lnDecimals?
      lnGamma := gammaFun
      lnBeta := betaFun
      embed := embedFun }
/-- Check that residual entries fall within LayerNorm bounds from pre-LN inputs. -/
def residWithinLayerNormBounds {seq : Nat} (slice : ModelLnSlice seq)
    (resid : Fin seq → Fin slice.dModel → Rat) : Bool :=
  let pow10 (d : Nat) : Nat := Nat.pow 10 d
  let ratScaledNum? (den : Nat) (x : Rat) : Option Int :=
    if x.den ∣ den then
      let mul := den / x.den
      some (x.num * (Int.ofNat mul))
    else
      none
  let ratMulNatFloor (x : Rat) (m : Nat) : Int :=
    let num := x.num
    let den := (x.den : Int)
    let prod := num * (Int.ofNat m)
    if prod ≥ 0 then
      prod / den
    else
      (prod - (den - 1)) / den
  let ratMulNatCeil (x : Rat) (m : Nat) : Int :=
    let num := x.num
    let den := (x.den : Int)
    let prod := num * (Int.ofNat m)
    if prod ≥ 0 then
      (prod + (den - 1)) / den
    else
      prod / den
  let fracLe (aNum aDen bNum bDen : Nat) : Bool :=
    aNum * bDen ≤ bNum * aDen
  let maxFrac (aNum aDen bNum bDen : Nat) : Nat × Nat :=
    if fracLe aNum aDen bNum bDen then (bNum, bDen) else (aNum, aDen)
  let minFrac (aNum aDen bNum bDen : Nat) : Nat × Nat :=
    if fracLe aNum aDen bNum bDen then (aNum, aDen) else (bNum, bDen)
  let sqrtLowerNumDen (numAbs den scale : Nat) : Nat × Nat :=
    let aBase := Nat.sqrt numAbs
    let bBase := Nat.sqrt den + 1
    let aAlt := Nat.sqrt (numAbs * den)
    let bAlt := den
    let aScaled := Nat.sqrt (numAbs * den * scale * scale)
    let bScaled := den * scale
    let m1 := maxFrac aBase bBase aAlt bAlt
    maxFrac m1.1 m1.2 aScaled bScaled
  let sqrtUpperNumDen (numAbs den scale : Nat) : Nat × Nat :=
    let aBase := Nat.sqrt numAbs + 1
    let bBase := Nat.sqrt den
    let aAlt := Nat.sqrt (numAbs * den) + 1
    let bAlt := den
    let aScaled := Nat.sqrt (numAbs * den * scale * scale) + 1
    let bScaled := den * scale
    let m1 := minFrac aBase bBase aAlt bAlt
    minFrac m1.1 m1.2 aScaled bScaled
  let residWithinFixed : Option Bool :=
    match slice.lnFast? with
    | some true => do
        let d ← slice.lnDecimals?
        let den := pow10 d
        if den = 0 then
          none
        else
          let gamma ←
            (List.finRange slice.dModel).foldlM (m := Option)
              (fun arr i => do
                let v ← ratScaledNum? den (slice.lnGamma i)
                return arr.push v)
              (Array.mkEmpty slice.dModel)
          let beta ←
            (List.finRange slice.dModel).foldlM (m := Option)
              (fun arr i => do
                let v ← ratScaledNum? den (slice.lnBeta i)
                return arr.push v)
              (Array.mkEmpty slice.dModel)
          let slackNum ← ratScaledNum? den slice.lnSlack
          let n := slice.dModel
          let nInt := Int.ofNat n
          let denNat := den
          let coeffDen : Nat := denNat * denNat * n
          let varDen : Nat := denNat * denNat * n * n * n
          let scale := slice.lnScale?.getD Bounds.sqrtLowerScale
          let mut ok := true
          for q in List.finRange seq do
            if ok then
              let rowEmbed? :=
                (List.finRange n).foldlM (m := Option)
                  (fun arr i => do
                    let v ← ratScaledNum? den (slice.embed q i)
                    return arr.push v)
                  (Array.mkEmpty n)
              let rowResid? :=
                (List.finRange n).foldlM (m := Option)
                  (fun arr i => do
                    let v ← ratScaledNum? den (resid q i)
                    return arr.push v)
                  (Array.mkEmpty n)
              match rowEmbed?, rowResid? with
              | some rowEmbed, some rowResid =>
                  let sum := rowEmbed.foldl (fun acc v => acc + v) 0
                  let varNum :=
                    rowEmbed.foldl (fun acc v =>
                      let centered := v * nInt - sum
                      acc + centered * centered) 0
                  let epsLo := ratMulNatFloor slice.lnEps varDen
                  let epsHi := ratMulNatCeil slice.lnEps varDen
                  let varLo := varNum + epsLo
                  let varHi := varNum + epsHi
                  let varLoAbs := Int.natAbs varLo
                  let varHiAbs := Int.natAbs varHi
                  let epsLoAbs := Int.natAbs epsLo
                  let sqrtLowerEps := sqrtLowerNumDen epsLoAbs varDen scale
                  let sqrtLowerVar := sqrtLowerNumDen varLoAbs varDen scale
                  let sqrtLowerBound :=
                    maxFrac sqrtLowerEps.1 sqrtLowerEps.2
                      sqrtLowerVar.1 sqrtLowerVar.2
                  let sqrtUpperBound := sqrtUpperNumDen varHiAbs varDen scale
                  let invUpperNum : Int :=
                    if sqrtLowerBound.1 = 0 then 0 else Int.ofNat sqrtLowerBound.2
                  let invUpperDen : Nat :=
                    if sqrtLowerBound.1 = 0 then 1 else sqrtLowerBound.1
                  let invLowerNum : Int :=
                    if sqrtUpperBound.1 = 0 then 0 else Int.ofNat sqrtUpperBound.2
                  let invLowerDen : Nat :=
                    if sqrtUpperBound.1 = 0 then 1 else sqrtUpperBound.1
                  for i in List.finRange n do
                    if ok then
                      let v := rowEmbed[i.1]!
                      let centered := v * nInt - sum
                      let coeffNum := (gamma[i.1]!) * centered
                      let residNum := rowResid[i.1]!
                      let betaNum := beta[i.1]!
                      let loInvNum := if coeffNum ≥ 0 then invLowerNum else invUpperNum
                      let loInvDen := if coeffNum ≥ 0 then invLowerDen else invUpperDen
                      let hiInvNum := if coeffNum ≥ 0 then invUpperNum else invLowerNum
                      let hiInvDen := if coeffNum ≥ 0 then invUpperDen else invLowerDen
                      let lhsLo :=
                        (residNum - betaNum + slackNum) *
                          (Int.ofNat coeffDen) * (Int.ofNat loInvDen)
                      let rhsLo :=
                        coeffNum * loInvNum * (Int.ofNat denNat)
                      let lhsHi :=
                        (residNum - betaNum - slackNum) *
                          (Int.ofNat coeffDen) * (Int.ofNat hiInvDen)
                      let rhsHi :=
                        coeffNum * hiInvNum * (Int.ofNat denNat)
                      if lhsLo < rhsLo || lhsHi > rhsHi then
                        ok := false
              | _, _ => ok := false
          return ok
    | _ => none
  match residWithinFixed with
  | some ok => ok
  | none =>
      (List.finRange seq).all (fun q =>
        let bounds :=
          match slice.lnScale? with
          | some scale =>
              Bounds.layerNormBoundsWithScale scale
                slice.lnEps slice.lnGamma slice.lnBeta (slice.embed q)
          | none =>
              Bounds.layerNormBounds slice.lnEps slice.lnGamma slice.lnBeta (slice.embed q)
        (List.finRange slice.dModel).all (fun i =>
          let lo := bounds.1 i - slice.lnSlack
          let hi := bounds.2 i + slice.lnSlack
          decide (lo ≤ resid q i) && decide (resid q i ≤ hi)))
/-- Check whether scores match the model slice computation. -/
def scoresMatchModelSlice {seq : Nat} (slice : ModelSlice seq)
    (scores : Fin seq → Fin seq → Rat) : Bool :=
  let scoresRef :=
    Sound.Induction.scoresRatOfSlice (seq := seq)
      (dModel := slice.dModel) (dHead := slice.headDim)
      slice.scoreScale slice.scoreMask slice.maskCausal
      slice.resid slice.wq slice.bq slice.wk slice.bk
  (List.finRange seq).all (fun q =>
    (List.finRange seq).all (fun k =>
      decide (scores q k = scoresRef q k)))
end InductionHeadCert
/-- Parse an explicit induction-head certificate from a text payload. -/
def parseInductionHeadCert (input : String) :
    Except String (Sigma InductionHeadCert.InductionHeadCertPayload) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let seq ← InductionHeadCert.parseSeq tokens
  match seq with
  | 0 => throw "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let hpos : 0 < seq := Nat.succ_pos n
      let st0 : InductionHeadCert.ParseState seq := InductionHeadCert.initState seq
      let st ← tokens.foldlM (fun st t =>
          match t with
          | ["seq", _] => pure st
          | _ => InductionHeadCert.parseLine st t) st0
      let kind ←
        match st.kind with
        | some v => pure v
        | none => throw "missing kind entry"
      if kind != "onehot-approx" && kind != "induction-aligned" then
        throw s!"unexpected kind: {kind}"
      let period? := st.period
      if kind = "onehot-approx" && period?.isSome then
        throw "unexpected period entry for kind onehot-approx"
      if kind = "induction-aligned" && period?.isNone then
        throw "missing period entry for kind induction-aligned"
      let tlExcludeBos? := st.tlExcludeBos
      let tlExcludeCurrent? := st.tlExcludeCurrent
      let tlScoreLB? := st.tlScoreLB
      if tlScoreLB?.isSome then
        match tlExcludeBos?, tlExcludeCurrent? with
        | some _, some _ => pure ()
        | _, _ =>
            throw "tl-score-lb requires tl-exclude-bos and tl-exclude-current-token"
      else
        if tlExcludeBos?.isSome || tlExcludeCurrent?.isSome then
          throw "tl-exclude-bos/current-token requires tl-score-lb"
      let cert ← InductionHeadCert.finalizeStateCore hpos st
      let copyLogits? ← InductionHeadCert.finalizeCopyLogits? st
      let modelSlice? ← InductionHeadCert.finalizeModelSlice? st
      let modelLnSlice? ← InductionHeadCert.finalizeModelLnSlice? st
      let payload : InductionHeadCert.InductionHeadCertPayload seq :=
        { kind := kind
          period? := period?
          modelSlice? := modelSlice?
          modelLnSlice? := modelLnSlice?
          tlExcludeBos? := tlExcludeBos?
          tlExcludeCurrent? := tlExcludeCurrent?
          tlScoreLB? := tlScoreLB?
          copyLogits? := copyLogits?
          cert := cert }
      return ⟨seq, payload⟩
/-- Load an induction-head certificate from disk. -/
def loadInductionHeadCert (path : System.FilePath) :
    IO (Except String (Sigma InductionHeadCert.InductionHeadCertPayload)) := do
  try
    let data ← IO.FS.readFile path
    return parseInductionHeadCert data
  catch e =>
    return Except.error s!"failed to read cert file: {e.toString}"
/-- Parse a token list payload. -/
def parseInductionHeadTokens (input : String) :
    Except String (Sigma fun seq => Fin seq → Nat) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let seq ← InductionHeadTokens.parseSeq tokens
  match seq with
  | 0 => throw "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let st0 : InductionHeadTokens.ParseState seq := InductionHeadTokens.initState seq
      let st ← tokens.foldlM (fun st t =>
          match t with
          | ["seq", _] => pure st
          | _ => InductionHeadTokens.parseLine st t) st0
      let tokensFun ← InductionHeadTokens.finalizeState st
      return ⟨seq, tokensFun⟩
/-- Load a token list from disk. -/
def loadInductionHeadTokens (path : System.FilePath) :
    IO (Except String (Sigma fun seq => Fin seq → Nat)) := do
  try
    let data ← IO.FS.readFile path
    return parseInductionHeadTokens data
  catch e =>
    return Except.error s!"failed to read tokens file: {e.toString}"
private def ratToString (x : Rat) : String :=
  toString x
private def tokensPeriodic {seq : Nat} (period : Nat) (tokens : Fin seq → Nat) : Bool :=
  (List.finRange seq).all (fun q =>
    if period ≤ q.val then
      decide (tokens q = tokens (Model.prevOfPeriod (seq := seq) period q))
    else
      true)
private def sumOver {seq : Nat} (f : Fin seq → Fin seq → Rat) : Rat :=
  (List.finRange seq).foldl (fun acc q =>
    acc + (List.finRange seq).foldl (fun acc' k => acc' + f q k) 0) 0
/-- Boolean TL induction pattern indicator (duplicate head shifted right). -/
private def tlPatternBool {seq : Nat} (tokens : Fin seq → Nat) (q k : Fin seq) : Bool :=
  (List.finRange seq).any (fun r =>
    decide (r < q) && decide (tokens r = tokens q) && decide (k.val = r.val + 1))
/-- Numeric TL induction pattern indicator (0/1). -/
private def tlPatternIndicator {seq : Nat} (tokens : Fin seq → Nat) (q k : Fin seq) : Rat :=
  if tlPatternBool (seq := seq) tokens q k then 1 else 0
/-- TL mul numerator with TL-style masking. -/
private def tlMulNumeratorMasked {seq : Nat} (excludeBos excludeCurrent : Bool)
    (weights : Fin seq → Fin seq → Rat) (tokens : Fin seq → Nat) : Rat :=
  sumOver (seq := seq) (fun q k =>
    Model.tlMaskedWeights (seq := seq) excludeBos excludeCurrent weights q k *
      tlPatternIndicator (seq := seq) tokens q k)
/-- TL mul score with TL-style masking. -/
private def tlMulScoreMasked {seq : Nat} (excludeBos excludeCurrent : Bool)
    (weights : Fin seq → Fin seq → Rat) (tokens : Fin seq → Nat) : Rat :=
  let masked := Model.tlMaskedWeights (seq := seq) excludeBos excludeCurrent weights
  let denom := sumOver (seq := seq) masked
  if denom = 0 then
    0
  else
    tlMulNumeratorMasked (seq := seq) excludeBos excludeCurrent weights tokens / denom
/-- Copying score utility (ratio scaled to [-1, 1]); not used by the checker. -/
def copyScore {seq : Nat} (weights copyLogits : Fin seq → Fin seq → Rat) : Option Rat :=
  let total := sumOver (seq := seq) copyLogits
  if total = 0 then
    none
  else
    let weighted := sumOver (seq := seq) (fun q k => weights q k * copyLogits q k)
    let ratio := weighted / total
    some ((4 : Rat) * ratio - 1)
/-- Check an explicit induction-head certificate from disk. -/
def runInductionHeadCertCheck (certPath : System.FilePath)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String)
    (tokensPath? : Option String)
    (minStripeMeanStr? : Option String) (minStripeTop1Str? : Option String)
    (timeLn : Bool) (timeScores : Bool) (timeParse : Bool) (timeStages : Bool) :
    IO UInt32 := do
  let minLogitDiff?E := parseRatOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseRatOpt "min-margin" minMarginStr?
  let maxEps?E := parseRatOpt "max-eps" maxEpsStr?
  let minStripeMean?E := parseRatOpt "min-stripe-mean" minStripeMeanStr?
  let minStripeTop1?E := parseRatOpt "min-stripe-top1" minStripeTop1Str?
  match minLogitDiff?E, minMargin?E, maxEps?E, minStripeMean?E, minStripeTop1?E with
  | Except.error msg, _, _, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg, _, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps?,
      Except.ok minStripeMean?, Except.ok minStripeTop1? =>
      let minMargin := minMargin?.getD (0 : Rat)
      let maxEps := maxEps?.getD (ratRoundDown (Rat.divInt 1 2))
      let tParse? ←
        if timeParse then
          some <$> IO.monoMsNow
        else
          pure none
      let parsed ← loadInductionHeadCert certPath
      if let some t0 := tParse? then
        let t1 ← IO.monoMsNow
        IO.eprintln s!"info: parse-ms {t1 - t0}"
      match parsed with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨seq, payload⟩ =>
          match seq with
          | 0 =>
              IO.eprintln "error: seq must be positive"
              return 2
          | Nat.succ n =>
              let seq := Nat.succ n
              let _ : NeZero seq := ⟨by simp⟩
              let cert := payload.cert
              let kind := payload.kind
              let period? := payload.period?
              let copyLogits? := payload.copyLogits?
              let tlExcludeBos? := payload.tlExcludeBos?
              let tlExcludeCurrent? := payload.tlExcludeCurrent?
              let tlScoreLB? := payload.tlScoreLB?
              let modelSlice? := payload.modelSlice?
              let modelLnSlice? := payload.modelLnSlice?
              if kind != "onehot-approx" && kind != "induction-aligned" then
                IO.eprintln s!"error: unexpected kind {kind}"
                return 2
              if kind = "onehot-approx" then
                if minStripeMeanStr?.isSome || minStripeTop1Str?.isSome then
                  IO.eprintln
                    "error: stripe thresholds are not used for onehot-approx"
                  return 2
              if kind = "induction-aligned" then
                let period ←
                  match period? with
                  | some v => pure v
                  | none =>
                      IO.eprintln "error: missing period entry for induction-aligned"
                      return 2
                if period = 0 then
                  IO.eprintln "error: period must be positive for induction-aligned"
                  return 2
                if seq ≤ period then
                  IO.eprintln "error: period must be less than seq for induction-aligned"
                  return 2
                let expectedActive := Model.activeOfPeriod (seq := seq) period
                if !decide (cert.active = expectedActive) then
                  IO.eprintln "error: active set does not match induction-aligned period"
                  return 2
                let prevOk :=
                  (List.finRange seq).all (fun q =>
                    decide (cert.prev q = Model.prevOfPeriod (seq := seq) period q))
                if !prevOk then
                  IO.eprintln "error: prev map does not match induction-aligned period"
                  return 2
                if minMarginStr?.isSome || maxEpsStr?.isSome then
                  IO.eprintln
                    "error: min-margin/max-eps are not used for induction-aligned"
                  return 2
                if minLogitDiffStr?.isSome && cert.values.direction.isNone then
                  IO.eprintln
                    "error: min-logit-diff requires direction metadata"
                  return 2
                if minStripeTop1Str?.isSome then
                  IO.eprintln "error: stripe-top1 is not used for induction-aligned"
                  return 2
              let activeCount := cert.active.card
              let defaultMinActive := max 1 (seq / 8)
              let minActive := minActive?.getD defaultMinActive
              if activeCount < minActive then
                IO.eprintln
                  s!"error: active queries {activeCount} below minimum {minActive}"
                return 2
              if let some modelLnSlice := modelLnSlice? then
                match modelSlice? with
                | none =>
                    IO.eprintln "error: model-ln data requires model-resid slice"
                    return 2
                | some modelSlice =>
                    if h : modelLnSlice.dModel = modelSlice.dModel then
                      if modelLnSlice.lnEps ≤ 0 then
                        IO.eprintln "error: model-ln-eps must be positive"
                        return 2
                      let resid' : Fin seq → Fin modelLnSlice.dModel → Rat := by
                        simpa [h] using modelSlice.resid
                      let t0? ←
                        if timeLn then
                          some <$> IO.monoMsNow
                        else
                          pure none
                      let okResid :=
                        InductionHeadCert.residWithinLayerNormBounds
                          (seq := seq) modelLnSlice resid'
                      if !okResid then
                        if let some t0 := t0? then
                          let t1 ← IO.monoMsNow
                          IO.eprintln s!"info: ln-check-ms {t1 - t0}"
                        IO.eprintln "error: residuals not within LayerNorm bounds"
                        return 2
                      if let some t0 := t0? then
                        let t1 ← IO.monoMsNow
                        IO.eprintln s!"info: ln-check-ms {t1 - t0}"
                    else
                      IO.eprintln "error: model-ln dModel does not match model slice"
                      return 2
              if let some modelSlice := modelSlice? then
                let t0? ←
                  if timeScores then
                    some <$> IO.monoMsNow
                  else
                    pure none
                let okScores :=
                  InductionHeadCert.scoresMatchModelSlice (seq := seq) modelSlice cert.scores
                if let some t0 := t0? then
                  let t1 ← IO.monoMsNow
                  IO.eprintln s!"info: scores-check-ms {t1 - t0}"
                if !okScores then
                  IO.eprintln "error: scores do not match model slice"
                  return 2
              if kind = "onehot-approx" then
                let ok ← timeIO timeStages "cert-check"
                  (fun _ => pure (Circuit.checkInductionHeadCert cert))
                if !ok then
                  IO.eprintln "error: induction-head certificate rejected"
                  return 2
                if cert.margin < minMargin then
                  IO.eprintln
                    s!"error: margin {ratToString cert.margin} \
                    below minimum {ratToString minMargin}"
                  return 2
                if maxEps < cert.eps then
                  IO.eprintln
                    s!"error: eps {ratToString cert.eps} \
                    above maximum {ratToString maxEps}"
                  return 2
              let tokensOpt ←
                match tokensPath? with
                | none =>
                    if tlScoreLB?.isSome then
                      IO.eprintln "error: tl-score-lb requires a tokens file"
                      return 2
                    else
                      pure none
                | some tokensPath =>
                    let tokensParsed ← timeIO timeStages "tokens-parse"
                      (fun _ => loadInductionHeadTokens tokensPath)
                    match tokensParsed with
                    | Except.error msg =>
                        IO.eprintln s!"error: {msg}"
                        return 2
                    | Except.ok ⟨seqTokens, tokens⟩ =>
                        if hseq : seqTokens = seq then
                          let tokens' : Fin seq → Nat := by
                            simpa [hseq] using tokens
                          pure (some tokens')
                        else
                          IO.eprintln
                            s!"error: tokens seq {seqTokens} does not match cert seq {seq}"
                          return 2
              if let some tokens' := tokensOpt then
                if kind = "induction-aligned" then
                  let period ←
                    match period? with
                    | some v => pure v
                    | none =>
                        IO.eprintln "error: missing period entry for induction-aligned"
                        return 2
                  if !tokensPeriodic (seq := seq) period tokens' then
                    IO.eprintln "error: tokens are not periodic for induction-aligned period"
                    return 2
                else
                  let activeTokens := Model.activeOfTokens (seq := seq) tokens'
                  if !decide (cert.active ⊆ activeTokens) then
                    IO.eprintln "error: active set not contained in token repeats"
                    return 2
                  let prevTokens := Model.prevOfTokens (seq := seq) tokens'
                  let prevOk :=
                    (List.finRange seq).all (fun q =>
                      if decide (q ∈ cert.active) then
                        decide (prevTokens q = cert.prev q)
                      else
                        true)
                  if !prevOk then
                    IO.eprintln "error: prev map does not match tokens on active queries"
                    return 2
              if let some tlScoreLB := tlScoreLB? then
                match tokensOpt with
                | none =>
                    IO.eprintln "error: tl-score-lb requires a tokens file"
                    return 2
                | some tokens' =>
                    let excludeBos := tlExcludeBos?.getD false
                    let excludeCurrent := tlExcludeCurrent?.getD false
                    let tlScore ← timeIO timeStages "tl-score" (fun _ =>
                      pure (tlMulScoreMasked (seq := seq) excludeBos excludeCurrent
                        cert.weights tokens'))
                    if tlScore < tlScoreLB then
                      IO.eprintln
                        s!"error: tl-score {ratToString tlScore} \
                        below minimum {ratToString tlScoreLB}"
                      return 2
              let stripeMeanOpt ←
                if kind = "induction-aligned" then
                  let period ←
                    match period? with
                    | some v => pure v
                    | none =>
                        IO.eprintln "error: missing period entry for induction-aligned"
                        return 2
                  let stripeCert : Stripe.StripeCert seq :=
                    { period := period
                      stripeMeanLB := 0
                      stripeTop1LB := 0
                      weights := cert.weights }
                  let stripeMean? ← timeIO timeStages "stripe-mean"
                    (fun _ => pure (Stripe.stripeMean (seq := seq) stripeCert))
                  let mean ←
                    match stripeMean? with
                    | some mean =>
                        let minMean := minStripeMean?.getD (0 : Rat)
                        if mean < minMean then
                          IO.eprintln
                            s!"error: stripe-mean {ratToString mean} below minimum \
                            {ratToString minMean}"
                          return 2
                        pure mean
                    | none =>
                        IO.eprintln "error: empty active set for stripe stats"
                        return 2
                  pure (some mean)
                else
                  pure none
              let effectiveMinLogitDiff :=
                match minLogitDiff?, cert.values.direction with
                | some v, _ => some v
                | none, some _ => some (0 : Rat)
                | none, none => none
              match effectiveMinLogitDiff with
              | none =>
                  if kind = "induction-aligned" then
                    match stripeMeanOpt with
                    | some mean =>
                        IO.println
                          s!"ok: induction-aligned certificate checked \
                          (seq={seq}, active={activeCount}, stripeMean={ratToString mean})"
                        return 0
                    | none =>
                        IO.eprintln "error: missing stripe mean for induction-aligned"
                        return 2
                  else
                    let kindLabel := "onehot-approx (proxy)"
                    IO.println
                      s!"ok: {kindLabel} certificate checked \
                      (seq={seq}, active={activeCount}, \
                      margin={ratToString cert.margin}, eps={ratToString cert.eps})"
                    return 0
              | some minLogitDiff =>
                  let logitDiffLB? ← timeIO timeStages "logit-diff" (fun _ =>
                    pure (Circuit.logitDiffLowerBoundAt cert.active cert.prev cert.epsAt
                      cert.values.lo cert.values.hi cert.values.vals))
                  match logitDiffLB? with
                  | none =>
                      IO.eprintln "error: empty active set for logit-diff bound"
                      return 2
                  | some logitDiffLB =>
                      if logitDiffLB < minLogitDiff then
                        IO.eprintln
                          s!"error: logitDiffLB {ratToString logitDiffLB} \
                          below minimum {ratToString minLogitDiff}"
                        return 2
                      else
                        if kind = "induction-aligned" then
                          match stripeMeanOpt with
                          | some mean =>
                              IO.println
                                s!"ok: induction-aligned certificate checked \
                                (seq={seq}, active={activeCount}, \
                                stripeMean={ratToString mean}, \
                                logitDiffLB={ratToString logitDiffLB})"
                              return 0
                          | none =>
                              IO.eprintln "error: missing stripe mean for induction-aligned"
                              return 2
                        else
                          let kindLabel := "onehot-approx (proxy)"
                          IO.println
                            s!"ok: {kindLabel} certificate checked \
                            (seq={seq}, active={activeCount}, \
                            margin={ratToString cert.margin}, eps={ratToString cert.eps}, \
                            logitDiffLB={ratToString logitDiffLB})"
                          return 0
end IO
end Nfp
