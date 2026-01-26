-- SPDX-License-Identifier: AGPL-3.0-or-later
module
public import Mathlib.Data.Finset.Insert
public import Nfp.Circuit.Cert.InductionHead
public import Nfp.Circuit.Cert.LogitDiff
public import Nfp.IO.InductionHead.Tokens
public import Nfp.IO.InductionHead.ModelDirectionSlice
public import Nfp.IO.InductionHead.LnCheck
public import Nfp.IO.InductionHead.ScoreCheck
public import Nfp.IO.InductionHead.ScoreUtils
public import Nfp.IO.InductionHead.ValueCheck
public import Nfp.IO.InductionHead.ModelLnSlice
public import Nfp.IO.InductionHead.LogitDiffBound
public import Nfp.IO.InductionHead.ActiveSelection
public import Nfp.IO.InductionHead.WeightedResidual
public import Nfp.IO.InductionHead.TokenChecks
public import Nfp.IO.InductionHead.ModelValueSlice
public import Nfp.IO.Pure.InductionHead.ModelDirectionSlice
public import Nfp.IO.Pure.InductionHead.ModelSlice
public import Nfp.IO.Pure.InductionHead.ModelLnSlice
public import Nfp.IO.Pure.InductionHead.ModelValueSlice
public import Nfp.Sound.Induction.ScoreBounds
public import Nfp.IO.InductionHead.ModelSlice
public import Nfp.IO.Parse.Basic
public import Nfp.IO.Stripe
public import Nfp.IO.Util
public import Nfp.Model.InductionPrompt
public section
namespace Nfp.IO
open Nfp.Circuit Nfp.IO.Parse
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
  /-- Optional model Wv slice (d_model × head_dim). -/
  modelWv : Array (Array (Option Rat))
  /-- Optional model Wo slice (d_model × head_dim). -/
  modelWo : Array (Array (Option Rat))
  /-- Optional model bv slice (head_dim). -/
  modelBv : Array (Option Rat)
  /-- Optional attention output bias (d_model). -/
  modelAttnBias : Array (Option Rat)
  /-- Optional unembedding row for the target token (d_model). -/
  modelUnembedTarget : Array (Option Rat)
  /-- Optional unembedding row for the negative token (d_model). -/
  modelUnembedNegative : Array (Option Rat)
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
    modelBk := #[]
    modelWv := #[]
    modelWo := #[]
    modelBv := #[]
    modelAttnBias := #[]
    modelUnembedTarget := #[]
    modelUnembedNegative := #[] }
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
  let qFin ← toIndex0 (seq := seq) "q" q; let kFin ← toIndex0 (seq := seq) "k" k
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
  let qFin ← toIndex0 (seq := seq) "q" q; let kFin ← toIndex0 (seq := seq) "k" k
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
private def allocModelAttnBias (dModel : Nat) : Array (Option Rat) :=
  Array.replicate dModel none
private def allocModelUnembed (dModel : Nat) : Array (Option Rat) :=
  Array.replicate dModel none
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
private def ensureModelV {seq : Nat} (st : ParseState seq) :
    Except String (ParseState seq × Nat × Nat) := do
  let dModel ←
    match st.modelDModel with
    | some v => pure v
    | none => throw "model-d-model required before model-wv/wo/bv/attn-bias"
  let headDim ←
    match st.modelHeadDim with
    | some v => pure v
    | none => throw "model-head-dim required before model-wv/wo/bv/attn-bias"
  if dModel = 0 then
    throw "model-d-model must be positive"
  if headDim = 0 then
    throw "model-head-dim must be positive"
  let st :=
    if st.modelWv.isEmpty then
      { st with modelWv := allocModelW dModel headDim, modelWo := allocModelW dModel headDim }
    else
      st
  let st :=
    if st.modelBv.isEmpty then
      { st with modelBv := allocModelB headDim }
    else
      st
  let st :=
    if st.modelAttnBias.isEmpty then
      { st with modelAttnBias := allocModelAttnBias dModel }
    else
      st
  return (st, dModel, headDim)
private def ensureModelUnembed {seq : Nat} (st : ParseState seq) :
    Except String (ParseState seq × Nat) := do
  let dModel ←
    match st.modelDModel with
    | some v => pure v
    | none => throw "model-d-model required before model-unembed"
  if dModel = 0 then
    throw "model-d-model must be positive"
  let st :=
    if st.modelUnembedTarget.isEmpty then
      { st with
        modelUnembedTarget := allocModelUnembed dModel
        modelUnembedNegative := allocModelUnembed dModel }
    else
      st
  return (st, dModel)
private def setModelResidEntry {seq : Nat} (st : ParseState seq) (q i : Nat) (v : Rat) :
    Except String (ParseState seq) := do
  let (st, dModel) ← ensureModelResid st; let qFin ← toIndex0 (seq := seq) "q" q
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
  let (st, dModel) ← ensureModelEmbed st; let qFin ← toIndex0 (seq := seq) "q" q
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
private def setModelVEntry {seq : Nat} (st : ParseState seq)
    (i j : Nat) (v : Rat) (which : String) :
    Except String (ParseState seq) := do
  let (st, dModel, headDim) ← ensureModelV st
  let iFin ← toIndex0Sized "i" dModel i
  let jFin ← toIndex0Sized "j" headDim j
  let mat := if which = "wv" then st.modelWv else st.modelWo
  let row := mat[iFin.1]!
  match row[jFin.1]! with
  | some _ =>
      throw s!"duplicate model-{which} entry at ({i}, {j})"
  | none =>
      let row' := row.set! jFin.1 (some v)
      let mat' := mat.set! iFin.1 row'
      if which = "wv" then
        return { st with modelWv := mat' }
      else
        return { st with modelWo := mat' }
private def setModelBvEntry {seq : Nat} (st : ParseState seq) (j : Nat) (v : Rat) :
    Except String (ParseState seq) := do
  let (st, _dModel, headDim) ← ensureModelV st
  let jFin ← toIndex0Sized "j" headDim j
  match st.modelBv[jFin.1]! with
  | some _ =>
      throw s!"duplicate model-bv entry for j={j}"
  | none =>
      return { st with modelBv := st.modelBv.set! jFin.1 (some v) }
private def setModelAttnBiasEntry {seq : Nat} (st : ParseState seq) (i : Nat) (v : Rat) :
    Except String (ParseState seq) := do
  let (st, dModel, _headDim) ← ensureModelV st
  let iFin ← toIndex0Sized "i" dModel i
  match st.modelAttnBias[iFin.1]! with
  | some _ =>
      throw s!"duplicate model-attn-bias entry for i={i}"
  | none =>
      return { st with modelAttnBias := st.modelAttnBias.set! iFin.1 (some v) }
private def setModelUnembedEntry {seq : Nat} (st : ParseState seq) (i : Nat) (v : Rat)
    (which : String) : Except String (ParseState seq) := do
  let (st, dModel) ← ensureModelUnembed st
  let iFin ← toIndex0Sized "i" dModel i
  let arr := if which = "target" then st.modelUnembedTarget else st.modelUnembedNegative
  match arr[iFin.1]! with
  | some _ =>
      throw s!"duplicate model-unembed-{which} entry for i={i}"
  | none =>
      let arr' := arr.set! iFin.1 (some v)
      if which = "target" then
        return { st with modelUnembedTarget := arr' }
      else
        return { st with modelUnembedNegative := arr' }
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
  | ["model-wv", i, j, val] =>
      setModelVEntry st (← parseNat i) (← parseNat j) (← parseRat val) "wv"
  | ["model-wo", i, j, val] =>
      setModelVEntry st (← parseNat i) (← parseNat j) (← parseRat val) "wo"
  | ["model-bv", j, val] =>
      setModelBvEntry st (← parseNat j) (← parseRat val)
  | ["model-attn-bias", i, val] =>
      setModelAttnBiasEntry st (← parseNat i) (← parseRat val)
  | ["model-unembed-target", i, val] =>
      setModelUnembedEntry st (← parseNat i) (← parseRat val) "target"
  | ["model-unembed-negative", i, val] =>
      setModelUnembedEntry st (← parseNat i) (← parseRat val) "negative"
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
  /-- Optional value-path inputs for anchoring per-key values. -/
  modelValueSlice? : Option InductionHeadCert.ModelValueSlice
  /-- Optional direction inputs anchored to unembedding rows. -/
  modelDirectionSlice? : Option InductionHeadCert.ModelDirectionSlice
  /-- Optional TL exclude_bos flag. -/
  tlExcludeBos? : Option Bool
  /-- Optional TL exclude_current_token flag. -/
  tlExcludeCurrent? : Option Bool
  /-- Optional TL mul score lower bound. -/
  tlScoreLB? : Option Rat
  /-- Optional copy-logit matrix (may be present for `kind induction-aligned`). -/
  copyLogits? : Option (Fin seq → Fin seq → Rat)
  /-- Whether all weight entries were present. -/
  weightsPresent : Bool
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
  let weightsFun : Fin seq → Fin seq → Rat :=
    if st.weights.all (fun row => row.all Option.isSome) then
      fun q k =>
        let row := st.weights[q.1]!
        (row[k.1]!).getD 0
    else
      fun _ _ => 0
  let weightBoundAtFun : Fin seq → Fin seq → Rat :=
    if st.weightBoundAt.all (fun row => row.all Option.isSome) then
      fun q k =>
        let row := st.weightBoundAt[q.1]!
        (row[k.1]!).getD 0
    else
      let scoreGapLo := Sound.scoreGapLoOfBounds prevFun scoresFun scoresFun
      Sound.weightBoundAtOfScoreGap prevFun scoreGapLo
  let epsAtFun : Fin seq → Rat :=
    if st.epsAt.all Option.isSome then
      fun q => (st.epsAt[q.1]!).getD 0
    else
      Sound.epsAtOfWeightBoundAt prevFun weightBoundAtFun
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
  if !st.modelResid.all (fun row => row.all Option.isSome) then
    throw "missing model-resid entries"
  if !st.modelWq.all (fun row => row.all Option.isSome) then
    throw "missing model-wq entries"
  if !st.modelWk.all (fun row => row.all Option.isSome) then
    throw "missing model-wk entries"
  if !st.modelBq.all Option.isSome then
    throw "missing model-bq entries"
  if !st.modelBk.all Option.isSome then
    throw "missing model-bk entries"
  let residArr : Array (Array Rat) :=
    st.modelResid.map (fun row => row.map (fun v => v.getD 0))
  let wqArr : Array (Array Rat) :=
    st.modelWq.map (fun row => row.map (fun v => v.getD 0))
  let wkArr : Array (Array Rat) :=
    st.modelWk.map (fun row => row.map (fun v => v.getD 0))
  let bqArr : Array Rat := st.modelBq.map (fun v => v.getD 0)
  let bkArr : Array Rat := st.modelBk.map (fun v => v.getD 0)
  let raw : Nfp.IO.Pure.InductionHeadCert.ModelSliceRaw seq :=
    { dModel := dModel
      headDim := headDim
      layer := layer
      head := head
      scoreScale := scoreScale
      scoreMask := scoreMask
      maskCausal := maskCausal
      decimals? := st.modelDecimals
      resid := residArr
      wq := wqArr
      wk := wkArr
      bq := bqArr
      bk := bkArr }
  let checked ←
    match Nfp.IO.Pure.InductionHeadCert.checkModelSliceRaw (seq := seq) raw with
    | Except.ok checked => pure checked
    | Except.error msg => throw msg
  return some checked.toSlice
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
  let lnSlack := st.modelLnSlack.getD 0; let lnScale? := st.modelLnScale
  let lnFast? := st.modelLnFast
  let lnDecimals? := st.modelDecimals
  if !st.modelEmbed.all (fun row => row.all Option.isSome) then
    throw "missing model-embed entries"
  if !st.modelLnGamma.all Option.isSome then
    throw "missing model-ln-gamma entries"
  if !st.modelLnBeta.all Option.isSome then
    throw "missing model-ln-beta entries"
  let embedArr : Array (Array Rat) :=
    st.modelEmbed.map (fun row => row.map (fun v => v.getD 0))
  let gammaArr : Array Rat := st.modelLnGamma.map (fun v => v.getD 0)
  let betaArr : Array Rat := st.modelLnBeta.map (fun v => v.getD 0)
  let raw : Nfp.IO.Pure.InductionHeadCert.ModelLnSliceRaw seq :=
    { dModel := dModel
      lnEps := lnEps
      lnSlack := lnSlack
      lnScale? := lnScale?
      lnFast? := lnFast?
      lnDecimals? := lnDecimals?
      lnGamma := gammaArr
      lnBeta := betaArr
      embed := embedArr }
  let checked ←
    match Nfp.IO.Pure.InductionHeadCert.checkModelLnSliceRaw (seq := seq) raw with
    | Except.ok checked => pure checked
    | Except.error msg => throw msg
  return some checked.toSlice
private def finalizeModelValueSlice? {seq : Nat} (st : ParseState seq) :
    Except String (Option ModelValueSlice) := do
  let any :=
    !st.modelWv.isEmpty || !st.modelWo.isEmpty || !st.modelBv.isEmpty ||
      !st.modelAttnBias.isEmpty
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
  if !st.modelWv.all (fun row => row.all Option.isSome) then
    throw "missing model-wv entries"
  if !st.modelWo.all (fun row => row.all Option.isSome) then
    throw "missing model-wo entries"
  if !st.modelBv.all Option.isSome then
    throw "missing model-bv entries"
  if !st.modelAttnBias.all Option.isSome then
    throw "missing model-attn-bias entries"
  let wvArr : Array (Array Rat) :=
    st.modelWv.map (fun row => row.map (fun v => v.getD 0))
  let woArr : Array (Array Rat) :=
    st.modelWo.map (fun row => row.map (fun v => v.getD 0))
  let bvArr : Array Rat := st.modelBv.map (fun v => v.getD 0)
  let attnBiasArr : Array Rat := st.modelAttnBias.map (fun v => v.getD 0)
  let raw : Nfp.IO.Pure.InductionHeadCert.ModelValueSliceRaw :=
    { dModel := dModel
      headDim := headDim
      layer := layer
      head := head
      wv := wvArr
      bv := bvArr
      wo := woArr
      attnBias := attnBiasArr }
  let checked ←
    match Nfp.IO.Pure.InductionHeadCert.checkModelValueSliceRaw raw with
    | Except.ok checked => pure checked
    | Except.error msg => throw msg
  return some checked.toSlice
private def finalizeModelDirectionSlice? {seq : Nat} (st : ParseState seq) :
    Except String (Option ModelDirectionSlice) := do
  let any :=
    !st.modelUnembedTarget.isEmpty || !st.modelUnembedNegative.isEmpty
  if !any then
    return none
  let dModel ←
    match st.modelDModel with
    | some v => pure v
    | none => throw "missing model-d-model entry"
  let target ←
    match st.directionTarget with
    | some v => pure v
    | none => throw "model-unembed requires direction-target/negative"
  let negative ←
    match st.directionNegative with
    | some v => pure v
    | none => throw "model-unembed requires direction-target/negative"
  if !st.modelUnembedTarget.all Option.isSome then
    throw "missing model-unembed-target entries"
  if !st.modelUnembedNegative.all Option.isSome then
    throw "missing model-unembed-negative entries"
  let targetArr : Array Rat := st.modelUnembedTarget.map (fun v => v.getD 0)
  let negativeArr : Array Rat := st.modelUnembedNegative.map (fun v => v.getD 0)
  let raw : Nfp.IO.Pure.InductionHeadCert.ModelDirectionSliceRaw :=
    { dModel := dModel
      target := target
      negative := negative
      unembedTarget := targetArr
      unembedNegative := negativeArr }
  let checked ←
    match Nfp.IO.Pure.InductionHeadCert.checkModelDirectionSliceRaw raw with
    | Except.ok checked => pure checked
    | Except.error msg => throw msg
  return some checked.toSlice
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
      let weightsPresent := st.weights.all (fun row => row.all Option.isSome)
      if kind = "onehot-approx" && !weightsPresent then
        throw "missing weight entries"
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
      let modelValueSlice? ← InductionHeadCert.finalizeModelValueSlice? st
      let modelDirectionSlice? ← InductionHeadCert.finalizeModelDirectionSlice? st
      let payload : InductionHeadCert.InductionHeadCertPayload seq :=
        { kind := kind
          period? := period?
          modelSlice? := modelSlice?
          modelLnSlice? := modelLnSlice?
          modelValueSlice? := modelValueSlice?
          modelDirectionSlice? := modelDirectionSlice?
          tlExcludeBos? := tlExcludeBos?
          tlExcludeCurrent? := tlExcludeCurrent?
          tlScoreLB? := tlScoreLB?
          copyLogits? := copyLogits?
          weightsPresent := weightsPresent
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
/-- Check an explicit induction-head certificate from disk. -/
def runInductionHeadCertCheck (certPath : System.FilePath)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String)
    (tokensPath? : Option String) (directionQ? : Option Nat)
    (minStripeMeanStr? : Option String) (minStripeTop1Str? : Option String)
    (minCoverageStr? : Option String)
    (reportWeightedResidual : Bool)
    (timeLn : Bool) (timeScores : Bool) (timeParse : Bool) (timeStages : Bool)
    (timeValues : Bool) (timeTotal : Bool) :
    IO UInt32 := do
  let tTotal? ← if timeTotal then some <$> IO.monoMsNow else pure none
  let finish (code : UInt32) : IO UInt32 := do
    if let some t0 := tTotal? then
      let t1 ← IO.monoMsNow
      IO.eprintln s!"info: total-ms {t1 - t0}"
    return code
  let minLogitDiff?E := parseRatOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseRatOpt "min-margin" minMarginStr?
  let maxEps?E := parseRatOpt "max-eps" maxEpsStr?
  let minStripeMean?E := parseRatOpt "min-stripe-mean" minStripeMeanStr?
  let minStripeTop1?E := parseRatOpt "min-stripe-top1" minStripeTop1Str?
  let minCoverage?E := parseRatOpt "min-coverage" minCoverageStr?
  match minLogitDiff?E, minMargin?E, maxEps?E, minStripeMean?E, minStripeTop1?E, minCoverage?E with
  | Except.error msg, _, _, _, _, _ =>
      IO.eprintln s!"error: {msg}"; return (← finish 2)
  | _, Except.error msg, _, _, _, _ =>
      IO.eprintln s!"error: {msg}"; return (← finish 2)
  | _, _, Except.error msg, _, _, _ =>
      IO.eprintln s!"error: {msg}"; return (← finish 2)
  | _, _, _, Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"; return (← finish 2)
  | _, _, _, _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"; return (← finish 2)
  | _, _, _, _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"; return (← finish 2)
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps?,
      Except.ok minStripeMean?, Except.ok minStripeTop1?, Except.ok minCoverage? =>
      let minMargin := minMargin?.getD (0 : Rat)
      let maxEps := maxEps?.getD (ratRoundDown (Rat.divInt 1 2))
      let tParse? ← if timeParse then some <$> IO.monoMsNow else pure none
      let parsed ← loadInductionHeadCert certPath
      if let some t0 := tParse? then
        let t1 ← IO.monoMsNow; IO.eprintln s!"info: parse-ms {t1 - t0}"
      match parsed with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"; return (← finish 1)
      | Except.ok ⟨seq, payload⟩ =>
          match seq with
          | 0 =>
              IO.eprintln "error: seq must be positive"; return (← finish 2)
          | Nat.succ n =>
              let seq := Nat.succ n; let _ : NeZero seq := ⟨by simp⟩
              let cert := payload.cert; let kind := payload.kind
              let weightsPresent := payload.weightsPresent
              let period? := payload.period?; let copyLogits? := payload.copyLogits?
              let tlExcludeBos? := payload.tlExcludeBos?
              let tlExcludeCurrent? := payload.tlExcludeCurrent?
              let tlScoreLB? := payload.tlScoreLB?; let modelSlice? := payload.modelSlice?
              let modelLnSlice? := payload.modelLnSlice?
              let modelValueSlice? := payload.modelValueSlice?
              let modelDirectionSlice? := payload.modelDirectionSlice?
              let tPre? ← if timeStages then some <$> IO.monoMsNow else pure none
              if kind != "onehot-approx" && kind != "induction-aligned" then
                IO.eprintln s!"error: unexpected kind {kind}"; return (← finish 2)
              if let some dirSlice := modelDirectionSlice? then
                match cert.values.direction with
                | none =>
                    IO.eprintln "error: model-unembed requires direction metadata"
                    return (← finish 2)
                | some dirSpec =>
                    if dirSpec.target != dirSlice.target ||
                        dirSpec.negative != dirSlice.negative then
                      IO.eprintln
                        "error: direction metadata does not match model-unembed target/negative"
                      return (← finish 2)
              if kind = "onehot-approx" then
                if minStripeMeanStr?.isSome || minStripeTop1Str?.isSome then
                  IO.eprintln "error: stripe thresholds are not used for onehot-approx"
                  return (← finish 2)
                if minCoverageStr?.isSome then
                  IO.eprintln "error: min-coverage is not used for onehot-approx"
                  return (← finish 2)
              let activeOverride? ←
                if kind = "induction-aligned" then
                  let period ←
                    match period? with
                    | some v => pure v
                    | none =>
                        IO.eprintln "error: missing period entry for induction-aligned"
                        return (← finish 2)
                  if period = 0 then
                    IO.eprintln "error: period must be positive for induction-aligned"
                    return (← finish 2)
                  if seq ≤ period then
                    IO.eprintln "error: period must be less than seq for induction-aligned"
                    return (← finish 2)
                  let expectedActive ← timeIO timeStages "period-active"
                    (fun _ => pure (Model.activeOfPeriod (seq := seq) period))
                  if !decide (cert.active = expectedActive) then
                    IO.eprintln "error: active set does not match induction-aligned period"
                    return (← finish 2)
                  let prevOk ← timeIO timeStages "period-prev"
                    (fun _ => pure ((List.finRange seq).all (fun q =>
                      decide (cert.prev q = Model.prevOfPeriod (seq := seq) period q))))
                  if !prevOk then
                    IO.eprintln "error: prev map does not match induction-aligned period"
                    return (← finish 2)
                  if minLogitDiffStr?.isSome && cert.values.direction.isNone then
                    IO.eprintln "error: min-logit-diff requires direction metadata"
                    return (← finish 2)
                  if minStripeTop1Str?.isSome then
                    IO.eprintln "error: stripe-top1 is not used for induction-aligned"
                    return (← finish 2)
                  if let some minCoverage := minCoverage? then
                    if minCoverage < 0 || (1 : Rat) < minCoverage then
                      IO.eprintln "error: min-coverage must be between 0 and 1"
                      return (← finish 2)
                    let activeGood :=
                      InductionHeadCert.derivedActive cert expectedActive minMargin maxEps
                    let coverage? := InductionHeadCert.activeCoverage activeGood expectedActive
                    match coverage? with
                    | none =>
                        IO.eprintln "error: empty induction-aligned active set"
                        return (← finish 2)
                    | some coverage =>
                        if coverage < minCoverage then
                          IO.eprintln
                            s!"error: active coverage {ratToString coverage} \
                            below minimum {ratToString minCoverage}"
                          return (← finish 2)
                    pure (some activeGood)
                  else
                    if minMarginStr?.isSome || maxEpsStr?.isSome then
                      IO.eprintln "error: min-margin/max-eps are not used for induction-aligned"
                      return (← finish 2)
                    pure none
                else
                  pure none
              let activeForChecks := activeOverride?.getD cert.active
              let activeCount := activeForChecks.card; let defaultMinActive := max 1 (seq / 8)
              let minActive := minActive?.getD defaultMinActive; if activeCount < minActive then
                IO.eprintln s!"error: active queries {activeCount} below minimum {minActive}"
                return (← finish 2)
              if let some t0 := tPre? then
                let t1 ← IO.monoMsNow; IO.eprintln s!"info: phase-pre-ms {t1 - t0}"
              let tModel? ← if timeStages then some <$> IO.monoMsNow else pure none
              let tLnPhase? ← if timeStages then some <$> IO.monoMsNow else pure none
              if let some modelLnSlice := modelLnSlice? then
                match modelSlice? with
                | none =>
                    IO.eprintln "error: model-ln data requires model-resid slice"
                    return (← finish 2)
                | some modelSlice =>
                    if h : modelLnSlice.dModel = modelSlice.dModel then
                      if modelLnSlice.lnEps ≤ 0 then
                        IO.eprintln "error: model-ln-eps must be positive"; return (← finish 2)
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
                          let t1 ← IO.monoMsNow; IO.eprintln s!"info: ln-check-ms {t1 - t0}"
                        IO.eprintln "error: residuals not within LayerNorm bounds"
                        return (← finish 2)
                      if let some t0 := t0? then
                        let t1 ← IO.monoMsNow; IO.eprintln s!"info: ln-check-ms {t1 - t0}"
                    else
                      IO.eprintln "error: model-ln dModel does not match model slice"
                      return (← finish 2)
              if let some t0 := tLnPhase? then
                let t1 ← IO.monoMsNow; IO.eprintln s!"info: phase-ln-ms {t1 - t0}"
              if let some modelValueSlice := modelValueSlice? then
                match modelLnSlice?, modelDirectionSlice? with
                | none, _ =>
                    IO.eprintln "error: model-value data requires model-ln slice"
                    return (← finish 2)
                | _, none =>
                    IO.eprintln "error: model-value data requires model-unembed slice"
                    return (← finish 2)
                | some modelLnSlice, some modelDirectionSlice =>
                    if hLn : modelLnSlice.dModel = modelValueSlice.dModel then
                      if hDir : modelDirectionSlice.dModel = modelValueSlice.dModel then
                        let okValues ← timeIO timeStages "values-model" (fun _ =>
                          if timeValues then
                            InductionHeadCert.valuesWithinModelBoundsProfile
                              modelLnSlice modelValueSlice modelDirectionSlice hLn hDir
                              cert.values true
                          else
                            pure
                              (InductionHeadCert.valuesWithinModelBounds
                                modelLnSlice modelValueSlice modelDirectionSlice hLn hDir
                                cert.values))
                        if !okValues then
                          IO.eprintln "error: values not within model bounds"
                          return (← finish 2)
                      else
                        IO.eprintln
                          "error: model-direction dModel does not match model-value slice"
                        return (← finish 2)
                    else
                      IO.eprintln "error: model-ln dModel does not match model-value slice"
                      return (← finish 2)
              let tScoresPhase? ← if timeStages then some <$> IO.monoMsNow else pure none
              if let some modelSlice := modelSlice? then
                let okScores ←
                  if timeScores then
                    let t0 ← IO.monoMsNow
                    let (fastUsed, okScores) :=
                      match InductionHeadCert.scoresMatchModelSliceFast?
                          (seq := seq) modelSlice cert.scores with
                      | some ok => (true, ok)
                      | none =>
                          (false,
                            InductionHeadCert.scoresMatchModelSliceSlow
                              (seq := seq) modelSlice cert.scores)
                    if okScores then pure () else IO.eprintln ""
                    let t1 ← IO.monoMsNow
                    IO.eprintln s!"info: scores-fast {fastUsed}"
                    IO.eprintln s!"info: scores-check-ms {t1 - t0}"
                    pure okScores
                  else
                    pure
                      (InductionHeadCert.scoresMatchModelSlice (seq := seq) modelSlice cert.scores)
                if !okScores then
                  IO.eprintln "error: scores do not match model slice"; return (← finish 2)
              let scoreBoundsRes ← timeIO timeStages "score-bounds" (fun _ =>
                pure (InductionHeadCert.checkScoreBoundsWithinModel
                  (seq := seq) modelSlice? cert))
              match scoreBoundsRes with
              | Except.error msg =>
                  IO.eprintln s!"error: {msg}"
                  return (← finish 2)
              | Except.ok () => pure ()
              if let some t0 := tScoresPhase? then
                let t1 ← IO.monoMsNow; IO.eprintln s!"info: phase-scores-ms {t1 - t0}"
              if let some t0 := tModel? then
                let t1 ← IO.monoMsNow; IO.eprintln s!"info: phase-model-ms {t1 - t0}"
              if kind = "onehot-approx" then
                let ok ← timeIO timeStages "cert-check"
                  (fun _ => pure (Circuit.checkInductionHeadCert cert))
                if !ok then
                  IO.eprintln "error: induction-head certificate rejected"; return (← finish 2)
                if cert.margin < minMargin then
                  IO.eprintln
                    s!"error: margin {ratToString cert.margin} \
                    below minimum {ratToString minMargin}"
                  return (← finish 2)
                if maxEps < cert.eps then
                  IO.eprintln
                    s!"error: eps {ratToString cert.eps} \
                    above maximum {ratToString maxEps}"
                  return (← finish 2)
              let tokensOpt ←
                match tokensPath? with
                | none =>
                    if tlScoreLB?.isSome then
                      IO.eprintln "error: tl-score-lb requires a tokens file"; return (← finish 2)
                    else
                      pure none
                | some tokensPath =>
                    let tokensParsed ← timeIO timeStages "tokens-parse"
                      (fun _ => loadInductionHeadTokens tokensPath)
                    match tokensParsed with
                    | Except.error msg =>
                        IO.eprintln s!"error: {msg}"; return (← finish 2)
                    | Except.ok ⟨seqTokens, tokens⟩ =>
                        if hseq : seqTokens = seq then
                          let tokens' : Fin seq → Nat := by
                            simpa [hseq] using tokens
                          pure (some tokens')
                        else
                          IO.eprintln
                            s!"error: tokens seq {seqTokens} does not match cert seq {seq}"
                          return (← finish 2)
              let tokensOk ← timeIO timeStages "tokens-check" (fun _ =>
                pure (InductionHeadCert.checkTokensAndDirection?
                  (seq := seq) kind period? directionQ? cert tokensOpt))
              match tokensOk with
              | Except.ok () => pure ()
              | Except.error msg =>
                  IO.eprintln s!"error: {msg}"
                  return (← finish 2)
              if let some tlScoreLB := tlScoreLB? then
                match tokensOpt with
                | none =>
                    IO.eprintln "error: tl-score-lb requires a tokens file"; return (← finish 2)
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
                      return (← finish 2)
              let stripeMeanOpt ←
                if kind = "induction-aligned" then
                  let period ←
                    match period? with
                    | some v => pure v
                    | none =>
                        IO.eprintln "error: missing period entry for induction-aligned"
                        return (← finish 2)
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
                          return (← finish 2)
                        pure mean
                    | none =>
                        IO.eprintln "error: empty active set for stripe stats"; return (← finish 2)
                  pure (some mean)
                else
                  pure none
              let activeForBounds := activeOverride?.getD cert.active
              if reportWeightedResidual then
                InductionHeadCert.reportWeightedResidualDir
                  modelLnSlice? modelValueSlice? modelDirectionSlice?
                  weightsPresent cert.weights activeForBounds timeStages
              let logitDiffLB? :=
                InductionHeadCert.logitDiffLowerBoundTightWithActive? activeForBounds cert
              match minLogitDiff? with
              | none =>
                  if kind = "induction-aligned" then
                    match stripeMeanOpt with
                    | some mean =>
                      let _ ← timeIO timeStages "print-ok" (fun _ =>
                        IO.println s!"ok: induction-aligned certificate checked \
                          (seq={seq}, active={activeCount}, stripeMean={ratToString mean})")
                        return (← finish 0)
                    | none =>
                        IO.eprintln "error: missing stripe mean for induction-aligned"
                        return (← finish 2)
                  else
                    let kindLabel := "onehot-approx (proxy)"
                    let _ ← timeIO timeStages "print-ok" (fun _ =>
                      IO.println s!"ok: {kindLabel} certificate checked \
                        (seq={seq}, active={activeCount}, \
                        margin={ratToString cert.margin}, eps={ratToString cert.eps})")
                    return (← finish 0)
              | some minLogitDiff =>
                  let logitDiffLB? ← timeIO timeStages "logit-diff" (fun _ =>
                    pure logitDiffLB?)
                  match logitDiffLB? with
                  | none =>
                      IO.eprintln "error: empty active set for logit-diff bound"
                      return (← finish 2)
                  | some logitDiffLB =>
                      if logitDiffLB < minLogitDiff then
                        IO.eprintln
                          s!"error: logitDiffLB {ratToString logitDiffLB} \
                          below minimum {ratToString minLogitDiff}"
                        return (← finish 2)
                      else
                        if kind = "induction-aligned" then
                          match stripeMeanOpt with
                          | some mean =>
                              let _ ← timeIO timeStages "print-ok" (fun _ =>
                                IO.println s!"ok: induction-aligned certificate checked \
                                  (seq={seq}, active={activeCount}, \
                                  stripeMean={ratToString mean}, \
                                  logitDiffLB={ratToString logitDiffLB})")
                              return (← finish 0)
                          | none =>
                              IO.eprintln "error: missing stripe mean for induction-aligned"
                              return (← finish 2)
                        else
                          let kindLabel := "onehot-approx (proxy)"
                          let _ ← timeIO timeStages "print-ok" (fun _ =>
                            IO.println s!"ok: {kindLabel} certificate checked \
                              (seq={seq}, active={activeCount}, \
                              margin={ratToString cert.margin}, eps={ratToString cert.eps}, \
                              logitDiffLB={ratToString logitDiffLB})")
                          return (← finish 0)
end IO
end Nfp
