-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Data.Finset.Insert
import Nfp.Circuit.Cert.SoftmaxMargin
import Nfp.Circuit.Cert.ValueRange
import Nfp.Circuit.Cert.DownstreamLinear
import Nfp.Circuit.Cert.ResidualBound
import Nfp.Circuit.Cert.ResidualInterval
import Nfp.Model.InductionHead

/-!
Pure parsing helpers for softmax-margin, value-range, and downstream certificates.
-/

namespace Nfp

namespace IO

namespace Pure

open Nfp.Circuit

private def splitWords (line : String) : List String :=
  line.splitToList (fun c => c = ' ' || c = '\t') |>.filter (· ≠ "")

private def cleanTokens (line : String) : Option (List String) :=
  let trimmed := line.trim
  if trimmed.isEmpty then
    none
  else if trimmed.startsWith "#" then
    none
  else
    some (splitWords trimmed)

private def parseNat (s : String) : Except String Nat :=
  match s.toNat? with
  | some n => Except.ok n
  | none => Except.error s!"expected Nat, got '{s}'"

private def parseInt (s : String) : Except String Int :=
  match s.toInt? with
  | some n => Except.ok n
  | none => Except.error s!"expected Int, got '{s}'"

/-- Parse a rational literal of the form `a` or `a/b`. -/
def parseRat (s : String) : Except String Rat := do
  match s.splitOn "/" with
  | [num] =>
      return Rat.ofInt (← parseInt num)
  | [num, den] =>
      let n ← parseInt num
      let d ← parseNat den
      if d = 0 then
        throw s!"invalid rational '{s}': zero denominator"
      else
        return Rat.ofInt n / Rat.ofInt (Int.ofNat d)
  | _ =>
      throw s!"invalid rational '{s}'"

private structure SoftmaxMarginParseState (seq : Nat) where
  eps : Option Rat
  margin : Option Rat
  active : Finset (Fin seq)
  activeSeen : Bool
  prev : Fin seq → Option (Fin seq)
  scores : Fin seq → Fin seq → Option Rat
  weights : Fin seq → Fin seq → Option Rat

private def initState (seq : Nat) : SoftmaxMarginParseState seq :=
  { eps := none
    margin := none
    active := ∅
    activeSeen := false
    prev := fun _ => none
    scores := fun _ _ => none
    weights := fun _ _ => none }

private def setPrev {seq : Nat} (st : SoftmaxMarginParseState seq)
    (q k : Nat) : Except String (SoftmaxMarginParseState seq) := do
  if hq : q < seq then
    if hk : k < seq then
      let qFin : Fin seq := ⟨q, hq⟩
      let kFin : Fin seq := ⟨k, hk⟩
      match st.prev qFin with
      | some _ =>
          throw s!"duplicate prev entry for q={q}"
      | none =>
          let prev' : Fin seq → Option (Fin seq) := fun q' =>
            if q' = qFin then
              some kFin
            else
              st.prev q'
          return { st with prev := prev' }
    else
      throw s!"prev index out of range: k={k}"
  else
    throw s!"prev index out of range: q={q}"

private def setActive {seq : Nat} (st : SoftmaxMarginParseState seq)
    (q : Nat) : Except String (SoftmaxMarginParseState seq) := do
  if hq : q < seq then
    let qFin : Fin seq := ⟨q, hq⟩
    if qFin ∈ st.active then
      throw s!"duplicate active entry for q={q}"
    else
      return { st with active := insert qFin st.active, activeSeen := true }
  else
    throw s!"active index out of range: q={q}"

private def setMatrixEntry {seq : Nat} (mat : Fin seq → Fin seq → Option Rat)
    (q k : Nat) (v : Rat) : Except String (Fin seq → Fin seq → Option Rat) := do
  if hq : q < seq then
    if hk : k < seq then
      let qFin : Fin seq := ⟨q, hq⟩
      let kFin : Fin seq := ⟨k, hk⟩
      match mat qFin kFin with
      | some _ =>
          throw s!"duplicate matrix entry at ({q}, {k})"
      | none =>
          let mat' : Fin seq → Fin seq → Option Rat := fun q' k' =>
            if q' = qFin then
              if k' = kFin then
                some v
              else
                mat q' k'
            else
              mat q' k'
          return mat'
    else
      throw s!"index out of range: k={k}"
  else
    throw s!"index out of range: q={q}"

private def parseLine {seq : Nat} (st : SoftmaxMarginParseState seq)
    (tokens : List String) : Except String (SoftmaxMarginParseState seq) := do
  match tokens with
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
      let mat ← setMatrixEntry st.scores (← parseNat q) (← parseNat k) (← parseRat val)
      return { st with scores := mat }
  | ["weight", q, k, val] =>
      let mat ← setMatrixEntry st.weights (← parseNat q) (← parseNat k) (← parseRat val)
      return { st with weights := mat }
  | _ =>
      throw s!"unrecognized line: '{String.intercalate " " tokens}'"

private def finalizeState {seq : Nat} (hpos : 0 < seq)
    (st : SoftmaxMarginParseState seq) : Except String (SoftmaxMarginCert seq) := do
  let eps ←
    match st.eps with
    | some v => pure v
    | none => throw "missing eps entry"
  let margin ←
    match st.margin with
    | some v => pure v
    | none => throw "missing margin entry"
  if !finsetAll (Finset.univ : Finset (Fin seq)) (fun q => (st.prev q).isSome) then
    throw "missing prev entries"
  if !finsetAll (Finset.univ : Finset (Fin seq)) (fun q =>
      finsetAll (Finset.univ : Finset (Fin seq)) (fun k => (st.scores q k).isSome)) then
    throw "missing score entries"
  if !finsetAll (Finset.univ : Finset (Fin seq)) (fun q =>
      finsetAll (Finset.univ : Finset (Fin seq)) (fun k => (st.weights q k).isSome)) then
    throw "missing weight entries"
  let defaultPrev : Fin seq := ⟨0, hpos⟩
  let prevFun : Fin seq → Fin seq := fun q =>
    (st.prev q).getD defaultPrev
  let scoresFun : Fin seq → Fin seq → Rat := fun q k =>
    (st.scores q k).getD 0
  let weightsFun : Fin seq → Fin seq → Rat := fun q k =>
    (st.weights q k).getD 0
  let active :=
    if st.activeSeen then
      st.active
    else
      (Finset.univ : Finset (Fin seq)).erase defaultPrev
  pure
    { eps := eps
      margin := margin
      active := active
      prev := prevFun
      scores := scoresFun
      weights := weightsFun }

/-- Parse a softmax-margin certificate from a text payload. -/
def parseSoftmaxMarginCert (input : String) :
    Except String (Sigma SoftmaxMarginCert) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let mut seq? : Option Nat := none
  for t in tokens do
    match t with
    | ["seq", n] =>
        if seq?.isSome then
          throw "duplicate seq entry"
        else
          seq? := some (← parseNat n)
    | _ => pure ()
  let seq ←
    match seq? with
    | some v => pure v
    | none => throw "missing seq entry"
  match seq with
  | 0 => throw "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let hpos : 0 < seq := Nat.succ_pos n
      let st0 : SoftmaxMarginParseState seq := initState seq
      let st ← tokens.foldlM (fun st t =>
          match t with
          | ["seq", _] => pure st
          | _ => parseLine st t) st0
      let cert ← finalizeState hpos st
      return ⟨seq, cert⟩

/-- Raw softmax-margin payload without `eps`/`margin`. -/
structure SoftmaxMarginRaw (seq : Nat) where
  /-- Active queries for which bounds are required. -/
  active : Finset (Fin seq)
  /-- `prev` selector for induction-style attention. -/
  prev : Fin seq → Fin seq
  /-- Score matrix entries. -/
  scores : Fin seq → Fin seq → Rat
  /-- Attention weight entries. -/
  weights : Fin seq → Fin seq → Rat

private def finalizeRawState {seq : Nat} (hpos : 0 < seq)
    (st : SoftmaxMarginParseState seq) : Except String (SoftmaxMarginRaw seq) := do
  if !finsetAll (Finset.univ : Finset (Fin seq)) (fun q => (st.prev q).isSome) then
    throw "missing prev entries"
  if !finsetAll (Finset.univ : Finset (Fin seq)) (fun q =>
      finsetAll (Finset.univ : Finset (Fin seq)) (fun k => (st.scores q k).isSome)) then
    throw "missing score entries"
  if !finsetAll (Finset.univ : Finset (Fin seq)) (fun q =>
      finsetAll (Finset.univ : Finset (Fin seq)) (fun k => (st.weights q k).isSome)) then
    throw "missing weight entries"
  let defaultPrev : Fin seq := ⟨0, hpos⟩
  let prevFun : Fin seq → Fin seq := fun q =>
    (st.prev q).getD defaultPrev
  let scoresFun : Fin seq → Fin seq → Rat := fun q k =>
    (st.scores q k).getD 0
  let weightsFun : Fin seq → Fin seq → Rat := fun q k =>
    (st.weights q k).getD 0
  let active :=
    if st.activeSeen then
      st.active
    else
      (Finset.univ : Finset (Fin seq)).erase defaultPrev
  pure
    { active := active
      prev := prevFun
      scores := scoresFun
      weights := weightsFun }

/-- Parse a raw softmax-margin payload from text (ignores any `eps`/`margin`). -/
def parseSoftmaxMarginRaw (input : String) :
    Except String (Sigma SoftmaxMarginRaw) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let mut seq? : Option Nat := none
  for t in tokens do
    match t with
    | ["seq", n] =>
        if seq?.isSome then
          throw "duplicate seq entry"
        else
          seq? := some (← parseNat n)
    | _ => pure ()
  let seq ←
    match seq? with
    | some v => pure v
    | none => throw "missing seq entry"
  match seq with
  | 0 => throw "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let hpos : 0 < seq := Nat.succ_pos n
      let st0 : SoftmaxMarginParseState seq := initState seq
      let st ← tokens.foldlM (fun st t =>
          match t with
          | ["seq", _] => pure st
          | _ => parseLine st t) st0
      let raw ← finalizeRawState hpos st
      return ⟨seq, raw⟩

private structure ValueRangeParseState (seq : Nat) where
  lo : Option Rat
  hi : Option Rat
  vals : Fin seq → Option Rat
  directionTarget : Option Nat
  directionNegative : Option Nat

private def initValueRangeState (seq : Nat) : ValueRangeParseState seq :=
  { lo := none
    hi := none
    vals := fun _ => none
    directionTarget := none
    directionNegative := none }

private def setVal {seq : Nat} (st : ValueRangeParseState seq)
    (k : Nat) (v : Rat) : Except String (ValueRangeParseState seq) := do
  if hk : k < seq then
    let kFin : Fin seq := ⟨k, hk⟩
    match st.vals kFin with
    | some _ =>
        throw s!"duplicate value entry for k={k}"
    | none =>
        let vals' : Fin seq → Option Rat := fun k' =>
          if k' = kFin then
            some v
          else
            st.vals k'
        return { st with vals := vals' }
  else
    throw s!"value index out of range: k={k}"

private def parseValueLine {seq : Nat} (st : ValueRangeParseState seq)
    (tokens : List String) : Except String (ValueRangeParseState seq) := do
  match tokens with
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
      setVal st (← parseNat k) (← parseRat val)
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
  | _ =>
      throw s!"unrecognized line: '{String.intercalate " " tokens}'"

private def finalizeValueState {seq : Nat} (st : ValueRangeParseState seq) :
    Except String (ValueRangeCert seq) := do
  let lo ←
    match st.lo with
    | some v => pure v
    | none => throw "missing lo entry"
  let hi ←
    match st.hi with
    | some v => pure v
    | none => throw "missing hi entry"
  if !finsetAll (Finset.univ : Finset (Fin seq)) (fun k => (st.vals k).isSome) then
    throw "missing value entries"
  let valsFun : Fin seq → Rat := fun k =>
    (st.vals k).getD 0
  let direction ←
    match st.directionTarget, st.directionNegative with
    | none, none => pure none
    | some target, some negative =>
        pure (some { target := target, negative := negative })
    | _, _ =>
        throw "direction metadata requires both direction-target and direction-negative"
  return { lo := lo, hi := hi, vals := valsFun, direction := direction }

/-- Parse a value-range certificate from a text payload. -/
def parseValueRangeCert (input : String) :
    Except String (Sigma ValueRangeCert) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let mut seq? : Option Nat := none
  for t in tokens do
    match t with
    | ["seq", n] =>
        if seq?.isSome then
          throw "duplicate seq entry"
        else
          seq? := some (← parseNat n)
    | _ => pure ()
  let seq ←
    match seq? with
    | some v => pure v
    | none => throw "missing seq entry"
  match seq with
  | 0 => throw "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let st0 : ValueRangeParseState seq := initValueRangeState seq
      let st ← tokens.foldlM (fun st t =>
          match t with
          | ["seq", _] => pure st
          | _ => parseValueLine st t) st0
      let cert ← finalizeValueState st
      return ⟨seq, cert⟩

/-- Raw value-range payload without `lo`/`hi` bounds. -/
structure ValueRangeRaw (seq : Nat) where
  /-- Value entries. -/
  vals : Fin seq → Rat
  /-- Optional logit-diff direction metadata. -/
  direction : Option Circuit.DirectionSpec

private def finalizeValueRawState {seq : Nat} (st : ValueRangeParseState seq) :
    Except String (ValueRangeRaw seq) := do
  if !finsetAll (Finset.univ : Finset (Fin seq)) (fun k => (st.vals k).isSome) then
    throw "missing value entries"
  let valsFun : Fin seq → Rat := fun k =>
    (st.vals k).getD 0
  let direction ←
    match st.directionTarget, st.directionNegative with
    | none, none => pure none
    | some target, some negative =>
        pure (some { target := target, negative := negative })
    | _, _ =>
        throw "direction metadata requires both direction-target and direction-negative"
  return { vals := valsFun, direction := direction }

/-- Parse a raw value-range payload from text (ignores any `lo`/`hi`). -/
def parseValueRangeRaw (input : String) :
    Except String (Sigma ValueRangeRaw) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let mut seq? : Option Nat := none
  for t in tokens do
    match t with
    | ["seq", n] =>
        if seq?.isSome then
          throw "duplicate seq entry"
        else
          seq? := some (← parseNat n)
    | _ => pure ()
  let seq ←
    match seq? with
    | some v => pure v
    | none => throw "missing seq entry"
  match seq with
  | 0 => throw "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let st0 : ValueRangeParseState seq := initValueRangeState seq
      let st ← tokens.foldlM (fun st t =>
          match t with
          | ["seq", _] => pure st
          | _ => parseValueLine st t) st0
      let raw ← finalizeValueRawState st
      return ⟨seq, raw⟩

private structure DownstreamLinearParseState where
  error : Option Rat
  gain : Option Rat
  inputBound : Option Rat

private def initDownstreamLinearState : DownstreamLinearParseState :=
  { error := none, gain := none, inputBound := none }

private def parseDownstreamLinearLine (st : DownstreamLinearParseState)
    (tokens : List String) : Except String DownstreamLinearParseState := do
  match tokens with
  | ["error", val] =>
      if st.error.isSome then
        throw "duplicate error entry"
      else
        return { st with error := some (← parseRat val) }
  | ["gain", val] =>
      if st.gain.isSome then
        throw "duplicate gain entry"
      else
        return { st with gain := some (← parseRat val) }
  | ["input-bound", val] =>
      if st.inputBound.isSome then
        throw "duplicate input-bound entry"
      else
        return { st with inputBound := some (← parseRat val) }
  | _ =>
      throw s!"unrecognized line: '{String.intercalate " " tokens}'"

private def finalizeDownstreamLinearState (st : DownstreamLinearParseState) :
    Except String Circuit.DownstreamLinearCert := do
  let error ←
    match st.error with
    | some v => pure v
    | none => throw "missing error entry"
  let gain ←
    match st.gain with
    | some v => pure v
    | none => throw "missing gain entry"
  let inputBound ←
    match st.inputBound with
    | some v => pure v
    | none => throw "missing input-bound entry"
  return { error := error, gain := gain, inputBound := inputBound }

/-- Parse a downstream linear certificate from a text payload. -/
def parseDownstreamLinearCert (input : String) :
    Except String Circuit.DownstreamLinearCert := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let st0 := initDownstreamLinearState
  let st ← tokens.foldlM (fun st t => parseDownstreamLinearLine st t) st0
  finalizeDownstreamLinearState st

private def setVecEntry {n : Nat} (vec : Fin n → Option Rat)
    (i : Nat) (v : Rat) : Except String (Fin n → Option Rat) := do
  if hi : i < n then
    let iFin : Fin n := ⟨i, hi⟩
    match vec iFin with
    | some _ =>
        throw s!"duplicate entry for index={i}"
    | none =>
        let vec' : Fin n → Option Rat := fun i' =>
          if i' = iFin then
            some v
          else
            vec i'
        return vec'
  else
    throw s!"index out of range: i={i}"

private def setMatEntry {m n : Nat} (mat : Fin m → Fin n → Option Rat)
    (i j : Nat) (v : Rat) : Except String (Fin m → Fin n → Option Rat) := do
  if hi : i < m then
    if hj : j < n then
      let iFin : Fin m := ⟨i, hi⟩
      let jFin : Fin n := ⟨j, hj⟩
      match mat iFin jFin with
      | some _ =>
          throw s!"duplicate entry for indices={i},{j}"
      | none =>
          let mat' : Fin m → Fin n → Option Rat := fun i' j' =>
            if i' = iFin then
              if j' = jFin then
                some v
              else
                mat i' j'
            else
              mat i' j'
          return mat'
    else
      throw s!"index out of range: j={j}"
  else
    throw s!"index out of range: i={i}"

private structure HeadParseState (seq dModel dHead : Nat) where
  scale : Option Rat
  active : Finset (Fin seq)
  activeSeen : Bool
  prev : Fin seq → Option (Fin seq)
  embed : Fin seq → Fin dModel → Option Rat
  lnEps : Option Rat
  ln1Gamma : Fin dModel → Option Rat
  ln1Beta : Fin dModel → Option Rat
  wq : Fin dModel → Fin dHead → Option Rat
  bq : Fin dHead → Option Rat
  wk : Fin dModel → Fin dHead → Option Rat
  bk : Fin dHead → Option Rat
  wv : Fin dModel → Fin dHead → Option Rat
  bv : Fin dHead → Option Rat
  wo : Fin dModel → Fin dHead → Option Rat
  attnBias : Fin dModel → Option Rat
  directionTarget : Option Nat
  directionNegative : Option Nat
  direction : Fin dModel → Option Rat

private def initHeadState (seq dModel dHead : Nat) : HeadParseState seq dModel dHead :=
  { scale := none
    active := ∅
    activeSeen := false
    prev := fun _ => none
    embed := fun _ _ => none
    lnEps := none
    ln1Gamma := fun _ => none
    ln1Beta := fun _ => none
    wq := fun _ _ => none
    bq := fun _ => none
    wk := fun _ _ => none
    bk := fun _ => none
    wv := fun _ _ => none
    bv := fun _ => none
    wo := fun _ _ => none
    attnBias := fun _ => none
    directionTarget := none
    directionNegative := none
    direction := fun _ => none }

private def setHeadActive {seq dModel dHead : Nat}
    (st : HeadParseState seq dModel dHead) (q : Nat) :
    Except String (HeadParseState seq dModel dHead) := do
  if hq : q < seq then
    let qFin : Fin seq := ⟨q, hq⟩
    return { st with active := st.active ∪ {qFin}, activeSeen := true }
  else
    throw s!"active index out of range: q={q}"

private def setHeadPrev {seq dModel dHead : Nat}
    (st : HeadParseState seq dModel dHead) (q k : Nat) :
    Except String (HeadParseState seq dModel dHead) := do
  if hq : q < seq then
    if hk : k < seq then
      let qFin : Fin seq := ⟨q, hq⟩
      let kFin : Fin seq := ⟨k, hk⟩
      match st.prev qFin with
      | some _ =>
          throw s!"duplicate prev entry for q={q}"
      | none =>
          let prev' : Fin seq → Option (Fin seq) := fun q' =>
            if q' = qFin then
              some kFin
            else
              st.prev q'
          return { st with prev := prev' }
    else
      throw s!"prev index out of range: k={k}"
  else
    throw s!"prev index out of range: q={q}"

private def parseHeadLine {seq dModel dHead : Nat} (st : HeadParseState seq dModel dHead)
    (tokens : List String) : Except String (HeadParseState seq dModel dHead) := do
  match tokens with
  | ["scale", val] =>
      if st.scale.isSome then
        throw "duplicate scale entry"
      else
        return { st with scale := some (← parseRat val) }
  | ["active", q] =>
      setHeadActive st (← parseNat q)
  | ["prev", q, k] =>
      setHeadPrev st (← parseNat q) (← parseNat k)
  | ["embed", q, d, val] =>
      let mat ← setMatEntry st.embed (← parseNat q) (← parseNat d) (← parseRat val)
      return { st with embed := mat }
  | ["ln_eps", val] =>
      if st.lnEps.isSome then
        throw "duplicate ln_eps entry"
      else
        return { st with lnEps := some (← parseRat val) }
  | ["ln1_gamma", d, val] =>
      let vec ← setVecEntry st.ln1Gamma (← parseNat d) (← parseRat val)
      return { st with ln1Gamma := vec }
  | ["ln1_beta", d, val] =>
      let vec ← setVecEntry st.ln1Beta (← parseNat d) (← parseRat val)
      return { st with ln1Beta := vec }
  | ["wq", i, j, val] =>
      let mat ← setMatEntry st.wq (← parseNat i) (← parseNat j) (← parseRat val)
      return { st with wq := mat }
  | ["bq", j, val] =>
      let vec ← setVecEntry st.bq (← parseNat j) (← parseRat val)
      return { st with bq := vec }
  | ["wk", i, j, val] =>
      let mat ← setMatEntry st.wk (← parseNat i) (← parseNat j) (← parseRat val)
      return { st with wk := mat }
  | ["bk", j, val] =>
      let vec ← setVecEntry st.bk (← parseNat j) (← parseRat val)
      return { st with bk := vec }
  | ["wv", i, j, val] =>
      let mat ← setMatEntry st.wv (← parseNat i) (← parseNat j) (← parseRat val)
      return { st with wv := mat }
  | ["bv", j, val] =>
      let vec ← setVecEntry st.bv (← parseNat j) (← parseRat val)
      return { st with bv := vec }
  | ["wo", i, j, val] =>
      let mat ← setMatEntry st.wo (← parseNat i) (← parseNat j) (← parseRat val)
      return { st with wo := mat }
  | ["attn_bias", d, val] =>
      let vec ← setVecEntry st.attnBias (← parseNat d) (← parseRat val)
      return { st with attnBias := vec }
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
  | ["direction", d, val] =>
      let vec ← setVecEntry st.direction (← parseNat d) (← parseRat val)
      return { st with direction := vec }
  | _ =>
      throw s!"unrecognized line: '{String.intercalate " " tokens}'"

private def finalizeHeadState {seq dModel dHead : Nat} (hpos : 0 < seq)
    (st : HeadParseState seq dModel dHead) :
    Except String (Model.InductionHeadInputs seq dModel dHead) := do
  let scale ←
    match st.scale with
    | some v => pure v
    | none => throw "missing scale entry"
  if !finsetAll (Finset.univ : Finset (Fin seq)) (fun q => (st.prev q).isSome) then
    throw "missing prev entries"
  if !finsetAll (Finset.univ : Finset (Fin seq)) (fun q =>
      finsetAll (Finset.univ : Finset (Fin dModel)) (fun d => (st.embed q d).isSome)) then
    throw "missing embed entries"
  let lnEps ←
    match st.lnEps with
    | some v => pure v
    | none => throw "missing ln_eps entry"
  if !finsetAll (Finset.univ : Finset (Fin dModel)) (fun d => (st.ln1Gamma d).isSome) then
    throw "missing ln1_gamma entries"
  if !finsetAll (Finset.univ : Finset (Fin dModel)) (fun d => (st.ln1Beta d).isSome) then
    throw "missing ln1_beta entries"
  if !finsetAll (Finset.univ : Finset (Fin dModel)) (fun i =>
      finsetAll (Finset.univ : Finset (Fin dHead)) (fun j => (st.wq i j).isSome)) then
    throw "missing wq entries"
  if !finsetAll (Finset.univ : Finset (Fin dHead)) (fun j => (st.bq j).isSome) then
    throw "missing bq entries"
  if !finsetAll (Finset.univ : Finset (Fin dModel)) (fun i =>
      finsetAll (Finset.univ : Finset (Fin dHead)) (fun j => (st.wk i j).isSome)) then
    throw "missing wk entries"
  if !finsetAll (Finset.univ : Finset (Fin dHead)) (fun j => (st.bk j).isSome) then
    throw "missing bk entries"
  if !finsetAll (Finset.univ : Finset (Fin dModel)) (fun i =>
      finsetAll (Finset.univ : Finset (Fin dHead)) (fun j => (st.wv i j).isSome)) then
    throw "missing wv entries"
  if !finsetAll (Finset.univ : Finset (Fin dHead)) (fun j => (st.bv j).isSome) then
    throw "missing bv entries"
  if !finsetAll (Finset.univ : Finset (Fin dModel)) (fun i =>
      finsetAll (Finset.univ : Finset (Fin dHead)) (fun j => (st.wo i j).isSome)) then
    throw "missing wo entries"
  if !finsetAll (Finset.univ : Finset (Fin dModel)) (fun d => (st.attnBias d).isSome) then
    throw "missing attn_bias entries"
  if !finsetAll (Finset.univ : Finset (Fin dModel)) (fun d => (st.direction d).isSome) then
    throw "missing direction entries"
  let directionSpec ←
    match st.directionTarget, st.directionNegative with
    | some target, some negative => pure { target := target, negative := negative }
    | _, _ =>
        throw "direction metadata requires both direction-target and direction-negative"
  let defaultPrev : Fin seq := ⟨0, hpos⟩
  let prevFun : Fin seq → Fin seq := fun q =>
    (st.prev q).getD defaultPrev
  let embedFun : Fin seq → Fin dModel → Rat := fun q d =>
    (st.embed q d).getD 0
  let ln1GammaFun : Fin dModel → Rat := fun d =>
    (st.ln1Gamma d).getD 0
  let ln1BetaFun : Fin dModel → Rat := fun d =>
    (st.ln1Beta d).getD 0
  let wqFun : Fin dModel → Fin dHead → Rat := fun i j =>
    (st.wq i j).getD 0
  let bqFun : Fin dHead → Rat := fun j =>
    (st.bq j).getD 0
  let wkFun : Fin dModel → Fin dHead → Rat := fun i j =>
    (st.wk i j).getD 0
  let bkFun : Fin dHead → Rat := fun j =>
    (st.bk j).getD 0
  let wvFun : Fin dModel → Fin dHead → Rat := fun i j =>
    (st.wv i j).getD 0
  let bvFun : Fin dHead → Rat := fun j =>
    (st.bv j).getD 0
  let woFun : Fin dModel → Fin dHead → Rat := fun i j =>
    (st.wo i j).getD 0
  let attnBiasFun : Fin dModel → Rat := fun d =>
    (st.attnBias d).getD 0
  let directionFun : Fin dModel → Rat := fun d =>
    (st.direction d).getD 0
  let active :=
    if st.activeSeen then
      st.active
    else
      (Finset.univ : Finset (Fin seq)).erase defaultPrev
  pure
    { scale := scale
      active := active
      prev := prevFun
      embed := embedFun
      lnEps := lnEps
      ln1Gamma := ln1GammaFun
      ln1Beta := ln1BetaFun
      wq := wqFun
      bq := bqFun
      wk := wkFun
      bk := bkFun
      wv := wvFun
      bv := bvFun
      wo := woFun
      attnBias := attnBiasFun
      directionSpec := directionSpec
      direction := directionFun }

/-- Parse a raw induction head input payload from text. -/
def parseInductionHeadInputs (input : String) :
    Except String (Sigma (fun seq =>
      Sigma (fun dModel => Sigma (fun dHead => Model.InductionHeadInputs seq dModel dHead)))) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let mut seq? : Option Nat := none
  let mut dModel? : Option Nat := none
  let mut dHead? : Option Nat := none
  for t in tokens do
    match t with
    | ["seq", n] =>
        if seq?.isSome then
          throw "duplicate seq entry"
        else
          seq? := some (← parseNat n)
    | ["d_model", n] =>
        if dModel?.isSome then
          throw "duplicate d_model entry"
        else
          dModel? := some (← parseNat n)
    | ["d_head", n] =>
        if dHead?.isSome then
          throw "duplicate d_head entry"
        else
          dHead? := some (← parseNat n)
    | _ => pure ()
  let seq ←
    match seq? with
    | some v => pure v
    | none => throw "missing seq entry"
  let dModel ←
    match dModel? with
    | some v => pure v
    | none => throw "missing d_model entry"
  let dHead ←
    match dHead? with
    | some v => pure v
    | none => throw "missing d_head entry"
  match seq, dModel, dHead with
  | 0, _, _ => throw "seq must be positive"
  | _, 0, _ => throw "d_model must be positive"
  | _, _, 0 => throw "d_head must be positive"
  | Nat.succ n, Nat.succ m, Nat.succ h =>
      let seq := Nat.succ n
      let dModel := Nat.succ m
      let dHead := Nat.succ h
      let hpos : 0 < seq := Nat.succ_pos n
      let st0 : HeadParseState seq dModel dHead := initHeadState seq dModel dHead
      let st ← tokens.foldlM (fun st t =>
          match t with
          | ["seq", _] => pure st
          | ["d_model", _] => pure st
          | ["d_head", _] => pure st
          | _ => parseHeadLine st t) st0
      let inputs ← finalizeHeadState hpos st
      return ⟨seq, ⟨dModel, ⟨dHead, inputs⟩⟩⟩

/-- Raw downstream matrix payload with an input bound. -/
structure DownstreamMatrixRaw (rows cols : Nat) where
  /-- Input magnitude bound. -/
  inputBound : Rat
  /-- Matrix entries. -/
  entries : Fin rows → Fin cols → Rat

private structure DownstreamMatrixParseState (rows cols : Nat) where
  inputBound : Option Rat
  entries : Fin rows → Fin cols → Option Rat

private def initDownstreamMatrixState (rows cols : Nat) :
    DownstreamMatrixParseState rows cols :=
  { inputBound := none, entries := fun _ _ => none }

private def setRectEntry {rows cols : Nat} (mat : Fin rows → Fin cols → Option Rat)
    (i j : Nat) (v : Rat) : Except String (Fin rows → Fin cols → Option Rat) := do
  if hi : i < rows then
    if hj : j < cols then
      let iFin : Fin rows := ⟨i, hi⟩
      let jFin : Fin cols := ⟨j, hj⟩
      match mat iFin jFin with
      | some _ =>
          throw s!"duplicate matrix entry at ({i}, {j})"
      | none =>
          let mat' : Fin rows → Fin cols → Option Rat := fun i' j' =>
            if i' = iFin then
              if j' = jFin then
                some v
              else
                mat i' j'
            else
              mat i' j'
          return mat'
    else
      throw s!"index out of range: col={j}"
  else
    throw s!"index out of range: row={i}"

private def parseDownstreamMatrixLine {rows cols : Nat}
    (st : DownstreamMatrixParseState rows cols) (tokens : List String) :
    Except String (DownstreamMatrixParseState rows cols) := do
  match tokens with
  | ["input-bound", val] =>
      if st.inputBound.isSome then
        throw "duplicate input-bound entry"
      else
        return { st with inputBound := some (← parseRat val) }
  | ["w", i, j, val] =>
      let mat ← setRectEntry st.entries (← parseNat i) (← parseNat j) (← parseRat val)
      return { st with entries := mat }
  | _ =>
      throw s!"unrecognized line: '{String.intercalate " " tokens}'"

private def finalizeDownstreamMatrixState {rows cols : Nat}
    (st : DownstreamMatrixParseState rows cols) :
    Except String (DownstreamMatrixRaw rows cols) := do
  let inputBound ←
    match st.inputBound with
    | some v => pure v
    | none => throw "missing input-bound entry"
  if !finsetAll (Finset.univ : Finset (Fin rows)) (fun i =>
      finsetAll (Finset.univ : Finset (Fin cols)) (fun j => (st.entries i j).isSome)) then
    throw "missing matrix entries"
  let entries : Fin rows → Fin cols → Rat := fun i j =>
    (st.entries i j).getD 0
  return { inputBound := inputBound, entries := entries }

/-- Parse a downstream matrix payload from text. -/
def parseDownstreamMatrixRaw (input : String) :
    Except String (Sigma (fun rows => Sigma (fun cols => DownstreamMatrixRaw rows cols))) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let mut rows? : Option Nat := none
  let mut cols? : Option Nat := none
  for t in tokens do
    match t with
    | ["rows", n] =>
        if rows?.isSome then
          throw "duplicate rows entry"
        else
          rows? := some (← parseNat n)
    | ["cols", n] =>
        if cols?.isSome then
          throw "duplicate cols entry"
        else
          cols? := some (← parseNat n)
    | _ => pure ()
  let rows ←
    match rows? with
    | some v => pure v
    | none => throw "missing rows entry"
  let cols ←
    match cols? with
    | some v => pure v
    | none => throw "missing cols entry"
  match rows, cols with
  | 0, _ => throw "rows must be positive"
  | _, 0 => throw "cols must be positive"
  | Nat.succ r, Nat.succ c =>
      let rows := Nat.succ r
      let cols := Nat.succ c
      let st0 := initDownstreamMatrixState rows cols
      let st ← tokens.foldlM (fun st t =>
          match t with
          | ["rows", _] => pure st
          | ["cols", _] => pure st
          | _ => parseDownstreamMatrixLine st t) st0
      let raw ← finalizeDownstreamMatrixState st
      return ⟨rows, ⟨cols, raw⟩⟩

private structure ResidualBoundParseState (n : Nat) where
  bounds : Fin n → Option Rat

private def initResidualBoundState (n : Nat) : ResidualBoundParseState n :=
  { bounds := fun _ => none }

private def setVectorEntry {n : Nat} (bounds : Fin n → Option Rat)
    (i : Nat) (v : Rat) : Except String (Fin n → Option Rat) := do
  if hi : i < n then
    let iFin : Fin n := ⟨i, hi⟩
    match bounds iFin with
    | some _ =>
        throw s!"duplicate bound entry at index {i}"
    | none =>
        let bounds' : Fin n → Option Rat := fun i' =>
          if i' = iFin then
            some v
          else
            bounds i'
        return bounds'
  else
    throw s!"index out of range: {i}"

private def parseResidualBoundLine {n : Nat} (st : ResidualBoundParseState n)
    (tokens : List String) : Except String (ResidualBoundParseState n) := do
  match tokens with
  | ["bound", i, val] =>
      let bounds ← setVectorEntry st.bounds (← parseNat i) (← parseRat val)
      return { st with bounds := bounds }
  | ["dim", _] =>
      throw "duplicate dim entry"
  | _ =>
      throw s!"unrecognized line: '{String.intercalate " " tokens}'"

private def finalizeResidualBoundState {n : Nat} (st : ResidualBoundParseState n) :
    Except String (Circuit.ResidualBoundCert n) := do
  if !finsetAll (Finset.univ : Finset (Fin n)) (fun i => (st.bounds i).isSome) then
    throw "missing bound entries"
  let bound : Fin n → Rat := fun i =>
    (st.bounds i).getD 0
  return { bound := bound }

/-- Parse a residual-bound payload from text. -/
def parseResidualBoundCert (input : String) :
    Except String (Sigma (fun n => Circuit.ResidualBoundCert n)) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  match tokens with
  | [] => throw "empty residual-bound payload"
  | ["dim", nStr] :: rest =>
      let n ← parseNat nStr
      match n with
      | 0 => throw "dim must be positive"
      | Nat.succ n' =>
          let dim := Nat.succ n'
          let st0 := initResidualBoundState dim
          let st ← rest.foldlM (fun st t => parseResidualBoundLine st t) st0
          let cert ← finalizeResidualBoundState st
          return ⟨dim, cert⟩
  | _ => throw "expected header 'dim <n>'"

private structure ResidualIntervalParseState (n : Nat) where
  lo : Fin n → Option Rat
  hi : Fin n → Option Rat

private def initResidualIntervalState (n : Nat) : ResidualIntervalParseState n :=
  { lo := fun _ => none, hi := fun _ => none }

private def parseResidualIntervalLine {n : Nat} (st : ResidualIntervalParseState n)
    (tokens : List String) : Except String (ResidualIntervalParseState n) := do
  match tokens with
  | ["lo", i, val] =>
      let lo ← setVectorEntry st.lo (← parseNat i) (← parseRat val)
      return { st with lo := lo }
  | ["hi", i, val] =>
      let hi ← setVectorEntry st.hi (← parseNat i) (← parseRat val)
      return { st with hi := hi }
  | ["dim", _] =>
      throw "duplicate dim entry"
  | _ =>
      throw s!"unrecognized line: '{String.intercalate " " tokens}'"

private def finalizeResidualIntervalState {n : Nat} (st : ResidualIntervalParseState n) :
    Except String (Circuit.ResidualIntervalCert n) := do
  if !finsetAll (Finset.univ : Finset (Fin n)) (fun i => (st.lo i).isSome) then
    throw "missing lo entries"
  if !finsetAll (Finset.univ : Finset (Fin n)) (fun i => (st.hi i).isSome) then
    throw "missing hi entries"
  let lo : Fin n → Rat := fun i =>
    (st.lo i).getD 0
  let hi : Fin n → Rat := fun i =>
    (st.hi i).getD 0
  return { lo := lo, hi := hi }

/-- Parse a residual-interval payload from text. -/
def parseResidualIntervalCert (input : String) :
    Except String (Sigma (fun n => Circuit.ResidualIntervalCert n)) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  match tokens with
  | [] => throw "empty residual-interval payload"
  | ["dim", nStr] :: rest =>
      let n ← parseNat nStr
      match n with
      | 0 => throw "dim must be positive"
      | Nat.succ n' =>
          let dim := Nat.succ n'
          let st0 := initResidualIntervalState dim
          let st ← rest.foldlM (fun st t => parseResidualIntervalLine st t) st0
          let cert ← finalizeResidualIntervalState st
          return ⟨dim, cert⟩
  | _ => throw "expected header 'dim <n>'"

end Pure

end IO

end Nfp
