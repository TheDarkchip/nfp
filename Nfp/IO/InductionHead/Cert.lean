-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Finset.Insert
public import Nfp.Circuit.Cert.InductionHead
public import Nfp.Circuit.Cert.LogitDiff
public import Nfp.IO.Parse.Basic
public import Nfp.IO.Util

/-!
Untrusted parsing and checking for explicit induction-head certificates.

All sequence indices in the certificate payload are 1-based (literature convention) and
are converted to `Fin` indices internally.
-/

public section

namespace Nfp

namespace IO

open Nfp.Circuit
open Nfp.IO.Parse

namespace InductionHeadCert

/-- State for parsing induction-head certificates. -/
structure ParseState (seq : Nat) where
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

/-- Initialize a parse state. -/
def initState (seq : Nat) : ParseState seq :=
  let row : Array (Option Rat) := Array.replicate seq none
  { eps := none
    margin := none
    active := ∅
    activeSeen := false
    prev := Array.replicate seq none
    scores := Array.replicate seq row
    weights := Array.replicate seq row
    epsAt := Array.replicate seq none
    weightBoundAt := Array.replicate seq row
    lo := none
    hi := none
    valsLo := Array.replicate seq none
    valsHi := Array.replicate seq none
    vals := Array.replicate seq none
    directionTarget := none
    directionNegative := none }

private def toIndex1 {seq : Nat} (label : String) (idx : Nat) : Except String (Fin seq) := do
  if idx = 0 then
    throw s!"{label} index must be >= 1"
  let idx' := idx - 1
  if h : idx' < seq then
    return ⟨idx', h⟩
  else
    throw s!"{label} index out of range: {idx}"

private def setActive {seq : Nat} (st : ParseState seq) (q : Nat) :
    Except String (ParseState seq) := do
  let qFin ← toIndex1 (seq := seq) "q" q
  if qFin ∈ st.active then
    throw s!"duplicate active entry for q={q}"
  else
    return { st with active := insert qFin st.active, activeSeen := true }

private def setPrev {seq : Nat} (st : ParseState seq) (q k : Nat) :
    Except String (ParseState seq) := do
  let qFin ← toIndex1 (seq := seq) "q" q
  let kFin ← toIndex1 (seq := seq) "k" k
  match st.prev[qFin.1]! with
  | some _ =>
      throw s!"duplicate prev entry for q={q}"
  | none =>
      let prev' := st.prev.set! qFin.1 (some kFin)
      return { st with prev := prev' }

private def setVecEntry {seq : Nat} (arr : Array (Option Rat)) (idx : Nat) (v : Rat) :
    Except String (Array (Option Rat)) := do
  let kFin ← toIndex1 (seq := seq) "k" idx
  match arr[kFin.1]! with
  | some _ =>
      throw s!"duplicate entry for k={idx}"
  | none =>
      return arr.set! kFin.1 (some v)

private def setMatrixEntry {seq : Nat} (mat : Array (Array (Option Rat)))
    (q k : Nat) (v : Rat) : Except String (Array (Array (Option Rat))) := do
  let qFin ← toIndex1 (seq := seq) "q" q
  let kFin ← toIndex1 (seq := seq) "k" k
  let row := mat[qFin.1]!
  match row[kFin.1]! with
  | some _ =>
      throw s!"duplicate matrix entry at ({q}, {k})"
  | none =>
      let row' := row.set! kFin.1 (some v)
      return mat.set! qFin.1 row'

/-- Parse a tokenized line into the parse state. -/
def parseLine {seq : Nat} (st : ParseState seq) (tokens : List String) :
    Except String (ParseState seq) := do
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
      let mat ← setMatrixEntry (seq := seq) st.scores (← parseNat q) (← parseNat k)
        (← parseRat val)
      return { st with scores := mat }
  | ["weight", q, k, val] =>
      let mat ← setMatrixEntry (seq := seq) st.weights (← parseNat q) (← parseNat k)
        (← parseRat val)
      return { st with weights := mat }
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

private def finalizeState {seq : Nat} (hpos : 0 < seq) (st : ParseState seq) :
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

end InductionHeadCert

/-- Parse an explicit induction-head certificate from a text payload. -/
def parseInductionHeadCert (input : String) :
    Except String (Sigma Circuit.InductionHeadCert) := do
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
      let cert ← InductionHeadCert.finalizeState hpos st
      return ⟨seq, cert⟩

/-- Load an induction-head certificate from disk. -/
def loadInductionHeadCert (path : System.FilePath) :
    IO (Except String (Sigma Circuit.InductionHeadCert)) := do
  let data ← IO.FS.readFile path
  return parseInductionHeadCert data

private def ratToString (x : Rat) : String :=
  toString x

/-- Check an explicit induction-head certificate from disk. -/
def runInductionHeadCertCheck (certPath : System.FilePath)
    (minActive? : Option Nat) (minLogitDiffStr? : Option String)
    (minMarginStr? : Option String) (maxEpsStr? : Option String) : IO UInt32 := do
  let minLogitDiff?E := parseRatOpt "min-logit-diff" minLogitDiffStr?
  let minMargin?E := parseRatOpt "min-margin" minMarginStr?
  let maxEps?E := parseRatOpt "max-eps" maxEpsStr?
  match minLogitDiff?E, minMargin?E, maxEps?E with
  | Except.error msg, _, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, Except.error msg, _ =>
      IO.eprintln s!"error: {msg}"
      return 2
  | _, _, Except.error msg =>
      IO.eprintln s!"error: {msg}"
      return 2
  | Except.ok minLogitDiff?, Except.ok minMargin?, Except.ok maxEps? =>
      let minMargin := minMargin?.getD (0 : Rat)
      let maxEps := maxEps?.getD (ratRoundDown (Rat.divInt 1 2))
      let parsed ← loadInductionHeadCert certPath
      match parsed with
      | Except.error msg =>
          IO.eprintln s!"error: {msg}"
          return 1
      | Except.ok ⟨seq, cert⟩ =>
          match seq with
          | 0 =>
              IO.eprintln "error: seq must be positive"
              return 2
          | Nat.succ n =>
              let seq := Nat.succ n
              let _ : NeZero seq := ⟨by simp⟩
              let ok := Circuit.checkInductionHeadCert cert
              if !ok then
                IO.eprintln "error: induction-head certificate rejected"
                return 2
              let activeCount := cert.active.card
              let defaultMinActive := max 1 (seq / 8)
              let minActive := minActive?.getD defaultMinActive
              if activeCount < minActive then
                IO.eprintln
                  s!"error: active queries {activeCount} below minimum {minActive}"
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
              let effectiveMinLogitDiff :=
                match minLogitDiff?, cert.values.direction with
                | some v, _ => some v
                | none, some _ => some (0 : Rat)
                | none, none => none
              match effectiveMinLogitDiff with
              | none =>
                  IO.println
                    s!"ok: induction head certificate checked \
                    (seq={seq}, active={activeCount}, \
                    margin={ratToString cert.margin}, eps={ratToString cert.eps})"
                  return 0
              | some minLogitDiff =>
                  let logitDiffLB? :=
                    Circuit.logitDiffLowerBoundAt cert.active cert.prev cert.epsAt
                      cert.values.lo cert.values.hi cert.values.vals
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
                        IO.println
                          s!"ok: induction head certificate checked \
                          (seq={seq}, active={activeCount}, \
                          margin={ratToString cert.margin}, eps={ratToString cert.eps}, \
                          logitDiffLB={ratToString logitDiffLB})"
                        return 0

end IO

end Nfp
