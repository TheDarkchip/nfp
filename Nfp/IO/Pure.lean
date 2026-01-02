-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Rat
import Nfp.Circuit.Cert.SoftmaxMargin

/-!
Pure parsing helpers for softmax-margin certificates.
-/

namespace Nfp

namespace IO

namespace Pure

open Nfp.Circuit

private def splitWords (line : String) : List String :=
  line.split (fun c => c = ' ' || c = '\t') |>.filter (· ≠ "")

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

private def parseRat (s : String) : Except String Rat :=
  match s.splitOn "/" with
  | [num] => return Rat.ofInt (← parseInt num)
  | [num, den] =>
      let n ← parseInt num
      let d ← parseNat den
      if d = 0 then
        throw s!"invalid rational '{s}': zero denominator"
      else
        return Rat.ofInt n / Rat.ofInt (Int.ofNat d)
  | _ => throw s!"invalid rational '{s}'"

private structure SoftmaxMarginParseState (seq : Nat) where
  eps : Option Rat
  margin : Option Rat
  prev : Array (Option (Fin seq))
  scores : Array (Array (Option Rat))
  weights : Array (Array (Option Rat))

private def initState (seq : Nat) : SoftmaxMarginParseState seq :=
  let prev := Array.mkArray seq none
  let row : Array (Option Rat) := Array.mkArray seq none
  let mat : Array (Array (Option Rat)) := Array.mkArray seq row
  { eps := none
    margin := none
    prev := prev
    scores := mat
    weights := mat }

private def setPrev {seq : Nat} (st : SoftmaxMarginParseState seq)
    (q k : Nat) : Except String (SoftmaxMarginParseState seq) := do
  if hq : q < seq then
    if hk : k < seq then
      let qFin : Fin seq := ⟨q, hq⟩
      let kFin : Fin seq := ⟨k, hk⟩
      match st.prev.get qFin with
      | some _ =>
          throw s!"duplicate prev entry for q={q}"
      | none =>
          let prev' := st.prev.set qFin (some kFin)
          return { st with prev := prev' }
    else
      throw s!"prev index out of range: k={k}"
  else
    throw s!"prev index out of range: q={q}"

private def setMatrixEntry {seq : Nat} (mat : Array (Array (Option Rat)))
    (q k : Nat) (v : Rat) : Except String (Array (Array (Option Rat))) := do
  if hq : q < seq then
    if hk : k < seq then
      let qFin : Fin seq := ⟨q, hq⟩
      let kFin : Fin seq := ⟨k, hk⟩
      let row := mat.get qFin
      match row.get kFin with
      | some _ =>
          throw s!"duplicate matrix entry at ({q}, {k})"
      | none =>
          let row' := row.set kFin (some v)
          return mat.set qFin row'
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

private def allSomeArray {α : Type} (arr : Array (Option α)) : Bool :=
  arr.all (fun v => v.isSome)

private def allSomeMatrix {α : Type} (mat : Array (Array (Option α))) : Bool :=
  mat.all (fun row => row.all (fun v => v.isSome))

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
  if !allSomeArray st.prev then
    throw "missing prev entries"
  if !allSomeMatrix st.scores then
    throw "missing score entries"
  if !allSomeMatrix st.weights then
    throw "missing weight entries"
  let defaultPrev : Fin seq := ⟨0, hpos⟩
  let prevFun : Fin seq → Fin seq := fun q =>
    (st.prev.get q).getD defaultPrev
  let scoresFun : Fin seq → Fin seq → Rat := fun q k =>
    (st.scores.get q).get k |>.getD 0
  let weightsFun : Fin seq → Fin seq → Rat := fun q k =>
    (st.weights.get q).get k |>.getD 0
  return
    { eps := eps
      margin := margin
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

end Pure

end IO

end Nfp
