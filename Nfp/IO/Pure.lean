-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Rat
import Nfp.Circuit.Cert.SoftmaxMargin
import Nfp.Circuit.Cert.ValueRange

/-!
Pure parsing helpers for softmax-margin and value-range certificates.
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

private def parseRat (s : String) : Except String Rat := do
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
  prev : Fin seq → Option (Fin seq)
  scores : Fin seq → Fin seq → Option Rat
  weights : Fin seq → Fin seq → Option Rat

private def initState (seq : Nat) : SoftmaxMarginParseState seq :=
  { eps := none
    margin := none
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
  pure
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

private structure ValueRangeParseState (seq : Nat) where
  lo : Option Rat
  hi : Option Rat
  vals : Fin seq → Option Rat

private def initValueRangeState (seq : Nat) : ValueRangeParseState seq :=
  { lo := none
    hi := none
    vals := fun _ => none }

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
  return { lo := lo, hi := hi, vals := valsFun }

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

end Pure

end IO

end Nfp
