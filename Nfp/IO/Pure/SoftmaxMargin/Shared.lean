-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Finset.Insert
import Nfp.IO.Pure.Basic

/-!
Shared parsing helpers for softmax-margin payloads.
-/

namespace Nfp

namespace IO

namespace Pure

namespace SoftmaxMargin

/-- State for parsing softmax-margin payloads. -/
structure ParseState (seq : Nat) where
  /-- Optional epsilon bound. -/
  eps : Option Dyadic
  /-- Optional margin bound. -/
  margin : Option Dyadic
  /-- Active query set. -/
  active : Finset (Fin seq)
  /-- Whether any active entries were parsed. -/
  activeSeen : Bool
  /-- Optional predecessor pointer per query. -/
  prev : Fin seq → Option (Fin seq)
  /-- Optional score matrix entries. -/
  scores : Fin seq → Fin seq → Option Dyadic
  /-- Optional weight matrix entries. -/
  weights : Fin seq → Fin seq → Option Dyadic

/-- Initialize a softmax-margin parse state. -/
def initState (seq : Nat) : ParseState seq :=
  { eps := none
    margin := none
    active := ∅
    activeSeen := false
    prev := fun _ => none
    scores := fun _ _ => none
    weights := fun _ _ => none }

/-- Set a predecessor entry from `(q, k)` tokens. -/
def setPrev {seq : Nat} (st : ParseState seq) (q k : Nat) : Except String (ParseState seq) := do
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

/-- Mark an active query index. -/
def setActive {seq : Nat} (st : ParseState seq) (q : Nat) : Except String (ParseState seq) := do
  if hq : q < seq then
    let qFin : Fin seq := ⟨q, hq⟩
    if qFin ∈ st.active then
      throw s!"duplicate active entry for q={q}"
    else
      return { st with active := insert qFin st.active, activeSeen := true }
  else
    throw s!"active index out of range: q={q}"

/-- Insert a matrix entry for scores/weights. -/
def setMatrixEntry {seq : Nat} (mat : Fin seq → Fin seq → Option Dyadic)
    (q k : Nat) (v : Dyadic) : Except String (Fin seq → Fin seq → Option Dyadic) := do
  if hq : q < seq then
    if hk : k < seq then
      let qFin : Fin seq := ⟨q, hq⟩
      let kFin : Fin seq := ⟨k, hk⟩
      match mat qFin kFin with
      | some _ =>
          throw s!"duplicate matrix entry at ({q}, {k})"
      | none =>
          let mat' : Fin seq → Fin seq → Option Dyadic := fun q' k' =>
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

/-- Parse a tokenized line into the softmax-margin parse state. -/
def parseLine {seq : Nat} (st : ParseState seq)
    (tokens : List String) : Except String (ParseState seq) := do
  match tokens with
  | ["eps", val] =>
      if st.eps.isSome then
        throw "duplicate eps entry"
      else
        return { st with eps := some (← parseDyadic val) }
  | ["margin", val] =>
      if st.margin.isSome then
        throw "duplicate margin entry"
      else
        return { st with margin := some (← parseDyadic val) }
  | ["active", q] =>
      setActive st (← parseNat q)
  | ["prev", q, k] =>
      setPrev st (← parseNat q) (← parseNat k)
  | ["score", q, k, val] =>
      let mat ← setMatrixEntry st.scores (← parseNat q) (← parseNat k) (← parseDyadic val)
      return { st with scores := mat }
  | ["weight", q, k, val] =>
      let mat ← setMatrixEntry st.weights (← parseNat q) (← parseNat k) (← parseDyadic val)
      return { st with weights := mat }
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

end SoftmaxMargin

end Pure

end IO

end Nfp
