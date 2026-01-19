-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Finset.Insert
public import Nfp.IO.Parse.Basic

/-!
Shared parsing helpers for softmax-margin payloads.

All sequence indices in the payload are 0-based (file-format convention) and are converted to
`Fin` indices internally.
-/

public section

namespace Nfp

namespace IO

namespace Parse

namespace SoftmaxMargin

/-- State for parsing softmax-margin payloads. -/
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

/-- Initialize a softmax-margin parse state. -/
def initState (seq : Nat) : ParseState seq :=
  let row : Array (Option Rat) := Array.replicate seq none
  { eps := none
    margin := none
    active := ∅
    activeSeen := false
    prev := Array.replicate seq none
    scores := Array.replicate seq row
    weights := Array.replicate seq row }

private def toIndex0 {seq : Nat} (label : String) (idx : Nat) : Except String (Fin seq) := do
  if h : idx < seq then
    return ⟨idx, h⟩
  else
    throw s!"{label} index out of range: {idx}"

/-- Set a predecessor entry from `(q, k)` tokens. -/
def setPrev {seq : Nat} (st : ParseState seq) (q k : Nat) : Except String (ParseState seq) := do
  let qFin ← toIndex0 (seq := seq) "q" q
  let kFin ← toIndex0 (seq := seq) "k" k
  match st.prev[qFin.1]! with
  | some _ =>
      throw s!"duplicate prev entry for q={q}"
  | none =>
      let prev' := st.prev.set! qFin.1 (some kFin)
      return { st with prev := prev' }

/-- Mark an active query index. -/
def setActive {seq : Nat} (st : ParseState seq) (q : Nat) : Except String (ParseState seq) := do
  let qFin ← toIndex0 (seq := seq) "q" q
  if qFin ∈ st.active then
    throw s!"duplicate active entry for q={q}"
  else
    return { st with active := insert qFin st.active, activeSeen := true }

/-- Insert a matrix entry for scores/weights. -/
def setMatrixEntry {seq : Nat} (mat : Array (Array (Option Rat)))
    (q k : Nat) (v : Rat) : Except String (Array (Array (Option Rat))) := do
  let qFin ← toIndex0 (seq := seq) "q" q
  let kFin ← toIndex0 (seq := seq) "k" k
  let row := mat[qFin.1]!
  match row[kFin.1]! with
  | some _ =>
      throw s!"duplicate matrix entry at ({q}, {k})"
  | none =>
      let row' := row.set! kFin.1 (some v)
      let mat' := mat.set! qFin.1 row'
      return mat'

/-- Parse a tokenized line into the softmax-margin parse state. -/
def parseLine {seq : Nat} (st : ParseState seq)
    (tokens : List String) : Except String (ParseState seq) := do
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

end Parse

end IO

end Nfp
