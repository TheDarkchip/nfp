-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Fintype.Basic
public import Nfp.IO.Parse.Basic

/-!
Untrusted parsing helpers for optional induction-head token lists.
-/

public section

namespace Nfp

namespace IO

open Nfp.IO.Parse

namespace InductionHeadTokens

/-- State for parsing token lists. -/
structure ParseState (seq : Nat) where
  /-- Optional per-position tokens. -/
  tokens : Array (Option Nat)

/-- Initialize a token parse state. -/
def initState (seq : Nat) : ParseState seq :=
  { tokens := Array.replicate seq none }

private def toIndex1 {seq : Nat} (label : String) (idx : Nat) : Except String (Fin seq) := do
  if idx = 0 then
    throw s!"{label} index must be >= 1"
  let idx' := idx - 1
  if h : idx' < seq then
    return ⟨idx', h⟩
  else
    throw s!"{label} index out of range: {idx}"

private def setToken {seq : Nat} (st : ParseState seq) (q tok : Nat) :
    Except String (ParseState seq) := do
  let qFin ← toIndex1 (seq := seq) "q" q
  match st.tokens[qFin.1]! with
  | some _ =>
      throw s!"duplicate token entry for q={q}"
  | none =>
      let tokens' := st.tokens.set! qFin.1 (some tok)
      return { st with tokens := tokens' }

/-- Parse a tokenized line into the token parse state. -/
def parseLine {seq : Nat} (st : ParseState seq) (tokens : List String) :
    Except String (ParseState seq) := do
  match tokens with
  | ["token", q, tok] =>
      setToken st (← parseNat q) (← parseNat tok)
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

/-- Finalize a token parse state into a total token map. -/
def finalizeState {seq : Nat} (st : ParseState seq) :
    Except String (Fin seq → Nat) := do
  if !st.tokens.all Option.isSome then
    throw "missing token entries"
  let tokensFun : Fin seq → Nat := fun q =>
    (st.tokens[q.1]!).getD 0
  pure tokensFun

end InductionHeadTokens

end IO

end Nfp
