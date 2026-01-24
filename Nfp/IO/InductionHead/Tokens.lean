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

private def toIndex0 {seq : Nat} (label : String) (idx : Nat) : Except String (Fin seq) := do
  if h : idx < seq then
    return ⟨idx, h⟩
  else
    throw s!"{label} index out of range: {idx}"

private def setToken {seq : Nat} (st : ParseState seq) (q tok : Nat) :
    Except String (ParseState seq) := do
  let qFin ← toIndex0 (seq := seq) "q" q
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

end IO

end Nfp
