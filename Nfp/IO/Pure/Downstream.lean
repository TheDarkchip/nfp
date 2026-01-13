-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Cert.DownstreamLinear
public import Nfp.IO.Pure.Basic

/-!
Pure parsing helpers for downstream linear and matrix payloads.
-/

public section

namespace Nfp

namespace IO

namespace Pure

open Nfp.Circuit

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

private def initPrevOpt (n : Nat) : Array (Option (Fin n)) :=
  Array.replicate n none

private def initActiveBits (n : Nat) : Array Bool :=
  Array.replicate n false

private def activeFromBits {n : Nat} (bits : Array Bool) : Finset (Fin n) :=
  (Finset.univ : Finset (Fin n)).filter (fun i => bits.getD i.1 false)

private def arrayAllSome {α : Type} (arr : Array (Option α)) : Bool :=
  (List.range arr.size).all (fun i => (arr.getD i none).isSome)

private def matAllSome {α : Type} (mat : Array (Array (Option α))) : Bool :=
  (List.range mat.size).all (fun i => arrayAllSome (mat.getD i #[]))

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

end Pure

end IO

end Nfp
