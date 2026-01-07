-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Circuit.Cert.ResidualBound
import Nfp.Circuit.Cert.ResidualInterval
import Nfp.IO.Pure.Basic

/-!
Pure parsing helpers for residual-bound and residual-interval certificates.
-/

namespace Nfp

namespace IO

namespace Pure

open Nfp.Circuit

private structure ResidualBoundParseState (n : Nat) where
  bounds : Fin n → Option Dyadic

private def initResidualBoundState (n : Nat) : ResidualBoundParseState n :=
  { bounds := fun _ => none }

private def setVectorEntry {n : Nat} (bounds : Fin n → Option Dyadic)
    (i : Nat) (v : Dyadic) : Except String (Fin n → Option Dyadic) := do
  if hi : i < n then
    let iFin : Fin n := ⟨i, hi⟩
    match bounds iFin with
    | some _ =>
        throw s!"duplicate bound entry at index {i}"
    | none =>
        let bounds' : Fin n → Option Dyadic := fun i' =>
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
      let bounds ← setVectorEntry st.bounds (← parseNat i) (← parseDyadic val)
      return { st with bounds := bounds }
  | ["dim", _] =>
      throw "duplicate dim entry"
  | _ =>
      throw s!"unrecognized line: '{String.intercalate " " tokens}'"

private def finalizeResidualBoundState {n : Nat} (st : ResidualBoundParseState n) :
    Except String (Circuit.ResidualBoundCert n) := do
  if !finsetAll (Finset.univ : Finset (Fin n)) (fun i => (st.bounds i).isSome) then
    throw "missing bound entries"
  let bound : Fin n → Dyadic := fun i =>
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
  lo : Fin n → Option Dyadic
  hi : Fin n → Option Dyadic

private def initResidualIntervalState (n : Nat) : ResidualIntervalParseState n :=
  { lo := fun _ => none, hi := fun _ => none }

private def parseResidualIntervalLine {n : Nat} (st : ResidualIntervalParseState n)
    (tokens : List String) : Except String (ResidualIntervalParseState n) := do
  match tokens with
  | ["lo", i, val] =>
      let lo ← setVectorEntry st.lo (← parseNat i) (← parseDyadic val)
      return { st with lo := lo }
  | ["hi", i, val] =>
      let hi ← setVectorEntry st.hi (← parseNat i) (← parseDyadic val)
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
  let lo : Fin n → Dyadic := fun i =>
    (st.lo i).getD 0
  let hi : Fin n → Dyadic := fun i =>
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
