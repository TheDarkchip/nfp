-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Circuit.Cert.ValueRange
import Nfp.IO.Pure.Basic

/-!
Shared parsing helpers for value-range payloads.
-/

namespace Nfp

namespace IO

namespace Pure

namespace ValueRange

open Nfp.Circuit

structure ParseState (seq : Nat) where
  lo : Option Dyadic
  hi : Option Dyadic
  vals : Fin seq → Option Dyadic
  directionTarget : Option Nat
  directionNegative : Option Nat


def initState (seq : Nat) : ParseState seq :=
  { lo := none
    hi := none
    vals := fun _ => none
    directionTarget := none
    directionNegative := none }


def setVal {seq : Nat} (st : ParseState seq) (k : Nat) (v : Dyadic) :
    Except String (ParseState seq) := do
  if hk : k < seq then
    let kFin : Fin seq := ⟨k, hk⟩
    match st.vals kFin with
    | some _ =>
        throw s!"duplicate value entry for k={k}"
    | none =>
        let vals' : Fin seq → Option Dyadic := fun k' =>
          if k' = kFin then
            some v
          else
            st.vals k'
        return { st with vals := vals' }
  else
    throw s!"value index out of range: k={k}"


def parseLine {seq : Nat} (st : ParseState seq)
    (tokens : List String) : Except String (ParseState seq) := do
  match tokens with
  | ["lo", val] =>
      if st.lo.isSome then
        throw "duplicate lo entry"
      else
        return { st with lo := some (← parseDyadic val) }
  | ["hi", val] =>
      if st.hi.isSome then
        throw "duplicate hi entry"
      else
        return { st with hi := some (← parseDyadic val) }
  | ["val", k, val] =>
      setVal st (← parseNat k) (← parseDyadic val)
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

end ValueRange

end Pure

end IO

end Nfp
