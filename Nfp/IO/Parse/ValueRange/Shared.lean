-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Cert.ValueRange
public import Nfp.IO.Parse.Basic

/-!
Shared parsing helpers for value-range payloads.

All sequence indices in the payload are 1-based (literature convention) and are converted to
`Fin` indices internally.
-/

public section

namespace Nfp

namespace IO

namespace Parse

namespace ValueRange

open Nfp.Circuit

/-- State for parsing value-range payloads. -/
structure ParseState (seq : Nat) where
  /-- Optional lower bound. -/
  lo : Option Rat
  /-- Optional upper bound. -/
  hi : Option Rat
  /-- Optional per-position values. -/
  vals : Fin seq → Option Rat
  /-- Optional direction target index. -/
  directionTarget : Option Nat
  /-- Optional direction negative index. -/
  directionNegative : Option Nat


/-- Initialize a value-range parse state. -/
def initState (seq : Nat) : ParseState seq :=
  { lo := none
    hi := none
    vals := fun _ => none
    directionTarget := none
    directionNegative := none }


/-- Set a value entry from `(k, v)` tokens. -/
def setVal {seq : Nat} (st : ParseState seq) (k : Nat) (v : Rat) :
    Except String (ParseState seq) := do
  if k = 0 then
    throw "value index must be >= 1"
  let k' := k - 1
  if hk : k' < seq then
    let kFin : Fin seq := ⟨k', hk⟩
    match st.vals kFin with
    | some _ =>
        throw s!"duplicate value entry for k={k}"
    | none =>
        let vals' : Fin seq → Option Rat := fun k'' =>
          if k'' = kFin then
            some v
          else
            st.vals k''
        return { st with vals := vals' }
  else
    throw s!"value index out of range: k={k}"


/-- Parse a tokenized line into the value-range parse state. -/
def parseLine {seq : Nat} (st : ParseState seq)
    (tokens : List String) : Except String (ParseState seq) := do
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

end ValueRange

end Parse

end IO

end Nfp
