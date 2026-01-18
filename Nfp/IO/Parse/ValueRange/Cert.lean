-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Cert.ValueRange
public import Nfp.IO.Parse.ValueRange.Shared

/-!
Parsing helpers for value-range certificates.
-/

public section

namespace Nfp

namespace IO

namespace Parse

open Nfp.Circuit

private def finalizeValueState {seq : Nat} (st : ValueRange.ParseState seq) :
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
  let direction ←
    match st.directionTarget, st.directionNegative with
    | none, none => pure none
    | some target, some negative =>
        pure (some { target := target, negative := negative })
    | _, _ =>
        throw "direction metadata requires both direction-target and direction-negative"
  return { lo := lo, hi := hi, vals := valsFun, direction := direction }

/-- Parse a value-range certificate from a text payload. -/
def parseValueRangeCert (input : String) :
    Except String (Sigma ValueRangeCert) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let seq ← ValueRange.parseSeq tokens
  match seq with
  | 0 => throw "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let st0 : ValueRange.ParseState seq := ValueRange.initState seq
      let st ← tokens.foldlM (fun st t =>
          match t with
          | ["seq", _] => pure st
          | _ => ValueRange.parseLine st t) st0
      let cert ← finalizeValueState st
      return ⟨seq, cert⟩

end Parse

end IO

end Nfp
