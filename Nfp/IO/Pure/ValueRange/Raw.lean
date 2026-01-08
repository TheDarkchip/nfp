-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Circuit.Cert.ValueRange
import Nfp.IO.Pure.ValueRange.Shared

/-!
Pure parsing helpers for raw value-range inputs.
-/

namespace Nfp

namespace IO

namespace Pure

open Nfp.Circuit

/-- Raw value-range payload without `lo`/`hi` bounds. -/
structure ValueRangeRaw (seq : Nat) where
  /-- Value entries. -/
  vals : Fin seq → Rat
  /-- Optional logit-diff direction metadata. -/
  direction : Option Circuit.DirectionSpec

private def finalizeValueRawState {seq : Nat} (st : ValueRange.ParseState seq) :
    Except String (ValueRangeRaw seq) := do
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
  return { vals := valsFun, direction := direction }

/-- Parse a raw value-range payload from text (ignores any `lo`/`hi`). -/
def parseValueRangeRaw (input : String) :
    Except String (Sigma ValueRangeRaw) := do
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
      let raw ← finalizeValueRawState st
      return ⟨seq, raw⟩

end Pure

end IO

end Nfp
