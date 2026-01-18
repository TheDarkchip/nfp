-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Cert.SoftmaxMargin
public import Nfp.IO.Parse.SoftmaxMargin.Shared

/-!
Parsing helpers for softmax-margin certificates.
-/

public section

namespace Nfp

namespace IO

namespace Parse

open Nfp.Circuit

private def finalizeState {seq : Nat} (hpos : 0 < seq)
    (st : SoftmaxMargin.ParseState seq) : Except String (SoftmaxMarginCert seq) := do
  let eps ←
    match st.eps with
    | some v => pure v
    | none => throw "missing eps entry"
  let margin ←
    match st.margin with
    | some v => pure v
    | none => throw "missing margin entry"
  if !st.prev.all Option.isSome then
    throw "missing prev entries"
  if !st.scores.all (fun row => row.all Option.isSome) then
    throw "missing score entries"
  if !st.weights.all (fun row => row.all Option.isSome) then
    throw "missing weight entries"
  let defaultPrev : Fin seq := ⟨0, hpos⟩
  let prevFun : Fin seq → Fin seq := fun q =>
    (st.prev[q.1]!).getD defaultPrev
  let scoresFun : Fin seq → Fin seq → Rat := fun q k =>
    let row := st.scores[q.1]!
    (row[k.1]!).getD 0
  let weightsFun : Fin seq → Fin seq → Rat := fun q k =>
    let row := st.weights[q.1]!
    (row[k.1]!).getD 0
  let active :=
    if st.activeSeen then
      st.active
    else
      (Finset.univ : Finset (Fin seq)).erase defaultPrev
  pure
    { eps := eps
      margin := margin
      active := active
      prev := prevFun
      scores := scoresFun
      weights := weightsFun }

/-- Parse a softmax-margin certificate from a text payload. -/
def parseSoftmaxMarginCert (input : String) :
    Except String (Sigma SoftmaxMarginCert) := do
  let lines := input.splitOn "\n"
  let tokens := lines.filterMap cleanTokens
  let seq ← SoftmaxMargin.parseSeq tokens
  match seq with
  | 0 => throw "seq must be positive"
  | Nat.succ n =>
      let seq := Nat.succ n
      let hpos : 0 < seq := Nat.succ_pos n
      let st0 : SoftmaxMargin.ParseState seq := SoftmaxMargin.initState seq
      let st ← tokens.foldlM (fun st t =>
          match t with
          | ["seq", _] => pure st
          | _ => SoftmaxMargin.parseLine st t) st0
      let cert ← finalizeState hpos st
      return ⟨seq, cert⟩

end Parse

end IO

end Nfp
