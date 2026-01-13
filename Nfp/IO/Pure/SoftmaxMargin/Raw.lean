-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Circuit.Cert.SoftmaxMargin
import Nfp.IO.Pure.SoftmaxMargin.Shared

/-!
Pure parsing helpers for raw softmax-margin inputs.
-/

namespace Nfp

namespace IO

namespace Pure

open Nfp.Circuit

/-- Raw softmax-margin payload without `eps`/`margin`. -/
structure SoftmaxMarginRaw (seq : Nat) where
  /-- Active queries for which bounds are required. -/
  active : Finset (Fin seq)
  /-- `prev` selector for induction-style attention. -/
  prev : Fin seq → Fin seq
  /-- Score matrix entries. -/
  scores : Fin seq → Fin seq → Rat
  /-- Attention weight entries. -/
  weights : Fin seq → Fin seq → Rat

private def finalizeRawState {seq : Nat} (hpos : 0 < seq)
    (st : SoftmaxMargin.ParseState seq) : Except String (SoftmaxMarginRaw seq) := do
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
    { active := active
      prev := prevFun
      scores := scoresFun
      weights := weightsFun }

/-- Parse a raw softmax-margin payload from text (ignores any `eps`/`margin`). -/
def parseSoftmaxMarginRaw (input : String) :
    Except String (Sigma SoftmaxMarginRaw) := do
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
      let raw ← finalizeRawState hpos st
      return ⟨seq, raw⟩

end Pure

end IO

end Nfp
