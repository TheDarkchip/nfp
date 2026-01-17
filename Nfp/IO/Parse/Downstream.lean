-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Cert.DownstreamLinear
public import Nfp.IO.Parse.Basic

/-!
Parse parsing helpers for downstream linear certificates.
-/

public section

namespace Nfp

namespace IO

namespace Parse

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

end Parse

end IO

end Nfp
