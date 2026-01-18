-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public meta import Lean.Elab.Command
public meta import Lean.Linter.Basic

/-!
Syntax linter forbidding heartbeat budget options.
-/

meta section

namespace Nfp
namespace Linter

open Lean Parser Elab Command Linter

/-- Enable the no-heartbeats linter. -/
public register_option linter.nfp.noHeartbeats : Bool := {
  defValue := false
  descr := "enable the noHeartbeats linter"
}

namespace NoHeartbeats

/-- Return the option name if syntax is a `set_option` command, term, or tactic. -/
def parseSetOption : Syntax → Option Name
  | `(command|set_option $name:ident $_val) => some name.getId
  | `(set_option $name:ident $_val in $_x) => some name.getId
  | `(tactic|set_option $name:ident $_val in $_x) => some name.getId
  | _ => none

/-- True if the option is a heartbeat budget. -/
def isHeartbeatOption (name : Name) : Bool :=
  name == `maxHeartbeats || name == `synthInstance.maxHeartbeats

/-- Linter that forbids heartbeat budget options in this repository. -/
def noHeartbeatsLinter : Linter where
  run stx := do
    unless getLinterValue linter.nfp.noHeartbeats (← getLinterOptions) do
      return
    if (← MonadState.get).messages.hasErrors then
      return
    if let some head := stx.find? (fun stx => (parseSetOption stx).isSome) then
      if let some name := parseSetOption head then
        if isHeartbeatOption name then
          logLint linter.nfp.noHeartbeats head
            m!"Setting option '{name}' is forbidden; refactor the proof instead."

initialize addLinter noHeartbeatsLinter

end NoHeartbeats

end Linter
end Nfp
