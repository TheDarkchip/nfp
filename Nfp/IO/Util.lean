-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.Pure

/-!
Small shared helpers for IO parsing.
-/

public section

namespace Nfp

namespace IO

/-- Parse an optional rational literal for CLI flags (rounded down if needed). -/
def parseRatOpt (label : String) (raw? : Option String) :
    Except String (Option Rat) :=
  match raw? with
  | none => Except.ok none
  | some raw =>
      match Pure.parseRat raw with
      | Except.ok v => Except.ok (some v)
      | Except.error msg => Except.error s!"invalid {label}: {msg}"

/-- Emit a deprecation warning on stderr. -/
def warnDeprecated (msg : String) : IO Unit := do
  IO.eprintln s!"warning: DEPRECATED: {msg}"

end IO

end Nfp
