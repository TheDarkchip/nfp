-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.IO.Parse

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
      match Parse.parseRat raw with
      | Except.ok v => Except.ok (some v)
      | Except.error msg => Except.error s!"invalid {label}: {msg}"

/-- Emit a deprecation warning on stderr. -/
def warnDeprecated (msg : String) : IO Unit := do
  IO.eprintln s!"warning: DEPRECATED: {msg}"

/-- Conditionally time an IO action and log the duration in ms. -/
def timeIO {α} (enabled : Bool) (label : String) (act : Unit → IO α) : IO α := do
  if enabled then
    let t0 ← IO.monoMsNow
    let out ← act ()
    let t1 ← IO.monoMsNow
    IO.eprintln s!"info: {label}-ms {t1 - t0}"
    return out
  else
    act ()

end IO

end Nfp
