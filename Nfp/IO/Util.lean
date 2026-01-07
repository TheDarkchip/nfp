-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.IO.Pure

/-!
Small shared helpers for IO parsing.
-/

namespace Nfp

namespace IO

/-- Parse an optional dyadic literal for CLI flags (rounded down if needed). -/
def parseDyadicOpt (label : String) (raw? : Option String) :
    Except String (Option Dyadic) :=
  match raw? with
  | none => Except.ok none
  | some raw =>
      match Pure.parseDyadic raw with
      | Except.ok v => Except.ok (some v)
      | Except.error msg => Except.error s!"invalid {label}: {msg}"

end IO

end Nfp
