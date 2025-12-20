-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Decimal

namespace Nfp.Sound

/-!
# Model header helpers (SOUND)

Pure parsing utilities for extracting trusted metadata (e.g., LayerNorm epsilon)
from `NFP_TEXT` model headers.
-/

/-- Parse `key=value` header lines. -/
def parseHeaderLine (line : String) : Option (String × String) :=
  let line := line.trim
  if line.isEmpty then none
  else
    match line.splitOn "=" with
    | [k, v] => some (k.trim, v.trim)
    | _ => none

/-- Parse `layer_norm_eps` (or `eps`) from a `NFP_TEXT` header. -/
def parseTextHeaderEps (lines : Array String) : Except String Rat :=
  Id.run do
    let mut i : Nat := 0
    while i < lines.size && lines[i]!.trim.isEmpty do
      i := i + 1
    if !(i < lines.size) then
      return .error "empty model file"
    let headerTag := lines[i]!.trim
    if !headerTag.startsWith "NFP_TEXT" then
      return .error s!"unexpected header '{headerTag}'"
    i := i + 1
    let mut eps? : Option Rat := none
    while i < lines.size do
      let line := lines[i]!.trim
      if line.isEmpty then
        i := lines.size
      else
        match parseHeaderLine line with
        | none =>
            i := i + 1
        | some (k, v) =>
            if k = "layer_norm_eps" || k = "eps" then
              match parseRat v with
              | .error e => return .error s!"invalid layer_norm_eps '{v}': {e}"
              | .ok r => eps? := some r
            i := i + 1
    match eps? with
    | some eps => return .ok eps
    | none => return .error "missing layer_norm_eps"

/-! ### Specs -/

theorem parseHeaderLine_spec : parseHeaderLine = parseHeaderLine := rfl
theorem parseTextHeaderEps_spec : parseTextHeaderEps = parseTextHeaderEps := rfl

end Nfp.Sound
