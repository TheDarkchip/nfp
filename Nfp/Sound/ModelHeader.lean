-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Activation
import Nfp.Sound.Decimal

namespace Nfp.Sound

/-!
# Model header helpers (SOUND)

Pure parsing utilities for extracting trusted metadata from `NFP_TEXT` model headers.
-/

/-- Parse `key=value` header lines. -/
def parseHeaderLine (line : String) : Option (String × String) :=
  let line := line.trim
  if line.isEmpty then none
  else
    match line.splitOn "=" with
    | [k, v] => some (k.trim, v.trim)
    | _ => none

/-- Split a string on `\n`, preserving empty lines. -/
def splitLines (s : String) : Array String :=
  Id.run do
    let mut out : Array String := #[]
    let mut start : String.Pos.Raw := 0
    let mut p : String.Pos.Raw := 0
    let stop := s.rawEndPos
    while p < stop do
      if p.get s = '\n' then
        out := out.push (String.Pos.Raw.extract s start p)
        p := p.next s
        start := p
      else
        p := p.next s
    out := out.push (String.Pos.Raw.extract s start stop)
    return out

/-- Minimal parsed header data for sound certification. -/
structure TextHeader where
  eps : Rat
  geluDerivTarget : GeluDerivTarget
  deriving Repr

private def parseGeluDerivTarget (v : String) : Except String GeluDerivTarget :=
  match geluDerivTargetOfString v with
  | some t => .ok t
  | none => .error s!"invalid gelu_kind '{v}' (expected tanh|exact)"

/-- Parse required metadata from a `NFP_TEXT` header. -/
def parseTextHeader (lines : Array String) : Except String TextHeader :=
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
    let mut gelu? : Option GeluDerivTarget := none
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
            else if k = "gelu_kind" || k = "gelu_deriv" then
              match parseGeluDerivTarget v with
              | .error e => return .error e
              | .ok t => gelu? := some t
            i := i + 1
    let some eps := eps? | return .error "missing layer_norm_eps"
    let some gelu := gelu? | return .error "missing gelu_kind"
    return .ok { eps := eps, geluDerivTarget := gelu }

/-- Parse `layer_norm_eps` (or `eps`) from a `NFP_TEXT` header. -/
def parseTextHeaderEps (lines : Array String) : Except String Rat := do
  let hdr ← parseTextHeader lines
  return hdr.eps

/-! ### Specs -/

theorem parseHeaderLine_spec : parseHeaderLine = parseHeaderLine := rfl
theorem splitLines_spec : splitLines = splitLines := rfl
theorem parseGeluDerivTarget_spec (v : String) :
    parseGeluDerivTarget v = parseGeluDerivTarget v := rfl
theorem parseTextHeader_spec : parseTextHeader = parseTextHeader := rfl
theorem parseTextHeaderEps_spec : parseTextHeaderEps = parseTextHeaderEps := rfl

end Nfp.Sound
