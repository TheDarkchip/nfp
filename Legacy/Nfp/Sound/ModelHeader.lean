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
def parseHeaderLine (line : String) : Option (String × String) := Id.run do
  let line := line.trim
  if line.isEmpty then
    return none
  -- Scan once to avoid `splitOn` allocations; require exactly one '='.
  let s := line
  let stop := s.rawEndPos
  let mut eqPos : Option String.Pos.Raw := none
  let mut eqCount : Nat := 0
  let mut p : String.Pos.Raw := 0
  while p < stop do
    if p.get s = '=' then
      eqCount := eqCount + 1
      if eqCount = 1 then
        eqPos := some p
    p := p.next s
  if eqCount ≠ 1 then
    return none
  let some eq := eqPos | return none
  let k := String.Pos.Raw.extract s 0 eq
  let v := String.Pos.Raw.extract s (eq.next s) stop
  return some (k.trim, v.trim)

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

/-- Whitespace predicate used in token scanners. -/
@[inline] def isWsChar (c : Char) : Bool :=
  c = ' ' || c = '\t' || c = '\n' || c = '\r'

/-- Count whitespace-separated tokens in a line. -/
def countWsTokens (s : String) : Nat :=
  Id.run do
    let mut p : String.Pos.Raw := 0
    let stop := s.rawEndPos
    let mut inTok : Bool := false
    let mut cnt : Nat := 0
    while p < stop do
      let c := p.get s
      if isWsChar c then
        inTok := false
      else if !inTok then
        inTok := true
        cnt := cnt + 1
      p := p.next s
    return cnt

/-- Find the first line index at or after `start` that satisfies `p`. -/
def findLineIdxFrom (lines : Array String) (start : Nat) (p : String → Bool) : Option Nat :=
  Id.run do
    let mut i := start
    while i < lines.size do
      if p (lines[i]!.trim) then
        return some i
      i := i + 1
    return none

/-- Skip to the next line satisfying `p`, or return `lines.size` if none. -/
def skipUntil (lines : Array String) (start : Nat) (p : String → Bool) : Nat :=
  match findLineIdxFrom lines start p with
  | some i => i
  | none => lines.size

/-- Skip blank (whitespace-only) lines starting at `start`. -/
def skipBlankLines (lines : Array String) (start : Nat) : Nat :=
  Id.run do
    let mut i := start
    while i < lines.size && lines[i]!.trim.isEmpty do
      i := i + 1
    return i

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
theorem isWsChar_spec : isWsChar = isWsChar := rfl
theorem countWsTokens_spec : countWsTokens = countWsTokens := rfl
theorem findLineIdxFrom_spec : findLineIdxFrom = findLineIdxFrom := rfl
theorem skipUntil_spec : skipUntil = skipUntil := rfl
theorem skipBlankLines_spec : skipBlankLines = skipBlankLines := rfl
theorem parseGeluDerivTarget_spec (v : String) :
    parseGeluDerivTarget v = parseGeluDerivTarget v := rfl
theorem parseTextHeader_spec : parseTextHeader = parseTextHeader := rfl
theorem parseTextHeaderEps_spec : parseTextHeaderEps = parseTextHeaderEps := rfl

end Nfp.Sound
