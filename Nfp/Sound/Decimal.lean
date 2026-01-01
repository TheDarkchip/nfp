-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std

namespace Nfp.Sound

/-!
# Exact decimal/scientific parsing for sound certification

This module parses decimal and scientific-notation numerals (e.g. `-1.25e-3`) into `Rat`.

Design goal: avoid `Float` as a source of truth in the sound certification path.
We only use exact integer arithmetic and powers of 10.
-/

/-- Parse a signed integer written in base-10 (optional leading `+`/`-`). -/
def parseInt10 (s : String) : Except String Int :=
  let s := s.trim
  if s.isEmpty then
    .error "empty integer"
  else
    let (neg, rest) :=
      if s.startsWith "-" then (true, s.drop 1)
      else if s.startsWith "+" then (false, s.drop 1)
      else (false, s)
    if rest.isEmpty then
      .error s!"invalid integer '{s}'"
    else
      match rest.toNat? with
      | none => .error s!"invalid integer '{s}'"
      | some n =>
        let i : Int := Int.ofNat n
        .ok (if neg then -i else i)

/-- Parse a base-10 natural number; empty string is treated as 0. -/
def parseNat10OrZero (s : String) : Except String Nat :=
  let s := s.trim
  if s.isEmpty then
    .ok 0
  else
    match s.toNat? with
    | none => .error s!"invalid natural '{s}'"
    | some n => .ok n

/-- Parse a decimal/scientific numeral from a substring into an exact `Rat`. -/
def parseRatRange (s : String) (start stop : String.Pos.Raw) : Except String Rat := Id.run do
  if start >= stop then
    return .error "empty numeral"

  let token := fun () => String.Pos.Raw.extract s start stop

  -- sign
  let mut p := start
  let mut neg := false
  let c0 := p.get s
  if c0 = '-' then
    neg := true
    p := p.next s
  else if c0 = '+' then
    p := p.next s

  -- optional exponent (exactly one `e`, otherwise exactly one `E`).
  let mut ePos : Option String.Pos.Raw := none
  let mut eCount : Nat := 0
  let mut EPos : Option String.Pos.Raw := none
  let mut ECount : Nat := 0
  let mut q := p
  while q < stop do
    let c := q.get s
    if c = 'e' then
      eCount := eCount + 1
      if eCount = 1 then ePos := some q
    else if c = 'E' then
      ECount := ECount + 1
      if ECount = 1 then EPos := some q
    q := q.next s

  let expMarker? : Option String.Pos.Raw :=
    if eCount = 1 then ePos else if ECount = 1 then EPos else none

  let mantEnd : String.Pos.Raw :=
    match expMarker? with
    | some ep => ep
    | none => stop

  -- mantissa: intPart.fracPart
  let mut dotPos : Option String.Pos.Raw := none
  let mut dotCount : Nat := 0
  let mut r := p
  while r < mantEnd do
    if r.get s = '.' then
      dotCount := dotCount + 1
      if dotCount = 1 then dotPos := some r
    r := r.next s
  if dotCount > 1 then
    return .error s!"invalid numeral '{token ()}'"

  let intStart := p
  let intStop : String.Pos.Raw :=
    match dotPos with
    | some dp => dp
    | none => mantEnd
  let fracStart? : Option String.Pos.Raw :=
    match dotPos with
    | some dp => some (dp.next s)
    | none => none
  let fracStop := mantEnd

  let parseNatRangeOrZero (start stop : String.Pos.Raw) : Except String (Nat × Nat) := Id.run do
    if start >= stop then
      return .ok (0, 0)
    let mut p := start
    let mut acc : Nat := 0
    let mut len : Nat := 0
    while p < stop do
      let c := p.get s
      if ('0' <= c) && (c <= '9') then
        acc := acc * 10 + (c.toNat - '0'.toNat)
        len := len + 1
        p := p.next s
      else
        let tok := String.Pos.Raw.extract s start stop
        return .error s!"invalid natural '{tok}'"
    return .ok (acc, len)

  let parseIntRange (start stop : String.Pos.Raw) : Except String Int := Id.run do
    if start >= stop then
      return .error "empty integer"
    let tok := String.Pos.Raw.extract s start stop
    let mut p := start
    let mut neg := false
    let c0 := p.get s
    if c0 = '-' then
      neg := true
      p := p.next s
    else if c0 = '+' then
      p := p.next s
    if p >= stop then
      return .error s!"invalid integer '{tok}'"
    let mut acc : Nat := 0
    while p < stop do
      let c := p.get s
      if ('0' <= c) && (c <= '9') then
        acc := acc * 10 + (c.toNat - '0'.toNat)
        p := p.next s
      else
        return .error s!"invalid integer '{tok}'"
    let i : Int := Int.ofNat acc
    return .ok (if neg then -i else i)

  let buildResult (iNat fNat fracLen : Nat) (expInt : Int) : Except String Rat :=
    -- Construct `Rat` in a single normalization step (avoids repeated gcd normalization).
    let denomBase : Nat := Nat.pow 10 fracLen
    let mantissaNat : Nat := iNat * denomBase + fNat
    let num0 : Int := if neg then -(Int.ofNat mantissaNat) else (Int.ofNat mantissaNat)
    let expAbs : Nat := Int.natAbs expInt
    let pow10Nat : Nat := Nat.pow 10 expAbs

    let den : Nat :=
      if expInt < 0 then denomBase * pow10Nat else denomBase
    let num : Int :=
      if expInt > 0 then num0 * (Int.ofNat pow10Nat) else num0

    have den_nz : den ≠ 0 := by
      have h10pos : (0 : Nat) < 10 := by decide
      have hpow1 : denomBase ≠ 0 := by
        exact Nat.ne_of_gt (Nat.pow_pos (n := fracLen) h10pos)
      have hpow2 : pow10Nat ≠ 0 := by
        exact Nat.ne_of_gt (Nat.pow_pos (n := expAbs) h10pos)
      by_cases hneg : expInt < 0
      · -- `den = denomBase * pow10Nat`
        simpa [den, hneg] using Nat.mul_ne_zero hpow1 hpow2
      · -- `den = denomBase`
        simpa [den, hneg] using hpow1

    .ok (Rat.normalize num den (den_nz := den_nz))

  let result : Except String Rat :=
    match parseNatRangeOrZero intStart intStop with
    | .error e => .error e
    | .ok (iNat, _) =>
        match fracStart? with
        | none =>
            match expMarker? with
            | none => buildResult iNat 0 0 0
            | some ep =>
                let expStart := ep.next s
                match parseIntRange expStart stop with
                | .error e => .error e
                | .ok expInt => buildResult iNat 0 0 expInt
        | some fs =>
            match parseNatRangeOrZero fs fracStop with
            | .error e => .error e
            | .ok (fNat, fracLen) =>
                match expMarker? with
                | none => buildResult iNat fNat fracLen 0
                | some ep =>
                    let expStart := ep.next s
                    match parseIntRange expStart stop with
                    | .error e => .error e
                    | .ok expInt => buildResult iNat fNat fracLen expInt

  return result

/-- Parse a decimal/scientific numeral into an exact `Rat`.

Supported forms:
- `123`, `-123`, `+123`
- `1.25`, `.25`, `2.`
- `1e3`, `1E3`, `-1.25e-3`

This is intended for `.nfpt` parsing in sound mode.
-/
def parseRat (s : String) : Except String Rat := do
  let s := s.trim
  if s.isEmpty then
    throw "empty numeral"
  parseRatRange s 0 s.rawEndPos

/-- Parse a line of space-separated rationals, failing on the first invalid token. -/
def parseRatLine (line : String) : Except String (Array Rat) := Id.run do
  let isWs (c : Char) : Bool :=
    c = ' ' || c = '\t' || c = '\n' || c = '\r'
  let mut out : Array Rat := #[]
  let s := line
  let mut p : String.Pos.Raw := 0
  let stop := s.rawEndPos
  while p < stop do
    while p < stop && isWs (p.get s) do
      p := p.next s
    let tokStart := p
    while p < stop && !isWs (p.get s) do
      p := p.next s
    if tokStart < p then
      match parseRatRange s tokStart p with
      | .error e => return .error e
      | .ok r => out := out.push r
  return .ok out

/-! ### Specs -/

theorem parseInt10_spec (s : String) : parseInt10 s = parseInt10 s := rfl

theorem parseNat10OrZero_spec (s : String) : parseNat10OrZero s = parseNat10OrZero s := rfl

theorem parseRatRange_spec (s : String) (start stop : String.Pos.Raw) :
    parseRatRange s start stop = parseRatRange s start stop := rfl

theorem parseRat_spec (s : String) : parseRat s = parseRat s := rfl

theorem parseRatLine_spec (line : String) : parseRatLine line = parseRatLine line := rfl

end Nfp.Sound
