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

  -- sign
  let (neg, rest) :=
    if s.startsWith "-" then (true, s.drop 1)
    else if s.startsWith "+" then (false, s.drop 1)
    else (false, s)

  -- optional exponent
  let (mantissaStr, expStr?) : String × Option String :=
    match rest.splitOn "e" with
    | [m, e] => (m, some e)
    | _ =>
      match rest.splitOn "E" with
      | [m, e] => (m, some e)
      | _ => (rest, none)

  -- mantissa: intPart.fracPart
  let parts := mantissaStr.splitOn "."
  let (intPart, fracPart) ←
    match parts with
    | [i] => pure (i, "")
    | [i, f] => pure (i, f)
    | _ => throw s!"invalid numeral '{s}'"

  let iNat ← parseNat10OrZero intPart
  let fNat ← parseNat10OrZero fracPart
  let fracLen := fracPart.trim.length

  let denomNat : Nat := Nat.pow 10 fracLen
  let mantissaNat : Nat := iNat * denomNat + fNat

  let baseNum : Int :=
    let m : Int := Int.ofNat mantissaNat
    if neg then -m else m

  let base : Rat := (baseNum : Rat) / (denomNat : Rat)

  let expInt : Int ←
    match expStr? with
    | none => pure 0
    | some e => parseInt10 e

  if expInt = 0 then
    return base
  else
    let expNeg : Bool := expInt < 0
    let expAbs : Nat := Int.natAbs expInt
    let pow10 : Rat := (Nat.pow 10 expAbs : Nat)
    if expNeg then
      return (base / pow10)
    else
      return (base * pow10)

/-- Parse a line of space-separated rationals, failing on the first invalid token. -/
def parseRatLine (line : String) : Except String (Array Rat) := do
  let parts := line.splitOn " " |>.filter (· ≠ "")
  let mut out : Array Rat := #[]
  for p in parts do
    let r ← parseRat p
    out := out.push r
  return out

end Nfp.Sound
