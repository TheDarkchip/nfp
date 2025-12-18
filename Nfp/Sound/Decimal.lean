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

  let expInt : Int ←
    match expStr? with
    | none => pure 0
    | some e => parseInt10 e

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

  return Rat.normalize num den (den_nz := den_nz)

/-- Parse a line of space-separated rationals, failing on the first invalid token. -/
def parseRatLine (line : String) : Except String (Array Rat) := do
  let parts := line.splitOn " " |>.filter (· ≠ "")
  let mut out : Array Rat := #[]
  for p in parts do
    let r ← parseRat p
    out := out.push r
  return out

end Nfp.Sound
