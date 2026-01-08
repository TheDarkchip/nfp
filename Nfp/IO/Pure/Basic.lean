-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Core.Basic

/-!
Shared parsing helpers for CLI inputs.
-/

namespace Nfp

namespace IO

namespace Pure

/-- Split a line into whitespace-separated tokens. -/
def splitWords (line : String) : List String :=
  line.splitToList (fun c => c = ' ' || c = '\t') |>.filter (· ≠ "")

/-- Drop empty/comment lines and return their whitespace tokens. -/
def cleanTokens (line : String) : Option (List String) :=
  let trimmed := line.trim
  if trimmed.isEmpty then
    none
  else if trimmed.startsWith "#" then
    none
  else
    some (splitWords trimmed)

/-- Parse a nonnegative decimal integer. -/
def parseNat (s : String) : Except String Nat := do
  if s.isEmpty then
    throw s!"expected Nat, got '{s}'"
  else
    let mut acc : Nat := 0
    for c in s.toList do
      if c.isDigit then
        acc := acc * 10 + c.toNat - '0'.toNat
      else
        throw s!"expected Nat, got '{s}'"
    return acc

/-- Parse a signed decimal integer. -/
def parseInt (s : String) : Except String Int := do
  if s.isEmpty then
    throw s!"expected Int, got '{s}'"
  else
    match s.toSlice.front? with
    | some '-' =>
        let rest := s.drop 1
        let n ← parseNat rest
        return -Int.ofNat n
    | _ =>
        let n ← parseNat s
        return Int.ofNat n

/-- Parse a rational literal from `a` or `a/b`, rounding down if needed. -/
def parseRat (s : String) : Except String Rat := do
  match s.splitOn "/" with
  | [num] =>
      return ratRoundDown (Rat.ofInt (← parseInt num))
  | [num, den] =>
      let n ← parseInt num
      let d ← parseNat den
      if d = 0 then
        throw s!"invalid rational '{s}': zero denominator"
      else
        return ratRoundDown (Rat.divInt n (Int.ofNat d))
  | _ =>
      throw s!"invalid rational '{s}'"

end Pure

end IO

end Nfp
