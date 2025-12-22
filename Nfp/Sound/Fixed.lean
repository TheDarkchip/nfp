-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Activation

namespace Nfp.Sound

/-!
# Fixed-point (base-10) arithmetic for SOUND-mode streaming

We represent real numbers as scaled integers with a **global** scale `S = 10^p`.

This file is intentionally `Int`/`Nat`-only: it is used to avoid `Rat.normalize`/gcd costs on
hot paths (matrix streaming / IBP), while preserving mathematical rigor by using conservative
outward rounding when rescaling after multiplication/division.
-/

structure Fixed10Cfg where
  /-- Scale exponent `p` in `S = 10^p`. -/
  scalePow10 : Nat
  deriving Repr

namespace Fixed10Cfg

def scaleNat (cfg : Fixed10Cfg) : Nat := Nat.pow 10 cfg.scalePow10
def scaleInt (cfg : Fixed10Cfg) : Int := Int.ofNat cfg.scaleNat

theorem scaleNat_def (cfg : Fixed10Cfg) : scaleNat cfg = Nat.pow 10 cfg.scalePow10 := rfl

theorem scaleInt_def (cfg : Fixed10Cfg) : scaleInt cfg = Int.ofNat cfg.scaleNat := rfl

end Fixed10Cfg

/-- Fixed-point scalar encoded as an `Int` meaning `x / S`. -/
abbrev Fixed10 := Int

/-- Closed fixed-point interval `[lo, hi]` (both in scaled integer units). -/
structure Fixed10Interval where
  lo : Fixed10
  hi : Fixed10
  deriving Repr

namespace Fixed10Interval

instance : Inhabited Fixed10Interval := ⟨{ lo := 0, hi := 0 }⟩

def const (x : Fixed10) : Fixed10Interval := { lo := x, hi := x }

def union (a b : Fixed10Interval) : Fixed10Interval :=
  { lo := min a.lo b.lo, hi := max a.hi b.hi }

def add (a b : Fixed10Interval) : Fixed10Interval :=
  { lo := a.lo + b.lo, hi := a.hi + b.hi }

def sub (a b : Fixed10Interval) : Fixed10Interval :=
  { lo := a.lo - b.hi, hi := a.hi - b.lo }

def relu (a : Fixed10Interval) : Fixed10Interval :=
  { lo := max 0 a.lo, hi := max 0 a.hi }

/-- Conservative GeLU hull: `GeLU(x) ∈ [min(x,0), max(x,0)]`. -/
def geluOverapprox (a : Fixed10Interval) : Fixed10Interval :=
  { lo := min a.lo 0, hi := max a.hi 0 }

private def absInt (x : Int) : Int := if x < 0 then -x else x

/-- Maximum absolute endpoint (in scaled integer units). -/
def absUpper (a : Fixed10Interval) : Int :=
  max (absInt a.lo) (absInt a.hi)

/-- Upper bound on `|x - μ|` for any `x, μ ∈ [lo, hi]` (scaled units). -/
def centeredAbsBound (a : Fixed10Interval) : Int :=
  absInt (a.hi - a.lo)

/-- Upper bound on `max |gelu'(x)|` over a fixed-point interval. -/
def geluDerivBound (cfg : Fixed10Cfg) (target : GeluDerivTarget) (a : Fixed10Interval) : Rat :=
  let maxAbsInt := absUpper a
  let maxAbsRat : Rat :=
    Rat.normalize maxAbsInt cfg.scaleNat (den_nz := by
      have h10pos : (0 : Nat) < 10 := by decide
      exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
  let maxAbsSq := maxAbsRat * maxAbsRat
  let half : Rat := (1 : Rat) / 2
  match target with
  | .exact =>
      min (1 + half * maxAbsRat) 2
  | .tanh =>
      let c : Rat := (44715 : Rat) / 1000000
      let slope := 1 + (3 : Rat) * c * maxAbsSq
      let localBound := 1 + half * maxAbsRat * slope
      min localBound 2

/-- Floor division by a positive `Nat` divisor. -/
private def floorDivNat (a : Int) (d : Nat) : Int :=
  -- `Int.ediv` is Euclidean division (for positive divisor): `a = q*d + r`, `0 ≤ r < d`.
  a.ediv (Int.ofNat d)

/-- Ceil division by a positive `Nat` divisor. -/
private def ceilDivNat (a : Int) (d : Nat) : Int :=
  let di : Int := Int.ofNat d
  let q := a.ediv di
  let r := a.emod di
  if r = 0 then q else q + 1

/-- Rescale an interval from scale `S^2` down to `S` using conservative rounding. -/
private def rescaleFromSq (cfg : Fixed10Cfg) (loSq hiSq : Int) : Fixed10Interval :=
  let S : Nat := cfg.scaleNat
  { lo := floorDivNat loSq S, hi := ceilDivNat hiSq S }

/-- Multiply two fixed-point intervals, returning an interval at the same scale.

If `a,b` are in units of `1/S`, then their product is in units of `1/S^2`; we rescale back to `1/S`
with outward rounding to remain conservative.
-/
def mul (cfg : Fixed10Cfg) (a b : Fixed10Interval) : Fixed10Interval :=
  Id.run do
    let p1 := a.lo * b.lo
    let p2 := a.lo * b.hi
    let p3 := a.hi * b.lo
    let p4 := a.hi * b.hi
    let loSq := min (min p1 p2) (min p3 p4)
    let hiSq := max (max p1 p2) (max p3 p4)
    return rescaleFromSq cfg loSq hiSq

/-- Add a constant vector to a vector of intervals. -/
def addConstVec (xs : Array Fixed10Interval) (c : Array Fixed10Interval) : Array Fixed10Interval :=
  Id.run do
    if xs.size ≠ c.size then
      return xs
    let mut out := Array.mkEmpty xs.size
    for i in [:xs.size] do
      out := out.push (add xs[i]! c[i]!)
    return out

/-- Elementwise union of two interval vectors. -/
def unionVec (a b : Array Fixed10Interval) : Array Fixed10Interval :=
  Id.run do
    if a.size ≠ b.size then
      return a
    let mut out := Array.mkEmpty a.size
    for i in [:a.size] do
      out := out.push (union a[i]! b[i]!)
    return out

/-! ### Specs -/

theorem Fixed10_spec : Fixed10 = Int := rfl

theorem const_def (x : Fixed10) : Fixed10Interval.const x = { lo := x, hi := x } := rfl

theorem union_def (a b : Fixed10Interval) :
    Fixed10Interval.union a b = { lo := min a.lo b.lo, hi := max a.hi b.hi } := rfl

theorem add_def (a b : Fixed10Interval) :
    Fixed10Interval.add a b = { lo := a.lo + b.lo, hi := a.hi + b.hi } := rfl

theorem sub_def (a b : Fixed10Interval) :
    Fixed10Interval.sub a b = { lo := a.lo - b.hi, hi := a.hi - b.lo } := rfl

theorem relu_def (a : Fixed10Interval) :
    Fixed10Interval.relu a = { lo := max 0 a.lo, hi := max 0 a.hi } := rfl

theorem geluOverapprox_def (a : Fixed10Interval) :
    Fixed10Interval.geluOverapprox a = { lo := min a.lo 0, hi := max a.hi 0 } := rfl

theorem absInt_spec (x : Int) : absInt x = absInt x := rfl

theorem absUpper_def (a : Fixed10Interval) :
    Fixed10Interval.absUpper a = max (absInt a.lo) (absInt a.hi) := rfl

theorem centeredAbsBound_def (a : Fixed10Interval) :
    Fixed10Interval.centeredAbsBound a = absInt (a.hi - a.lo) := rfl

theorem geluDerivBound_def (cfg : Fixed10Cfg) (target : GeluDerivTarget) (a : Fixed10Interval) :
    Fixed10Interval.geluDerivBound cfg target a =
      let maxAbsInt := absUpper a
      let maxAbsRat : Rat :=
        Rat.normalize maxAbsInt cfg.scaleNat (den_nz := by
          have h10pos : (0 : Nat) < 10 := by decide
          exact Nat.ne_of_gt (Nat.pow_pos (n := cfg.scalePow10) h10pos))
      let maxAbsSq := maxAbsRat * maxAbsRat
      let half : Rat := (1 : Rat) / 2
      match target with
      | .exact =>
          min (1 + half * maxAbsRat) 2
      | .tanh =>
          let c : Rat := (44715 : Rat) / 1000000
          let slope := 1 + (3 : Rat) * c * maxAbsSq
          let localBound := 1 + half * maxAbsRat * slope
          min localBound 2 := rfl

theorem floorDivNat_spec (a : Int) (d : Nat) : floorDivNat a d = floorDivNat a d := rfl

theorem ceilDivNat_spec (a : Int) (d : Nat) : ceilDivNat a d = ceilDivNat a d := rfl

theorem rescaleFromSq_spec (cfg : Fixed10Cfg) (loSq hiSq : Int) :
    rescaleFromSq cfg loSq hiSq = rescaleFromSq cfg loSq hiSq := rfl

theorem mul_spec (cfg : Fixed10Cfg) (a b : Fixed10Interval) :
    Fixed10Interval.mul cfg a b = Fixed10Interval.mul cfg a b := rfl

theorem addConstVec_spec (xs : Array Fixed10Interval) (c : Array Fixed10Interval) :
    Fixed10Interval.addConstVec xs c = Fixed10Interval.addConstVec xs c := rfl

theorem unionVec_spec (a b : Array Fixed10Interval) :
    Fixed10Interval.unionVec a b = Fixed10Interval.unionVec a b := rfl

end Fixed10Interval

/-!
## Fast decimal → fixed-point parsing

We parse a decimal/scientific numeral token into a **rounded** scaled integer at scale `S = 10^p`
without constructing a `Rat` (and therefore without gcd normalization).

Correctness contract (soundness):
- The returned integer `r` is a rounding of the exact scaled value `x*S`.
- If later we treat the true scaled value as lying in `[r-1, r+1]`, then this interval always
  contains the exact scaled value (since the exact value lies between `floor` and `ceil`).
-/

private def isDigit (b : UInt8) : Bool := (48 ≤ b) && (b ≤ 57)
private def digitVal (b : UInt8) : Nat := (b.toNat - 48)

private def pow10Nat (k : Nat) : Nat := Nat.pow 10 k

/-- Parse an `Int` exponent written in base-10 from a byte slice. -/
private def parseExpInt (bytes : ByteArray) (start stop : Nat) : Except String Int :=
  if start ≥ stop then
    .error "invalid exponent"
  else
    Id.run do
      let mut i := start
      let mut neg : Bool := false
      let b0 := bytes[i]!
      if b0 = 45 then -- '-'
        neg := true
        i := i + 1
      else if b0 = 43 then -- '+'
        i := i + 1
      if i ≥ stop then
        return .error "invalid exponent"
      let mut acc : Int := 0
      while i < stop do
        let b := bytes[i]!
        if !isDigit b then
          return .error "invalid exponent digit"
        acc := acc * 10 + (Int.ofNat (digitVal b))
        i := i + 1
      return .ok (if neg then -acc else acc)

/-- Parse a token into a rounded scaled integer at scale `10^scalePow10`. -/
def parseFixed10Rounded (scalePow10 : Nat) (bytes : ByteArray) (start stop : Nat) :
    Except String Int :=
  if start ≥ stop then
    .error "empty token"
  else
    Id.run do
      let mut i := start
      -- sign
      let mut neg : Bool := false
      let b0 := bytes[i]!
      if b0 = 45 then -- '-'
        neg := true
        i := i + 1
      else if b0 = 43 then
        i := i + 1

      -- mantissa with optional '.'
      let mut mant : Int := 0
      let mut fracLen : Nat := 0
      let mut seenDot : Bool := false
      let mut anyDigit : Bool := false
      while i < stop do
        let b := bytes[i]!
        if b = 46 then -- '.'
          if seenDot then
            return .error "invalid numeral (multiple dots)"
          seenDot := true
          i := i + 1
        else if b = 101 || b = 69 then -- 'e' or 'E'
          break
        else if isDigit b then
          anyDigit := true
          mant := mant * 10 + (Int.ofNat (digitVal b))
          if seenDot then
            fracLen := fracLen + 1
          i := i + 1
        else
          return .error "invalid numeral"
      if !anyDigit then
        return .error "invalid numeral (no digits)"

      -- optional exponent
      let mut exp : Int := 0
      if i < stop then
        let b := bytes[i]!
        if b = 101 || b = 69 then
          match parseExpInt bytes (i + 1) stop with
          | .error e => return .error e
          | .ok e => exp := e

      -- scaled value: mant * 10^(scalePow10 + exp - fracLen)
      let expTotal : Int := (Int.ofNat scalePow10) + exp - (Int.ofNat fracLen)
      let num0 : Int := if neg then -mant else mant
      if expTotal ≥ 0 then
        let eNat : Nat := Int.toNat expTotal
        let pow : Int := Int.ofNat (pow10Nat eNat)
        return .ok (num0 * pow)
      else
        let eNat : Nat := Int.toNat (-expTotal)
        let denNat : Nat := pow10Nat eNat
        let den : Int := Int.ofNat denNat
        let q := num0.ediv den
        let r := num0.emod den
        if r = 0 then
          return .ok q
        -- Round-to-nearest (ties up). Always within 1 of the exact scaled value.
        let twoR := (2 : Int) * r
        if twoR < den then
          return .ok q
        else
          return .ok (q + 1)

/-!
### Token folding helpers (line-based)

These helpers mirror the `foldRatTokens` pattern used elsewhere, but avoid allocating token
substrings by scanning whitespace boundaries in the UTF-8 byte array of each line.
-/

private def isWs (b : UInt8) : Bool := b = 32 || b = 9  -- ' ' or '\t'

def foldFixed10Tokens {α : Type}
    (scalePow10 : Nat)
    (lines : Array String)
    (start : Nat)
    (count : Nat)
    (state : α)
    (step : α → Int → α) : Except String (α × Nat) :=
  Id.run do
    let mut i := start
    let mut remaining := count
    let mut st := state
    while remaining > 0 do
      if i ≥ lines.size then
        return .error "unexpected end of file while reading fixed tokens"
      let line := lines[i]!.trim
      i := i + 1
      if line.isEmpty then
        pure ()
      else
        let bytes := line.toUTF8
        let mut j : Nat := 0
        while j < bytes.size && remaining > 0 do
          while j < bytes.size && isWs (bytes[j]!) do
            j := j + 1
          if j ≥ bytes.size then
            break
          let tokStart := j
          while j < bytes.size && !isWs (bytes[j]!) do
            j := j + 1
          let tokStop := j
          match parseFixed10Rounded scalePow10 bytes tokStart tokStop with
          | .error e => return .error e
          | .ok x =>
              st := step st x
              remaining := remaining - 1
    return .ok (st, i)

/-! ### Specs -/

theorem isDigit_spec (b : UInt8) : isDigit b = isDigit b := rfl

theorem digitVal_spec (b : UInt8) : digitVal b = digitVal b := rfl

theorem pow10Nat_spec (k : Nat) : pow10Nat k = pow10Nat k := rfl

theorem parseExpInt_spec (bytes : ByteArray) (start stop : Nat) :
    parseExpInt bytes start stop = parseExpInt bytes start stop := rfl

theorem parseFixed10Rounded_spec (scalePow10 : Nat) (bytes : ByteArray) (start stop : Nat) :
    parseFixed10Rounded scalePow10 bytes start stop =
      parseFixed10Rounded scalePow10 bytes start stop := rfl

theorem isWs_spec (b : UInt8) : isWs b = isWs b := rfl

theorem foldFixed10Tokens_spec {α : Type}
    (scalePow10 : Nat)
    (lines : Array String)
    (start : Nat)
    (count : Nat)
    (state : α)
    (step : α → Int → α) :
    foldFixed10Tokens scalePow10 lines start count state step =
      foldFixed10Tokens scalePow10 lines start count state step := rfl

end Nfp.Sound
