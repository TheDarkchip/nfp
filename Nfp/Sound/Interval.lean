-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Bounds

namespace Nfp.Sound

/-!
# Rational intervals for SOUND-mode local certification

This file provides a minimal-but-complete `Rat` interval arithmetic library used by
streaming Interval Bound Propagation (IBP) in `Nfp.Sound.IO`.

All operations are conservative (over-approximations).
-/

/-- Closed rational interval `[lo, hi]`. -/
structure RatInterval where
  lo : Rat
  hi : Rat
  deriving Repr

namespace RatInterval

instance : Inhabited RatInterval := ⟨{ lo := 0, hi := 0 }⟩

/-- Singleton interval `[r, r]`. -/
def const (r : Rat) : RatInterval := { lo := r, hi := r }

/-- Interval addition: `[a.lo + b.lo, a.hi + b.hi]`. -/
def add (a b : RatInterval) : RatInterval :=
  { lo := a.lo + b.lo, hi := a.hi + b.hi }

/-- Interval subtraction: `[a.lo - b.hi, a.hi - b.lo]`. -/
def sub (a b : RatInterval) : RatInterval :=
  { lo := a.lo - b.hi, hi := a.hi - b.lo }

/-- Scale an interval by a rational `c`, handling sign. -/
def scale (c : Rat) (a : RatInterval) : RatInterval :=
  if c ≥ 0 then
    { lo := c * a.lo, hi := c * a.hi }
  else
    { lo := c * a.hi, hi := c * a.lo }

/-- ReLU over-approximation: `[max(0, lo), max(0, hi)]`. -/
def relu (a : RatInterval) : RatInterval :=
  { lo := max 0 a.lo, hi := max 0 a.hi }

/-- Union (hull) of intervals: `[min(a.lo,b.lo), max(a.hi,b.hi)]`. -/
def union (a b : RatInterval) : RatInterval :=
  { lo := min a.lo b.lo, hi := max a.hi b.hi }

/-- Whether the interval contains 0. -/
def containsZero (a : RatInterval) : Bool :=
  a.lo ≤ 0 ∧ 0 ≤ a.hi

private def ratSq (x : Rat) : Rat := x * x

/-- Lower bound on `x^2` over an interval.

If the interval contains 0, this is 0. Otherwise it is the squared distance to 0 of the
endpoint with smaller absolute value.
-/
def squareLowerBound (a : RatInterval) : Rat :=
  if containsZero a then
    0
  else
    let alo := ratAbs a.lo
    let ahi := ratAbs a.hi
    let m := min alo ahi
    ratSq m

/-- Mean interval of a coordinate-wise box.

For `xᵢ ∈ [loᵢ, hiᵢ]`, we have `μ = (1/n)∑ xᵢ ∈ [(1/n)∑ loᵢ, (1/n)∑ hiᵢ]`.
-/
def mean (xs : Array RatInterval) : RatInterval :=
  if xs.isEmpty then
    const 0
  else
    let n : Nat := xs.size
    let nRat : Rat := (n : Nat)
    let (loSum, hiSum) :=
      xs.foldl (fun (acc : Rat × Rat) x => (acc.1 + x.lo, acc.2 + x.hi)) (0, 0)
    { lo := loSum / nRat, hi := hiSum / nRat }

/-- Sound lower bound on the variance of a coordinate-wise box.

We compute a mean interval `μ ∈ [μ_lo, μ_hi]`, then for each coordinate interval `xᵢ` we bound
`(xᵢ - μ)^2` from below by using `squareLowerBound` on the interval subtraction.

If a centered interval contains 0, we take the square lower bound to be 0.
-/
def varianceLowerBound (xs : Array RatInterval) : Rat :=
  if xs.isEmpty then
    0
  else
    Id.run do
      let n : Nat := xs.size
      let nRat : Rat := (n : Nat)
      let μ := mean xs
      let mut acc : Rat := 0
      for x in xs do
        let centered := sub x μ
        acc := acc + squareLowerBound centered
      return acc / nRat

/-- Over-approximate GeLU on an interval without any transcendental facts.

For all real `x`, `GeLU(x) = x·Φ(x)` lies between `x` and `0`.
Therefore `GeLU([lo,hi]) ⊆ [min(lo,0), max(hi,0)]`.
-/
def geluOverapprox (a : RatInterval) : RatInterval :=
  { lo := min a.lo 0, hi := max a.hi 0 }

end RatInterval

end Nfp.Sound
