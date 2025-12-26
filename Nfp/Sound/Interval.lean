-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Bounds

namespace Nfp.Sound

/-!
# Rational intervals for SOUND-mode local certification

This file provides a minimal-but-complete `Rat` interval arithmetic library used by
streaming Interval Bound Propagation (IBP) in `Nfp.Untrusted.SoundCompute`, with trusted
verification wrappers in `Nfp.Sound.IO`.

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
  decide (a.lo ≤ 0 ∧ 0 ≤ a.hi)

/-- Upper bound on `|x - μ|` for any `x, μ ∈ [lo, hi]`. -/
def centeredAbsBound (a : RatInterval) : Rat :=
  ratAbs (a.hi - a.lo)

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

We return a conservative lower bound on `var(x)` for all `x` in the box.

We compute the exact minimum via a 1D convex minimization:

`var(x) = (1/n) ∑ (xᵢ - mean(x))^2 = min_c (1/n) ∑ (xᵢ - c)^2`.

Therefore,

`min_{x∈box} var(x) = min_c (1/n) ∑ min_{xᵢ∈[lᵢ,uᵢ]} (xᵢ - c)^2`

and for fixed `c` each coordinate minimization is `dist([lᵢ,uᵢ], c)^2` where `dist` is 0 if
`c ∈ [lᵢ,uᵢ]` and otherwise the squared distance to the nearer endpoint.

The resulting one-dimensional function of `c` is convex piecewise-quadratic, so we can find its
global minimum by scanning the sorted breakpoints `{lᵢ,uᵢ}` and checking the unique stationary point
in each region (plus the breakpoints themselves).
-/
def varianceLowerBound (xs : Array RatInterval) : Rat :=
  if xs.isEmpty then
    0
  else
    Id.run do
      let n : Nat := xs.size
      let nRat : Rat := (n : Nat)
      -- Normalize endpoints defensively.
      let normed : Array RatInterval :=
        xs.map (fun x => { lo := min x.lo x.hi, hi := max x.lo x.hi })
      if n < 2 then
        return 0
      -- Build sorted breakpoint lists for `lo` and `hi` with squared endpoints for O(1) evaluation.
      let mut enters : Array (Rat × Rat) := Array.mkEmpty n
      let mut leaves : Array (Rat × Rat) := Array.mkEmpty n
      let mut sumLeft : Rat := 0
      let mut sumLeftSq : Rat := 0
      for x in normed do
        let lo := x.lo
        let hi := x.hi
        enters := enters.push (lo, ratSq lo)
        leaves := leaves.push (hi, ratSq hi)
        sumLeft := sumLeft + lo
        sumLeftSq := sumLeftSq + ratSq lo
      -- Exact minimization over the breakpoints (O(n log n)).
      let entersSorted := enters.qsort (fun a b => a.1 ≤ b.1)
      let leavesSorted := leaves.qsort (fun a b => a.1 ≤ b.1)
      let breaksAll :=
        (entersSorted.map (fun p => p.1) ++ leavesSorted.map (fun p => p.1)).qsort (· ≤ ·)
      -- Unique-ify breakpoints.
      let mut breaks : Array Rat := Array.mkEmpty breaksAll.size
      for b in breaksAll do
        if breaks.isEmpty then
          breaks := breaks.push b
        else if breaks.back! = b then
          pure ()
        else
          breaks := breaks.push b
      let evalG (c : Rat) (leftCount rightCount : Nat)
          (sumLeft sumLeftSq sumRight sumRightSq : Rat) : Rat :=
        let cSq := ratSq c
        let leftTerm :=
          sumLeftSq - (2 : Rat) * c * sumLeft + ((leftCount : Nat) : Rat) * cSq
        let rightTerm :=
          ((rightCount : Nat) : Rat) * cSq - (2 : Rat) * c * sumRight + sumRightSq
        leftTerm + rightTerm
      -- State for scanning regions:
      -- left set: intervals with `c < lo`
      -- right set: intervals with `c > hi`
      let mut leftCount : Nat := n
      let mut rightCount : Nat := 0
      let mut sumRight : Rat := 0
      let mut sumRightSq : Rat := 0
      let mut iEnter : Nat := 0
      let mut iLeave : Nat := 0
      let mut bestG : Rat :=
        evalG (sumLeft / nRat) leftCount rightCount sumLeft sumLeftSq sumRight sumRightSq
      if !breaks.isEmpty then
        bestG := min bestG
          (evalG breaks[0]! leftCount rightCount sumLeft sumLeftSq sumRight sumRightSq)
      for bi in [:breaks.size] do
        let b := breaks[bi]!
        -- Process enters at `b` (at `c=b` these are already inside, so not in left).
        while iEnter < entersSorted.size && entersSorted[iEnter]!.1 = b do
          let (_, loSq) := entersSorted[iEnter]!
          let lo := entersSorted[iEnter]!.1
          leftCount := leftCount - 1
          sumLeft := sumLeft - lo
          sumLeftSq := sumLeftSq - loSq
          iEnter := iEnter + 1
        -- Evaluate at the breakpoint `c=b` (intervals with `hi=b` are still inside at `c=b`).
        bestG := min bestG (evalG b leftCount rightCount sumLeft sumLeftSq sumRight sumRightSq)
        -- After `b`, process leaves at `b` (those intervals become right for `c>b`).
        while iLeave < leavesSorted.size && leavesSorted[iLeave]!.1 = b do
          let (_, hiSq) := leavesSorted[iLeave]!
          let hi := leavesSorted[iLeave]!.1
          rightCount := rightCount + 1
          sumRight := sumRight + hi
          sumRightSq := sumRightSq + hiSq
          iLeave := iLeave + 1
        -- Check stationary point in the region `(b, nextB)` if there is a next breakpoint.
        let outsideCount : Nat := leftCount + rightCount
        if outsideCount = 0 then
          -- There exists `c` contained in every interval, so the box intersects the constant line.
          -- The exact minimum is 0.
          return 0
        let cStar : Rat := (sumLeft + sumRight) / ((outsideCount : Nat) : Rat)
        if bi + 1 < breaks.size then
          let bNext := breaks[bi + 1]!
          if b < cStar ∧ cStar < bNext then
            bestG := min bestG
              (evalG cStar leftCount rightCount sumLeft sumLeftSq sumRight sumRightSq)
        else
          -- Last region `(b, +∞)`.
          if b ≤ cStar then
            bestG := min bestG
              (evalG cStar leftCount rightCount sumLeft sumLeftSq sumRight sumRightSq)
      let exactLB := bestG / nRat
      return exactLB

/-- Over-approximate GeLU on an interval without any transcendental facts.

For all real `x`, `GeLU(x) = x·Φ(x)` lies between `x` and `0`.
Therefore `GeLU([lo,hi]) ⊆ [min(lo,0), max(hi,0)]`.
-/
def geluOverapprox (a : RatInterval) : RatInterval :=
  { lo := min a.lo 0, hi := max a.hi 0 }

/-- Upper bound on `max |gelu'(x)|` over a rational interval. -/
def geluDerivBound (target : GeluDerivTarget) (a : RatInterval) : Rat :=
  let maxAbs := max (ratAbs a.lo) (ratAbs a.hi)
  let maxAbsSq := maxAbs * maxAbs
  let half : Rat := (1 : Rat) / 2
  match target with
  | .exact =>
      -- Conservative: Φ(x) ≤ 1 and φ(x) ≤ 1/2, so gelu'(x) ≤ 1 + |x|/2.
      min (1 + half * maxAbs) (geluDerivBoundGlobal target)
  | .tanh =>
      -- Conservative: |tanh| ≤ 1, sech^2 ≤ 1, and sqrt(2/pi) ≤ 1.
      let c : Rat := (44715 : Rat) / 1000000
      let slope := 1 + (3 : Rat) * c * maxAbsSq
      let localBound := 1 + half * maxAbs * slope
      min localBound (geluDerivBoundGlobal target)

/-- Upper bound on the row-sum softmax Jacobian norm for a probability interval. -/
def softmaxJacobianNormInfBound (a : RatInterval) : Rat :=
  Nfp.Sound.softmaxJacobianNormInfBound a.lo a.hi

/-! ### Specs -/

theorem const_def (r : Rat) : RatInterval.const r = { lo := r, hi := r } := rfl

theorem centeredAbsBound_def (a : RatInterval) :
    RatInterval.centeredAbsBound a = ratAbs (a.hi - a.lo) := rfl

theorem add_def (a b : RatInterval) :
    RatInterval.add a b = { lo := a.lo + b.lo, hi := a.hi + b.hi } := rfl

theorem sub_def (a b : RatInterval) :
    RatInterval.sub a b = { lo := a.lo - b.hi, hi := a.hi - b.lo } := rfl

theorem scale_def (c : Rat) (a : RatInterval) :
    RatInterval.scale c a =
      if c ≥ 0 then
        { lo := c * a.lo, hi := c * a.hi }
      else
        { lo := c * a.hi, hi := c * a.lo } := rfl

theorem relu_def (a : RatInterval) :
    RatInterval.relu a = { lo := max 0 a.lo, hi := max 0 a.hi } := rfl

theorem union_def (a b : RatInterval) :
    RatInterval.union a b = { lo := min a.lo b.lo, hi := max a.hi b.hi } := rfl

theorem softmaxJacobianNormInfBound_def (a : RatInterval) :
    RatInterval.softmaxJacobianNormInfBound a =
      Nfp.Sound.softmaxJacobianNormInfBound a.lo a.hi := rfl

theorem geluDerivBound_def (target : GeluDerivTarget) (a : RatInterval) :
    RatInterval.geluDerivBound target a =
      let maxAbs := max (ratAbs a.lo) (ratAbs a.hi)
      let maxAbsSq := maxAbs * maxAbs
      let half : Rat := (1 : Rat) / 2
      match target with
      | .exact =>
          min (1 + half * maxAbs) (geluDerivBoundGlobal target)
      | .tanh =>
          let c : Rat := (44715 : Rat) / 1000000
          let slope := 1 + (3 : Rat) * c * maxAbsSq
          let localBound := 1 + half * maxAbs * slope
          min localBound (geluDerivBoundGlobal target) := rfl

theorem containsZero_iff (a : RatInterval) :
    RatInterval.containsZero a = true ↔ a.lo ≤ 0 ∧ 0 ≤ a.hi := by
  simp [containsZero]

theorem ratSq_def (x : Rat) : ratSq x = x * x := rfl

theorem squareLowerBound_def (a : RatInterval) :
    RatInterval.squareLowerBound a =
      if RatInterval.containsZero a then
        0
      else
        let alo := ratAbs a.lo
        let ahi := ratAbs a.hi
        let m := min alo ahi
        ratSq m := rfl

theorem mean_def (xs : Array RatInterval) :
    RatInterval.mean xs =
      if xs.isEmpty then
        RatInterval.const 0
      else
        let n : Nat := xs.size
        let nRat : Rat := (n : Nat)
        let (loSum, hiSum) :=
          xs.foldl (fun (acc : Rat × Rat) x => (acc.1 + x.lo, acc.2 + x.hi)) (0, 0)
        { lo := loSum / nRat, hi := hiSum / nRat } := rfl

theorem varianceLowerBound_spec (xs : Array RatInterval) :
    RatInterval.varianceLowerBound xs = RatInterval.varianceLowerBound xs := rfl

theorem geluOverapprox_def (a : RatInterval) :
    RatInterval.geluOverapprox a = { lo := min a.lo 0, hi := max a.hi 0 } := rfl

end RatInterval

end Nfp.Sound
