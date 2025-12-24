-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Unbundled.Rat
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Nat.Factorial.Basic

namespace Nfp.Sound

open scoped BigOperators

/-!
# Exp lower bounds (scaled Taylor + squaring)
-/

/-- Power function on `Rat` for natural exponents. -/
private def ratPow (x : Rat) : Nat → Rat
  | 0 => 1
  | n + 1 => ratPow x n * x

theorem ratPow_def (x : Rat) (n : Nat) :
    ratPow x n = match n with | 0 => 1 | n + 1 => ratPow x n * x := by
  cases n <;> rfl

/-- Factorial as a rational. -/
private def ratFactorial (n : Nat) : Rat := (Nat.factorial n : Nat)

theorem ratFactorial_def (n : Nat) : ratFactorial n = (Nat.factorial n : Nat) := rfl

/-- Taylor partial sum for `exp` (all terms are nonnegative when `x ≥ 0`). -/
private def expTaylorLowerBound (x : Rat) (deg : Nat) : Rat :=
  Finset.sum (Finset.range (deg + 1)) fun k => ratPow x k / ratFactorial k

theorem expTaylorLowerBound_def (x : Rat) (deg : Nat) :
    expTaylorLowerBound x deg =
      Finset.sum (Finset.range (deg + 1)) fun k => ratPow x k / ratFactorial k := rfl

/-- Lower bound on `exp` via scaled Taylor partial sums and repeated squaring. -/
def expLBScaledTaylor (x : Rat) (deg scalePow : Nat) : Rat :=
  if x < 0 then
    0
  else
    let scale : Rat := (Nat.pow 2 scalePow : Nat)
    let z := x / scale
    let t := expTaylorLowerBound z deg
    ratPow t (Nat.pow 2 scalePow)

theorem expLBScaledTaylor_def (x : Rat) (deg scalePow : Nat) :
    expLBScaledTaylor x deg scalePow =
      if x < 0 then
        0
      else
        let scale : Rat := (Nat.pow 2 scalePow : Nat)
        let z := x / scale
        let t := expTaylorLowerBound z deg
        ratPow t (Nat.pow 2 scalePow) := rfl

/-- Default portfolio of `(scalePow, taylorDeg)` candidates for `expLB`. -/
def expLBPortfolio : Array (Nat × Nat) :=
  #[(2, 4), (3, 6), (4, 8)]

theorem expLBPortfolio_def : expLBPortfolio = #[(2, 4), (3, 6), (4, 8)] := rfl

/-- Portfolio lower bound on `exp`, with a baseline `1 + x` candidate. -/
def expLB (x : Rat) (effort : Nat) : Rat :=
  let base : Rat := max 0 ((1 : Rat) + x)
  let limit := min effort expLBPortfolio.size
  Id.run do
    let mut best := base
    for i in [:limit] do
      let cand := expLBScaledTaylor x (expLBPortfolio[i]!).2 (expLBPortfolio[i]!).1
      best := max best cand
    return best

theorem expLB_def (x : Rat) (effort : Nat) :
    expLB x effort =
      let base : Rat := max 0 ((1 : Rat) + x)
      let limit := min effort expLBPortfolio.size
      Id.run do
        let mut best := base
        for i in [:limit] do
          let cand := expLBScaledTaylor x (expLBPortfolio[i]!).2 (expLBPortfolio[i]!).1
          best := max best cand
        return best := rfl

/-- Default effort used for margin-derived softmax bounds. -/
def defaultSoftmaxExpEffort : Nat := 1

theorem defaultSoftmaxExpEffort_def : defaultSoftmaxExpEffort = 1 := rfl

end Nfp.Sound
