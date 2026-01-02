-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Unbundled.Rat
import Mathlib.Data.Finset.Lattice.Fold
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Nat.Factorial.Basic
import Nfp.Sound.Bounds.Portfolio

namespace Nfp.Sound

open scoped BigOperators

/-!
# Exp lower bounds (scaled Taylor + squaring)
-/

/-- Power function on `Rat` for natural exponents (iterative to avoid deep recursion). -/
private def ratPow (x : Rat) (n : Nat) : Rat :=
  Id.run do
    let mut acc : Rat := 1
    let mut base : Rat := x
    let mut exp : Nat := n
    while exp > 0 do
      if exp % 2 = 1 then
        acc := acc * base
      base := base * base
      exp := exp / 2
    return acc

theorem ratPow_def (x : Rat) (n : Nat) :
    ratPow x n =
      Id.run do
        let mut acc : Rat := 1
        let mut base : Rat := x
        let mut exp : Nat := n
        while exp > 0 do
          if exp % 2 = 1 then
            acc := acc * base
          base := base * base
          exp := exp / 2
        return acc := rfl

/-- Factorial as a rational. -/
private def ratFactorial (n : Nat) : Rat := (Nat.factorial n : Nat)

theorem ratFactorial_def (n : Nat) : ratFactorial n = (Nat.factorial n : Nat) := rfl

/-- Taylor partial sum for `exp` (all terms are nonnegative when `x ≥ 0`). -/
private def expTaylorLowerBound (x : Rat) (deg : Nat) : Rat :=
  Id.run do
    let mut term : Rat := 1
    let mut sum : Rat := 1
    let mut k : Nat := 1
    while k ≤ deg do
      let kRat : Rat := (k : Nat)
      term := term * x / kRat
      sum := sum + term
      k := k + 1
    return sum

theorem expTaylorLowerBound_def (x : Rat) (deg : Nat) :
    expTaylorLowerBound x deg =
      Id.run do
        let mut term : Rat := 1
        let mut sum : Rat := 1
        let mut k : Nat := 1
        while k ≤ deg do
          let kRat : Rat := (k : Nat)
          term := term * x / kRat
          sum := sum + term
          k := k + 1
        return sum := rfl

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

/-- Portfolio of `expLBScaledTaylor` candidates, truncated by effort. -/
def expLBCandidates (x : Rat) (effort : Nat) : Array Rat :=
  let limit := min effort expLBPortfolio.size
  Array.ofFn fun (i : Fin limit) =>
    let pair := expLBPortfolio[i.val]!
    expLBScaledTaylor x pair.2 pair.1

theorem expLBCandidates_def (x : Rat) (effort : Nat) :
    expLBCandidates x effort =
      let limit := min effort expLBPortfolio.size
      Array.ofFn fun (i : Fin limit) =>
        let pair := expLBPortfolio[i.val]!
        expLBScaledTaylor x pair.2 pair.1 := rfl

/-- Portfolio lower bound on `exp`, with a baseline `1 + x` candidate. -/
def expLB (x : Rat) (effort : Nat) : Rat :=
  let base : Rat := max 0 ((1 : Rat) + x)
  lbBest base (expLBCandidates x effort)

theorem expLB_def (x : Rat) (effort : Nat) :
    expLB x effort =
      let base : Rat := max 0 ((1 : Rat) + x)
      lbBest base (expLBCandidates x effort) := rfl

/-- `expLB` never undercuts its baseline `1 + x` lower bound. -/
theorem expLB_ge_base (x : Rat) (effort : Nat) :
    max 0 ((1 : Rat) + x) ≤ expLB x effort := by
  dsimp [expLB]
  exact lbBest_ge_base (base := max 0 ((1 : Rat) + x)) (cands := expLBCandidates x effort)

/-- Scaling exponent so `x / 2^s ≤ 1/2` for `x ≥ 0`. -/
private def expUBScalePow (x : Rat) : Nat :=
  let half : Rat := (1 : Rat) / 2
  if x ≤ half then
    0
  else
    Id.run do
      let mut s : Nat := 0
      let mut y : Rat := x
      while y > half do
        s := s + 1
        y := y / (2 : Rat)
      return s

theorem expUBScalePow_def (x : Rat) :
    expUBScalePow x =
      let half : Rat := (1 : Rat) / 2
      if x ≤ half then
        0
      else
        Id.run do
          let mut s : Nat := 0
          let mut y : Rat := x
          while y > half do
            s := s + 1
            y := y / (2 : Rat)
          return s := rfl

/-!
### Exp upper bounds (geometric series + squaring)
-/

/-- Upper bound on `exp(x)` for `x ≥ 0` using `exp(z) ≤ 1/(1-z)` with scaling. -/
def expUBScaledGeom (x : Rat) : Rat :=
  if x ≤ 0 then
    1
  else
    let scalePow := expUBScalePow x
    let scale : Rat := (Nat.pow 2 scalePow : Nat)
    let z := x / scale
    let denom := (1 : Rat) - z
    if denom ≤ 0 then
      0
    else
      let base := (1 : Rat) / denom
      ratPow base (Nat.pow 2 scalePow)

theorem expUBScaledGeom_def (x : Rat) :
    expUBScaledGeom x =
      if x ≤ 0 then
        1
      else
        let scalePow := expUBScalePow x
        let scale : Rat := (Nat.pow 2 scalePow : Nat)
        let z := x / scale
        let denom := (1 : Rat) - z
        if denom ≤ 0 then
          0
        else
          let base := (1 : Rat) / denom
          ratPow base (Nat.pow 2 scalePow) := rfl

/-- Default effort used for margin-derived softmax bounds. -/
def defaultSoftmaxExpEffort : Nat := 1

theorem defaultSoftmaxExpEffort_def : defaultSoftmaxExpEffort = 1 := rfl

end Nfp.Sound
