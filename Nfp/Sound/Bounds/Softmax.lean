-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Unbundled.Rat
import Nfp.Sound.Bounds.Exp
import Nfp.Sound.Bounds.Portfolio

namespace Nfp.Sound

/-!
# Softmax Jacobian bounds
-/

/-- Worst-case bound on the row-sum operator norm of a softmax Jacobian row.

For a probability row `p`, the softmax Jacobian is `J = diag(p) - p pᵀ`.
For row `i`, the absolute row-sum is:

`∑ⱼ |Jᵢⱼ| = pᵢ(1-pᵢ) + ∑_{j≠i} pᵢ pⱼ = 2 pᵢ (1-pᵢ) ≤ 1/2`.

This bound is universal (independent of sequence length).
-/
def softmaxJacobianNormInfWorst : Rat := (1 : Rat) / 2

theorem softmaxJacobianNormInfWorst_def : softmaxJacobianNormInfWorst = (1 : Rat) / 2 := rfl

/-- Clamp a rational to the unit interval `[0,1]`. -/
private def clamp01 (x : Rat) : Rat :=
  max 0 (min x 1)

theorem clamp01_def (x : Rat) : clamp01 x = max 0 (min x 1) := rfl

/-- Local upper bound on the row-sum softmax Jacobian norm given `p ∈ [pLo, pHi]`. -/
def softmaxJacobianNormInfBound (pLo pHi : Rat) : Rat :=
  let lo0 := min pLo pHi
  let hi0 := max pLo pHi
  let lo := clamp01 lo0
  let hi := clamp01 hi0
  if hi < lo then
    0
  else
    let half : Rat := (1 : Rat) / 2
    let f : Rat → Rat := fun p => (2 : Rat) * p * (1 - p)
    if lo ≤ half ∧ half ≤ hi then
      half
    else
      max (f lo) (f hi)

theorem softmaxJacobianNormInfBound_def (pLo pHi : Rat) :
    softmaxJacobianNormInfBound pLo pHi =
      let lo0 := min pLo pHi
      let hi0 := max pLo pHi
      let lo := clamp01 lo0
      let hi := clamp01 hi0
      if hi < lo then
        0
      else
        let half : Rat := (1 : Rat) / 2
        let f : Rat → Rat := fun p => (2 : Rat) * p * (1 - p)
        if lo ≤ half ∧ half ≤ hi then
          half
        else
          max (f lo) (f hi) := rfl

/-! ### Margin-derived softmax bounds -/

/-- Lower bound on the maximum softmax probability from a logit margin.

Uses a portfolio `expLB` to lower bound `exp(m)` and maps it to
`p_max ≥ exp(m) / (exp(m) + (n-1))` for `m > 0`, with `n = seqLen`. -/
def softmaxMaxProbLowerBound (seqLen : Nat) (margin : Rat) (expEffort : Nat) : Rat :=
  if seqLen = 0 then
    0
  else if margin > 0 then
    let nRat : Rat := (seqLen : Nat)
    let e := expLB margin expEffort
    e / (e + (nRat - 1))
  else
    0

theorem softmaxMaxProbLowerBound_def (seqLen : Nat) (margin : Rat) (expEffort : Nat) :
    softmaxMaxProbLowerBound seqLen margin expEffort =
      if seqLen = 0 then
        0
      else if margin > 0 then
        let nRat : Rat := (seqLen : Nat)
        let e := expLB margin expEffort
        e / (e + (nRat - 1))
      else
        0 := rfl

/-- Lower bound on total target softmax weight from a logit margin.

If at least `targetCount` logits exceed the rest by `margin`, then the total
target weight is at least `t*exp(m)/(t*exp(m)+(n-t))`.
-/
def softmaxTargetWeightLowerBound (seqLen targetCount : Nat) (margin : Rat)
    (expEffort : Nat) : Rat :=
  if seqLen = 0 || targetCount = 0 then
    0
  else if margin > 0 then
    let nRat : Rat := (seqLen : Nat)
    let tRat : Rat := (targetCount : Nat)
    let base := tRat / nRat
    let e := expLB margin expEffort
    let cand := (tRat * e) / (tRat * e + (nRat - tRat))
    lbBest base #[cand]
  else
    0

theorem softmaxTargetWeightLowerBound_def (seqLen targetCount : Nat) (margin : Rat)
    (expEffort : Nat) :
    softmaxTargetWeightLowerBound seqLen targetCount margin expEffort =
      if seqLen = 0 || targetCount = 0 then
        0
      else if margin > 0 then
        let nRat : Rat := (seqLen : Nat)
        let tRat : Rat := (targetCount : Nat)
        let base := tRat / nRat
        let e := expLB margin expEffort
        let cand := (tRat * e) / (tRat * e + (nRat - tRat))
        lbBest base #[cand]
      else
        0 := rfl

/-- Upper bound on the row-sum softmax Jacobian norm from a max-probability lower bound.

If the maximum probability is at least `pLo` and `pLo > 1/2`, then every row
satisfies `2 p (1-p) ≤ 2 pLo (1-pLo)`; otherwise the universal `1/2` bound applies. -/
def softmaxJacobianNormInfBoundFromMaxProb (pLo : Rat) : Rat :=
  let half : Rat := (1 : Rat) / 2
  let p := clamp01 pLo
  if p > half then
    (2 : Rat) * p * (1 - p)
  else
    half

theorem softmaxJacobianNormInfBoundFromMaxProb_def (pLo : Rat) :
    softmaxJacobianNormInfBoundFromMaxProb pLo =
      let half : Rat := (1 : Rat) / 2
      let p := clamp01 pLo
      if p > half then
        (2 : Rat) * p * (1 - p)
      else
        half := rfl

/-- Upper bound on the row-sum softmax Jacobian norm from a logit margin. -/
def softmaxJacobianNormInfBoundFromMargin (seqLen : Nat) (margin : Rat) (expEffort : Nat) : Rat :=
  softmaxJacobianNormInfBoundFromMaxProb (softmaxMaxProbLowerBound seqLen margin expEffort)

theorem softmaxJacobianNormInfBoundFromMargin_def (seqLen : Nat) (margin : Rat)
    (expEffort : Nat) :
    softmaxJacobianNormInfBoundFromMargin seqLen margin expEffort =
      softmaxJacobianNormInfBoundFromMaxProb
        (softmaxMaxProbLowerBound seqLen margin expEffort) := rfl

end Nfp.Sound
