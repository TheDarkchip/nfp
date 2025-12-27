-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Unbundled.Rat
import Mathlib.Data.Nat.Sqrt

namespace Nfp.Sound

/-!
# Attention pattern-term helpers
-/

/-- Upper bound on `sqrt(n)` using `Nat.sqrt` (floor) plus one. -/
def sqrtUpperNat (n : Nat) : Nat := Nat.sqrt n + 1

theorem sqrtUpperNat_def (n : Nat) : sqrtUpperNat n = Nat.sqrt n + 1 := rfl

/-- Upper bound on `sqrt(n)` as a rational. -/
def sqrtUpperRat (n : Nat) : Rat := (sqrtUpperNat n : Nat)

theorem sqrtUpperRat_def (n : Nat) : sqrtUpperRat n = (sqrtUpperNat n : Nat) := rfl

/-- Upper bound on `1 / sqrt(n)` using `Nat.sqrt` (floor). -/
def invSqrtUpperBound (n : Nat) : Rat :=
  if n = 0 then 0 else (1 : Rat) / (Nat.sqrt n : Nat)

theorem invSqrtUpperBound_def (n : Nat) :
    invSqrtUpperBound n = if n = 0 then 0 else (1 : Rat) / (Nat.sqrt n : Nat) := rfl

/-- Conservative bound on `max |LayerNorm(x)|` after affine (uses only `γ`, `β`, and `dim`). -/
def layerNormOutputMaxAbsBound (dim : Nat) (maxAbsGamma maxAbsBeta : Rat) : Rat :=
  maxAbsGamma * sqrtUpperRat dim + maxAbsBeta

theorem layerNormOutputMaxAbsBound_def (dim : Nat) (maxAbsGamma maxAbsBeta : Rat) :
    layerNormOutputMaxAbsBound dim maxAbsGamma maxAbsBeta =
      maxAbsGamma * sqrtUpperRat dim + maxAbsBeta := rfl

/-- Score-gradient L1 bound for attention pattern terms. -/
def attnScoreGradBound (seqLen modelDim headDim : Nat)
    (ln1OutMaxAbs wqBound wkBound : Rat) : Rat :=
  let scale := invSqrtUpperBound headDim
  (seqLen : Rat) * scale *
    ((2 : Rat) * (modelDim : Rat) * ln1OutMaxAbs * wqBound * wkBound)

theorem attnScoreGradBound_def (seqLen modelDim headDim : Nat)
    (ln1OutMaxAbs wqBound wkBound : Rat) :
    attnScoreGradBound seqLen modelDim headDim ln1OutMaxAbs wqBound wkBound =
      let scale := invSqrtUpperBound headDim
      (seqLen : Rat) * scale *
        ((2 : Rat) * (modelDim : Rat) * ln1OutMaxAbs * wqBound * wkBound) := rfl

/-- Pattern-term coefficient bound from value and score-gradient bounds. -/
def attnPatternCoeffBound (seqLen modelDim headDim : Nat)
    (ln1OutMaxAbs wqBound wkBound valueCoeff : Rat) : Rat :=
  let inputL1 := (modelDim : Rat) * ln1OutMaxAbs
  (seqLen : Rat) *
    attnScoreGradBound seqLen modelDim headDim ln1OutMaxAbs wqBound wkBound *
      (inputL1 * valueCoeff)

theorem attnPatternCoeffBound_def (seqLen modelDim headDim : Nat)
    (ln1OutMaxAbs wqBound wkBound valueCoeff : Rat) :
    attnPatternCoeffBound seqLen modelDim headDim ln1OutMaxAbs wqBound wkBound valueCoeff =
      let inputL1 := (modelDim : Rat) * ln1OutMaxAbs
      (seqLen : Rat) *
        attnScoreGradBound seqLen modelDim headDim ln1OutMaxAbs wqBound wkBound *
          (inputL1 * valueCoeff) := rfl

/-- Conservative bound on `|q·k|/sqrt(d_head)` using max-abs LN1 output and W_Q/W_K norms. -/
def attnScoreAbsBound (modelDim headDim : Nat)
    (ln1OutMaxAbs wqBound wkBound : Rat) : Rat :=
  let dRat : Rat := (modelDim : Nat)
  let qMax := dRat * ln1OutMaxAbs * wqBound
  let kMax := dRat * ln1OutMaxAbs * wkBound
  sqrtUpperRat headDim * qMax * kMax

theorem attnScoreAbsBound_def (modelDim headDim : Nat)
    (ln1OutMaxAbs wqBound wkBound : Rat) :
    attnScoreAbsBound modelDim headDim ln1OutMaxAbs wqBound wkBound =
      let dRat : Rat := (modelDim : Nat)
      let qMax := dRat * ln1OutMaxAbs * wqBound
      let kMax := dRat * ln1OutMaxAbs * wkBound
      sqrtUpperRat headDim * qMax * kMax := rfl

end Nfp.Sound
