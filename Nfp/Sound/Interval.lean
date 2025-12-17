-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Bounds

namespace Nfp.Sound

/-!
# Minimal `Rat` interval helpers for local certification

This module is intentionally tiny: it provides only what we need to obtain a **lower bound**
on row-wise variance for LayerNorm over an ℓ∞ ball around a concrete row `x₀`.

If the row is `x₀ : ℚ^d` and we certify on the region `‖x - x₀‖∞ ≤ δ`, then:
- the mean shifts by at most `δ`, and
- each centered coordinate shifts by at most `2δ`.

For a centered coordinate `c = x₀[j] - mean(x₀)`, the interval is `[c-2δ, c+2δ]`.
If it does not include `0`, then `min |c'| ≥ |c| - 2δ`, so we get a variance lower bound by
summing `(max (|c|-2δ) 0)^2`.
-/

private def ratSq (x : Rat) : Rat := x * x

/-- Lower bound on the per-row variance of LayerNorm on an ℓ∞ ball around `row0`.

This is a *sound* lower bound in exact rational arithmetic.
-/
def varianceLowerBoundLinfBall (row0 : Array Rat) (delta : Rat) : Rat :=
  if row0.isEmpty then
    0
  else
    Id.run do
      let d : Nat := row0.size
      let dRat : Rat := (d : Nat)
      let sum : Rat := row0.foldl (fun acc x => acc + x) 0
      let mu0 : Rat := sum / dRat
      let twoDelta : Rat := 2 * delta
      let mut acc : Rat := 0
      for x in row0 do
        let c := x - mu0
        let a := ratAbs c
        if a > twoDelta then
          let m := a - twoDelta
          acc := acc + ratSq m
        else
          pure ()
      return acc / dRat

end Nfp.Sound
