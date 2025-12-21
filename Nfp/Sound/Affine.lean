-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Interval

namespace Nfp.Sound

/-!
# Affine arithmetic scaffolding (SOUND)

This module provides a minimal affine-form representation for future local
certification improvements. It is not yet integrated into the SOUND pipeline.
-/

/-- Affine form `x = center + sum coeffs[i] * eps_i` with `eps_i in [-1, 1]`. -/
structure AffineForm where
  center : Rat
  coeffs : Array Rat
  deriving Repr

namespace AffineForm

/-- Constant affine form. -/
def const (x : Rat) : AffineForm := { center := x, coeffs := #[] }

private def combineCoeffs (a b : Array Rat) (f : Rat → Rat → Rat) : Array Rat :=
  Id.run do
    let n := max a.size b.size
    let mut out := Array.mkEmpty n
    for i in [:n] do
      let ai := a.get? i |>.getD 0
      let bi := b.get? i |>.getD 0
      out := out.push (f ai bi)
    return out

/-- Add two affine forms, aligning noise terms by index. -/
def add (a b : AffineForm) : AffineForm :=
  { center := a.center + b.center
    coeffs := combineCoeffs a.coeffs b.coeffs (· + ·) }

/-- Subtract two affine forms, aligning noise terms by index. -/
def sub (a b : AffineForm) : AffineForm :=
  { center := a.center - b.center
    coeffs := combineCoeffs a.coeffs b.coeffs (· - ·) }

/-- Scale an affine form by a rational constant. -/
def scale (c : Rat) (a : AffineForm) : AffineForm :=
  { center := c * a.center
    coeffs := a.coeffs.map (fun k => c * k) }

/-- Sum of absolute noise coefficients (radius of the interval hull). -/
def radius (a : AffineForm) : Rat :=
  a.coeffs.foldl (fun acc c => acc + ratAbs c) 0

/-- Interval hull of an affine form. -/
def toInterval (a : AffineForm) : RatInterval :=
  let r := radius a
  { lo := a.center - r, hi := a.center + r }

/-! ### Specs -/

theorem AffineForm_spec : AffineForm = AffineForm := rfl
theorem const_spec : const = const := rfl
theorem combineCoeffs_spec : combineCoeffs = combineCoeffs := rfl
theorem add_spec : add = add := rfl
theorem sub_spec : sub = sub := rfl
theorem scale_spec : scale = scale := rfl
theorem radius_spec : radius = radius := rfl
theorem toInterval_spec : toInterval = toInterval := rfl

end AffineForm

end Nfp.Sound
