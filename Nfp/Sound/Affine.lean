-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std
import Nfp.Sound.Interval

namespace Nfp.Sound

/-!
# Affine arithmetic scaffolding (SOUND)

This module provides a minimal affine-form representation for local certification
improvements, used by optional affine Q/K bounds in SOUND best-match paths.
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
      let ai := a.getD i 0
      let bi := b.getD i 0
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

/-- Append a fresh independent noise coefficient (skipped if zero). -/
def appendNoise (a : AffineForm) (coeff : Rat) : AffineForm :=
  if coeff = 0 then
    a
  else
    { center := a.center, coeffs := a.coeffs.push coeff }

/-- Sum of absolute noise coefficients (radius of the interval hull). -/
def radius (a : AffineForm) : Rat :=
  a.coeffs.foldl (fun acc c => acc + ratAbs c) 0

/-- Interval hull of an affine form. -/
def toInterval (a : AffineForm) : RatInterval :=
  let r := radius a
  { lo := a.center - r, hi := a.center + r }

/-- Affine multiplication with aligned noise terms and a single remainder noise. -/
def mul (a b : AffineForm) : AffineForm :=
  let coeffs := combineCoeffs a.coeffs b.coeffs
    (fun ai bi => b.center * ai + a.center * bi)
  let rem := radius a * radius b
  appendNoise { center := a.center * b.center, coeffs := coeffs } rem

/-- Affine multiplication treating noise terms as disjoint. -/
def mulDisjoint (a b : AffineForm) : AffineForm :=
  Id.run do
    let mut coeffs : Array Rat := Array.mkEmpty (a.coeffs.size + b.coeffs.size)
    for ai in a.coeffs do
      coeffs := coeffs.push (b.center * ai)
    for bi in b.coeffs do
      coeffs := coeffs.push (a.center * bi)
    let rem := radius a * radius b
    return appendNoise { center := a.center * b.center, coeffs := coeffs } rem

/-! ### Specs -/

theorem AffineForm_spec : AffineForm = AffineForm := rfl
theorem const_spec : const = const := rfl
theorem combineCoeffs_spec : combineCoeffs = combineCoeffs := rfl
theorem add_spec : add = add := rfl
theorem sub_spec : sub = sub := rfl
theorem scale_spec : scale = scale := rfl
theorem appendNoise_spec : appendNoise = appendNoise := rfl
theorem radius_spec : radius = radius := rfl
theorem toInterval_spec : toInterval = toInterval := rfl
theorem mul_spec : mul = mul := rfl
theorem mulDisjoint_spec : mulDisjoint = mulDisjoint := rfl

end AffineForm

end Nfp.Sound
