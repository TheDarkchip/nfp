-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Core.Basic
public import Nfp.Sound.Linear.FinFold

/-!
Unnormalized rational arithmetic.

Rat values already avoid gcd normalization, so this module provides a
lightweight alias and helper API used by older code paths.
-/

@[expose] public section

namespace Nfp

namespace Sound

namespace Bounds

/-- Unnormalized rational value (alias). -/
abbrev UnnormRat := Rat

/-- Interpret an unnormalized rational as a rational. -/
def UnnormRat.toRat (q : UnnormRat) : Rat :=
  q

/-- Embed a rational as an unnormalized rational. -/
def UnnormRat.ofRat (q : Rat) : UnnormRat :=
  q

/-- Unnormalized zero. -/
def UnnormRat.zero : UnnormRat := 0

/-- Unnormalized addition. -/
def UnnormRat.add (a b : UnnormRat) : UnnormRat :=
  a + b

/-- Unnormalized multiplication. -/
def UnnormRat.mul (a b : UnnormRat) : UnnormRat :=
  a * b

/-- `toRat` respects multiplication. -/
theorem UnnormRat.toRat_mul_ofRat (a b : Rat) :
    UnnormRat.toRat (UnnormRat.mul (UnnormRat.ofRat a) (UnnormRat.ofRat b)) = a * b := by
  rfl

/-- Tail-recursive sum of unnormalized rationals. -/
def UnnormRat.sumFin (n : Nat) (f : Fin n → UnnormRat) : UnnormRat :=
  Linear.sumFin n f

/-- `toRat` commutes with `sumFin`. -/
theorem UnnormRat.toRat_sumFin (n : Nat) (f : Fin n → UnnormRat) :
    UnnormRat.toRat (UnnormRat.sumFin n f) =
      Linear.sumFin n (fun i => UnnormRat.toRat (f i)) := by
  rfl

end Bounds

end Sound

end Nfp
