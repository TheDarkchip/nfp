-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Core.Basic
import Nfp.Sound.Linear.FinFold

/-!
Unnormalized dyadic arithmetic.

Dyadic values already avoid gcd normalization, so this module provides a
lightweight alias and helper API used by older code paths.
-/

namespace Nfp

namespace Sound

namespace Bounds

/-- Unnormalized dyadic value (alias). -/
abbrev UnnormDyadic := Dyadic

/-- Interpret an unnormalized dyadic as a dyadic. -/
def UnnormDyadic.toDyadic (q : UnnormDyadic) : Dyadic :=
  q

/-- Embed a dyadic as an unnormalized dyadic. -/
def UnnormDyadic.ofDyadic (q : Dyadic) : UnnormDyadic :=
  q

/-- Unnormalized zero. -/
def UnnormDyadic.zero : UnnormDyadic := 0

/-- Unnormalized addition. -/
def UnnormDyadic.add (a b : UnnormDyadic) : UnnormDyadic :=
  a + b

/-- Unnormalized multiplication. -/
def UnnormDyadic.mul (a b : UnnormDyadic) : UnnormDyadic :=
  a * b

/-- `toDyadic` respects multiplication. -/
theorem UnnormDyadic.toDyadic_mul_ofDyadic (a b : Dyadic) :
    UnnormDyadic.toDyadic (UnnormDyadic.mul (UnnormDyadic.ofDyadic a)
      (UnnormDyadic.ofDyadic b)) = a * b := by
  rfl

/-- Tail-recursive sum of unnormalized dyadics. -/
def UnnormDyadic.sumFin (n : Nat) (f : Fin n → UnnormDyadic) : UnnormDyadic :=
  Linear.sumFin n f

/-- `toDyadic` commutes with `sumFin`. -/
theorem UnnormDyadic.toDyadic_sumFin (n : Nat) (f : Fin n → UnnormDyadic) :
    UnnormDyadic.toDyadic (UnnormDyadic.sumFin n f) =
      Linear.sumFin n (fun i => UnnormDyadic.toDyadic (f i)) := by
  rfl

end Bounds

end Sound

end Nfp
