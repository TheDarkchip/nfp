-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Unbundled.Rat

namespace Nfp.Sound

/-!
# Basic Rat helpers

Small utilities used across the sound bounds modules.
-/

/-- Exact absolute value on `Rat`. -/
def ratAbs (x : Rat) : Rat :=
  if x < 0 then -x else x

theorem ratAbs_def (x : Rat) : ratAbs x = if x < 0 then -x else x := rfl

end Nfp.Sound
