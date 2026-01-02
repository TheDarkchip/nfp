-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Sound.Activation

namespace Nfp.Sound

/-!
# GeLU derivative bounds
-/

/-- Global conservative GeLU derivative bound (independent of interval). -/
def geluDerivBoundGlobal : GeluDerivTarget → Rat
  | .tanh => 2
  | .exact => 2

theorem geluDerivBoundGlobal_def (t : GeluDerivTarget) :
    geluDerivBoundGlobal t = match t with | .tanh => 2 | .exact => 2 := rfl

end Nfp.Sound
