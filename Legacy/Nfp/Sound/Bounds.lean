-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Sound.Bounds.Basic
import Nfp.Sound.Bounds.MatrixNorm
import Nfp.Sound.Bounds.Gelu
import Nfp.Sound.Bounds.Exp
import Nfp.Sound.Bounds.Softmax
import Nfp.Sound.Bounds.Attention
import Nfp.Sound.Bounds.LayerNorm
import Nfp.Sound.Bounds.Portfolio
import Nfp.Sound.Bounds.Effort

/-!
# Sound bounds in exact arithmetic

This module is an umbrella import for the sound bound utilities.
Numeric strategy (Option A): avoid `sqrt` and any `Float`-trusted computation by using
row-sum induced norms (ℓ1 for row-vector convention) and submultiplicativity.
-/
