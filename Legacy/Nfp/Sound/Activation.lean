-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std

namespace Nfp.Sound

/-!
# Activation metadata (SOUND)

This module defines activation-derivative targets used by SOUND certification and
parsing helpers for `.nfpt` headers.
-/

/-- Which GeLU derivative formula the model uses. -/
inductive GeluDerivTarget
  | tanh
  | exact
  deriving Repr, DecidableEq

/-- Parse a GeLU derivative target string (case-insensitive). -/
def geluDerivTargetOfString (s : String) : Option GeluDerivTarget :=
  let v := s.trim.toLower
  if v = "tanh" || v = "gelu_tanh" then
    some .tanh
  else if v = "exact" || v = "gelu_exact" then
    some .exact
  else
    none

/-- Render a GeLU derivative target for headers/logging. -/
def geluDerivTargetToString : GeluDerivTarget → String
  | .tanh => "tanh"
  | .exact => "exact"

/-! ### Specs -/

theorem GeluDerivTarget_spec : GeluDerivTarget = GeluDerivTarget := rfl
theorem geluDerivTargetOfString_spec :
    geluDerivTargetOfString = geluDerivTargetOfString := rfl
theorem geluDerivTargetToString_spec :
    geluDerivTargetToString = geluDerivTargetToString := rfl

end Nfp.Sound
