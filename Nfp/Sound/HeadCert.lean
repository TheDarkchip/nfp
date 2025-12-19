-- SPDX-License-Identifier: AGPL-3.0-or-later

import Std

namespace Nfp.Sound

/-!
# Sound per-head contribution certificates

This module defines a minimal, checkable certificate for per-head weight-only
contribution bounds. These are intended as a lightweight starting point for
sound circuit certification.
-/

/-- Weight-only per-head operator-norm bounds and derived factors. -/
structure HeadContributionCert where
  layerIdx : Nat
  headIdx : Nat
  wqOpBound : Rat
  wkOpBound : Rat
  wvOpBound : Rat
  woOpBound : Rat
  qkFactorBound : Rat
  voFactorBound : Rat
  deriving Repr

namespace HeadContributionCert

/-- Internal consistency checks for derived factor bounds. -/
def Valid (c : HeadContributionCert) : Prop :=
  c.qkFactorBound = c.wqOpBound * c.wkOpBound ∧
    c.voFactorBound = c.wvOpBound * c.woOpBound

instance (c : HeadContributionCert) : Decidable (Valid c) := by
  unfold Valid
  infer_instance

/-- Boolean checker for `Valid`. -/
def check (c : HeadContributionCert) : Bool :=
  decide (Valid c)

theorem check_iff (c : HeadContributionCert) : c.check = true ↔ c.Valid := by
  simp [check, Valid]

end HeadContributionCert

end Nfp.Sound
