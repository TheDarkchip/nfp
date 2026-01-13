-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Core.Basic
public import Nfp.Circuit.Cert.Basic

/-!
Downstream linear certificates for end-to-end induction bounds.

These certificates record a nonnegative error bound computed externally.
The checker only verifies arithmetic consistency (`error = gain * inputBound`)
and nonnegativity of the reported quantities.
-/

public section

namespace Nfp

namespace Circuit

/-- Certificate payload for downstream linear error bounds. -/
structure DownstreamLinearCert where
  /-- Upper bound on the downstream logit-diff error. -/
  error : Rat
  /-- Operator gain bound used to justify the error. -/
  gain : Rat
  /-- Input magnitude bound used to justify the error. -/
  inputBound : Rat

/-- Arithmetic properties enforced by `checkDownstreamLinearCert`. -/
structure DownstreamLinearBounds (c : DownstreamLinearCert) : Prop where
  /-- Error bound is nonnegative. -/
  error_nonneg : 0 ≤ c.error
  /-- Gain bound is nonnegative. -/
  gain_nonneg : 0 ≤ c.gain
  /-- Input bound is nonnegative. -/
  input_nonneg : 0 ≤ c.inputBound
  /-- Error bound matches the reported gain/input product. -/
  error_eq : c.error = c.gain * c.inputBound

/-- Boolean checker for downstream linear certificates. -/
def checkDownstreamLinearCert (c : DownstreamLinearCert) : Bool :=
  decide (0 ≤ c.error) &&
    decide (0 ≤ c.gain) &&
    decide (0 ≤ c.inputBound) &&
    decide (c.error = c.gain * c.inputBound)

/-- `checkDownstreamLinearCert` is sound for `DownstreamLinearBounds`. -/
theorem checkDownstreamLinearCert_sound (c : DownstreamLinearCert) :
    checkDownstreamLinearCert c = true → DownstreamLinearBounds c := by
  intro h
  have h' :
      ((0 ≤ c.error ∧ 0 ≤ c.gain) ∧ 0 ≤ c.inputBound) ∧
        c.error = c.gain * c.inputBound := by
    simpa [checkDownstreamLinearCert, Bool.and_eq_true, decide_eq_true_iff] using h
  rcases h' with ⟨⟨⟨herror, hgain⟩, hinput⟩, heq⟩
  refine
    { error_nonneg := herror
      gain_nonneg := hgain
      input_nonneg := hinput
      error_eq := heq }

end Circuit

end Nfp
