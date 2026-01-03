-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.Order.Ring.Rat
import Nfp.Circuit.Cert

/-!
Residual-stream bound certificates.

These certificates record per-coordinate absolute bounds for residual vectors.
-/

namespace Nfp

namespace Circuit

/-- Certificate payload for per-coordinate residual absolute bounds. -/
structure ResidualBoundCert (n : Nat) where
  /-- Absolute bound per coordinate. -/
  bound : Fin n → Rat

/-- Properties enforced by `checkResidualBoundCert`. -/
structure ResidualBoundBounds {n : Nat} (c : ResidualBoundCert n) : Prop where
  /-- Residual bounds are nonnegative. -/
  bound_nonneg : ∀ i, 0 ≤ c.bound i

/-- Boolean checker for residual-bound certificates. -/
def checkResidualBoundCert {n : Nat} (c : ResidualBoundCert n) : Bool :=
  finsetAll (Finset.univ : Finset (Fin n)) (fun i => decide (0 ≤ c.bound i))

/-- `checkResidualBoundCert` is sound for `ResidualBoundBounds`. -/
theorem checkResidualBoundCert_sound {n : Nat} (c : ResidualBoundCert n) :
    checkResidualBoundCert c = true → ResidualBoundBounds c := by
  intro hcheck
  have hall :
      finsetAll (Finset.univ : Finset (Fin n)) (fun i =>
        decide (0 ≤ c.bound i)) = true := by
    simpa [checkResidualBoundCert] using hcheck
  have hall' :=
    (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin n)))).1 hall
  refine { bound_nonneg := ?_ }
  intro i
  have hi := hall' i (by simp)
  exact (decide_eq_true_iff).1 hi

end Circuit

end Nfp
