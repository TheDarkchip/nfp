-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Core.Basic
import Nfp.Circuit.Cert

/-!
Residual-stream interval certificates.

These certificates record per-coordinate lower/upper bounds for residual vectors.
-/

namespace Nfp

namespace Circuit

/-- Certificate payload for per-coordinate residual intervals. -/
structure ResidualIntervalCert (n : Nat) where
  /-- Lower bound per coordinate. -/
  lo : Fin n → Rat
  /-- Upper bound per coordinate. -/
  hi : Fin n → Rat

/-- Properties enforced by `checkResidualIntervalCert`. -/
structure ResidualIntervalBounds {n : Nat} (c : ResidualIntervalCert n) : Prop where
  /-- Lower bounds are at most upper bounds. -/
  lo_le_hi : ∀ i, c.lo i ≤ c.hi i

/-- Boolean checker for residual-interval certificates. -/
def checkResidualIntervalCert {n : Nat} (c : ResidualIntervalCert n) : Bool :=
  finsetAll (Finset.univ : Finset (Fin n)) (fun i => decide (c.lo i ≤ c.hi i))

/-- `checkResidualIntervalCert` is sound for `ResidualIntervalBounds`. -/
theorem checkResidualIntervalCert_sound {n : Nat} (c : ResidualIntervalCert n) :
    checkResidualIntervalCert c = true → ResidualIntervalBounds c := by
  intro hcheck
  have hall :
      finsetAll (Finset.univ : Finset (Fin n)) (fun i =>
        decide (c.lo i ≤ c.hi i)) = true := by
    simpa [checkResidualIntervalCert] using hcheck
  have hall' :=
    (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin n)))).1 hall
  refine { lo_le_hi := ?_ }
  intro i
  have hi := hall' i (by simp)
  simpa [decide_eq_true_iff] using hi

end Circuit

end Nfp
