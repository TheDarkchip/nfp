-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Nfp.Core.Basic
public import Nfp.Circuit.Cert.Basic
public import Nfp.Circuit.Layers.Induction

/-!
Value-range certificates for attention value vectors.
-/

@[expose] public section

namespace Nfp

namespace Circuit

open scoped BigOperators

variable {seq : Nat}

/-- Metadata describing a logit-diff direction (target minus negative token). -/
structure DirectionSpec where
  /-- Target token id for the logit-diff direction. -/
  target : Nat
  /-- Negative token id for the logit-diff direction. -/
  negative : Nat

/-- Certificate payload for value-range bounds (Rat-valued). -/
structure ValueRangeCert (seq : Nat) where
  /-- Lower bound for values. -/
  lo : Rat
  /-- Upper bound for values. -/
  hi : Rat
  /-- Value entries. -/
  vals : Fin seq → Rat
  /-- Optional logit-diff direction metadata (ignored by the checker). -/
  direction : Option DirectionSpec

/-- Boolean checker for value-range certificates. -/
def checkValueRangeCert [NeZero seq] (c : ValueRangeCert seq) : Bool :=
  decide (c.lo ≤ c.hi) &&
    finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
      decide (c.lo ≤ c.vals k) && decide (c.vals k ≤ c.hi))

/-- `checkValueRangeCert` is sound for `ValueRangeBounds`. -/
theorem checkValueRangeCert_sound [NeZero seq] (c : ValueRangeCert seq) :
    checkValueRangeCert c = true →
      Layers.ValueRangeBounds (Val := Rat) c.lo c.hi c.vals := by
  classical
  intro hcheck
  have hcheck' :
      decide (c.lo ≤ c.hi) = true ∧
        finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
          decide (c.lo ≤ c.vals k) && decide (c.vals k ≤ c.hi)) = true := by
    simpa [checkValueRangeCert, Bool.and_eq_true] using hcheck
  rcases hcheck' with ⟨hlohi, hall⟩
  have hlohi' : c.lo ≤ c.hi := by
    simpa [decide_eq_true_iff] using hlohi
  have hall' :=
    (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hall
  have hbounds : ∀ k, c.lo ≤ c.vals k ∧ c.vals k ≤ c.hi := by
    intro k
    have hk := hall' k (by simp)
    simpa [Bool.and_eq_true, decide_eq_true_iff] using hk
  exact
    { lo_le_hi := hlohi'
      lo_le := fun k => (hbounds k).1
      le_hi := fun k => (hbounds k).2 }

end Circuit

end Nfp
