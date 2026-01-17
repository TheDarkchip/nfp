-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Circuit.Layers.Induction.Basic

/-!
Circuit-level induction specs: previous-token head feeding an induction head.

These are definitional wrappers that name the canonical two-head induction
mechanism used in the literature: a previous-token head writes the prior token
representation into the residual stream, and the induction head then copies the
appropriate continuation.
-/

public section

namespace Nfp

namespace Circuit

namespace Layers

universe v

section Specs

variable {Val : Type v}
variable {n : Nat}

/--
Two-head induction circuit spec.

The previous-token head copies `vals (q-1)` into `prevOut`, and the induction
head copies from `prevOut (prev q)` into `indOut`.
-/
def InductionCircuitSpec
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (prevOut indOut vals : Fin (Nat.succ n) → Val) : Prop :=
  PrevTokenSpec (n := n) prevOut vals ∧
    InductionSpec (n := n) prev indOut prevOut

/-- Unfolding lemma for `InductionCircuitSpec`. -/
theorem InductionCircuitSpec_def
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (prevOut indOut vals : Fin (Nat.succ n) → Val) :
    InductionCircuitSpec (n := n) prev prevOut indOut vals =
      (PrevTokenSpec (n := n) prevOut vals ∧
        InductionSpec (n := n) prev indOut prevOut) := by
  rfl

/--
Circuit composition: the induction head copies the predecessor of the predecessor.

This is the direct consequence of a previous-token head feeding the induction head.
-/
theorem InductionCircuitSpec_compose
    (prev : Fin (Nat.succ n) → Fin (Nat.succ n))
    (prevOut indOut vals : Fin (Nat.succ n) → Val)
    (h : InductionCircuitSpec (n := n) prev prevOut indOut vals) :
    ∀ q, q ≠ 0 → prev q ≠ 0 →
      indOut q = vals (prevIndex (n := n) (prev q)) := by
  intro q hq hprev
  rcases h with ⟨hprevTok, hind⟩
  have hprevTok' :
      ∀ q, q ≠ 0 → prevOut q = vals (prevIndex (n := n) q) := by
    simpa [PrevTokenSpec_def, InductionSpec_def] using hprevTok
  have hind' : ∀ q, q ≠ 0 → indOut q = prevOut (prev q) := by
    simpa [InductionSpec_def] using hind
  calc
    indOut q = prevOut (prev q) := hind' q hq
    _ = vals (prevIndex (n := n) (prev q)) := hprevTok' (prev q) hprev

end Specs

end Layers

end Circuit

end Nfp
