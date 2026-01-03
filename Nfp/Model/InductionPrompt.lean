-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Fintype.Basic

/-!
Helpers for induction-style prompts.

These are small, deterministic utilities for constructing the `prev` map and
active-query set from a fixed period. They keep the prompt bookkeeping
separate from the model weights.
-/

namespace Nfp

namespace Model

/-- `prev` map for a periodic induction prompt: `q ↦ q - period` (truncated at 0). -/
def prevOfPeriod {seq : Nat} (period : Nat) (q : Fin seq) : Fin seq :=
  ⟨q.val - period, lt_of_le_of_lt (Nat.sub_le _ _) q.isLt⟩

/-- Active queries for a periodic induction prompt (`period ≤ q`). -/
def activeOfPeriod {seq : Nat} (period : Nat) : Finset (Fin seq) :=
  (Finset.univ : Finset (Fin seq)).filter (fun q => period ≤ q.val)

/-- Membership characterization for `activeOfPeriod`. -/
theorem mem_activeOfPeriod {seq : Nat} {period : Nat} {q : Fin seq} :
    q ∈ activeOfPeriod (seq := seq) period ↔ period ≤ q.val := by
  simp [activeOfPeriod]

end Model

end Nfp
