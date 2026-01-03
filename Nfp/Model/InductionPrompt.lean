-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Data.Finset.Max
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

/-- `prev` map induced by token repeats (defaulting to `0` when no prior match exists). -/
def prevOfTokens {seq : Nat} (tokens : Fin seq → Nat) (q : Fin seq) : Fin seq := by
  classical
  let hpos : 0 < seq := lt_of_le_of_lt (Nat.zero_le _) q.isLt
  let zero : Fin seq := ⟨0, hpos⟩
  let candidates : Finset (Fin seq) :=
    (Finset.univ : Finset (Fin seq)).filter (fun k =>
      k.val < q.val ∧ tokens k = tokens q)
  by_cases h : candidates.Nonempty
  · exact Finset.max' candidates h
  · exact zero

/-- Active queries induced by token repeats. -/
def activeOfTokens {seq : Nat} (tokens : Fin seq → Nat) : Finset (Fin seq) :=
  (Finset.univ : Finset (Fin seq)).filter (fun q =>
    ∃ k, k.val < q.val ∧ tokens k = tokens q)

/-- Membership characterization for `activeOfTokens`. -/
theorem mem_activeOfTokens {seq : Nat} {tokens : Fin seq → Nat} {q : Fin seq} :
    q ∈ activeOfTokens tokens ↔ ∃ k, k.val < q.val ∧ tokens k = tokens q := by
  simp [activeOfTokens]

end Model

end Nfp
