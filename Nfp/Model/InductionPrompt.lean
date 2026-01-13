-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Finset.Max
public import Mathlib.Data.Fintype.Basic

/-!
Helpers for induction-style prompts.

These are small, deterministic utilities for constructing the `prev` map and
active-query set from a fixed period. They keep the prompt bookkeeping
separate from the model weights.
-/

@[expose] public section

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

/-- If a prior matching token exists, `prevOfTokens` picks a matching index and is maximal. -/
theorem prevOfTokens_spec {seq : Nat} {tokens : Fin seq → Nat} {q : Fin seq}
    (h : ∃ k, k < q ∧ tokens k = tokens q) :
    let p := prevOfTokens tokens q
    p < q ∧ tokens p = tokens q ∧
      ∀ k, k < q → tokens k = tokens q → k ≤ p := by
  classical
  let candidates : Finset (Fin seq) :=
    (Finset.univ : Finset (Fin seq)).filter (fun k =>
      k < q ∧ tokens k = tokens q)
  have hnonempty : candidates.Nonempty := by
    rcases h with ⟨k, hk, htok⟩
    exact ⟨k, by simp [candidates, hk, htok]⟩
  by_cases h' : candidates.Nonempty
  · have hmem : Finset.max' candidates h' ∈ candidates :=
      Finset.max'_mem candidates h'
    have hcond :
        Finset.max' candidates h' < q ∧
          tokens (Finset.max' candidates h') = tokens q := by
      have hmem' := (Finset.mem_filter.1 hmem).2
      simpa using hmem'
    have hmax :
        ∀ k, k < q → tokens k = tokens q →
          k ≤ Finset.max' candidates h' := by
      intro k hk htok
      have hk_mem : k ∈ candidates := by
        simp [candidates, hk, htok]
      have hk_mem' : k ∈ (candidates : Set (Fin seq)) := by
        simpa using hk_mem
      exact (Finset.isGreatest_max' (s := candidates) h').2 hk_mem'
    simpa [prevOfTokens, candidates, h'] using
      And.intro hcond.1 (And.intro hcond.2 hmax)
  · exact (h' hnonempty).elim

/-- Active queries imply the `prevOfTokens` maximal-match specification. -/
theorem prevOfTokens_spec_of_active {seq : Nat} {tokens : Fin seq → Nat} {q : Fin seq}
    (hq : q ∈ activeOfTokens tokens) :
    let p := prevOfTokens tokens q
    p < q ∧ tokens p = tokens q ∧
      ∀ k, k < q → tokens k = tokens q → k ≤ p := by
  have h := (mem_activeOfTokens (tokens := tokens) (q := q)).1 hq
  rcases h with ⟨k, hk, htok⟩
  have hk' : k < q := by
    exact (Fin.lt_def).2 hk
  exact prevOfTokens_spec (tokens := tokens) (q := q) ⟨k, hk', htok⟩

end Model

end Nfp
