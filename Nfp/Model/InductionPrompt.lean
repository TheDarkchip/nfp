-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Data.Finset.Max
public import Mathlib.Data.Fintype.Basic
public import Nfp.Model.InductionHead

/-!
Helpers for induction-style prompts.

These are small, deterministic utilities for constructing the `prev` map and
active-query set from a fixed period. They keep the prompt bookkeeping
separate from the model weights.
-/

public section

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

/--
Shifted `prev` map for a periodic induction prompt: if `0 < period` and
`period ≤ q`, return `q - period + 1`; otherwise default to `0`.
-/
def prevOfPeriodShift {seq : Nat} (period : Nat) (q : Fin seq) : Fin seq := by
  classical
  by_cases hq : period ≤ q.val
  · by_cases hper : 0 < period
    · have hlt : q.val - period + 1 < seq := by
        have hsub : q.val - period < q.val := Nat.sub_lt_of_pos_le hper hq
        have hle : q.val - period + 1 ≤ q.val := Nat.succ_le_of_lt hsub
        exact lt_of_le_of_lt hle q.isLt
      exact ⟨q.val - period + 1, hlt⟩
    · have hpos : 0 < seq := lt_of_le_of_lt (Nat.zero_le _) q.isLt
      exact ⟨0, hpos⟩
  · have hpos : 0 < seq := lt_of_le_of_lt (Nat.zero_le _) q.isLt
    exact ⟨0, hpos⟩

/-- Active queries for shifted periodic induction prompts (`0 < period ≤ q`). -/
def activeOfPeriodShift {seq : Nat} (period : Nat) : Finset (Fin seq) :=
  (Finset.univ : Finset (Fin seq)).filter (fun q => 0 < period ∧ period ≤ q.val)

/-- Membership characterization for `activeOfPeriodShift`. -/
theorem mem_activeOfPeriodShift {seq : Nat} {period : Nat} {q : Fin seq} :
    q ∈ activeOfPeriodShift (seq := seq) period ↔ 0 < period ∧ period ≤ q.val := by
  simp [activeOfPeriodShift]

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

/--
Shifted `prev` map for induction: match the current token to its previous
occurrence and return the following position (`A B ... A -> B`).

Example (tokens = [1,2,1,3,2,1]):
  prevShift = [0,0,0,1,0,2], activeShift = {3,5}
-/
def prevOfTokensShift {seq : Nat} (tokens : Fin seq → Nat) (q : Fin seq) : Fin seq := by
  classical
  by_cases hq : q ∈ activeOfTokens tokens
  · let p := prevOfTokens tokens q
    have hp :
        p < q ∧ tokens p = tokens q ∧
          ∀ k, k < q → tokens k = tokens q → k ≤ p := by
      simpa [p] using
        (prevOfTokens_spec_of_active (tokens := tokens) (q := q) hq)
    have hpv : p.val < q.val := (Fin.lt_def).1 hp.1
    have hle : p.val + 1 ≤ q.val := Nat.succ_le_of_lt hpv
    have hlt : p.val + 1 < seq := lt_of_le_of_lt hle q.isLt
    exact ⟨p.val + 1, hlt⟩
  · let hpos : 0 < seq := lt_of_le_of_lt (Nat.zero_le _) q.isLt
    exact ⟨0, hpos⟩

/-- Active queries for shifted-token induction (same witness condition). -/
def activeOfTokensShift {seq : Nat} (tokens : Fin seq → Nat) : Finset (Fin seq) :=
  activeOfTokens tokens

/-- Membership characterization for `activeOfTokensShift`. -/
theorem mem_activeOfTokensShift {seq : Nat} {tokens : Fin seq → Nat} {q : Fin seq} :
    q ∈ activeOfTokensShift tokens ↔ ∃ k, k.val < q.val ∧ tokens k = tokens q := by
  simp [activeOfTokensShift, activeOfTokens]

/-- Shifted `prev` agrees with the maximal previous match, advanced by one. -/
theorem prevOfTokensShift_spec {seq : Nat} {tokens : Fin seq → Nat} {q : Fin seq}
    (h : ∃ k, k < q ∧ tokens k = tokens q) :
    let p := prevOfTokensShift tokens q
    let p0 := prevOfTokens tokens q
    p.val = p0.val + 1 ∧
      p0 < q ∧ tokens p0 = tokens q ∧
        ∀ k, k < q → tokens k = tokens q → k ≤ p0 := by
  classical
  have hactive : q ∈ activeOfTokens tokens := by
    rcases h with ⟨k, hk, htok⟩
    exact (mem_activeOfTokens (tokens := tokens) (q := q)).2
      ⟨k, (Fin.lt_def).1 hk, htok⟩
  let p0 := prevOfTokens tokens q
  have hp0 :
      p0 < q ∧ tokens p0 = tokens q ∧
        ∀ k, k < q → tokens k = tokens q → k ≤ p0 := by
    simpa [p0] using (prevOfTokens_spec (tokens := tokens) (q := q) h)
  have hpval : (prevOfTokensShift tokens q).val = p0.val + 1 := by
    simpa [prevOfTokensShift, hactive, p0]
  simpa [p0, hpval, hp0]

/-- Active shifted queries imply the shifted `prev` maximal-match specification. -/
theorem prevOfTokensShift_spec_of_active {seq : Nat} {tokens : Fin seq → Nat} {q : Fin seq}
    (hq : q ∈ activeOfTokensShift tokens) :
    let p := prevOfTokensShift tokens q
    let p0 := prevOfTokens tokens q
    p.val = p0.val + 1 ∧
      p0 < q ∧ tokens p0 = tokens q ∧
        ∀ k, k < q → tokens k = tokens q → k ≤ p0 := by
  have h := (mem_activeOfTokensShift (tokens := tokens) (q := q)).1 hq
  rcases h with ⟨k, hk, htok⟩
  have hk' : k < q := by
    exact (Fin.lt_def).2 hk
  exact prevOfTokensShift_spec (tokens := tokens) (q := q) ⟨k, hk', htok⟩

/-- Active queries select a `prev` strictly in the past. -/
def InductionPrevInPast {seq dModel dHead : Nat}
    (inputs : InductionHeadInputs seq dModel dHead) : Prop :=
  ∀ q, q ∈ inputs.active → inputs.prev q < q

/--
Canonical shifted-prev spec for periodic prompts.

Note: when `1 < period`, every active query has `prev q < q`.
-/
structure InductionPrevSpecPeriodShift {seq dModel dHead : Nat}
    (period : Nat) (inputs : InductionHeadInputs seq dModel dHead) : Prop where
  /-- Active queries are the shifted-period active set. -/
  active_eq : inputs.active = activeOfPeriodShift (seq := seq) period
  /-- Prev map matches the shifted-period definition. -/
  prev_eq : inputs.prev = prevOfPeriodShift (seq := seq) period

/--
Canonical shifted-prev spec for token-based prompts.

Note: if successive tokens repeat, the shifted target can coincide with `q`.
-/
structure InductionPrevSpecTokensShift {seq dModel dHead : Nat}
    (tokens : Fin seq → Nat) (inputs : InductionHeadInputs seq dModel dHead) : Prop where
  /-- Active queries match the shifted-token definition. -/
  active_eq : inputs.active = activeOfTokensShift (seq := seq) tokens
  /-- Prev map matches the shifted-token definition. -/
  prev_eq : inputs.prev = prevOfTokensShift (seq := seq) tokens

end Model

end Nfp
