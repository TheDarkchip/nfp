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
    simp [prevOfTokensShift, hactive, p0]
  have hpval' :
      (prevOfTokensShift tokens q).val = (prevOfTokens tokens q).val + 1 := by
    simpa [p0] using hpval
  have hp0' :
      prevOfTokens tokens q < q ∧ tokens (prevOfTokens tokens q) = tokens q ∧
        ∀ k, k < q → tokens k = tokens q → k ≤ prevOfTokens tokens q := by
    simpa [p0] using hp0
  simpa using And.intro hpval' hp0'

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

/-- Helper: lift a first-half index into `Fin (2 * period)`. -/
lemma lt_double_of_lt_period {period i : Nat} (hi : i < period) : i < 2 * period := by
  have hle : period ≤ 2 * period := by
    have hpos : 0 < (2 : Nat) := by decide
    exact Nat.le_mul_of_pos_left period hpos
  exact Nat.lt_of_lt_of_le hi hle

/--
Tokens are a repeated pattern of length `period` with no repeats in the first half.

This matches the usual induction diagnostic: a random pattern of length `period`
repeated twice.
-/
structure InductionDiagnosticTokens (period : Nat) (tokens : Fin (2 * period) → Nat) : Prop where
  /-- Second half repeats the first half with period `period`. -/
  repeat_tok : ∀ q : Fin (2 * period), period ≤ q.val →
      tokens q = tokens ⟨q.val - period, lt_of_le_of_lt (Nat.sub_le _ _) q.isLt⟩
  /-- The first-half tokens are pairwise distinct. -/
  inj : ∀ {i j : Nat} (hi : i < period) (hj : j < period),
      tokens ⟨i, lt_double_of_lt_period hi⟩ =
        tokens ⟨j, lt_double_of_lt_period hj⟩ → i = j

/--
In a diagnostic prompt (repeated distinct pattern), the previous matching token
for any query in the second half is exactly `q - period`.
-/
theorem prevOfTokens_eq_prevOfPeriod_of_diag {period : Nat}
    {tokens : Fin (2 * period) → Nat} (hdiag : InductionDiagnosticTokens period tokens)
    (hper : 0 < period) {q : Fin (2 * period)} (hq : period ≤ q.val) :
    prevOfTokens tokens q = prevOfPeriod (seq := 2 * period) period q := by
  classical
  let kq : Fin (2 * period) :=
    ⟨q.val - period, lt_of_le_of_lt (Nat.sub_le _ _) q.isLt⟩
  have hklt : kq < q := by
    have hklt' : kq.val < q.val := by
      simpa [kq] using (Nat.sub_lt_of_pos_le hper hq)
    exact (Fin.lt_def).2 hklt'
  have htok : tokens kq = tokens q := by
    have := hdiag.repeat_tok q hq
    simpa [kq] using this.symm
  have hspec := prevOfTokens_spec (tokens := tokens) (q := q) ⟨kq, hklt, htok⟩
  let p := prevOfTokens tokens q
  have hp :
      p < q ∧ tokens p = tokens q ∧
        ∀ k, k < q → tokens k = tokens q → k ≤ p := by
    simpa [p] using hspec
  have hqsub : q.val - period < period := by
    have hq2 : q.val < 2 * period := q.isLt
    have hq2' : q.val < period + period := by simpa [two_mul] using hq2
    exact (Nat.sub_lt_iff_lt_add hq).2 (by simpa [Nat.add_comm] using hq2')
  have huniq :
      ∀ r : Fin (2 * period), r < q → tokens r = tokens q → r.val = q.val - period := by
    intro r hr htokr
    by_cases hrper : period ≤ r.val
    · have hrsub : r.val - period < period := by
        have hr2 : r.val < 2 * period := r.isLt
        have hr2' : r.val < period + period := by simpa [two_mul] using hr2
        exact (Nat.sub_lt_iff_lt_add hrper).2 (by simpa [Nat.add_comm] using hr2')
      have htok_r :
          tokens ⟨r.val - period, lt_of_le_of_lt (Nat.sub_le _ _) r.isLt⟩ = tokens r := by
        have := hdiag.repeat_tok r hrper
        simpa using this.symm
      have htok_q :
          tokens q = tokens ⟨q.val - period, lt_of_le_of_lt (Nat.sub_le _ _) q.isLt⟩ := by
        simpa using hdiag.repeat_tok q hq
      have htok_first :
          tokens ⟨r.val - period, lt_of_le_of_lt (Nat.sub_le _ _) r.isLt⟩ =
            tokens ⟨q.val - period, lt_of_le_of_lt (Nat.sub_le _ _) q.isLt⟩ := by
        calc
          tokens ⟨r.val - period, lt_of_le_of_lt (Nat.sub_le _ _) r.isLt⟩
              = tokens r := by simpa using htok_r
          _ = tokens q := by simpa using htokr
          _ = tokens ⟨q.val - period, lt_of_le_of_lt (Nat.sub_le _ _) q.isLt⟩ := by
              simpa using htok_q
      have hkeq : r.val - period = q.val - period := by
        apply hdiag.inj hrsub hqsub
        simpa using htok_first
      have hrval : r.val = q.val := by
        have h := congrArg (fun x => x + period) hkeq
        simpa [Nat.sub_add_cancel hrper, Nat.sub_add_cancel hq] using h
      have hrlt : r.val < q.val := (Fin.lt_def).1 hr
      have hrlt' : r.val < r.val := by
        have hrlt' := hrlt
        rw [← hrval] at hrlt'
        exact hrlt'
      exact (False.elim (lt_irrefl _ hrlt'))
    · have hrlt : r.val < period := lt_of_not_ge hrper
      have hrfin :
          (⟨r.val, lt_double_of_lt_period hrlt⟩ : Fin (2 * period)) = r := by
        apply Fin.ext
        rfl
      have htok_q :
          tokens q = tokens ⟨q.val - period, lt_of_le_of_lt (Nat.sub_le _ _) q.isLt⟩ := by
        simpa using hdiag.repeat_tok q hq
      have htok_first :
          tokens ⟨r.val, lt_double_of_lt_period hrlt⟩ =
            tokens ⟨q.val - period, lt_of_le_of_lt (Nat.sub_le _ _) q.isLt⟩ := by
        calc
          tokens ⟨r.val, lt_double_of_lt_period hrlt⟩ = tokens r := by
            simp [hrfin]
          _ = tokens q := by simpa using htokr
          _ = tokens ⟨q.val - period, lt_of_le_of_lt (Nat.sub_le _ _) q.isLt⟩ := by
            simpa using htok_q
      have hkeq : r.val = q.val - period := by
        apply hdiag.inj hrlt hqsub
        simpa using htok_first
      exact hkeq
  have hpval : p.val = q.val - period := by
    have := hp.2.2 p hp.1 hp.2.1
    have huniq' := huniq p hp.1 hp.2.1
    exact huniq'
  apply Fin.ext
  simp [prevOfPeriod, hpval, p]

/-- Shifted `prev` map matches the period-shifted map under diagnostic tokens. -/
theorem prevOfTokensShift_eq_prevOfPeriodShift_of_diag {period : Nat}
    {tokens : Fin (2 * period) → Nat} (hdiag : InductionDiagnosticTokens period tokens)
    (hper : 0 < period) {q : Fin (2 * period)} (hq : period ≤ q.val) :
    prevOfTokensShift tokens q = prevOfPeriodShift (seq := 2 * period) period q := by
  have hprev :=
    prevOfTokens_eq_prevOfPeriod_of_diag (tokens := tokens) hdiag hper (q := q) hq
  have hactive : q ∈ activeOfTokensShift tokens := by
    let kq : Fin (2 * period) :=
      ⟨q.val - period, lt_of_le_of_lt (Nat.sub_le _ _) q.isLt⟩
    have hklt : kq < q := by
      have hklt' : kq.val < q.val := by
        simpa [kq] using (Nat.sub_lt_of_pos_le hper hq)
      exact (Fin.lt_def).2 hklt'
    have htok : tokens kq = tokens q := by
      have := hdiag.repeat_tok q hq
      simpa [kq] using this.symm
    exact (mem_activeOfTokensShift (tokens := tokens) (q := q)).2
      ⟨kq, hklt, htok⟩
  have hactive' : q ∈ activeOfTokens tokens := by
    simpa [activeOfTokensShift] using hactive
  simp [prevOfTokensShift, hactive', hprev, prevOfPeriodShift, prevOfPeriod, hq, hper]

/-- Active shifted queries coincide with the periodic active set in diagnostics. -/
theorem activeOfTokensShift_eq_activeOfPeriodShift_of_diag {period : Nat}
    {tokens : Fin (2 * period) → Nat} (hdiag : InductionDiagnosticTokens period tokens)
    (hper : 0 < period) :
    activeOfTokensShift (seq := 2 * period) tokens =
      activeOfPeriodShift (seq := 2 * period) period := by
  ext q
  constructor
  · intro hq
    have hq' := (mem_activeOfTokensShift (tokens := tokens) (q := q)).1 hq
    rcases hq' with ⟨k, hk, htok⟩
    have hkper : period ≤ q.val := by
      by_contra hlt
      have hqlt : q.val < period := lt_of_not_ge hlt
      have hklt : k.val < period := lt_of_lt_of_le hk (Nat.le_of_lt hqlt)
      have hkfin :
          (⟨k.val, lt_double_of_lt_period hklt⟩ : Fin (2 * period)) = k := by
        apply Fin.ext
        rfl
      have hqfin :
          (⟨q.val, lt_double_of_lt_period hqlt⟩ : Fin (2 * period)) = q := by
        apply Fin.ext
        rfl
      have hkeq : k.val = q.val := by
        apply hdiag.inj hklt hqlt
        simpa [hkfin, hqfin] using htok
      have hk' : k.val < k.val := by
        have hk' := hk
        rw [← hkeq] at hk'
        exact hk'
      exact (False.elim (lt_irrefl _ hk'))
    exact (mem_activeOfPeriodShift (seq := 2 * period) (period := period) (q := q)).2
      ⟨hper, hkper⟩
  · intro hq
    have hq' := (mem_activeOfPeriodShift (seq := 2 * period) (period := period) (q := q)).1 hq
    rcases hq' with ⟨_, hqper⟩
    let kq : Fin (2 * period) :=
      ⟨q.val - period, lt_of_le_of_lt (Nat.sub_le _ _) q.isLt⟩
    have hklt : kq.val < q.val := by
      simpa [kq] using (Nat.sub_lt_of_pos_le hper hqper)
    have htok : tokens kq = tokens q := by
      have := hdiag.repeat_tok q hqper
      simpa [kq] using this.symm
    exact (mem_activeOfTokensShift (tokens := tokens) (q := q)).2
      ⟨kq, hklt, htok⟩

/-- Diagnostic prompts align shifted-token `prev` with the period-shifted map. -/
theorem prevOfTokensShift_eq_prevOfPeriodShift_of_diag_all {period : Nat}
    {tokens : Fin (2 * period) → Nat} (hdiag : InductionDiagnosticTokens period tokens)
    (hper : 0 < period) :
    prevOfTokensShift tokens = prevOfPeriodShift (seq := 2 * period) period := by
  funext q
  by_cases hqper : period ≤ q.val
  · simpa using
      (prevOfTokensShift_eq_prevOfPeriodShift_of_diag (tokens := tokens) hdiag hper
        (q := q) hqper)
  · have hqnot_period : q ∉ activeOfPeriodShift (seq := 2 * period) period := by
      intro hqmem
      have hcond :=
        (mem_activeOfPeriodShift (seq := 2 * period) (period := period) (q := q)).1 hqmem
      exact (hqper hcond.2).elim
    have hactive_eq :=
      activeOfTokensShift_eq_activeOfPeriodShift_of_diag (tokens := tokens) hdiag hper
    have hqnot_tokens_shift : q ∉ activeOfTokensShift (seq := 2 * period) tokens := by
      simpa [hactive_eq] using hqnot_period
    have hqnot_tokens : q ∉ activeOfTokens tokens := by
      simpa [activeOfTokensShift] using hqnot_tokens_shift
    simp [prevOfTokensShift, hqnot_tokens, prevOfPeriodShift, hqper]

/-- Diagnostic prompts let period-shift specs re-express as token-shift specs. -/
theorem InductionPrevSpecTokensShift_of_diag {dModel dHead : Nat} {period : Nat}
    {tokens : Fin (2 * period) → Nat} (hdiag : InductionDiagnosticTokens period tokens)
    (hper : 0 < period) {inputs : InductionHeadInputs (2 * period) dModel dHead}
    (hspec : InductionPrevSpecPeriodShift (seq := 2 * period) period inputs) :
    InductionPrevSpecTokensShift (seq := 2 * period) tokens inputs := by
  refine ⟨?active, ?prev⟩
  · have hactive_eq :=
      activeOfTokensShift_eq_activeOfPeriodShift_of_diag (tokens := tokens) hdiag hper
    calc
      inputs.active = activeOfPeriodShift (seq := 2 * period) period := hspec.active_eq
      _ = activeOfTokensShift (seq := 2 * period) tokens := by
        symm
        exact hactive_eq
  · have hprev_eq :=
      prevOfTokensShift_eq_prevOfPeriodShift_of_diag_all (tokens := tokens) hdiag hper
    calc
      inputs.prev = prevOfPeriodShift (seq := 2 * period) period := hspec.prev_eq
      _ = prevOfTokensShift tokens := by
        symm
        exact hprev_eq

end Model

end Nfp
