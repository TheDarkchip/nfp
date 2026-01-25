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

/--
Canonical logit-diff direction metadata for a shifted-token induction query.

The target token is the shifted previous match (`prevOfTokensShift`), while the
negative token is the current token at `q`.
-/
def directionSpecOfTokensShift {seq : Nat} (tokens : Fin seq → Nat) (q : Fin seq) :
    Nfp.Circuit.DirectionSpec :=
  { target := tokens (prevOfTokensShift (seq := seq) tokens q)
    negative := tokens q }

/-- `directionSpecOfTokensShift` exposes the shifted-prev target and current token. -/
theorem directionSpecOfTokensShift_spec {seq : Nat} (tokens : Fin seq → Nat) (q : Fin seq) :
    let dir := directionSpecOfTokensShift (seq := seq) tokens q
    dir.target = tokens (prevOfTokensShift (seq := seq) tokens q) ∧
      dir.negative = tokens q := by
  simp [directionSpecOfTokensShift]

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

/--
In a diagnostic prompt, any prior match for a second-half query is at `q - period`.
-/
theorem prevMatch_val_of_diag {period : Nat}
    {tokens : Fin (2 * period) → Nat} (hdiag : InductionDiagnosticTokens period tokens)
    {q : Fin (2 * period)} (hq : period ≤ q.val)
    {r : Fin (2 * period)} (hr : r < q) (htok : tokens r = tokens q) :
    r.val = q.val - period := by
  have hqsub : q.val - period < period := by
    have hq2 : q.val < 2 * period := q.isLt
    have hq2' : q.val < period + period := by simpa [two_mul] using hq2
    exact (Nat.sub_lt_iff_lt_add hq).2 (by simpa [Nat.add_comm] using hq2')
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
        _ = tokens q := by simpa using htok
        _ = tokens ⟨q.val - period, lt_of_le_of_lt (Nat.sub_le _ _) q.isLt⟩ := by
            simpa using htok_q
    have hkeq : r.val - period = q.val - period := by
      apply hdiag.inj hrsub hqsub
      simpa using htok_first
    have hrval : r.val = q.val := by
      have h := congrArg (fun x => x + period) hkeq
      simpa [Nat.sub_add_cancel hrper, Nat.sub_add_cancel hq] using h
    have hrlt' : r.val < r.val := by
      have hrlt : r.val < q.val := (Fin.lt_def).1 hr
      rw [← hrval] at hrlt
      exact hrlt
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
        _ = tokens q := by simpa using htok
        _ = tokens ⟨q.val - period, lt_of_le_of_lt (Nat.sub_le _ _) q.isLt⟩ := by
          simpa using htok_q
    have hkeq : r.val = q.val - period := by
      apply hdiag.inj hrlt hqsub
      simpa using htok_first
    exact hkeq

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

/-- Total attention mass across all query-key pairs. -/
def totalAttentionMass {seq : Nat} (weights : Fin seq → Fin seq → Rat) : Rat :=
  (Finset.univ : Finset (Fin seq)).sum (fun q =>
    (Finset.univ : Finset (Fin seq)).sum (fun k => weights q k))

/-- Sum of weights on the shifted-period stripe. -/
def shiftedStripeSum {seq : Nat} (period : Nat) (weights : Fin seq → Fin seq → Rat) : Rat :=
  (activeOfPeriodShift (seq := seq) period).sum (fun q =>
    weights q (prevOfPeriodShift (seq := seq) period q))

/--
Mean attention on the shifted-period stripe (defaults to `0` when the active set is empty).
-/
def shiftedStripeMean {seq : Nat} (period : Nat) (weights : Fin seq → Fin seq → Rat) : Rat :=
  if _ : (activeOfPeriodShift (seq := seq) period).card = 0 then
    0
  else
    shiftedStripeSum (seq := seq) period weights /
      ((activeOfPeriodShift (seq := seq) period).card : Rat)

/--
TL-style "mul" score computed on the shifted-token stripe.

This matches TransformerLens induction detection on prompts where each query has
at most one previous occurrence (e.g. diagnostic prompts).
-/
def tlInductionStripeNumerator {seq : Nat} (weights : Fin seq → Fin seq → Rat)
    (tokens : Fin seq → Nat) : Rat :=
  (activeOfTokensShift (seq := seq) tokens).sum (fun q =>
    weights q (prevOfTokensShift (seq := seq) tokens q))

/-- TL-style "mul" score computed on the shifted-token stripe. -/
def tlInductionStripeScore {seq : Nat} (weights : Fin seq → Fin seq → Rat)
    (tokens : Fin seq → Nat) : Rat :=
  let denom := totalAttentionMass (seq := seq) weights
  if _ : denom = 0 then 0 else tlInductionStripeNumerator (seq := seq) weights tokens / denom

/--
TL-style induction detection pattern (duplicate head shifted right).

This marks a key position `k` whenever there exists a prior position `r < q` with
the same token, and `k = r + 1`.
-/
def tlInductionPattern {seq : Nat} (tokens : Fin seq → Nat) (q k : Fin seq) : Prop :=
  ∃ r, r < q ∧ tokens r = tokens q ∧ k.val = r.val + 1

/-- Indicator for the TL induction pattern (0/1). -/
noncomputable def tlInductionPatternIndicator {seq : Nat} (tokens : Fin seq → Nat)
    (q k : Fin seq) : Rat :=
  by
    classical
    exact if tlInductionPattern (seq := seq) tokens q k then 1 else 0

/-- TL "mul" numerator: attention mass on the induction detection pattern. -/
noncomputable def tlInductionMulNumerator {seq : Nat} (weights : Fin seq → Fin seq → Rat)
    (tokens : Fin seq → Nat) : Rat :=
  (Finset.univ : Finset (Fin seq)).sum (fun q =>
    (Finset.univ : Finset (Fin seq)).sum (fun k =>
      weights q k * tlInductionPatternIndicator (seq := seq) tokens q k))

/-- TL "mul" score for induction detection. -/
noncomputable def tlInductionMulScore {seq : Nat} (weights : Fin seq → Fin seq → Rat)
    (tokens : Fin seq → Nat) : Rat :=
  let denom := totalAttentionMass (seq := seq) weights
  if _ : denom = 0 then 0 else tlInductionMulNumerator (seq := seq) weights tokens / denom

/-- Mask condition matching TL's `exclude_bos` and `exclude_current_token` switches. -/
def tlMaskCond {seq : Nat} (excludeBos excludeCurrent : Bool) (q k : Fin seq) : Bool :=
  (excludeBos && decide (k.val = 0)) || (excludeCurrent && decide (k = q))

/-- Apply TL-style masking to attention weights. -/
def tlMaskedWeights {seq : Nat} (excludeBos excludeCurrent : Bool)
    (weights : Fin seq → Fin seq → Rat) : Fin seq → Fin seq → Rat :=
  fun q k => if tlMaskCond (seq := seq) excludeBos excludeCurrent q k then 0 else weights q k

/-- Masked weights are nonnegative when the original weights are nonnegative. -/
theorem tlMaskedWeights_nonneg {seq : Nat} (excludeBos excludeCurrent : Bool)
    (weights : Fin seq → Fin seq → Rat) (hnonneg : ∀ q k, 0 ≤ weights q k) :
    ∀ q k, 0 ≤ tlMaskedWeights (seq := seq) excludeBos excludeCurrent weights q k := by
  intro q k
  by_cases h : tlMaskCond (seq := seq) excludeBos excludeCurrent q k
  · simp [tlMaskedWeights, h]
  · simp [tlMaskedWeights, h, hnonneg]

/-- TL mul score with masking switches. -/
noncomputable def tlInductionMulScoreMasked {seq : Nat} (excludeBos excludeCurrent : Bool)
    (weights : Fin seq → Fin seq → Rat) (tokens : Fin seq → Nat) : Rat :=
  tlInductionMulScore (seq := seq)
    (weights := tlMaskedWeights excludeBos excludeCurrent weights) tokens

/-- Stripe score with masking switches. -/
def tlInductionStripeScoreMasked {seq : Nat} (excludeBos excludeCurrent : Bool)
    (weights : Fin seq → Fin seq → Rat) (tokens : Fin seq → Nat) : Rat :=
  tlInductionStripeScore (seq := seq)
    (weights := tlMaskedWeights excludeBos excludeCurrent weights) tokens

/--
If a query is active, the TL induction pattern selects the shifted previous match.
-/
theorem tlInductionPattern_of_prevOfTokensShift {seq : Nat} {tokens : Fin seq → Nat}
    {q : Fin seq} (hq : q ∈ activeOfTokensShift tokens) :
    tlInductionPattern (seq := seq) tokens q (prevOfTokensShift (seq := seq) tokens q) := by
  classical
  let p := prevOfTokensShift (seq := seq) tokens q
  let p0 := prevOfTokens (seq := seq) tokens q
  have hspec :
      p.val = p0.val + 1 ∧
        p0 < q ∧ tokens p0 = tokens q ∧
          ∀ k, k < q → tokens k = tokens q → k ≤ p0 := by
    simpa [p, p0] using
      (prevOfTokensShift_spec_of_active (tokens := tokens) (q := q) hq)
  refine ⟨p0, hspec.2.1, hspec.2.2.1, ?_⟩
  simpa [p, p0] using hspec.1

/-- TL induction pattern indicator is always nonnegative. -/
theorem tlInductionPatternIndicator_nonneg {seq : Nat} (tokens : Fin seq → Nat)
    (q k : Fin seq) : 0 ≤ tlInductionPatternIndicator (seq := seq) tokens q k := by
  classical
  by_cases h : tlInductionPattern (seq := seq) tokens q k
  · simp [tlInductionPatternIndicator, h]
  · simp [tlInductionPatternIndicator, h]

/--
The TL mul numerator dominates the stripe numerator when attention weights are nonnegative.
-/
theorem tlInductionStripeNumerator_le_tlInductionMulNumerator {seq : Nat}
    (weights : Fin seq → Fin seq → Rat) (tokens : Fin seq → Nat)
    (hnonneg : ∀ q k, 0 ≤ weights q k) :
    tlInductionStripeNumerator (seq := seq) weights tokens ≤
      tlInductionMulNumerator (seq := seq) weights tokens := by
  classical
  let rowSum : Fin seq → Rat := fun q =>
    (Finset.univ : Finset (Fin seq)).sum (fun k =>
      weights q k * tlInductionPatternIndicator (seq := seq) tokens q k)
  have hrow_nonneg : ∀ q, 0 ≤ rowSum q := by
    intro q
    refine Finset.sum_nonneg ?_
    intro k hk
    exact
      mul_nonneg (hnonneg q k)
        (tlInductionPatternIndicator_nonneg (tokens := tokens) (q := q) (k := k))
  have hrow_ge :
      ∀ q ∈ activeOfTokensShift (seq := seq) tokens,
        weights q (prevOfTokensShift (seq := seq) tokens q) ≤ rowSum q := by
    intro q hq
    have hpat :
        tlInductionPattern (seq := seq) tokens q
          (prevOfTokensShift (seq := seq) tokens q) := by
      exact tlInductionPattern_of_prevOfTokensShift (tokens := tokens) hq
    have hind :
        tlInductionPatternIndicator (seq := seq) tokens q
            (prevOfTokensShift (seq := seq) tokens q) = 1 := by
      simp [tlInductionPatternIndicator, hpat]
    have hle :
        weights q (prevOfTokensShift (seq := seq) tokens q) *
            tlInductionPatternIndicator (seq := seq) tokens q
              (prevOfTokensShift (seq := seq) tokens q) ≤ rowSum q := by
      refine
        (Finset.single_le_sum
          (s := (Finset.univ : Finset (Fin seq)))
          (f := fun k =>
            weights q k * tlInductionPatternIndicator (seq := seq) tokens q k)
          ?_ ?_)
      · intro k hk
        exact
          mul_nonneg (hnonneg q k)
            (tlInductionPatternIndicator_nonneg (tokens := tokens) (q := q) (k := k))
      · simp
    simpa [hind, rowSum] using hle
  have hsum_active :
      (activeOfTokensShift (seq := seq) tokens).sum (fun q =>
        weights q (prevOfTokensShift (seq := seq) tokens q)) ≤
        (activeOfTokensShift (seq := seq) tokens).sum rowSum := by
    refine Finset.sum_le_sum ?_
    intro q hq
    exact hrow_ge q hq
  have hsubset :
      activeOfTokensShift (seq := seq) tokens ⊆ (Finset.univ : Finset (Fin seq)) := by
    intro q hq
    exact Finset.mem_univ q
  have hsum_univ :
      (activeOfTokensShift (seq := seq) tokens).sum rowSum ≤
        (Finset.univ : Finset (Fin seq)).sum rowSum := by
    refine Finset.sum_le_sum_of_subset_of_nonneg hsubset ?_
    intro q hq hqnot
    exact hrow_nonneg q
  calc
    tlInductionStripeNumerator (seq := seq) weights tokens =
        (activeOfTokensShift (seq := seq) tokens).sum (fun q =>
          weights q (prevOfTokensShift (seq := seq) tokens q)) := rfl
    _ ≤ (activeOfTokensShift (seq := seq) tokens).sum rowSum := hsum_active
    _ ≤ (Finset.univ : Finset (Fin seq)).sum rowSum := hsum_univ
    _ = tlInductionMulNumerator (seq := seq) weights tokens := by
      rfl

/--
The TL mul score is at least the stripe score when attention weights are nonnegative.
-/
theorem tlInductionStripeScore_le_tlInductionMulScore {seq : Nat}
    (weights : Fin seq → Fin seq → Rat) (tokens : Fin seq → Nat)
    (hnonneg : ∀ q k, 0 ≤ weights q k) :
    tlInductionStripeScore (seq := seq) weights tokens ≤
      tlInductionMulScore (seq := seq) weights tokens := by
  classical
  by_cases hdenom : totalAttentionMass (seq := seq) weights = 0
  · simp [tlInductionStripeScore, tlInductionMulScore, hdenom]
  · have hdenom_nonneg : 0 ≤ totalAttentionMass (seq := seq) weights := by
      unfold totalAttentionMass
      refine Finset.sum_nonneg ?_
      intro q hq
      refine Finset.sum_nonneg ?_
      intro k hk
      exact hnonneg q k
    have hnum_le :
        tlInductionStripeNumerator (seq := seq) weights tokens ≤
          tlInductionMulNumerator (seq := seq) weights tokens := by
      exact tlInductionStripeNumerator_le_tlInductionMulNumerator
        (weights := weights) (tokens := tokens) hnonneg
    have hstripe :
        tlInductionStripeScore (seq := seq) weights tokens =
          tlInductionStripeNumerator (seq := seq) weights tokens /
            totalAttentionMass (seq := seq) weights := by
      simp [tlInductionStripeScore, hdenom]
    have hmul :
        tlInductionMulScore (seq := seq) weights tokens =
          tlInductionMulNumerator (seq := seq) weights tokens /
            totalAttentionMass (seq := seq) weights := by
      simp [tlInductionMulScore, hdenom]
    calc
      tlInductionStripeScore (seq := seq) weights tokens =
          tlInductionStripeNumerator (seq := seq) weights tokens /
            totalAttentionMass (seq := seq) weights := hstripe
      _ ≤ tlInductionMulNumerator (seq := seq) weights tokens /
            totalAttentionMass (seq := seq) weights := by
          exact div_le_div_of_nonneg_right hnum_le hdenom_nonneg
      _ = tlInductionMulScore (seq := seq) weights tokens := by
          symm
          exact hmul

/--
Each query has at most one previous matching token.
-/
def UniquePreviousMatch {seq : Nat} (tokens : Fin seq → Nat) : Prop :=
  ∀ q r₁ r₂, r₁ < q → tokens r₁ = tokens q → r₂ < q → tokens r₂ = tokens q → r₁ = r₂

/--
Diagnostic prompts have unique previous matches.
-/
theorem uniquePreviousMatch_of_diag {period : Nat}
    {tokens : Fin (2 * period) → Nat} (hdiag : InductionDiagnosticTokens period tokens) :
    UniquePreviousMatch (seq := 2 * period) tokens := by
  intro q r1 r2 hr1 htok1 hr2 htok2
  by_cases hqper : period ≤ q.val
  · have hr1val :=
      prevMatch_val_of_diag (tokens := tokens) hdiag (q := q) hqper hr1 htok1
    have hr2val :=
      prevMatch_val_of_diag (tokens := tokens) hdiag (q := q) hqper hr2 htok2
    apply Fin.ext
    simp [hr1val, hr2val]
  · have hqlt : q.val < period := lt_of_not_ge hqper
    have hr1lt : r1.val < period := lt_of_lt_of_le ((Fin.lt_def).1 hr1) (Nat.le_of_lt hqlt)
    have hr2lt : r2.val < period := lt_of_lt_of_le ((Fin.lt_def).1 hr2) (Nat.le_of_lt hqlt)
    have htok12 : tokens r1 = tokens r2 := by
      calc
        tokens r1 = tokens q := htok1
        _ = tokens r2 := by symm; exact htok2
    have hr1fin :
        (⟨r1.val, lt_double_of_lt_period hr1lt⟩ : Fin (2 * period)) = r1 := by
      apply Fin.ext
      rfl
    have hr2fin :
        (⟨r2.val, lt_double_of_lt_period hr2lt⟩ : Fin (2 * period)) = r2 := by
      apply Fin.ext
      rfl
    have hkeq : r1.val = r2.val := by
      apply hdiag.inj hr1lt hr2lt
      simpa [hr1fin, hr2fin] using htok12
    apply Fin.ext
    simp [hkeq]

/--
Under unique previous matches, the TL pattern for an active query hits exactly the shifted prev.
-/
theorem tlInductionPattern_iff_prevOfTokensShift_of_unique {seq : Nat}
    {tokens : Fin seq → Nat} (huniq : UniquePreviousMatch (seq := seq) tokens)
    {q k : Fin seq} (hq : q ∈ activeOfTokensShift (seq := seq) tokens) :
    tlInductionPattern (seq := seq) tokens q k ↔
      k = prevOfTokensShift (seq := seq) tokens q := by
  classical
  constructor
  · intro hpat
    rcases hpat with ⟨r, hr, htok, hk⟩
    have hspec := prevOfTokensShift_spec_of_active (tokens := tokens) (q := q) hq
    have hp0_lt : prevOfTokens (seq := seq) tokens q < q := by
      simpa using hspec.2.1
    have hp0_tok : tokens (prevOfTokens (seq := seq) tokens q) = tokens q := by
      simpa using hspec.2.2.1
    have hprev_eq : prevOfTokens (seq := seq) tokens q = r := by
      apply huniq q (prevOfTokens (seq := seq) tokens q) r
      · exact hp0_lt
      · exact hp0_tok
      · exact hr
      · exact htok
    have hval :
        (prevOfTokensShift (seq := seq) tokens q).val =
          (prevOfTokens (seq := seq) tokens q).val + 1 := by
      simpa using hspec.1
    apply Fin.ext
    calc
      k.val = r.val + 1 := hk
      _ = (prevOfTokens (seq := seq) tokens q).val + 1 := by
        simp [hprev_eq]
      _ = (prevOfTokensShift (seq := seq) tokens q).val := by
        symm
        exact hval
  · intro hk
    have hpat :=
      tlInductionPattern_of_prevOfTokensShift (tokens := tokens) (q := q) hq
    simpa [hk] using hpat

/--
If a query is inactive, the TL induction pattern never holds.
-/
theorem tlInductionPattern_false_of_inactive {seq : Nat} {tokens : Fin seq → Nat}
    {q : Fin seq} (hq : q ∉ activeOfTokensShift (seq := seq) tokens) :
    ∀ k : Fin seq, ¬ tlInductionPattern (seq := seq) tokens q k := by
  intro k hpat
  rcases hpat with ⟨r, hr, htok, hk⟩
  have hactive : q ∈ activeOfTokensShift (seq := seq) tokens := by
    exact (mem_activeOfTokensShift (tokens := tokens) (q := q)).2
      ⟨r, (Fin.lt_def).1 hr, htok⟩
  exact hq hactive

/--
With unique previous matches, the TL mul numerator equals the stripe numerator.
-/
theorem tlInductionMulNumerator_eq_stripeNumerator_of_unique {seq : Nat}
    (weights : Fin seq → Fin seq → Rat) (tokens : Fin seq → Nat)
    (huniq : UniquePreviousMatch (seq := seq) tokens) :
    tlInductionMulNumerator (seq := seq) weights tokens =
      tlInductionStripeNumerator (seq := seq) weights tokens := by
  classical
  let rowSum : Fin seq → Rat := fun q =>
    (Finset.univ : Finset (Fin seq)).sum (fun k =>
      weights q k * tlInductionPatternIndicator (seq := seq) tokens q k)
  have hrow_active :
      ∀ q ∈ activeOfTokensShift (seq := seq) tokens,
        rowSum q = weights q (prevOfTokensShift (seq := seq) tokens q) := by
    intro q hq
    let a := prevOfTokensShift (seq := seq) tokens q
    have hsum :
        rowSum q =
          weights q a * tlInductionPatternIndicator (seq := seq) tokens q a := by
      refine (Finset.sum_eq_single a ?_ ?_)
      · intro k hk hkne
        have hpat_false :
            ¬ tlInductionPattern (seq := seq) tokens q k := by
          have hiff :=
            tlInductionPattern_iff_prevOfTokensShift_of_unique
              (tokens := tokens) huniq (q := q) (k := k) hq
          exact fun hpat => hkne (hiff.1 hpat)
        simp [tlInductionPatternIndicator, hpat_false]
      · simp
    have hpat :
        tlInductionPattern (seq := seq) tokens q a := by
      simpa [a] using
        tlInductionPattern_of_prevOfTokensShift (tokens := tokens) (q := q) hq
    have hind :
        tlInductionPatternIndicator (seq := seq) tokens q a = 1 := by
      simp [tlInductionPatternIndicator, hpat]
    simp [a, hsum, hind]
  have hrow_inactive :
      ∀ q, q ∉ activeOfTokensShift (seq := seq) tokens → rowSum q = 0 := by
    intro q hq
    have hpat_false := tlInductionPattern_false_of_inactive (tokens := tokens) (q := q) hq
    unfold rowSum
    simp [tlInductionPatternIndicator, hpat_false]
  have hrow_if :
      ∀ q,
        rowSum q =
          if q ∈ activeOfTokensShift (seq := seq) tokens then
            weights q (prevOfTokensShift (seq := seq) tokens q)
          else
            0 := by
    intro q
    by_cases hq : q ∈ activeOfTokensShift (seq := seq) tokens
    · simp [hrow_active q hq, hq]
    · simp [hrow_inactive q hq, hq]
  let f := fun q =>
    if q ∈ activeOfTokensShift (seq := seq) tokens then
      weights q (prevOfTokensShift (seq := seq) tokens q)
    else
      0
  have hsum_univ :
      (Finset.univ : Finset (Fin seq)).sum rowSum =
        (Finset.univ : Finset (Fin seq)).sum f := by
    refine Finset.sum_congr rfl ?_
    intro q hq
    simpa [f] using hrow_if q
  have hsum_sub :
      (Finset.univ : Finset (Fin seq)).sum f =
        (activeOfTokensShift (seq := seq) tokens).sum f := by
    symm
    refine Finset.sum_subset
      (s₁ := activeOfTokensShift (seq := seq) tokens)
      (s₂ := (Finset.univ : Finset (Fin seq))) ?_ ?_
    · intro q hq
      exact Finset.mem_univ q
    · intro q hq hqnot
      simp [hqnot]
  have hsum_eq :
      (activeOfTokensShift (seq := seq) tokens).sum f =
        (activeOfTokensShift (seq := seq) tokens).sum (fun q =>
          weights q (prevOfTokensShift (seq := seq) tokens q)) := by
    refine Finset.sum_congr rfl ?_
    intro q hq
    simp [f, hq]
  calc
    tlInductionMulNumerator (seq := seq) weights tokens =
        (Finset.univ : Finset (Fin seq)).sum rowSum := rfl
    _ = (Finset.univ : Finset (Fin seq)).sum f := hsum_univ
    _ = (activeOfTokensShift (seq := seq) tokens).sum f := hsum_sub
    _ =
        (activeOfTokensShift (seq := seq) tokens).sum (fun q =>
          weights q (prevOfTokensShift (seq := seq) tokens q)) := hsum_eq
    _ = tlInductionStripeNumerator (seq := seq) weights tokens := rfl

/--
Under unique previous matches, the TL mul score equals the stripe score.
-/
theorem tlInductionMulScore_eq_stripeScore_of_unique {seq : Nat}
    (weights : Fin seq → Fin seq → Rat) (tokens : Fin seq → Nat)
    (huniq : UniquePreviousMatch (seq := seq) tokens) :
    tlInductionMulScore (seq := seq) weights tokens =
      tlInductionStripeScore (seq := seq) weights tokens := by
  classical
  by_cases hdenom : totalAttentionMass (seq := seq) weights = 0
  · simp [tlInductionMulScore, tlInductionStripeScore, hdenom]
  · have hnum :
        tlInductionMulNumerator (seq := seq) weights tokens =
          tlInductionStripeNumerator (seq := seq) weights tokens := by
        exact
          tlInductionMulNumerator_eq_stripeNumerator_of_unique
            (weights := weights) (tokens := tokens) huniq
    simp [tlInductionMulScore, tlInductionStripeScore, hdenom, hnum]

/--
Masked TL mul score is at least the masked stripe score under nonnegative weights.
-/
theorem tlInductionStripeScoreMasked_le_tlInductionMulScoreMasked {seq : Nat}
    (excludeBos excludeCurrent : Bool) (weights : Fin seq → Fin seq → Rat) (tokens : Fin seq → Nat)
    (hnonneg : ∀ q k, 0 ≤ weights q k) :
    tlInductionStripeScoreMasked (seq := seq) excludeBos excludeCurrent weights tokens ≤
      tlInductionMulScoreMasked (seq := seq) excludeBos excludeCurrent weights tokens := by
  have hnonneg' :
      ∀ q k,
        0 ≤ tlMaskedWeights (seq := seq) excludeBos excludeCurrent weights q k := by
    exact tlMaskedWeights_nonneg (seq := seq) excludeBos excludeCurrent weights hnonneg
  simpa [tlInductionStripeScoreMasked, tlInductionMulScoreMasked] using
    (tlInductionStripeScore_le_tlInductionMulScore
      (seq := seq)
      (weights := tlMaskedWeights (seq := seq) excludeBos excludeCurrent weights)
      (tokens := tokens) hnonneg')

/--
Masked TL mul score equals the masked stripe score under unique previous matches.
-/
theorem tlInductionMulScoreMasked_eq_stripeScoreMasked_of_unique {seq : Nat}
    (excludeBos excludeCurrent : Bool) (weights : Fin seq → Fin seq → Rat) (tokens : Fin seq → Nat)
    (huniq : UniquePreviousMatch (seq := seq) tokens) :
    tlInductionMulScoreMasked (seq := seq) excludeBos excludeCurrent weights tokens =
      tlInductionStripeScoreMasked (seq := seq) excludeBos excludeCurrent weights tokens := by
  simpa [tlInductionMulScoreMasked, tlInductionStripeScoreMasked] using
    (tlInductionMulScore_eq_stripeScore_of_unique
      (seq := seq)
      (weights := tlMaskedWeights (seq := seq) excludeBos excludeCurrent weights)
      (tokens := tokens) huniq)

/--
For diagnostic prompts, the TL-style stripe score equals the active fraction times
the shifted-stripe mean.
-/
theorem tlInductionStripeScore_of_diag {period : Nat}
    {tokens : Fin (2 * period) → Nat} (hdiag : InductionDiagnosticTokens period tokens)
    (hper : 0 < period) (weights : Fin (2 * period) → Fin (2 * period) → Rat)
    (hsum : ∀ q,
      (Finset.univ : Finset (Fin (2 * period))).sum (fun k => weights q k) = 1) :
    tlInductionStripeScore (seq := 2 * period) weights tokens =
      ((activeOfPeriodShift (seq := 2 * period) period).card : Rat) / (2 * period) *
        shiftedStripeMean (seq := 2 * period) period weights := by
  classical
  have hdenom : totalAttentionMass (seq := 2 * period) weights = (2 * period : Rat) := by
    simp [totalAttentionMass, hsum]
  have hseq_pos : 0 < 2 * period := Nat.mul_pos (by decide) hper
  have hseq_ne : (2 * period : Rat) ≠ 0 := by
    exact_mod_cast (Nat.ne_of_gt hseq_pos)
  have hlt : period < 2 * period := by
    simpa [two_mul] using (Nat.lt_add_of_pos_right (n := period) (k := period) hper)
  have hmem :
      (⟨period, hlt⟩ : Fin (2 * period)) ∈
        activeOfPeriodShift (seq := 2 * period) period := by
    exact
      (mem_activeOfPeriodShift (seq := 2 * period) (period := period) (q := ⟨period, hlt⟩)).2
        ⟨hper, le_rfl⟩
  have hactive_ne : (activeOfPeriodShift (seq := 2 * period) period).card ≠ 0 := by
    exact Finset.card_ne_zero.mpr ⟨⟨period, hlt⟩, hmem⟩
  have hmean :
      shiftedStripeMean (seq := 2 * period) period weights =
        shiftedStripeSum (seq := 2 * period) period weights /
          ((activeOfPeriodShift (seq := 2 * period) period).card : Rat) := by
    simp [shiftedStripeMean, shiftedStripeSum, hactive_ne]
  have hscore :
      tlInductionStripeScore (seq := 2 * period) weights tokens =
        shiftedStripeSum (seq := 2 * period) period weights / (2 * period : Rat) := by
    have hactive_eq :=
      activeOfTokensShift_eq_activeOfPeriodShift_of_diag (tokens := tokens) hdiag hper
    have hprev_eq :=
      prevOfTokensShift_eq_prevOfPeriodShift_of_diag_all (tokens := tokens) hdiag hper
    simp [tlInductionStripeScore, tlInductionStripeNumerator, shiftedStripeSum,
      hactive_eq, hprev_eq, hdenom, hseq_ne]
  have hactive_ne_rat :
      ((activeOfPeriodShift (seq := 2 * period) period).card : Rat) ≠ 0 := by
    exact_mod_cast hactive_ne
  have hcalc :
      ((activeOfPeriodShift (seq := 2 * period) period).card : Rat) / (2 * period : Rat) *
          (shiftedStripeSum (seq := 2 * period) period weights /
            ((activeOfPeriodShift (seq := 2 * period) period).card : Rat)) =
        shiftedStripeSum (seq := 2 * period) period weights / (2 * period : Rat) := by
    calc
      ((activeOfPeriodShift (seq := 2 * period) period).card : Rat) / (2 * period : Rat) *
          (shiftedStripeSum (seq := 2 * period) period weights /
            ((activeOfPeriodShift (seq := 2 * period) period).card : Rat)) =
          ((activeOfPeriodShift (seq := 2 * period) period).card : Rat) *
            (shiftedStripeSum (seq := 2 * period) period weights /
              ((activeOfPeriodShift (seq := 2 * period) period).card : Rat)) /
              (2 * period : Rat) := by
          simp [div_mul_eq_mul_div]
      _ =
          (((activeOfPeriodShift (seq := 2 * period) period).card : Rat) *
              shiftedStripeSum (seq := 2 * period) period weights /
                ((activeOfPeriodShift (seq := 2 * period) period).card : Rat)) /
              (2 * period : Rat) := by
          simpa using
            congrArg (fun x => x / (2 * period : Rat))
              (mul_div_assoc
                ((activeOfPeriodShift (seq := 2 * period) period).card : Rat)
                (shiftedStripeSum (seq := 2 * period) period weights)
                ((activeOfPeriodShift (seq := 2 * period) period).card : Rat)).symm
      _ =
          shiftedStripeSum (seq := 2 * period) period weights / (2 * period : Rat) := by
          simp [hactive_ne_rat]
  calc
    tlInductionStripeScore (seq := 2 * period) weights tokens =
        shiftedStripeSum (seq := 2 * period) period weights / (2 * period : Rat) := hscore
    _ =
        ((activeOfPeriodShift (seq := 2 * period) period).card : Rat) / (2 * period : Rat) *
          (shiftedStripeSum (seq := 2 * period) period weights /
            ((activeOfPeriodShift (seq := 2 * period) period).card : Rat)) := by
        simpa using hcalc.symm
    _ = ((activeOfPeriodShift (seq := 2 * period) period).card : Rat) / (2 * period) *
        shiftedStripeMean (seq := 2 * period) period weights := by
      simp [hmean]
end Model

end Nfp
