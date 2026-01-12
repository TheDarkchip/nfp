-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Rat.BigOperators
import Nfp.Core.Basic
import Nfp.Circuit.Layers.Induction
import Nfp.Circuit.Layers.Softmax

/-!
Per-query one-hot bounds derived from score margins.
-/

namespace Nfp

namespace Sound

open scoped BigOperators

open Nfp.Circuit

variable {seq : Nat} [NeZero seq]

/-- One-hot bounds on a single active query, derived from a per-query margin. -/
theorem oneHot_bounds_at_of_marginAt
    (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (scoresReal : Fin seq → Fin seq → Real)
    (marginAt : Fin seq → Rat)
    (epsAt : Fin seq → Rat)
    (hepsAt :
      ∀ q, epsAt q =
        if marginAt q < 0 then (1 : Rat) else
          ratDivUp (seq - 1) (1 + marginAt q))
    (hseq : (1 : Nat) ≤ seq)
    (hscore_margin_real_at :
      ∀ q, q ∈ active → ∀ k, k ≠ prev q →
        scoresReal q k + (marginAt q : Real) ≤ scoresReal q (prev q)) :
    ∀ q, q ∈ active →
      Layers.OneHotApproxBoundsOnActive (Val := Real) (epsAt q : Real)
        (fun q' => q' = q) prev
        (fun q k => Circuit.softmax (scoresReal q) k) := by
  classical
  intro q hq
  let weights : Fin seq → Fin seq → Real := fun q k =>
    Circuit.softmax (scoresReal q) k
  let others : Fin seq → Finset (Fin seq) := fun q =>
    (Finset.univ : Finset (Fin seq)).erase (prev q)
  have hweights_nonneg : ∀ k, 0 ≤ weights q k := by
    intro k
    simpa [weights] using
      (Circuit.softmax_nonneg (scores := scoresReal q) k)
  have hsum_one : (∑ k, weights q k) = 1 := by
    simpa [weights] using
      (Circuit.softmax_sum_one (scores := scoresReal q))
  have hsum_others_le : (∑ k ∈ others q, weights q k) ≤ (epsAt q : Real) := by
    by_cases hneg : marginAt q < 0
    · have heps : (epsAt q : Real) = 1 := by
        simp [hepsAt, hneg]
      have hsubset : others q ⊆ (Finset.univ : Finset (Fin seq)) := by
        intro k hk
        simp
      have hsum_le :
          (∑ k ∈ others q, weights q k) ≤
            ∑ k ∈ (Finset.univ : Finset (Fin seq)), weights q k :=
        Finset.sum_le_sum_of_subset_of_nonneg hsubset (by
          intro k _ _
          exact hweights_nonneg k)
      simpa [heps, hsum_one] using hsum_le
    · have hnonneg : 0 ≤ marginAt q := le_of_not_gt hneg
      have hnonneg_real : 0 ≤ (marginAt q : Real) := by
        exact ratToReal_nonneg_of_nonneg hnonneg
      have hbound :
          ∀ k ∈ others q,
            weights q k ≤ (1 + (marginAt q : Real))⁻¹ := by
        intro k hk
        have hkne : k ≠ prev q := (Finset.mem_erase.mp hk).1
        have hscore := hscore_margin_real_at q hq k hkne
        simpa [weights] using
          (Circuit.softmax_other_le_inv_one_add (scores := scoresReal q)
            (prev := prev q) (k := k) (m := (marginAt q : Real))
            hnonneg_real hscore)
      have hsum_le :
          (∑ k ∈ others q, weights q k) ≤
            ∑ k ∈ others q, (1 + (marginAt q : Real))⁻¹ :=
        Finset.sum_le_sum hbound
      have hsum_const :
          (∑ k ∈ others q, (1 + (marginAt q : Real))⁻¹) =
            (others q).card * (1 + (marginAt q : Real))⁻¹ := by
        simp
      have hcard : (others q).card = seq - 1 := by
        simp [others, Finset.card_erase_of_mem]
      have hsum_le' :
          (∑ k ∈ others q, weights q k) ≤
            (seq - 1 : Real) * (1 + (marginAt q : Real))⁻¹ := by
        have hsum_le'' := hsum_le.trans_eq hsum_const
        have hsum_le''' := hsum_le''
        simp only [hcard, Nat.cast_sub hseq, Nat.cast_one] at hsum_le'''
        exact hsum_le'''
      have heps :
          (seq - 1 : Real) * (1 + (marginAt q : Real))⁻¹ ≤ (epsAt q : Real) := by
        have hden : (1 + marginAt q) ≠ 0 := by
          intro hzero
          have hrat : (1 : Rat) + marginAt q = 0 := by
            simpa using hzero
          have hnonneg_rat : (0 : Rat) ≤ marginAt q := hnonneg
          linarith
        have hrat :
            (seq - 1 : Real) * (1 + (marginAt q : Real))⁻¹ ≤
              (ratDivUp (seq - 1) (1 + marginAt q) : Real) := by
          have hrat' := ratDivUp_ge_real (seq - 1) (1 + marginAt q) hden
          simpa [ratToReal, Rat.cast_div, Rat.cast_add, Rat.cast_natCast,
            div_eq_mul_inv] using hrat'
        simpa [hepsAt, hneg] using hrat
      exact le_trans hsum_le' heps
  refine
    { nonneg := ?_
      sum_one := ?_
      prev_large := ?_
      other_le := ?_ }
  · intro q' hq' k
    subst q'
    change 0 ≤ Circuit.softmax (scoresReal q) k
    exact Circuit.softmax_nonneg (scores := scoresReal q) k
  · intro q' hq'
    subst q'
    change (∑ k, Circuit.softmax (scoresReal q) k) = 1
    exact Circuit.softmax_sum_one (scores := scoresReal q)
  · intro q' hq'
    subst q'
    have hsum_eq :
        weights q (prev q) + ∑ k ∈ others q, weights q k = 1 := by
      have hsum' :
          weights q (prev q) + ∑ k ∈ others q, weights q k =
            ∑ k, weights q k := by
        simp [others]
      calc
        weights q (prev q) + ∑ k ∈ others q, weights q k =
            ∑ k, weights q k := hsum'
        _ = 1 := hsum_one
    have hsum_le' :
        weights q (prev q) + ∑ k ∈ others q, weights q k ≤
          weights q (prev q) + (epsAt q : Real) := by
      simpa [add_comm, add_left_comm, add_assoc] using
        (add_le_add_left hsum_others_le (weights q (prev q)))
    have hprev :
        1 ≤ weights q (prev q) + (epsAt q : Real) := by
      simpa [hsum_eq] using hsum_le'
    exact hprev
  · intro q' hq' k hk
    subst q'
    have hk' : k ∈ others q := by
      simp [others, hk]
    have hnonneg :
        ∀ j ∈ others q, 0 ≤ weights q j := by
      intro j _
      exact hweights_nonneg j
    have hle :
        weights q k ≤ ∑ j ∈ others q, weights q j := by
      have h := Finset.single_le_sum hnonneg hk'
      simpa using h
    exact hle.trans hsum_others_le

/-- One-hot bounds on a single active query, derived from per-key score gaps. -/
theorem oneHot_bounds_at_of_scoreGapLo
    (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (scoresReal : Fin seq → Fin seq → Real)
    (scoreGapLo : Fin seq → Fin seq → Rat)
    (epsAt : Fin seq → Rat)
    (hepsAt :
      ∀ q, epsAt q =
        min (1 : Rat)
          ((Finset.univ : Finset (Fin seq)).erase (prev q) |>.sum (fun k =>
            if scoreGapLo q k < 0 then
              (1 : Rat)
            else
              ratDivUp 1 (1 + scoreGapLo q k))))
    (hscore_gap_real_at :
      ∀ q, q ∈ active → ∀ k, k ≠ prev q →
        scoresReal q k + (scoreGapLo q k : Real) ≤ scoresReal q (prev q)) :
    ∀ q, q ∈ active →
      Layers.OneHotApproxBoundsOnActive (Val := Real) (epsAt q : Real)
        (fun q' => q' = q) prev
        (fun q k => Circuit.softmax (scoresReal q) k) := by
  classical
  intro q hq
  let weights : Fin seq → Fin seq → Real := fun q k =>
    Circuit.softmax (scoresReal q) k
  let others : Fin seq → Finset (Fin seq) := fun q =>
    (Finset.univ : Finset (Fin seq)).erase (prev q)
  let bound : Fin seq → Rat := fun k =>
    if scoreGapLo q k < 0 then
      (1 : Rat)
    else
      ratDivUp 1 (1 + scoreGapLo q k)
  have hweights_nonneg : ∀ k, 0 ≤ weights q k := by
    intro k
    simpa [weights] using
      (Circuit.softmax_nonneg (scores := scoresReal q) k)
  have hsum_one : (∑ k, weights q k) = 1 := by
    simpa [weights] using
      (Circuit.softmax_sum_one (scores := scoresReal q))
  have hsum_others_le_one : (∑ k ∈ others q, weights q k) ≤ 1 := by
    have hsubset : others q ⊆ (Finset.univ : Finset (Fin seq)) := by
      intro k hk
      simp
    have hsum_le :
        (∑ k ∈ others q, weights q k) ≤
          ∑ k ∈ (Finset.univ : Finset (Fin seq)), weights q k :=
      Finset.sum_le_sum_of_subset_of_nonneg hsubset (by
        intro k _ _
        exact hweights_nonneg k)
    simpa [hsum_one] using hsum_le
  have hbound :
      ∀ k ∈ others q, weights q k ≤ (bound k : Real) := by
    intro k hk
    have hkne : k ≠ prev q := (Finset.mem_erase.mp hk).1
    by_cases hneg : scoreGapLo q k < 0
    · have hle : weights q k ≤ 1 := by
        simpa [weights] using
          (Circuit.softmax_le_one (scores := scoresReal q) k)
      simpa [bound, hneg] using hle
    · have hnonneg : 0 ≤ scoreGapLo q k := le_of_not_gt hneg
      have hnonneg_real : 0 ≤ (scoreGapLo q k : Real) := by
        exact ratToReal_nonneg_of_nonneg hnonneg
      have hscore := hscore_gap_real_at q hq k hkne
      have hsoft :
          weights q k ≤ 1 / (1 + (scoreGapLo q k : Real)) := by
        simpa [weights] using
          (Circuit.softmax_other_le_inv_one_add (scores := scoresReal q)
            (prev := prev q) (k := k) (m := (scoreGapLo q k : Real))
            hnonneg_real hscore)
      have hpos : (0 : Rat) < 1 + scoreGapLo q k := by
        have hle : (1 : Rat) ≤ 1 + scoreGapLo q k := by
          exact le_add_of_nonneg_right hnonneg
        exact lt_of_lt_of_le zero_lt_one hle
      have hden : (1 + scoreGapLo q k) ≠ 0 := by
        exact ne_of_gt hpos
      have hrat :
          1 / (1 + (scoreGapLo q k : Real)) ≤
            ratToReal (ratDivUp 1 (1 + scoreGapLo q k)) := by
        simpa [ratToReal] using
          (ratDivUp_ge_real 1 (1 + scoreGapLo q k) hden)
      have hbound' :
          weights q k ≤ ratToReal (ratDivUp 1 (1 + scoreGapLo q k)) :=
        hsoft.trans hrat
      simpa [bound, hneg] using hbound'
  have hsum_others_le : (∑ k ∈ others q, weights q k) ≤ (epsAt q : Real) := by
    have hsum_le :
        (∑ k ∈ others q, weights q k) ≤
          ∑ k ∈ others q, (bound k : Real) :=
      Finset.sum_le_sum hbound
    have hsum_le_min :
        (∑ k ∈ others q, weights q k) ≤
          min (1 : Real) (∑ k ∈ others q, (bound k : Real)) := by
      exact le_min hsum_others_le_one hsum_le
    have hepsAtReal :
        (epsAt q : Real) = min (1 : Real) (∑ k ∈ others q, (bound k : Real)) := by
      have h' : epsAt q = min 1 ((others q).sum bound) := by
        simpa only [others, bound] using hepsAt q
      have h'' :
          ratToReal (epsAt q) = ratToReal (min 1 ((others q).sum bound)) := by
        exact congrArg ratToReal h'
      -- Avoid rewriting the erased-sum into a difference.
      simpa [ratToReal_min, ratToReal, Rat.cast_sum] using h''
    simpa [hepsAtReal] using hsum_le_min
  refine
    { nonneg := ?_
      sum_one := ?_
      prev_large := ?_
      other_le := ?_ }
  · intro q' hq' k
    subst q'
    change 0 ≤ Circuit.softmax (scoresReal q) k
    exact Circuit.softmax_nonneg (scores := scoresReal q) k
  · intro q' hq'
    subst q'
    change (∑ k, Circuit.softmax (scoresReal q) k) = 1
    exact Circuit.softmax_sum_one (scores := scoresReal q)
  · intro q' hq'
    subst q'
    have hsum_eq :
        weights q (prev q) + ∑ k ∈ others q, weights q k = 1 := by
      have hsum' :
          weights q (prev q) + ∑ k ∈ others q, weights q k =
            ∑ k, weights q k := by
        simp [others]
      calc
        weights q (prev q) + ∑ k ∈ others q, weights q k =
            ∑ k, weights q k := hsum'
        _ = 1 := hsum_one
    have hsum_le' :
        weights q (prev q) + ∑ k ∈ others q, weights q k ≤
          weights q (prev q) + (epsAt q : Real) := by
      simpa [add_comm, add_left_comm, add_assoc] using
        (add_le_add_left hsum_others_le (weights q (prev q)))
    have hprev :
        1 ≤ weights q (prev q) + (epsAt q : Real) := by
      simpa [hsum_eq] using hsum_le'
    exact hprev
  · intro q' hq' k hk
    subst q'
    have hk' : k ∈ others q := by
      simp [others, hk]
    have hnonneg :
        ∀ j ∈ others q, 0 ≤ weights q j := by
      intro j _
      exact hweights_nonneg j
    have hle :
        weights q k ≤ ∑ j ∈ others q, weights q j := by
      have h := Finset.single_le_sum hnonneg hk'
      simpa using h
    exact hle.trans hsum_others_le

/-- Per-key weight bounds on a single active query, derived from per-key score gaps. -/
theorem weight_bound_at_of_scoreGapLo
    (active : Finset (Fin seq))
    (prev : Fin seq → Fin seq)
    (scoresReal : Fin seq → Fin seq → Real)
    (scoreGapLo : Fin seq → Fin seq → Rat)
    (weightBoundAt : Fin seq → Fin seq → Rat)
    (hweightBoundAt :
      ∀ q k, k ≠ prev q →
        weightBoundAt q k =
          if scoreGapLo q k < 0 then
            (1 : Rat)
          else
            ratDivUp 1 (1 + scoreGapLo q k))
    (hscore_gap_real_at :
      ∀ q, q ∈ active → ∀ k, k ≠ prev q →
        scoresReal q k + (scoreGapLo q k : Real) ≤ scoresReal q (prev q)) :
    ∀ q, q ∈ active → ∀ k, k ≠ prev q →
      Circuit.softmax (scoresReal q) k ≤ (weightBoundAt q k : Real) := by
  classical
  intro q hq k hk
  by_cases hneg : scoreGapLo q k < 0
  · have hle : Circuit.softmax (scoresReal q) k ≤ 1 := by
      simpa using (Circuit.softmax_le_one (scores := scoresReal q) k)
    simpa [hweightBoundAt q k hk, hneg] using hle
  · have hnonneg : 0 ≤ scoreGapLo q k := le_of_not_gt hneg
    have hnonneg_real : 0 ≤ (scoreGapLo q k : Real) := by
      exact ratToReal_nonneg_of_nonneg hnonneg
    have hscore := hscore_gap_real_at q hq k hk
    have hsoft :
        Circuit.softmax (scoresReal q) k ≤ 1 / (1 + (scoreGapLo q k : Real)) := by
      simpa using
        (Circuit.softmax_other_le_inv_one_add (scores := scoresReal q)
          (prev := prev q) (k := k) (m := (scoreGapLo q k : Real))
          hnonneg_real hscore)
    have hpos : (0 : Rat) < 1 + scoreGapLo q k := by
      have hle : (1 : Rat) ≤ 1 + scoreGapLo q k := by
        exact le_add_of_nonneg_right hnonneg
      exact lt_of_lt_of_le zero_lt_one hle
    have hden : (1 + scoreGapLo q k) ≠ 0 := by
      exact ne_of_gt hpos
    have hrat :
        1 / (1 + (scoreGapLo q k : Real)) ≤
          ratToReal (ratDivUp 1 (1 + scoreGapLo q k)) := by
      simpa [ratToReal] using
        (ratDivUp_ge_real 1 (1 + scoreGapLo q k) hden)
    have hbound' :
        Circuit.softmax (scoresReal q) k ≤ ratToReal (ratDivUp 1 (1 + scoreGapLo q k)) :=
      hsoft.trans hrat
    simpa [hweightBoundAt q k hk, hneg] using hbound'

end Sound

end Nfp
