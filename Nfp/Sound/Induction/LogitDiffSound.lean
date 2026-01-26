-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Nfp.Sound.Induction.LogitDiff

/-!
Soundness lemmas for logit-diff lower bounds with custom eps/values.
-/

public section

namespace Nfp

namespace Sound

open Nfp.Circuit

variable {seq dModel dHead : Nat}

section WithNeZero

variable [NeZero seq]

/-- The unweighted logit-diff lower bound is sound for any valid per-query `epsAt`. -/
theorem logitDiffLowerBoundFromCacheWithEps_le
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (c : InductionHeadCert seq) (epsAtCustom : Fin seq → Rat)
    (hsound : InductionHeadCertSound inputs c)
    (honeHot :
      ∀ q, q ∈ c.active →
        Layers.OneHotApproxBoundsOnActive (Val := Real) (epsAtCustom q : Real)
          (fun q' => q' = q) c.prev
          (weightsRealOfInputs inputs))
    {lb : Rat}
    (hbound :
      logitDiffLowerBoundFromCacheWithEps c (logitDiffCache c) epsAtCustom = some lb)
    {q : Fin seq} (hq : q ∈ c.active) :
    (lb : Real) ≤ headLogitDiff inputs q := by
  classical
  cases seq with
  | zero =>
      cases (NeZero.ne (n := (0 : Nat)) rfl)
  | succ n =>
      let weights : Fin (Nat.succ n) → Fin (Nat.succ n) → Real := weightsRealOfInputs inputs
      let vals : Fin (Nat.succ n) → Real := valsRealOfInputs inputs
      let epsArr : Array Rat := Array.ofFn epsAtCustom
      let valsLoArr : Array Rat := Array.ofFn (logitDiffCache c).valsLo
      let epsAt : Fin (Nat.succ n) → Rat := fun q =>
        epsArr[q.1]'(by
          simp [epsArr, q.isLt])
      let valsLo : Fin (Nat.succ n) → Rat := fun q =>
        valsLoArr[q.1]'(by
          simp [valsLoArr, q.isLt])
      let loAt : Fin (Nat.succ n) → Rat := fun q =>
        let others : Finset (Fin (Nat.succ n)) :=
          (Finset.univ : Finset (Fin (Nat.succ n))).erase (c.prev q)
        if h : others.Nonempty then
          others.inf' h valsLo
        else
          c.values.lo
      let others : Finset (Fin (Nat.succ n)) :=
        (Finset.univ : Finset (Fin (Nat.succ n))).erase (c.prev q)
      let sumOthers : Real := ∑ k ∈ others, weights q k
      let valsLoPrev : Real := (c.values.valsLo (c.prev q) : Real)
      let loAtRat : Rat := loAt q
      let loAtReal : Real := (loAtRat : Real)
      have hboundRat :
          lb ≤ valsLo (c.prev q) -
            epsAt q * max (0 : Rat) (valsLo (c.prev q) - loAt q) := by
        refine
          Circuit.logitDiffLowerBoundAtLoAt_le
            (active := c.active)
            (prev := c.prev)
            (epsAt := epsAt)
            (loAt := loAt)
            (valsLo := valsLo)
            q hq lb ?_
        simpa [logitDiffLowerBoundFromCacheWithEps_def, logitDiffCache_def, loAt, epsAt, valsLo,
          valsLoArr, epsArr] using hbound
      have hepsAt : epsAt q = epsAtCustom q := by
        simp [epsAt, epsArr]
      have hvalsLo : ∀ k, valsLo k = c.values.valsLo k := by
        intro k
        simp [valsLo, valsLoArr, logitDiffCache_def, Bounds.cacheBoundTask_apply]
      have hboundRat' :
          lb ≤ c.values.valsLo (c.prev q) -
            epsAtCustom q * max (0 : Rat) (c.values.valsLo (c.prev q) - loAt q) := by
        simpa [hepsAt, hvalsLo] using hboundRat
      have hboundReal :
          (lb : Real) ≤
            valsLoPrev - (epsAtCustom q : Real) *
              max (0 : Real) (valsLoPrev - loAtReal) := by
        simpa [loAtRat, loAtReal, ratToReal_sub, ratToReal_mul, ratToReal_max, ratToReal_def]
          using ratToReal_le_of_le hboundRat'
      have hweights_nonneg : ∀ k, 0 ≤ weights q k := by
        have hweights := honeHot q hq
        simpa [weights] using hweights.nonneg q rfl
      have hweights := honeHot q hq
      have hsum_decomp :
          weights q (c.prev q) + ∑ k ∈ others, weights q k = ∑ k, weights q k := by
        simp [others]
      have hsum :
          weights q (c.prev q) + ∑ k ∈ others, weights q k = 1 := by
        have hsum_one : (∑ k, weights q k) = 1 := by
          simpa [weights] using hweights.sum_one q rfl
        calc
          weights q (c.prev q) + ∑ k ∈ others, weights q k = ∑ k, weights q k := hsum_decomp
          _ = 1 := hsum_one
      have hsum_others_le : sumOthers ≤ (epsAtCustom q : Real) := by
        have hprev : 1 ≤ weights q (c.prev q) + (epsAtCustom q : Real) :=
          hweights.prev_large q rfl
        have hprev' :
            weights q (c.prev q) + sumOthers ≤
              weights q (c.prev q) + (epsAtCustom q : Real) := by
          simpa [hsum, sumOthers] using hprev
        exact (add_le_add_iff_left (weights q (c.prev q))).1 hprev'
      have hloAt_le_valsLo : ∀ k ∈ others, loAtRat ≤ c.values.valsLo k := by
        intro k hk
        have hnonempty : others.Nonempty := ⟨k, hk⟩
        have hmin : others.inf' hnonempty valsLo ≤ valsLo k :=
          Finset.inf'_le (s := others) (f := valsLo) hk
        have hnonempty' : (Finset.univ.erase (c.prev q)).Nonempty := by
          simpa [others] using hnonempty
        have hloAt : loAtRat = others.inf' hnonempty valsLo := by
          dsimp [loAtRat, loAt]
          simp [hnonempty', others]
        have hvalsLo' : valsLo k = c.values.valsLo k := hvalsLo k
        calc
          loAtRat = others.inf' hnonempty valsLo := hloAt
          _ ≤ valsLo k := hmin
          _ = c.values.valsLo k := hvalsLo'
      have hvals_lo : ∀ k ∈ others, loAtReal ≤ vals k := by
        intro k hk
        have hloRat := hloAt_le_valsLo k hk
        have hloReal : loAtReal ≤ (c.values.valsLo k : Real) := by
          simpa [loAtReal, ratToReal_def] using (ratToReal_le_of_le hloRat)
        have hvals : (c.values.valsLo k : Real) ≤ vals k := by
          simpa using (hsound.value_bounds.vals_bounds k).1
        exact le_trans hloReal hvals
      have hvalsLo_prev : valsLoPrev ≤ vals (c.prev q) := by
        exact (hsound.value_bounds.vals_bounds (c.prev q)).1
      have hsum_lo :
          sumOthers * loAtReal = ∑ k ∈ others, weights q k * loAtReal := by
        have hsum_lo' :
            (∑ k ∈ others, weights q k) * loAtReal =
              ∑ k ∈ others, weights q k * loAtReal := by
          simpa using
            (Finset.sum_mul (s := others) (f := fun k => weights q k) (a := loAtReal))
        simpa [sumOthers] using hsum_lo'
      have hsum_vals_ge :
          sumOthers * loAtReal ≤ ∑ k ∈ others, weights q k * vals k := by
        have hle :
            ∀ k ∈ others, weights q k * loAtReal ≤ weights q k * vals k := by
          intro k _hk
          have hval := hvals_lo k _hk
          have hnonneg := hweights_nonneg k
          exact mul_le_mul_of_nonneg_left hval hnonneg
        have hsum' :
            ∑ k ∈ others, weights q k * loAtReal ≤
              ∑ k ∈ others, weights q k * vals k := by
          exact Finset.sum_le_sum hle
        simpa [hsum_lo] using hsum'
      have hsum_vals_ge' :
          ∑ k ∈ others, weights q k * loAtReal ≤
            ∑ k ∈ others, weights q k * vals k := by
        simpa [hsum_lo] using hsum_vals_ge
      have hsum_nonneg : 0 ≤ sumOthers := by
        have hnonneg : ∀ k ∈ others, 0 ≤ weights q k := by
          intro k hk
          exact hweights_nonneg k
        have hsum_nonneg' : 0 ≤ ∑ k ∈ others, weights q k := by
          exact Finset.sum_nonneg hnonneg
        simpa [sumOthers] using hsum_nonneg'
      have hsplit :
          weights q (c.prev q) = 1 - sumOthers := by
        have hsum' : weights q (c.prev q) + sumOthers = 1 := by
          simpa [sumOthers] using hsum
        exact (eq_sub_iff_add_eq).2 hsum'
      have hdiff_le : valsLoPrev - loAtReal ≤ max (0 : Real) (valsLoPrev - loAtReal) := by
        exact le_max_right _ _
      have hsum_mul_le_left :
          sumOthers * (valsLoPrev - loAtReal) ≤
            sumOthers * max (0 : Real) (valsLoPrev - loAtReal) := by
        exact mul_le_mul_of_nonneg_left hdiff_le hsum_nonneg
      have hmax_nonneg : 0 ≤ max (0 : Real) (valsLoPrev - loAtReal) := by
        exact le_max_left _ _
      have hsum_mul_le :
          sumOthers * (valsLoPrev - loAtReal) ≤
            (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) := by
        have hsum_mul_le_right :
            sumOthers * max (0 : Real) (valsLoPrev - loAtReal) ≤
              (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) := by
          exact mul_le_mul_of_nonneg_right hsum_others_le hmax_nonneg
        exact le_trans hsum_mul_le_left hsum_mul_le_right
      have hsub_le :
          valsLoPrev - (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) ≤
            valsLoPrev - sumOthers * (valsLoPrev - loAtReal) := by
        exact sub_le_sub_left hsum_mul_le valsLoPrev
      have hdot_lower :
          valsLoPrev - (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) ≤
            dotProduct (weights q) vals := by
        calc
          valsLoPrev - (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) ≤
              valsLoPrev - sumOthers * (valsLoPrev - loAtReal) := hsub_le
          _ = weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal := by
              have hsplit_calc :
                  valsLoPrev - sumOthers * (valsLoPrev - loAtReal) =
                    (1 - sumOthers) * valsLoPrev + sumOthers * loAtReal := by
                ring
              simpa [hsplit] using hsplit_calc
          _ ≤ dotProduct (weights q) vals := by
              have hprev_le := mul_le_mul_of_nonneg_left hvalsLo_prev (hweights_nonneg (c.prev q))
              have hdot_ge :
                  weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal ≤
                    weights q (c.prev q) * vals (c.prev q) + sumOthers * loAtReal := by
                simpa [add_comm, add_left_comm, add_assoc] using
                  (add_le_add_right hprev_le (sumOthers * loAtReal))
              have hdot_ge' :
                  weights q (c.prev q) * vals (c.prev q) + sumOthers * loAtReal ≤
                    dotProduct (weights q) vals := by
                calc
                  weights q (c.prev q) * vals (c.prev q) + sumOthers * loAtReal
                      = weights q (c.prev q) * vals (c.prev q) +
                          ∑ k ∈ others, weights q k * loAtReal := by
                            simp [hsum_lo]
                  _ ≤ weights q (c.prev q) * vals (c.prev q) +
                        ∑ k ∈ others, weights q k * vals k := by
                          simpa [add_comm, add_left_comm, add_assoc] using
                            (add_le_add_left hsum_vals_ge'
                              (weights q (c.prev q) * vals (c.prev q)))
                  _ = dotProduct (weights q) vals := by
                        simp [dotProduct, others]
              exact le_trans hdot_ge hdot_ge'
      have hle : (lb : Real) ≤ dotProduct (weights q) vals :=
        le_trans hboundReal hdot_lower
      simpa [headLogitDiff_def, weights, vals] using hle

/-- The unweighted logit-diff lower bound is sound for custom eps and values. -/
theorem logitDiffLowerBoundFromCacheWithEpsVals_le
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (c : InductionHeadCert seq) (epsAtCustom valsLoCustom : Fin seq → Rat)
    (hsound : InductionHeadCertSound inputs c)
    (honeHot :
      ∀ q, q ∈ c.active →
        Layers.OneHotApproxBoundsOnActive (Val := Real) (epsAtCustom q : Real)
          (fun q' => q' = q) c.prev
          (weightsRealOfInputs inputs))
    (hvalsLo :
      ∀ k, (valsLoCustom k : Real) ≤ valsRealOfInputs inputs k)
    {lb : Rat}
    (hbound :
      logitDiffLowerBoundFromCacheWithEpsVals c epsAtCustom valsLoCustom = some lb)
    {q : Fin seq} (hq : q ∈ c.active) :
    (lb : Real) ≤ headLogitDiff inputs q := by
  classical
  cases seq with
  | zero =>
      cases (NeZero.ne (n := (0 : Nat)) rfl)
  | succ n =>
      let weights : Fin (Nat.succ n) → Fin (Nat.succ n) → Real := weightsRealOfInputs inputs
      let vals : Fin (Nat.succ n) → Real := valsRealOfInputs inputs
      let epsArr : Array Rat := Array.ofFn epsAtCustom
      let valsLoArr : Array Rat := Array.ofFn valsLoCustom
      let epsAt : Fin (Nat.succ n) → Rat := fun q =>
        epsArr[q.1]'(by
          simp [epsArr, q.isLt])
      let valsLo : Fin (Nat.succ n) → Rat := fun q =>
        valsLoArr[q.1]'(by
          simp [valsLoArr, q.isLt])
      let loAt : Fin (Nat.succ n) → Rat := fun q =>
        let others : Finset (Fin (Nat.succ n)) :=
          (Finset.univ : Finset (Fin (Nat.succ n))).erase (c.prev q)
        if h : others.Nonempty then
          others.inf' h valsLo
        else
          c.values.lo
      let others : Finset (Fin (Nat.succ n)) :=
        (Finset.univ : Finset (Fin (Nat.succ n))).erase (c.prev q)
      let sumOthers : Real := ∑ k ∈ others, weights q k
      let valsLoPrev : Real := (valsLo (c.prev q) : Real)
      let loAtRat : Rat := loAt q
      let loAtReal : Real := (loAtRat : Real)
      have hvalsLo_eq : ∀ k, valsLo k = valsLoCustom k := by
        intro k
        simp [valsLo, valsLoArr]
      have hboundRat :
          lb ≤ valsLo (c.prev q) -
            epsAt q * max (0 : Rat) (valsLo (c.prev q) - loAt q) := by
        refine
          Circuit.logitDiffLowerBoundAtLoAt_le
            (active := c.active)
            (prev := c.prev)
            (epsAt := epsAt)
            (loAt := loAt)
            (valsLo := valsLo)
            q hq lb ?_
        simpa [logitDiffLowerBoundFromCacheWithEpsVals_def, loAt, epsAt, valsLo, valsLoArr, epsArr]
          using hbound
      have hepsAt : epsAt q = epsAtCustom q := by
        simp [epsAt, epsArr]
      have hboundRat' :
          lb ≤ valsLoCustom (c.prev q) -
            epsAtCustom q * max (0 : Rat) (valsLoCustom (c.prev q) - loAt q) := by
        simpa [hepsAt, hvalsLo_eq] using hboundRat
      have hboundReal :
          (lb : Real) ≤
            valsLoPrev - (epsAtCustom q : Real) *
              max (0 : Real) (valsLoPrev - loAtReal) := by
        have hvalsLoPrev_eq : valsLoPrev = (valsLoCustom (c.prev q) : Real) := by
          simp [valsLoPrev, valsLo, valsLoArr]
        simpa [hvalsLoPrev_eq, loAtRat, loAtReal, ratToReal_sub, ratToReal_mul, ratToReal_max,
          ratToReal_def]
          using ratToReal_le_of_le hboundRat'
      have hweights_nonneg : ∀ k, 0 ≤ weights q k := by
        have hweights := honeHot q hq
        simpa [weights] using hweights.nonneg q rfl
      have hweights := honeHot q hq
      have hsum_decomp :
          weights q (c.prev q) + ∑ k ∈ others, weights q k = ∑ k, weights q k := by
        simp [others]
      have hsum :
          weights q (c.prev q) + ∑ k ∈ others, weights q k = 1 := by
        have hsum_one : (∑ k, weights q k) = 1 := by
          simpa [weights] using hweights.sum_one q rfl
        calc
          weights q (c.prev q) + ∑ k ∈ others, weights q k = ∑ k, weights q k := hsum_decomp
          _ = 1 := hsum_one
      have hsum_others_le : sumOthers ≤ (epsAtCustom q : Real) := by
        have hprev : 1 ≤ weights q (c.prev q) + (epsAtCustom q : Real) :=
          hweights.prev_large q rfl
        have hprev' :
            weights q (c.prev q) + sumOthers ≤
              weights q (c.prev q) + (epsAtCustom q : Real) := by
          simpa [hsum, sumOthers] using hprev
        exact (add_le_add_iff_left (weights q (c.prev q))).1 hprev'
      have hvalsLo_real : ∀ k, (valsLo k : Real) ≤ vals k := by
        intro k
        have hvals := hvalsLo k
        simpa [valsLo, valsLoArr, vals] using hvals
      have hprev_lo : valsLoPrev ≤ vals (c.prev q) := by
        simpa [valsLoPrev] using hvalsLo_real (c.prev q)
      have hloAt_le_valsLo : ∀ k ∈ others, loAtRat ≤ valsLo k := by
        intro k hk
        have hnonempty : others.Nonempty := ⟨k, hk⟩
        have hmin : others.inf' hnonempty valsLo ≤ valsLo k :=
          Finset.inf'_le (s := others) (f := valsLo) hk
        have hnonempty' : (Finset.univ.erase (c.prev q)).Nonempty := by
          simpa [others] using hnonempty
        have hloAt : loAtRat = others.inf' hnonempty valsLo := by
          dsimp [loAtRat, loAt]
          simp [hnonempty', others]
        calc
          loAtRat = others.inf' hnonempty valsLo := hloAt
          _ ≤ valsLo k := hmin
      have hvals_lo : ∀ k ∈ others, loAtReal ≤ vals k := by
        intro k hk
        have hloRat := hloAt_le_valsLo k hk
        have hloReal : loAtReal ≤ (valsLo k : Real) := by
          simpa [loAtReal, ratToReal_def] using (ratToReal_le_of_le hloRat)
        have hvalsReal : (valsLo k : Real) ≤ vals k := hvalsLo_real k
        exact le_trans hloReal hvalsReal
      have hsum_nonneg : 0 ≤ sumOthers := by
        have hnonneg : ∀ k ∈ others, 0 ≤ weights q k := by
          intro k hk
          exact hweights_nonneg k
        have hsum_nonneg' : 0 ≤ ∑ k ∈ others, weights q k := by
          exact Finset.sum_nonneg hnonneg
        simpa [sumOthers] using hsum_nonneg'
      have hsplit :
          weights q (c.prev q) = 1 - sumOthers := by
        have hsum' : weights q (c.prev q) + sumOthers = 1 := by
          simpa [sumOthers] using hsum
        exact (eq_sub_iff_add_eq).2 hsum'
      have hdiff_le : valsLoPrev - loAtReal ≤ max (0 : Real) (valsLoPrev - loAtReal) := by
        exact le_max_right _ _
      have hsum_mul_le_left :
          sumOthers * (valsLoPrev - loAtReal) ≤
            sumOthers * max (0 : Real) (valsLoPrev - loAtReal) := by
        exact mul_le_mul_of_nonneg_left hdiff_le hsum_nonneg
      have hmax_nonneg : 0 ≤ max (0 : Real) (valsLoPrev - loAtReal) := by
        exact le_max_left _ _
      have hsum_mul_le :
          sumOthers * (valsLoPrev - loAtReal) ≤
            (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) := by
        have hsum_mul_le_right :
            sumOthers * max (0 : Real) (valsLoPrev - loAtReal) ≤
              (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) := by
          exact mul_le_mul_of_nonneg_right hsum_others_le hmax_nonneg
        exact le_trans hsum_mul_le_left hsum_mul_le_right
      have hsub_le :
          valsLoPrev - (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) ≤
            valsLoPrev - sumOthers * (valsLoPrev - loAtReal) := by
        exact sub_le_sub_left hsum_mul_le valsLoPrev
      have hsum_lo :
          sumOthers * loAtReal = ∑ k ∈ others, weights q k * loAtReal := by
        have hsum' :
            (∑ k ∈ others, weights q k) * loAtReal =
              ∑ k ∈ others, weights q k * loAtReal := by
          simpa using
            (Finset.sum_mul (s := others) (f := fun k => weights q k) (a := loAtReal))
        simpa [sumOthers] using hsum'
      have hsum_vals_ge :
          sumOthers * loAtReal ≤ ∑ k ∈ others, weights q k * vals k := by
        have hle : ∀ k ∈ others, weights q k * loAtReal ≤ weights q k * vals k := by
          intro k hk
          have hlo := hvals_lo k hk
          have hnonneg := hweights_nonneg k
          exact mul_le_mul_of_nonneg_left hlo hnonneg
        have hle' :
            ∑ k ∈ others, weights q k * loAtReal ≤
              ∑ k ∈ others, weights q k * vals k := by
          exact Finset.sum_le_sum hle
        simpa [hsum_lo] using hle'
      have hsum_vals_ge' :
          ∑ k ∈ others, weights q k * loAtReal ≤
            ∑ k ∈ others, weights q k * vals k := by
        simpa [hsum_lo] using hsum_vals_ge
      have hdot_ge'' :
          weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal ≤
            dotProduct (weights q) vals := by
        have hdot_ge :
            weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal ≤
              weights q (c.prev q) * vals (c.prev q) + sumOthers * loAtReal := by
          have hprev_le := mul_le_mul_of_nonneg_left hprev_lo (hweights_nonneg (c.prev q))
          simpa [add_comm, add_left_comm, add_assoc] using
            (add_le_add_right hprev_le (sumOthers * loAtReal))
        have hdot_ge' :
            weights q (c.prev q) * vals (c.prev q) + sumOthers * loAtReal ≤
              dotProduct (weights q) vals := by
          calc
            weights q (c.prev q) * vals (c.prev q) + sumOthers * loAtReal =
              weights q (c.prev q) * vals (c.prev q) +
                ∑ k ∈ others, weights q k * loAtReal := by
                  simp [hsum_lo]
            _ ≤ weights q (c.prev q) * vals (c.prev q) +
                  ∑ k ∈ others, weights q k * vals k := by
                  simpa [add_comm, add_left_comm, add_assoc] using
                    (add_le_add_left hsum_vals_ge'
                      (weights q (c.prev q) * vals (c.prev q)))
            _ = dotProduct (weights q) vals := by
                  simp [dotProduct, others]
        exact le_trans hdot_ge hdot_ge'
      have hdot_lower :
          valsLoPrev - (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) ≤
            dotProduct (weights q) vals := by
        calc
          valsLoPrev - (epsAtCustom q : Real) * max (0 : Real) (valsLoPrev - loAtReal) ≤
              valsLoPrev - sumOthers * (valsLoPrev - loAtReal) := hsub_le
          _ = weights q (c.prev q) * valsLoPrev + sumOthers * loAtReal := by
              have hsplit_calc :
                  valsLoPrev - sumOthers * (valsLoPrev - loAtReal) =
                    (1 - sumOthers) * valsLoPrev + sumOthers * loAtReal := by
                ring
              simpa [hsplit] using hsplit_calc
          _ ≤ dotProduct (weights q) vals := hdot_ge''
      have hle : (lb : Real) ≤ dotProduct (weights q) vals :=
        le_trans hboundReal hdot_lower
      simpa [headLogitDiff_def, weights, vals] using hle

end WithNeZero

end Sound

end Nfp
