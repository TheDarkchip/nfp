-- SPDX-License-Identifier: AGPL-3.0-or-later

import Aesop
import Nfp.Circuit.Cert.LogitDiff
import Nfp.Sound.Induction

/-!
Logit-diff bounds derived from induction certificates.
-/

namespace Nfp

namespace Sound

open Nfp.Circuit

variable {seq : Nat}

section LogitDiffLowerBound

variable {seq dModel dHead : Nat} [NeZero seq]

section

/-- Real-valued logit-diff contribution for a query. -/
noncomputable def headLogitDiff (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) : Real :=
  let weights : Fin seq → Fin seq → Real := fun q k =>
    Circuit.softmax (scoresRealOfInputs inputs q) k
  dotProduct (weights q) (valsRealOfInputs inputs)

/-- Lower bound computed from the per-key lower bounds in an induction certificate. -/
def logitDiffLowerBoundFromCert (c : InductionHeadCert seq) : Option Rat :=
  Circuit.logitDiffLowerBoundAtLo c.active c.prev c.epsAt
    c.values.lo c.values.valsLo

theorem logitDiffLowerBoundFromCert_le
    (inputs : Model.InductionHeadInputs seq dModel dHead)
    (c : InductionHeadCert seq) (hsound : InductionHeadCertSound inputs c)
    {lb : Rat} (hbound : logitDiffLowerBoundFromCert c = some lb)
    {q : Fin seq} (hq : q ∈ c.active) :
    (lb : Real) ≤ headLogitDiff inputs q := by
  classical
  cases seq with
  | zero =>
      cases (NeZero.ne (n := (0 : Nat)) rfl)
  | succ n =>
      let weights : Fin (Nat.succ n) → Fin (Nat.succ n) → Real := fun q k =>
        Circuit.softmax (scoresRealOfInputs inputs q) k
      let vals : Fin (Nat.succ n) → Real := valsRealOfInputs inputs
      let others : Finset (Fin (Nat.succ n)) :=
        (Finset.univ : Finset (Fin (Nat.succ n))).erase (c.prev q)
      let sumOthers : Real := ∑ k ∈ others, weights q k
      let valsLoPrev : Real := (c.values.valsLo (c.prev q) : Real)
      let lo : Real := (c.values.lo : Real)
      have hboundRat :
          lb ≤ c.values.valsLo (c.prev q) -
            c.epsAt q * (c.values.valsLo (c.prev q) - c.values.lo) := by
        refine
          Circuit.logitDiffLowerBoundAtLo_le
            (active := c.active)
            (prev := c.prev)
            (epsAt := c.epsAt)
            (lo := c.values.lo)
            (valsLo := c.values.valsLo)
            q hq lb ?_
        simpa [logitDiffLowerBoundFromCert] using hbound
      have hboundReal :
          (lb : Real) ≤ valsLoPrev - (c.epsAt q : Real) * (valsLoPrev - lo) := by
        simpa [ratToReal_sub, ratToReal_mul] using ratToReal_le_of_le hboundRat
      have hweights_nonneg : ∀ k, 0 ≤ weights q k :=
        hsound.softmax_bounds.nonneg q hq
      have hweights := hsound.oneHot_bounds_at q hq
      have hsum_decomp :
          weights q (c.prev q) + ∑ k ∈ others, weights q k = ∑ k, weights q k := by
        simp [others]
      have hsum :
          weights q (c.prev q) + ∑ k ∈ others, weights q k = 1 := by
        calc
          weights q (c.prev q) + ∑ k ∈ others, weights q k = ∑ k, weights q k := hsum_decomp
          _ = 1 := hweights.sum_one q rfl
      have hsum_others_le : sumOthers ≤ (c.epsAt q : Real) := by
        have hprev : 1 ≤ weights q (c.prev q) + (c.epsAt q : Real) :=
          hweights.prev_large q rfl
        have hprev' :
            weights q (c.prev q) + sumOthers ≤
              weights q (c.prev q) + (c.epsAt q : Real) := by
          simpa [hsum, sumOthers] using hprev
        exact (add_le_add_iff_left (weights q (c.prev q))).1 hprev'
      have hvals_lo : ∀ k, lo ≤ vals k := by
        intro k
        have hlo := hsound.value_bounds.lo_le_valsLo k
        have hvals := (hsound.value_bounds.vals_bounds k).1
        exact le_trans hlo hvals
      have hvalsLo_prev : valsLoPrev ≤ vals (c.prev q) := by
        exact (hsound.value_bounds.vals_bounds (c.prev q)).1
      have hsum_vals_ge :
          sumOthers * lo ≤ ∑ k ∈ others, weights q k * vals k := by
        have hsum_lo :
            sumOthers * lo = ∑ k ∈ others, weights q k * lo := by
          have hsum_lo' :
              (∑ k ∈ others, weights q k) * lo =
                ∑ k ∈ others, weights q k * lo := by
            simpa using
              (Finset.sum_mul (s := others) (f := fun k => weights q k) (a := lo))
          simpa [sumOthers] using hsum_lo'
        have hle :
            ∀ k ∈ others, weights q k * lo ≤ weights q k * vals k := by
          intro k _hk
          have hval := hvals_lo k
          have hnonneg := hweights_nonneg k
          exact mul_le_mul_of_nonneg_left hval hnonneg
        have hsum' :
            ∑ k ∈ others, weights q k * lo ≤
              ∑ k ∈ others, weights q k * vals k := by
          exact Finset.sum_le_sum hle
        simpa [hsum_lo] using hsum'
      have hsum_prod :
          weights q (c.prev q) * vals (c.prev q) +
              ∑ k ∈ others, weights q k * vals k =
            ∑ k, weights q k * vals k := by
        simp [others]
      have hout_eq :
          dotProduct (weights q) vals =
            weights q (c.prev q) * vals (c.prev q) +
              ∑ k ∈ others, weights q k * vals k := by
        simpa [dotProduct] using hsum_prod.symm
      have hdot_ge :
          weights q (c.prev q) * vals (c.prev q) + sumOthers * lo ≤
            dotProduct (weights q) vals := by
        have hle :
            weights q (c.prev q) * vals (c.prev q) + sumOthers * lo ≤
              weights q (c.prev q) * vals (c.prev q) +
                ∑ k ∈ others, weights q k * vals k := by
          simpa [add_comm, add_left_comm, add_assoc] using
            (add_le_add_left hsum_vals_ge (weights q (c.prev q) * vals (c.prev q)))
        simpa [sumOthers, hout_eq, add_comm, add_left_comm, add_assoc] using hle
      have hprev_lo :
          weights q (c.prev q) * valsLoPrev ≤
            weights q (c.prev q) * vals (c.prev q) := by
        exact mul_le_mul_of_nonneg_left hvalsLo_prev (hweights_nonneg (c.prev q))
      have hdot_ge' :
          weights q (c.prev q) * valsLoPrev + sumOthers * lo ≤
            dotProduct (weights q) vals := by
        have hle :
            weights q (c.prev q) * valsLoPrev + sumOthers * lo ≤
              weights q (c.prev q) * vals (c.prev q) + sumOthers * lo := by
          simpa [add_comm, add_left_comm, add_assoc] using
            (add_le_add_right hprev_lo (sumOthers * lo))
        exact hle.trans hdot_ge
      have hsplit :
          weights q (c.prev q) * valsLoPrev + sumOthers * lo =
            valsLoPrev - sumOthers * (valsLoPrev - lo) := by
        have hsplit' :
            weights q (c.prev q) * valsLoPrev + sumOthers * lo =
              (weights q (c.prev q) + sumOthers) * valsLoPrev -
                sumOthers * (valsLoPrev - lo) := by
          ring
        calc
          weights q (c.prev q) * valsLoPrev + sumOthers * lo =
              (weights q (c.prev q) + sumOthers) * valsLoPrev -
                sumOthers * (valsLoPrev - lo) := hsplit'
          _ = valsLoPrev - sumOthers * (valsLoPrev - lo) := by
              simp [hsum, sumOthers]
      have hdiff_nonneg : 0 ≤ valsLoPrev - lo := by
        exact sub_nonneg.mpr (hsound.value_bounds.lo_le_valsLo (c.prev q))
      have hsum_mul_le :
          sumOthers * (valsLoPrev - lo) ≤
            (c.epsAt q : Real) * (valsLoPrev - lo) := by
        exact mul_le_mul_of_nonneg_right hsum_others_le hdiff_nonneg
      have hsub_le :
          valsLoPrev - (c.epsAt q : Real) * (valsLoPrev - lo) ≤
            valsLoPrev - sumOthers * (valsLoPrev - lo) := by
        exact sub_le_sub_left hsum_mul_le valsLoPrev
      have hdot_lower :
          valsLoPrev - (c.epsAt q : Real) * (valsLoPrev - lo) ≤
            dotProduct (weights q) vals := by
        calc
          valsLoPrev - (c.epsAt q : Real) * (valsLoPrev - lo) ≤
              valsLoPrev - sumOthers * (valsLoPrev - lo) := hsub_le
          _ = weights q (c.prev q) * valsLoPrev + sumOthers * lo := by
              simp [hsplit]
          _ ≤ dotProduct (weights q) vals := hdot_ge'
      have hle : (lb : Real) ≤ dotProduct (weights q) vals :=
        le_trans hboundReal hdot_lower
      simpa [headLogitDiff, weights, vals] using hle

/-- Certified logit-diff lower bound derived from exact head inputs. -/
structure InductionLogitLowerBoundResult
    (inputs : Model.InductionHeadInputs seq dModel dHead) where
  /-- Induction certificate built from the head inputs. -/
  cert : InductionHeadCert seq
  /-- Soundness proof for the induction certificate. -/
  sound : InductionHeadCertSound inputs cert
  /-- Reported lower bound on logit diff. -/
  lb : Rat
  /-- `lb` is computed from `logitDiffLowerBoundFromCert`. -/
  lb_def : logitDiffLowerBoundFromCert cert = some lb
  /-- The lower bound is sound on active queries. -/
  lb_sound : ∀ q, q ∈ cert.active → (lb : Real) ≤ headLogitDiff inputs q

/-- Nonvacuous logit-diff bound (strictly positive). -/
structure InductionLogitLowerBoundNonvacuous
    (inputs : Model.InductionHeadInputs seq dModel dHead) where
  /-- Base logit-diff bound data. -/
  base : InductionLogitLowerBoundResult inputs
  /-- The reported bound is strictly positive. -/
  lb_pos : 0 < base.lb

/-- Build a logit-diff lower bound from exact head inputs. -/
def buildInductionLogitLowerBoundFromHead?
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option (InductionLogitLowerBoundResult inputs) := by
  classical
  cases hcert : buildInductionCertFromHead? inputs with
  | none => exact none
  | some certWithProof =>
      rcases certWithProof with ⟨cert, hsound⟩
      cases hlb : logitDiffLowerBoundFromCert cert with
      | none => exact none
      | some lb =>
          refine some ?_
          refine
            { cert := cert
              sound := hsound
              lb := lb
              lb_def := hlb
              lb_sound := ?_ }
          intro q hq
          exact
            logitDiffLowerBoundFromCert_le
              (inputs := inputs)
              (c := cert)
              (hsound := hsound)
              (lb := lb)
              (hbound := hlb)
              (q := q)
              hq

/-- Build a strictly positive logit-diff lower bound from exact head inputs. -/
def buildInductionLogitLowerBoundNonvacuous?
    (inputs : Model.InductionHeadInputs seq dModel dHead) :
    Option (InductionLogitLowerBoundNonvacuous inputs) := by
  classical
  cases hbase : buildInductionLogitLowerBoundFromHead? inputs with
  | none => exact none
  | some base =>
      by_cases hpos : 0 < base.lb
      · exact some ⟨base, hpos⟩
      · exact none

end

end LogitDiffLowerBound

end Sound

end Nfp
