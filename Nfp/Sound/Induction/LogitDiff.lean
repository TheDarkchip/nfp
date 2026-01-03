-- SPDX-License-Identifier: AGPL-3.0-or-later

import Nfp.Circuit.Cert.LogitDiff
import Nfp.Sound.Induction

/-!
Logit-diff bounds derived from induction certificates.
-/

namespace Nfp

namespace Sound

open Nfp.Circuit

section LogitDiffLowerBound

variable {seq dModel dHead : Nat} [NeZero seq]

/-- Real-valued logit-diff contribution for a query. -/
noncomputable def headLogitDiff (inputs : Model.InductionHeadInputs seq dModel dHead)
    (q : Fin seq) : Real :=
  let weights : Fin seq → Fin seq → Real := fun q k =>
    Circuit.softmax (scoresRealOfInputs inputs q) k
  dotProduct (weights q) (valsRealOfInputs inputs)

/-- Lower bound computed from the per-key lower bounds in an induction certificate. -/
def logitDiffLowerBoundFromCert (c : InductionHeadCert seq) : Option Rat :=
  Circuit.logitDiffLowerBoundAt c.active c.prev c.epsAt
    c.values.lo c.values.hi c.values.valsLo

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
      have hweights :
          Layers.OneHotApproxBoundsOnActive (Val := Real) (c.epsAt q : Real)
            (fun q' => q' = q) c.prev weights :=
        hsound.oneHot_bounds_at q hq
      have hvalsRange :
          Layers.ValueRangeBounds (Val := Real) (c.values.lo : Real) (c.values.hi : Real)
            (valsRealOfInputs inputs) := by
        refine { lo_le_hi := ?_, lo_le := ?_, le_hi := ?_ }
        · exact (Rat.cast_le (K := Real)).2 hsound.value_bounds.lo_le_hi
        · intro k
          exact
            le_trans (hsound.value_bounds.lo_le_valsLo k)
              (hsound.value_bounds.vals_bounds k).1
        · intro k
          exact
            le_trans (hsound.value_bounds.vals_bounds k).2
              (hsound.value_bounds.valsHi_le_hi k)
      have happrox :=
        Layers.inductionSpecApproxOn_of_oneHotApprox_valueRange
          (Val := Real)
          (n := n)
          (ε := (c.epsAt q : Real))
          (lo := (c.values.lo : Real))
          (hi := (c.values.hi : Real))
          (active := fun q' => q' = q)
          (prev := c.prev)
          (weights := weights)
          (vals := valsRealOfInputs inputs)
          hweights hvalsRange
      have hboundRat :
          lb ≤ c.values.valsLo (c.prev q) -
            c.epsAt q * (c.values.hi - c.values.lo) := by
        refine
          Circuit.logitDiffLowerBoundAt_le
            (active := c.active)
            (prev := c.prev)
            (epsAt := c.epsAt)
            (lo := c.values.lo)
            (hi := c.values.hi)
            (vals := c.values.valsLo)
            q hq lb ?_
        simpa [logitDiffLowerBoundFromCert] using hbound
      have hboundReal :
          (lb : Real) ≤
            (c.values.valsLo (c.prev q) : Real) -
              (c.epsAt q : Real) * ((c.values.hi : Real) - (c.values.lo : Real)) := by
        have hboundReal' :
            (lb : Real) ≤
              (c.values.valsLo (c.prev q) - c.epsAt q * (c.values.hi - c.values.lo) : Rat) := by
          exact (Rat.cast_le (K := Real)).2 hboundRat
        simpa [Rat.cast_sub, Rat.cast_mul] using hboundReal'
      have hvalsLo :
          (c.values.valsLo (c.prev q) : Real) ≤
            valsRealOfInputs inputs (c.prev q) := by
        exact (hsound.value_bounds.vals_bounds (c.prev q)).1
      have hvalsLo' :
          (c.values.valsLo (c.prev q) : Real) -
              (c.epsAt q : Real) * ((c.values.hi : Real) - (c.values.lo : Real)) ≤
            valsRealOfInputs inputs (c.prev q) -
              (c.epsAt q : Real) * ((c.values.hi : Real) - (c.values.lo : Real)) := by
        exact
          sub_le_sub_right hvalsLo
            ((c.epsAt q : Real) * ((c.values.hi : Real) - (c.values.lo : Real)))
      have hlow :
          valsRealOfInputs inputs (c.prev q) -
              (c.epsAt q : Real) * ((c.values.hi : Real) - (c.values.lo : Real)) ≤
            dotProduct (weights q) (valsRealOfInputs inputs) := by
        exact (sub_le_iff_le_add).2 (happrox q rfl).2
      have hle :
          (lb : Real) ≤ dotProduct (weights q) (valsRealOfInputs inputs) :=
        le_trans hboundReal (le_trans hvalsLo' hlow)
      simpa [headLogitDiff, weights] using hle

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
      · exact some { base := base, lb_pos := hpos }
      · exact none

end LogitDiffLowerBound

end Sound

end Nfp
