-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Nfp.Core.Basic
public import Nfp.Circuit.Cert.Basic
public import Nfp.Circuit.Cert.SoftmaxMargin
public import Nfp.Circuit.Cert.ValueRange
public import Nfp.Circuit.Layers.Induction

/-!
Induction-head certificates with explicit scores, weights, and value bounds.
-/

public section

namespace Nfp

namespace Circuit

open scoped BigOperators

variable {seq : Nat}

/-- Certificate payload for value-interval bounds (Rat-valued). -/
structure ValueIntervalCert (seq : Nat) where
  /-- Lower bound for values. -/
  lo : Rat
  /-- Upper bound for values. -/
  hi : Rat
  /-- Lower bounds on per-key values. -/
  valsLo : Fin seq → Rat
  /-- Upper bounds on per-key values. -/
  valsHi : Fin seq → Rat
  /-- Exact per-key values. -/
  vals : Fin seq → Rat
  /-- Optional logit-diff direction metadata (ignored by the checker). -/
  direction : Option DirectionSpec

/-- Internal consistency predicate for value-interval certificates. -/
structure ValueIntervalCertBounds {seq : Nat} (c : ValueIntervalCert seq) : Prop where
  /-- Interval endpoints are ordered. -/
  lo_le_hi : c.lo ≤ c.hi
  /-- `lo` is below every lower bound. -/
  lo_le_valsLo : ∀ k, c.lo ≤ c.valsLo k
  /-- Lower bounds are below the values. -/
  valsLo_le_vals : ∀ k, c.valsLo k ≤ c.vals k
  /-- Values are below the upper bounds. -/
  vals_le_valsHi : ∀ k, c.vals k ≤ c.valsHi k
  /-- Upper bounds are below `hi`. -/
  valsHi_le_hi : ∀ k, c.valsHi k ≤ c.hi

/-- Boolean checker for value-interval certificates. -/
def checkValueIntervalCert [NeZero seq] (c : ValueIntervalCert seq) : Bool :=
  decide (c.lo ≤ c.hi) &&
    finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
      decide (c.lo ≤ c.valsLo k) &&
        decide (c.valsLo k ≤ c.vals k) &&
        decide (c.vals k ≤ c.valsHi k) &&
        decide (c.valsHi k ≤ c.hi))

/-- `checkValueIntervalCert` is sound for `ValueIntervalCertBounds`. -/
theorem checkValueIntervalCert_sound [NeZero seq] (c : ValueIntervalCert seq) :
    checkValueIntervalCert c = true → ValueIntervalCertBounds c := by
  classical
  intro hcheck
  have hcheck' :
      decide (c.lo ≤ c.hi) = true ∧
        finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
          decide (c.lo ≤ c.valsLo k) &&
            decide (c.valsLo k ≤ c.vals k) &&
            decide (c.vals k ≤ c.valsHi k) &&
            decide (c.valsHi k ≤ c.hi)) = true := by
    simpa [checkValueIntervalCert, Bool.and_eq_true] using hcheck
  rcases hcheck' with ⟨hlohi, hall⟩
  have hlohi' : c.lo ≤ c.hi := by
    simpa [decide_eq_true_iff] using hlohi
  have hall' :=
    (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hall
  have hbounds :
      ∀ k, c.lo ≤ c.valsLo k ∧ c.valsLo k ≤ c.vals k ∧
        c.vals k ≤ c.valsHi k ∧ c.valsHi k ≤ c.hi := by
    intro k
    have hk := hall' k (by simp)
    have hk' :
        decide (c.lo ≤ c.valsLo k) = true ∧
          decide (c.valsLo k ≤ c.vals k) = true ∧
            decide (c.vals k ≤ c.valsHi k) = true ∧
              decide (c.valsHi k ≤ c.hi) = true := by
      simpa [Bool.and_eq_true, and_assoc] using hk
    rcases hk' with ⟨hlo, hloVals, hvalsHi, hhi⟩
    refine ⟨?_, ?_, ?_, ?_⟩
    · simpa [decide_eq_true_iff] using hlo
    · simpa [decide_eq_true_iff] using hloVals
    · simpa [decide_eq_true_iff] using hvalsHi
    · simpa [decide_eq_true_iff] using hhi
  refine
    { lo_le_hi := hlohi'
      lo_le_valsLo := fun k => (hbounds k).1
      valsLo_le_vals := fun k => (hbounds k).2.1
      vals_le_valsHi := fun k => (hbounds k).2.2.1
      valsHi_le_hi := fun k => (hbounds k).2.2.2 }

/-- Certificate payload for induction-head bounds (Rat-valued). -/
structure InductionHeadCert (seq : Nat) where
  /-- Weight tolerance. -/
  eps : Rat
  /-- Per-query weight tolerance. -/
  epsAt : Fin seq → Rat
  /-- Per-key weight bounds derived from score gaps. -/
  weightBoundAt : Fin seq → Fin seq → Rat
  /-- Score margin used to justify weight bounds. -/
  margin : Rat
  /-- Active queries for which bounds are checked. -/
  active : Finset (Fin seq)
  /-- `prev` selector for induction-style attention. -/
  prev : Fin seq → Fin seq
  /-- Score matrix entries. -/
  scores : Fin seq → Fin seq → Rat
  /-- Attention weight entries. -/
  weights : Fin seq → Fin seq → Rat
  /-- Value-interval certificate for direction values. -/
  values : ValueIntervalCert seq

/-- View an induction certificate as a softmax-margin certificate. -/
def InductionHeadCert.softmaxMargin (c : InductionHeadCert seq) : SoftmaxMarginCert seq :=
  { eps := c.eps
    margin := c.margin
    active := c.active
    prev := c.prev
    scores := c.scores
    weights := c.weights }

private def weightsOkAt [NeZero seq] (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Rat) (epsAt : Fin seq → Rat) (q : Fin seq) : Bool :=
  finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
    decide (0 ≤ weights q k) &&
      (if k = prev q then
        true
      else
        decide (weights q k ≤ epsAt q)))

private def checkOneHotAt [NeZero seq] (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Rat) (epsAt : Fin seq → Rat) (q : Fin seq) : Bool :=
  weightsOkAt prev weights epsAt q &&
    decide (1 ≤ weights q (prev q) + epsAt q) &&
    decide ((∑ k, weights q k) = 1)

private def checkWeightBoundsAt [NeZero seq] (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Rat) (weightBoundAt : Fin seq → Fin seq → Rat) (q : Fin seq) :
    Bool :=
  finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
    if k = prev q then
      true
    else
      decide (weights q k ≤ weightBoundAt q k))

/-- `checkOneHotAt` yields per-query approximate one-hot bounds. -/
private theorem checkOneHotAt_sound [NeZero seq] (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Rat) (epsAt : Fin seq → Rat) (q : Fin seq) :
    checkOneHotAt prev weights epsAt q = true →
      Layers.OneHotApproxBoundsOnActive (Val := Rat) (epsAt q : Rat)
        (fun q' => q' = q) prev weights := by
  classical
  intro hOneHot
  have hOneHot' :
      weightsOkAt prev weights epsAt q = true ∧
        decide (1 ≤ weights q (prev q) + epsAt q) = true ∧
          decide ((∑ k, weights q k) = 1) = true := by
    simpa [checkOneHotAt, Bool.and_eq_true, and_assoc] using hOneHot
  rcases hOneHot' with ⟨hweights, hprev, hsum⟩
  have hweights' :=
    (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hweights
  refine
    { nonneg := ?_
      sum_one := ?_
      prev_large := ?_
      other_le := ?_ }
  · intro q' hq' k
    cases hq'
    have hk := hweights' k (by simp)
    have hk' :
        decide (0 ≤ weights q k) = true ∧
          (if k = prev q then
            true
          else
            decide (weights q k ≤ epsAt q)) = true := by
      simpa [Bool.and_eq_true] using hk
    simpa [decide_eq_true_iff] using hk'.1
  · intro q' hq'
    cases hq'
    simpa [decide_eq_true_iff] using hsum
  · intro q' hq'
    cases hq'
    simpa [decide_eq_true_iff] using hprev
  · intro q' hq' k hk
    cases hq'
    have hk' := hweights' k (by simp)
    have hk'' :
        decide (0 ≤ weights q k) = true ∧
          (if k = prev q then
            true
          else
            decide (weights q k ≤ epsAt q)) = true := by
      simpa [Bool.and_eq_true] using hk'
    have hother :
        decide (weights q k ≤ epsAt q) = true := by
      simpa [hk] using hk''.2
    simpa [decide_eq_true_iff] using hother

/-- `checkWeightBoundsAt` yields per-key upper bounds on non-`prev` weights. -/
private theorem checkWeightBoundsAt_sound [NeZero seq] (prev : Fin seq → Fin seq)
    (weights : Fin seq → Fin seq → Rat) (weightBoundAt : Fin seq → Fin seq → Rat) (q : Fin seq) :
    checkWeightBoundsAt prev weights weightBoundAt q = true →
      ∀ k, k ≠ prev q → weights q k ≤ weightBoundAt q k := by
  classical
  intro hweights k hk
  have hweights' :=
    (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hweights
  have hk' := hweights' k (by simp)
  have hk'' : decide (weights q k ≤ weightBoundAt q k) = true := by
    simpa [hk] using hk'
  simpa [decide_eq_true_iff] using hk''

/-- Boolean checker for induction-head certificates. -/
def checkInductionHeadCert [NeZero seq] (c : InductionHeadCert seq) : Bool :=
  checkSoftmaxMarginCert c.softmaxMargin &&
    finsetAll (Finset.univ : Finset (Fin seq)) (fun q =>
      if q ∈ c.active then
        checkOneHotAt c.prev c.weights c.epsAt q &&
          checkWeightBoundsAt c.prev c.weights c.weightBoundAt q
      else
        true) &&
    checkValueIntervalCert c.values

/-- Soundness predicate for induction-head certificates. -/
structure InductionHeadCertBounds [NeZero seq] (c : InductionHeadCert seq) : Prop where
  /-- Softmax-margin bounds on active queries. -/
  softmax_bounds :
    Layers.SoftmaxMarginBoundsOn (Val := Rat) c.eps c.margin (fun q => q ∈ c.active)
      c.prev c.scores c.weights
  /-- Per-query one-hot bounds for the weights. -/
  oneHot_bounds_at :
    ∀ q, q ∈ c.active →
      Layers.OneHotApproxBoundsOnActive (Val := Rat) (c.epsAt q : Rat)
        (fun q' => q' = q) c.prev c.weights
  /-- Per-key weight bounds for non-`prev` keys. -/
  weight_bounds_at :
    ∀ q, q ∈ c.active → ∀ k, k ≠ c.prev q →
      c.weights q k ≤ c.weightBoundAt q k
  /-- Value-interval bounds are internally consistent. -/
  value_bounds : ValueIntervalCertBounds c.values

/-- `checkInductionHeadCert` is sound for `InductionHeadCertBounds`. -/
theorem checkInductionHeadCert_sound [NeZero seq] (c : InductionHeadCert seq) :
    checkInductionHeadCert c = true → InductionHeadCertBounds c := by
  classical
  intro hcheck
  have hsplit :
      checkSoftmaxMarginCert c.softmaxMargin = true ∧
        finsetAll (Finset.univ : Finset (Fin seq)) (fun q =>
          if q ∈ c.active then
            checkOneHotAt c.prev c.weights c.epsAt q &&
              checkWeightBoundsAt c.prev c.weights c.weightBoundAt q
          else
            true) = true ∧
        checkValueIntervalCert c.values = true := by
    simpa [checkInductionHeadCert, Bool.and_eq_true, and_assoc] using hcheck
  rcases hsplit with ⟨hsoftmax, hactive, hvalues⟩
  have hsoftmax' :
      Layers.SoftmaxMarginBoundsOn (Val := Rat) c.eps c.margin (fun q => q ∈ c.active)
        c.prev c.scores c.weights :=
    checkSoftmaxMarginCert_sound c.softmaxMargin hsoftmax
  have hactive' :=
    (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hactive
  have honehot :
      ∀ q, q ∈ c.active →
        Layers.OneHotApproxBoundsOnActive (Val := Rat) (c.epsAt q : Rat)
          (fun q' => q' = q) c.prev c.weights := by
    intro q hq
    have hq' := hactive' q (by simp)
    have hq'' :
        checkOneHotAt c.prev c.weights c.epsAt q = true ∧
          checkWeightBoundsAt c.prev c.weights c.weightBoundAt q = true := by
      simpa [hq, Bool.and_eq_true] using hq'
    have hOneHot : checkOneHotAt c.prev c.weights c.epsAt q = true := hq''.1
    exact checkOneHotAt_sound c.prev c.weights c.epsAt q hOneHot
  have hweightBounds :
      ∀ q, q ∈ c.active → ∀ k, k ≠ c.prev q →
        c.weights q k ≤ c.weightBoundAt q k := by
    intro q hq k hk
    have hq' := hactive' q (by simp)
    have hq'' :
        checkOneHotAt c.prev c.weights c.epsAt q = true ∧
          checkWeightBoundsAt c.prev c.weights c.weightBoundAt q = true := by
      simpa [hq, Bool.and_eq_true] using hq'
    have hweights : checkWeightBoundsAt c.prev c.weights c.weightBoundAt q = true := hq''.2
    exact checkWeightBoundsAt_sound c.prev c.weights c.weightBoundAt q hweights k hk
  have hvals : ValueIntervalCertBounds c.values :=
    checkValueIntervalCert_sound c.values hvalues
  exact
    { softmax_bounds := hsoftmax'
      oneHot_bounds_at := honehot
      weight_bounds_at := hweightBounds
      value_bounds := hvals }

end Circuit

end Nfp
