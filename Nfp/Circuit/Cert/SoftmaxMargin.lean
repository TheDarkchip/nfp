-- SPDX-License-Identifier: AGPL-3.0-or-later

module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Nfp.Core.Basic
public import Nfp.Circuit.Cert.Basic
public import Nfp.Circuit.Layers.Induction

/-!
Softmax-margin certificates for approximate one-hot attention weights.
-/

@[expose] public section

namespace Nfp

namespace Circuit

open scoped BigOperators

variable {seq : Nat}

/-- Certificate payload for softmax-margin bounds (Rat-valued). -/
structure SoftmaxMarginCert (seq : Nat) where
  /-- Weight tolerance. -/
  eps : Rat
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

/-- Boolean checker for softmax-margin certificates. -/
def checkSoftmaxMarginCert [NeZero seq] (c : SoftmaxMarginCert seq) : Bool :=
  finsetAll (Finset.univ : Finset (Fin seq)) (fun q =>
    if q ∈ c.active then
      finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
          decide (0 ≤ c.weights q k) &&
            (if k = c.prev q then
              true
            else
              decide (c.weights q k ≤ c.eps))) &&
        finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
          if k = c.prev q then
            true
          else
            decide (c.scores q k + c.margin ≤ c.scores q (c.prev q))) &&
        decide (1 ≤ c.weights q (c.prev q) + c.eps) &&
        decide ((∑ k, c.weights q k) = 1)
    else
      true)

/-- `checkSoftmaxMarginCert` is sound for `SoftmaxMarginBoundsOn`. -/
theorem checkSoftmaxMarginCert_sound [NeZero seq] (c : SoftmaxMarginCert seq) :
    checkSoftmaxMarginCert c = true →
      Layers.SoftmaxMarginBoundsOn (Val := Rat) c.eps c.margin (fun q => q ∈ c.active)
        c.prev c.scores c.weights := by
  classical
  intro hcheck
  let weightsOk (q : Fin seq) : Bool :=
    finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
      decide (0 ≤ c.weights q k) &&
        (if k = c.prev q then
          true
        else
          decide (c.weights q k ≤ c.eps)))
  let scoresOk (q : Fin seq) : Bool :=
    finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
      if k = c.prev q then
        true
      else
        decide (c.scores q k + c.margin ≤ c.scores q (c.prev q)))
  have hqall :
      ∀ q ∈ (Finset.univ : Finset (Fin seq)),
        (if q ∈ c.active then
          weightsOk q &&
            scoresOk q &&
            decide (1 ≤ c.weights q (c.prev q) + c.eps) &&
            decide ((∑ k, c.weights q k) = 1)
        else
          true) = true := by
    have hcheck'' :
        finsetAll (Finset.univ : Finset (Fin seq)) (fun q =>
            if q ∈ c.active then
              weightsOk q &&
                scoresOk q &&
                decide (1 ≤ c.weights q (c.prev q) + c.eps) &&
                decide ((∑ k, c.weights q k) = 1)
            else
              true) = true := by
      simpa [checkSoftmaxMarginCert, weightsOk, scoresOk] using hcheck
    exact (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hcheck''
  have hqchecks {q} (hq : q ∈ c.active) :
      weightsOk q = true ∧
        scoresOk q = true ∧
          decide (1 ≤ c.weights q (c.prev q) + c.eps) = true ∧
            decide ((∑ k, c.weights q k) = 1) = true := by
    have hqall' := hqall q (by simp)
    have hqall'' :
        weightsOk q &&
          scoresOk q &&
          decide (1 ≤ c.weights q (c.prev q) + c.eps) &&
          decide ((∑ k, c.weights q k) = 1) = true := by
      simpa [hq] using hqall'
    simpa [Bool.and_eq_true, and_assoc] using hqall''
  refine
    { score_margin := ?_
      nonneg := ?_
      sum_one := ?_
      prev_large := ?_
      other_le := ?_ }
  · intro q hq k hk
    rcases hqchecks hq with ⟨_, hscore, _, _⟩
    have hscoreall :=
      (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hscore
    have hscorek := hscoreall k (by simp)
    have hscorek' :
        decide (c.scores q k + c.margin ≤ c.scores q (c.prev q)) = true := by
      simpa [hk] using hscorek
    simpa [decide_eq_true_iff] using hscorek'
  · intro q hq k
    rcases hqchecks hq with ⟨hweights, _, _, _⟩
    have hweightsall :=
      (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hweights
    have hweightsk := hweightsall k (by simp)
    have hweightsk' :
        decide (0 ≤ c.weights q k) = true ∧
          (if k = c.prev q then
            true
          else
            decide (c.weights q k ≤ c.eps)) = true := by
      simpa [Bool.and_eq_true] using hweightsk
    simpa [decide_eq_true_iff] using hweightsk'.1
  · intro q hq
    rcases hqchecks hq with ⟨_, _, _, hsum⟩
    simpa [decide_eq_true_iff] using hsum
  · intro q hq
    rcases hqchecks hq with ⟨_, _, hprev, _⟩
    simpa [decide_eq_true_iff] using hprev
  · intro q hq k hk
    rcases hqchecks hq with ⟨hweights, _, _, _⟩
    have hweightsall :=
      (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hweights
    have hweightsk := hweightsall k (by simp)
    have hweightsk' :
        decide (0 ≤ c.weights q k) = true ∧
          (if k = c.prev q then
            true
          else
            decide (c.weights q k ≤ c.eps)) = true := by
      simpa [Bool.and_eq_true] using hweightsk
    have hother :
        decide (c.weights q k ≤ c.eps) = true := by
      simpa [hk] using hweightsk'.2
    simpa [decide_eq_true_iff] using hother

end Circuit

end Nfp
