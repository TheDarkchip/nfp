-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Nfp.Core.Basic
import Nfp.Circuit.Cert
import Nfp.Circuit.Layers.Induction

/-!
Softmax-margin certificates for approximate one-hot attention weights.
-/

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
  have hqall :
      ∀ q ∈ (Finset.univ : Finset (Fin seq)),
        (if q ∈ c.active then
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
          true) = true := by
    have hcheck' : checkSoftmaxMarginCert c = true := hcheck
    have hcheck'' :
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
              true) = true := by
      simpa [checkSoftmaxMarginCert] using hcheck'
    exact (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hcheck''
  refine
    { score_margin := ?_
      nonneg := ?_
      sum_one := ?_
      prev_large := ?_
      other_le := ?_ }
  · intro q hq k hk
    have hqcheck := hqall q (by simp)
    have hqcheck' :
        finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
            if k = c.prev q then
              true
            else
              decide (c.scores q k + c.margin ≤ c.scores q (c.prev q))) = true := by
      have hqcheck'' :
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
            decide ((∑ k, c.weights q k) = 1) = true := by
        simpa [hq] using hqcheck
      have hqcheck''' :
          finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
              decide (0 ≤ c.weights q k) &&
                (if k = c.prev q then
                  true
                else
                  decide (c.weights q k ≤ c.eps))) = true ∧
            finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
              if k = c.prev q then
                true
              else
                decide (c.scores q k + c.margin ≤ c.scores q (c.prev q))) = true ∧
              decide (1 ≤ c.weights q (c.prev q) + c.eps) = true ∧
                decide ((∑ k, c.weights q k) = 1) = true := by
        simpa [Bool.and_eq_true, and_assoc] using hqcheck''
      rcases hqcheck''' with ⟨_, hscoreOk, _, _⟩
      exact hscoreOk
    have hscoreall :=
      (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hqcheck'
    have hscorek := hscoreall k (by simp)
    have hscorek' :
        decide (c.scores q k + c.margin ≤ c.scores q (c.prev q)) = true := by
      simpa [hk] using hscorek
    exact (decide_eq_true_iff).1 hscorek'
  · intro q hq k
    have hqcheck := hqall q (by simp)
    have hqcheck' :
        finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
            decide (0 ≤ c.weights q k) &&
              (if k = c.prev q then
                true
              else
                decide (c.weights q k ≤ c.eps))) = true := by
      have hqcheck'' :
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
            decide ((∑ k, c.weights q k) = 1) = true := by
        simpa [hq] using hqcheck
      have hqcheck''' :
          finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
              decide (0 ≤ c.weights q k) &&
                (if k = c.prev q then
                  true
                else
                  decide (c.weights q k ≤ c.eps))) = true ∧
            finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
              if k = c.prev q then
                true
              else
                decide (c.scores q k + c.margin ≤ c.scores q (c.prev q))) = true ∧
              decide (1 ≤ c.weights q (c.prev q) + c.eps) = true ∧
                decide ((∑ k, c.weights q k) = 1) = true := by
        simpa [Bool.and_eq_true, and_assoc] using hqcheck''
      rcases hqcheck''' with ⟨hweightsOk, _, _, _⟩
      exact hweightsOk
    have hweightsall :=
      (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hqcheck'
    have hweightsk := hweightsall k (by simp)
    have hweightsk' :
        decide (0 ≤ c.weights q k) = true ∧
          (if k = c.prev q then
            true
          else
            decide (c.weights q k ≤ c.eps)) = true := by
      simpa [Bool.and_eq_true] using hweightsk
    exact (decide_eq_true_iff).1 hweightsk'.1
  · intro q hq
    have hqcheck := hqall q (by simp)
    have hqcheck' :
        decide ((∑ k, c.weights q k) = 1) = true := by
      have hqcheck'' :
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
            decide ((∑ k, c.weights q k) = 1) = true := by
        simpa [hq] using hqcheck
      have hqcheck''' :
          finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
              decide (0 ≤ c.weights q k) &&
                (if k = c.prev q then
                  true
                else
                  decide (c.weights q k ≤ c.eps))) = true ∧
            finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
              if k = c.prev q then
                true
              else
                decide (c.scores q k + c.margin ≤ c.scores q (c.prev q))) = true ∧
              decide (1 ≤ c.weights q (c.prev q) + c.eps) = true ∧
                decide ((∑ k, c.weights q k) = 1) = true := by
        simpa [Bool.and_eq_true, and_assoc] using hqcheck''
      rcases hqcheck''' with ⟨_, _, _, hsumOk⟩
      exact hsumOk
    exact (decide_eq_true_iff).1 hqcheck'
  · intro q hq
    have hqcheck := hqall q (by simp)
    have hqcheck' :
        decide (1 ≤ c.weights q (c.prev q) + c.eps) = true := by
      have hqcheck'' :
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
            decide ((∑ k, c.weights q k) = 1) = true := by
        simpa [hq] using hqcheck
      have hqcheck''' :
          finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
              decide (0 ≤ c.weights q k) &&
                (if k = c.prev q then
                  true
                else
                  decide (c.weights q k ≤ c.eps))) = true ∧
            finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
              if k = c.prev q then
                true
              else
                decide (c.scores q k + c.margin ≤ c.scores q (c.prev q))) = true ∧
              decide (1 ≤ c.weights q (c.prev q) + c.eps) = true ∧
                decide ((∑ k, c.weights q k) = 1) = true := by
        simpa [Bool.and_eq_true, and_assoc] using hqcheck''
      rcases hqcheck''' with ⟨_, _, hprevOk, _⟩
      exact hprevOk
    exact (decide_eq_true_iff).1 hqcheck'
  · intro q hq k hk
    have hqcheck := hqall q (by simp)
    have hqcheck' :
        finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
            decide (0 ≤ c.weights q k) &&
              (if k = c.prev q then
                true
              else
                decide (c.weights q k ≤ c.eps))) = true := by
      have hqcheck'' :
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
            decide ((∑ k, c.weights q k) = 1) = true := by
        simpa [hq] using hqcheck
      have hqcheck''' :
          finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
              decide (0 ≤ c.weights q k) &&
                (if k = c.prev q then
                  true
                else
                  decide (c.weights q k ≤ c.eps))) = true ∧
            finsetAll (Finset.univ : Finset (Fin seq)) (fun k =>
              if k = c.prev q then
                true
              else
                decide (c.scores q k + c.margin ≤ c.scores q (c.prev q))) = true ∧
              decide (1 ≤ c.weights q (c.prev q) + c.eps) = true ∧
                decide ((∑ k, c.weights q k) = 1) = true := by
        simpa [Bool.and_eq_true, and_assoc] using hqcheck''
      rcases hqcheck''' with ⟨hweightsOk, _, _, _⟩
      exact hweightsOk
    have hweightsall :=
      (finsetAll_eq_true_iff (s := (Finset.univ : Finset (Fin seq)))).1 hqcheck'
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
    exact (decide_eq_true_iff).1 hother

end Circuit

end Nfp
