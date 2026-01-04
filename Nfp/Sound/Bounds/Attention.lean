-- SPDX-License-Identifier: AGPL-3.0-or-later

import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Algebra.Order.Ring.Rat
import Mathlib.Data.Rat.Cast.Order
import Mathlib.Data.Real.Basic
import Nfp.Circuit.Layers.Softmax
import Nfp.Model.Gpt2
import Nfp.Sound.Bounds.LayerNorm
import Nfp.Sound.Bounds.MatrixNorm
import Nfp.Sound.Bounds.Mlp

/-!
Interval bounds for multi-head attention and transformer layers.
-/

namespace Nfp

namespace Sound

namespace Bounds

open scoped BigOperators

/-- Real-valued attention output for a query token and model coordinate. -/
noncomputable def attentionOutputReal {seq dModel dHead numHeads : Nat} [NeZero seq]
    (eps : Rat) (ln1Gamma ln1Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (scores : Fin numHeads → Fin seq → Fin seq → Real)
    (x : Fin seq → Fin dModel → Real)
    (q : Fin seq) (i : Fin dModel) : Real :=
  let lnOut : Fin seq → Fin dModel → Real := fun k j =>
    layerNormRealOfReal eps ln1Gamma ln1Beta (x k) j
  let headValue : Fin numHeads → Fin seq → Fin dHead → Real := fun h k d =>
    dotProduct (fun j => ((heads h).wv j d : Real)) (lnOut k) + (heads h).bv d
  let headWeights : Fin numHeads → Fin seq → Fin seq → Real := fun h q k =>
    Circuit.softmax (scores h q) k
  let headOutput : Fin numHeads → Fin seq → Fin dHead → Real := fun h q d =>
    dotProduct (headWeights h q) (fun k => headValue h k d)
  let headProj : Fin numHeads → Fin seq → Fin dModel → Real := fun h q j =>
    dotProduct (fun d => ((heads h).wo j d : Real)) (fun d => headOutput h q d)
  (∑ h, headProj h q i) + (attnBias i : Real)

/-- Unfolding lemma for `attentionOutputReal`. -/
theorem attentionOutputReal_def {seq dModel dHead numHeads : Nat} [NeZero seq]
    (eps : Rat) (ln1Gamma ln1Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (scores : Fin numHeads → Fin seq → Fin seq → Real)
    (x : Fin seq → Fin dModel → Real)
    (q : Fin seq) (i : Fin dModel) :
    attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i =
      let lnOut : Fin seq → Fin dModel → Real := fun k j =>
        layerNormRealOfReal eps ln1Gamma ln1Beta (x k) j
      let headValue : Fin numHeads → Fin seq → Fin dHead → Real := fun h k d =>
        dotProduct (fun j => ((heads h).wv j d : Real)) (lnOut k) + (heads h).bv d
      let headWeights : Fin numHeads → Fin seq → Fin seq → Real := fun h q k =>
        Circuit.softmax (scores h q) k
      let headOutput : Fin numHeads → Fin seq → Fin dHead → Real := fun h q d =>
        dotProduct (headWeights h q) (fun k => headValue h k d)
      let headProj : Fin numHeads → Fin seq → Fin dModel → Real := fun h q j =>
        dotProduct (fun d => ((heads h).wo j d : Real)) (fun d => headOutput h q d)
      (∑ h, headProj h q i) + (attnBias i : Real) := rfl

/-- Interval bounds for multi-head attention outputs from interval inputs. -/
def attentionOutputBounds {dModel dHead numHeads : Nat}
    (eps : Rat) (ln1Gamma ln1Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (lo hi : Fin dModel → Rat) :
    (Fin dModel → Rat) × (Fin dModel → Rat) :=
  let absBound := intervalAbsBound lo hi
  let ln := layerNormAbsBounds eps ln1Gamma ln1Beta absBound
  let lnLo := ln.1
  let lnHi := ln.2
  let vLo : Fin numHeads → Fin dHead → Rat := fun h d =>
    dotIntervalLower (fun j => (heads h).wv j d) lnLo lnHi + (heads h).bv d
  let vHi : Fin numHeads → Fin dHead → Rat := fun h d =>
    dotIntervalUpper (fun j => (heads h).wv j d) lnLo lnHi + (heads h).bv d
  let headLo : Fin numHeads → Fin dModel → Rat := fun h i =>
    dotIntervalLower (fun d => (heads h).wo i d) (vLo h) (vHi h)
  let headHi : Fin numHeads → Fin dModel → Rat := fun h i =>
    dotIntervalUpper (fun d => (heads h).wo i d) (vLo h) (vHi h)
  let sumLo : Fin dModel → Rat := fun i => ∑ h, headLo h i
  let sumHi : Fin dModel → Rat := fun i => ∑ h, headHi h i
  (fun i => sumLo i + attnBias i, fun i => sumHi i + attnBias i)

/-- `attentionOutputBounds` soundness for real attention outputs. -/
theorem attentionOutputBounds_spec {seq dModel dHead numHeads : Nat} [NeZero seq]
    (eps : Rat) (ln1Gamma ln1Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (scores : Fin numHeads → Fin seq → Fin seq → Real)
    (lo hi : Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps)
    (hlo : ∀ q i, (lo i : Real) ≤ x q i) (hhi : ∀ q i, x q i ≤ (hi i : Real)) :
    let bounds := attentionOutputBounds eps ln1Gamma ln1Beta heads attnBias lo hi
    ∀ q i,
      (bounds.1 i : Real) ≤
          attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i ∧
        attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i ≤
          (bounds.2 i : Real) := by
  classical
  intro bounds q i
  let absBound := intervalAbsBound lo hi
  let lnBounds := layerNormAbsBounds eps ln1Gamma ln1Beta absBound
  let lnLo := lnBounds.1
  let lnHi := lnBounds.2
  let lnOut : Fin seq → Fin dModel → Real := fun k j =>
    layerNormRealOfReal eps ln1Gamma ln1Beta (x k) j
  let vLo : Fin numHeads → Fin dHead → Rat := fun h d =>
    dotIntervalLower (fun j => (heads h).wv j d) lnLo lnHi + (heads h).bv d
  let vHi : Fin numHeads → Fin dHead → Rat := fun h d =>
    dotIntervalUpper (fun j => (heads h).wv j d) lnLo lnHi + (heads h).bv d
  let headLo : Fin numHeads → Fin dModel → Rat := fun h j =>
    dotIntervalLower (fun d => (heads h).wo j d) (vLo h) (vHi h)
  let headHi : Fin numHeads → Fin dModel → Rat := fun h j =>
    dotIntervalUpper (fun d => (heads h).wo j d) (vLo h) (vHi h)
  let sumLo : Fin dModel → Rat := fun j => ∑ h, headLo h j
  let sumHi : Fin dModel → Rat := fun j => ∑ h, headHi h j
  let headValue : Fin numHeads → Fin seq → Fin dHead → Real := fun h k d =>
    dotProduct (fun j => ((heads h).wv j d : Real)) (lnOut k) + (heads h).bv d
  let headWeights : Fin numHeads → Fin seq → Fin seq → Real := fun h q k =>
    Circuit.softmax (scores h q) k
  let headOutput : Fin numHeads → Fin seq → Fin dHead → Real := fun h q d =>
    dotProduct (headWeights h q) (fun k => headValue h k d)
  let headProj : Fin numHeads → Fin seq → Fin dModel → Real := fun h q j =>
    dotProduct (fun d => ((heads h).wo j d : Real)) (fun d => headOutput h q d)
  have habs : ∀ q i, |x q i| ≤ (absBound : Real) := by
    intro q i
    have hbound :
        |x q i| ≤ max |(lo i : Real)| |(hi i : Real)| :=
      abs_le_max_abs_abs_of_interval_real (hlo q i) (hhi q i)
    have hnonempty : (Finset.univ : Finset (Fin dModel)).Nonempty := ⟨i, by simp⟩
    have hsup :
        max |lo i| |hi i| ≤ intervalAbsBound lo hi := by
      have hsup' :
          max |lo i| |hi i| ≤
            (Finset.univ).sup' hnonempty (fun k => max |lo k| |hi k|) := by
        simpa using
          (Finset.le_sup'
            (s := (Finset.univ : Finset (Fin dModel)))
            (f := fun k => max |lo k| |hi k|)
            (by simp : i ∈ (Finset.univ : Finset (Fin dModel))))
      simpa [intervalAbsBound, hnonempty] using hsup'
    have hsup_real :
        max |(lo i : Real)| |(hi i : Real)| ≤ (absBound : Real) := by
      exact_mod_cast hsup
    exact le_trans hbound hsup_real
  have hln_bounds : ∀ q i, (lnLo i : Real) ≤ lnOut q i ∧ lnOut q i ≤ (lnHi i : Real) := by
    intro q i
    have hln := layerNormAbsBounds_spec_real eps ln1Gamma ln1Beta absBound (x q) hne heps
      (fun j => habs q j)
    simpa [lnBounds, lnLo, lnHi, lnOut] using hln i
  have hval_bounds :
      ∀ h k d,
        (vLo h d : Real) ≤ headValue h k d ∧
          headValue h k d ≤ (vHi h d : Real) := by
    intro h k d
    have hln := hln_bounds k
    have hlo' : ∀ j, (lnLo j : Real) ≤ lnOut k j := fun j => (hln j).1
    have hhi' : ∀ j, lnOut k j ≤ (lnHi j : Real) := fun j => (hln j).2
    have hlow :=
      dotIntervalLower_le_dotProduct_real (v := fun j => (heads h).wv j d)
        (lo := lnLo) (hi := lnHi) (x := lnOut k) hlo' hhi'
    have hhigh :=
      dotProduct_le_dotIntervalUpper_real (v := fun j => (heads h).wv j d)
        (lo := lnLo) (hi := lnHi) (x := lnOut k) hlo' hhi'
    have hlow' := add_le_add_right hlow ((heads h).bv d : Real)
    have hhigh' := add_le_add_right hhigh ((heads h).bv d : Real)
    constructor
    · simpa [headValue, vLo, Rat.cast_add] using hlow'
    · simpa [headValue, vHi, Rat.cast_add] using hhigh'
  have weighted_bounds :
      ∀ {lo hi : Real} {vals : Fin seq → Real} {w : Fin seq → Real},
        (∀ k, lo ≤ vals k) → (∀ k, vals k ≤ hi) →
        (∀ k, 0 ≤ w k) → (∑ k, w k = 1) →
        lo ≤ dotProduct w vals ∧ dotProduct w vals ≤ hi := by
    intro lo hi vals w hlo' hhi' hnonneg hsum
    have hsum_lo : ∑ k, w k * lo ≤ ∑ k, w k * vals k := by
      refine Finset.sum_le_sum ?_
      intro k _
      exact mul_le_mul_of_nonneg_left (hlo' k) (hnonneg k)
    have hsum_lo' : ∑ k, w k * lo = lo := by
      calc
        ∑ k, w k * lo = (∑ k, w k) * lo := by
          simpa using
            (Finset.sum_mul (s := (Finset.univ : Finset (Fin seq))) (f := w) (a := lo)).symm
        _ = lo := by simp [hsum]
    have hlow : lo ≤ dotProduct w vals := by
      have hsum_le : lo ≤ ∑ k, w k * vals k := by
        simpa [hsum_lo'] using hsum_lo
      simpa [dotProduct] using hsum_le
    have hsum_hi : ∑ k, w k * vals k ≤ ∑ k, w k * hi := by
      refine Finset.sum_le_sum ?_
      intro k _
      exact mul_le_mul_of_nonneg_left (hhi' k) (hnonneg k)
    have hsum_hi' : ∑ k, w k * hi = hi := by
      calc
        ∑ k, w k * hi = (∑ k, w k) * hi := by
          simpa using
            (Finset.sum_mul (s := (Finset.univ : Finset (Fin seq))) (f := w) (a := hi)).symm
        _ = hi := by simp [hsum]
    have hhigh : dotProduct w vals ≤ hi := by
      have hsum_le : ∑ k, w k * vals k ≤ hi := by
        simpa [hsum_hi'] using hsum_hi
      simpa [dotProduct] using hsum_le
    exact ⟨hlow, hhigh⟩
  have hhead_output_bounds :
      ∀ h q d,
        (vLo h d : Real) ≤ headOutput h q d ∧
          headOutput h q d ≤ (vHi h d : Real) := by
    intro h q d
    have hvals := hval_bounds h
    have hlo' : ∀ k,
        (vLo h d : Real) ≤ headValue h k d := fun k => (hvals k d).1
    have hhi' : ∀ k,
        headValue h k d ≤ (vHi h d : Real) := fun k => (hvals k d).2
    have hnonneg : ∀ k, 0 ≤ headWeights h q k := by
      intro k
      exact Circuit.softmax_nonneg (scores h q) k
    have hsum : ∑ k, headWeights h q k = 1 := by
      simpa [headWeights] using Circuit.softmax_sum_one (scores h q)
    have h := weighted_bounds (lo := (vLo h d : Real)) (hi := (vHi h d : Real))
      (vals := fun k => headValue h k d) (w := headWeights h q)
      hlo' hhi' hnonneg hsum
    simpa [headOutput] using h
  have hproj_bounds :
      ∀ h q i,
        (headLo h i : Real) ≤ headProj h q i ∧ headProj h q i ≤ (headHi h i : Real) := by
    intro h q i
    have hout := hhead_output_bounds h q
    have hlo' : ∀ d,
        (vLo h d : Real) ≤ headOutput h q d := fun d => (hout d).1
    have hhi' : ∀ d,
        headOutput h q d ≤ (vHi h d : Real) := fun d => (hout d).2
    have hlow :=
      dotIntervalLower_le_dotProduct_real (v := fun d => (heads h).wo i d)
        (lo := vLo h) (hi := vHi h)
        (x := fun d => headOutput h q d) hlo' hhi'
    have hhigh :=
      dotProduct_le_dotIntervalUpper_real (v := fun d => (heads h).wo i d)
        (lo := vLo h) (hi := vHi h)
        (x := fun d => headOutput h q d) hlo' hhi'
    constructor
    · simpa [headProj, headLo] using hlow
    · simpa [headProj, headHi] using hhigh
  have hsum_bounds :
      (sumLo i : Real) ≤ ∑ h, headProj h q i ∧
        ∑ h, headProj h q i ≤ (sumHi i : Real) := by
    have hlow : (sumLo i : Real) ≤ ∑ h, headProj h q i := by
      have hsum :=
        Finset.sum_le_sum (s := (Finset.univ : Finset (Fin numHeads)))
          (fun h _ => (hproj_bounds h q i).1)
      simpa [sumLo] using hsum
    have hhigh : ∑ h, headProj h q i ≤ (sumHi i : Real) := by
      have hsum :=
        Finset.sum_le_sum (s := (Finset.univ : Finset (Fin numHeads)))
          (fun h _ => (hproj_bounds h q i).2)
      simpa [sumHi] using hsum
    exact ⟨hlow, hhigh⟩
  have hlow :
      (sumLo i : Real) + (attnBias i : Real) ≤
        (∑ h, headProj h q i) + (attnBias i : Real) := by
    have h := add_le_add_left hsum_bounds.1 (attnBias i : Real)
    simpa [add_comm, add_left_comm, add_assoc] using h
  have hhigh :
      (∑ h, headProj h q i) + (attnBias i : Real) ≤
        (sumHi i : Real) + (attnBias i : Real) := by
    have h := add_le_add_left hsum_bounds.2 (attnBias i : Real)
    simpa [add_comm, add_left_comm, add_assoc] using h
  have hreal :
      attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i =
        (∑ h, headProj h q i) + (attnBias i : Real) := by
    simp [attentionOutputReal, lnOut, headValue, headWeights, headOutput, headProj]
  have hlo :
      (bounds.1 i : Real) ≤
        attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i := by
    simpa [bounds, attentionOutputBounds, absBound, lnBounds, lnLo, lnHi, vLo, vHi, headLo, headHi,
      sumLo, sumHi, hreal] using hlow
  have hhi :
      attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i ≤
        (bounds.2 i : Real) := by
    simpa [bounds, attentionOutputBounds, absBound, lnBounds, lnLo, lnHi, vLo, vHi, headLo, headHi,
      sumLo, sumHi, hreal] using hhigh
  exact And.intro hlo hhi

/-- Interval bounds for the attention residual path. -/
def attentionResidualBounds {dModel dHead numHeads : Nat}
    (eps : Rat) (ln1Gamma ln1Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (lo hi : Fin dModel → Rat) :
    (Fin dModel → Rat) × (Fin dModel → Rat) :=
  let attn := attentionOutputBounds eps ln1Gamma ln1Beta heads attnBias lo hi
  (fun i => lo i + attn.1 i, fun i => hi i + attn.2 i)

/-- `attentionResidualBounds` soundness for attention residual outputs. -/
theorem attentionResidualBounds_spec {seq dModel dHead numHeads : Nat} [NeZero seq]
    (eps : Rat) (ln1Gamma ln1Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (scores : Fin numHeads → Fin seq → Fin seq → Real)
    (lo hi : Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps)
    (hlo : ∀ q i, (lo i : Real) ≤ x q i) (hhi : ∀ q i, x q i ≤ (hi i : Real)) :
    let bounds := attentionResidualBounds eps ln1Gamma ln1Beta heads attnBias lo hi
    ∀ q i,
      (bounds.1 i : Real) ≤
        x q i + attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i ∧
        x q i + attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i ≤
          (bounds.2 i : Real) := by
  classical
  intro bounds q i
  let attn := attentionOutputBounds eps ln1Gamma ln1Beta heads attnBias lo hi
  have hattn :=
    attentionOutputBounds_spec eps ln1Gamma ln1Beta heads attnBias scores lo hi x
      hne heps hlo hhi q i
  have hlow := add_le_add (hlo q i) hattn.1
  have hhigh := add_le_add (hhi q i) hattn.2
  constructor
  · simpa [bounds, attentionResidualBounds, attn, Rat.cast_add] using hlow
  · simpa [bounds, attentionResidualBounds, attn, Rat.cast_add] using hhigh

/-- Interval bounds for a full transformer layer (attention + MLP). -/
def transformerLayerBounds {dModel dHead numHeads hidden : Nat}
    (eps : Rat)
    (ln1Gamma ln1Beta ln2Gamma ln2Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (mlpWIn : Fin dModel → Fin hidden → Rat) (mlpBIn : Fin hidden → Rat)
    (mlpWOut : Fin hidden → Fin dModel → Rat) (mlpBOut : Fin dModel → Rat)
    (lo hi : Fin dModel → Rat) :
    (Fin dModel → Rat) × (Fin dModel → Rat) :=
  let attn := attentionResidualBounds eps ln1Gamma ln1Beta heads attnBias lo hi
  layerNormAbsMlpResidualBounds eps ln2Gamma ln2Beta mlpWIn mlpBIn mlpWOut mlpBOut attn.1 attn.2

/-- `transformerLayerBounds` soundness for full transformer-layer outputs. -/
theorem transformerLayerBounds_spec {seq dModel dHead numHeads hidden : Nat} [NeZero seq]
    (eps : Rat)
    (ln1Gamma ln1Beta ln2Gamma ln2Beta : Fin dModel → Rat)
    (heads : Fin numHeads → Model.Gpt2HeadWeights dModel dHead)
    (attnBias : Fin dModel → Rat)
    (mlpWIn : Fin dModel → Fin hidden → Rat) (mlpBIn : Fin hidden → Rat)
    (mlpWOut : Fin hidden → Fin dModel → Rat) (mlpBOut : Fin dModel → Rat)
    (scores : Fin numHeads → Fin seq → Fin seq → Real)
    (lo hi : Fin dModel → Rat) (x : Fin seq → Fin dModel → Real)
    (hne : dModel ≠ 0) (heps : 0 < eps)
    (hlo : ∀ q i, (lo i : Real) ≤ x q i) (hhi : ∀ q i, x q i ≤ (hi i : Real)) :
    let bounds := transformerLayerBounds eps ln1Gamma ln1Beta ln2Gamma ln2Beta heads attnBias
      mlpWIn mlpBIn mlpWOut mlpBOut lo hi
    ∀ q i,
      (bounds.1 i : Real) ≤
        x q i + attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i +
          mlpReal mlpWIn mlpBIn mlpWOut mlpBOut
            (layerNormRealOfReal eps ln2Gamma ln2Beta
              (fun j =>
                x q j + attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q j)) i ∧
        x q i + attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q i +
          mlpReal mlpWIn mlpBIn mlpWOut mlpBOut
            (layerNormRealOfReal eps ln2Gamma ln2Beta
              (fun j =>
                x q j + attentionOutputReal eps ln1Gamma ln1Beta heads attnBias scores x q j)) i ≤
          (bounds.2 i : Real) := by
  classical
  intro bounds q i
  let attn := attentionResidualBounds eps ln1Gamma ln1Beta heads attnBias lo hi
  have hattn := attentionResidualBounds_spec eps ln1Gamma ln1Beta heads attnBias scores lo hi x
    hne heps hlo hhi q
  have hmlp := layerNormAbsMlpResidualBounds_spec eps ln2Gamma ln2Beta mlpWIn mlpBIn mlpWOut
    mlpBOut attn.1 attn.2 (fun j => x q j + attentionOutputReal eps ln1Gamma ln1Beta heads
      attnBias scores x q j) hne heps
    (fun j => (hattn j).1) (fun j => (hattn j).2)
  have hmlp_i := hmlp i
  simpa [bounds, transformerLayerBounds, attn] using hmlp_i

end Bounds

end Sound

end Nfp
